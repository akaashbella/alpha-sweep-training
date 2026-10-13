"""One full-budget training engine; durable persistence is a required service."""

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import math
import time

import torch
from torch import nn
from torch.optim import Optimizer

from .artifacts import RunHandle
from .component_contracts import ComponentError
from .config import RunConfig
from .data import DataLoaders, TrainingView
from .evaluation import EvaluationFailure, classification_loss, evaluate_clean
from .identity import identity_metadata
from .metrics import (
    AmpSkip, BestSelection, Counters, EpochRecord, FailureRecord, MetricAccumulator,
    RecordContext, TestRecord, append_train_log, clean_layer_statistics,
)
from .models import build_model, describe_model
from .numerical import NumericalFailure, gradient_nonfinite_names, require_clean_model, require_finite_optimizer
from .optimizers import describe_optimizer, validate_optimizer
from .perturbation import training_forward
from .randomness import RuntimeStreams
from .runtime import RuntimePolicy
from .scheduler import WarmupCosineScheduler
from .spec import BATCH_SIZE, EPOCHS, TEST_SIZE, TRAIN_SIZE, VALIDATION_SIZE
from .training_contracts import (
    BestLoadReceipt, CommitReceipt, CompletionReceipt, ContinuationState, EpochBoundary,
    PersistenceContractError, TrainingPersistence, TrainingSnapshot, owned_cpu_snapshot, require_persistence,
)


class TrainingRunError(RuntimeError):
    """Computation stopped; the last acknowledged checkpoint remains authoritative."""
    def __init__(self, failure: FailureRecord, *, failure_published: bool, publication_error: str | None):
        self.failure = failure
        self.failure_published = failure_published
        self.publication_error = publication_error
        super().__init__(f"{failure.status} at {failure.phase}: {failure.reason}"
                         + (f"; failure publication also failed: {publication_error}" if publication_error else ""))


@dataclass(frozen=True)
class TrainingResult:
    status: str
    counters: Counters
    best: BestSelection
    test: TestRecord
    final_commit: CommitReceipt
    completion: CompletionReceipt


def _validate_data(data: DataLoaders, config: RunConfig) -> None:
    if not isinstance(data, DataLoaders):
        raise ComponentError("Training requires the existing build_loaders DataLoaders")
    science, metadata = config.scientific, data.metadata
    counts = metadata.counts
    if (metadata.dataset != science["dataset"] or metadata.num_classes != science["model"]["num_classes"]
            or metadata.training_seed != science["training_seed"]
            or metadata.split_identity != science["data"]["split"]["identity"]
            or f"normalization-v1:{metadata.normalization_identity}" != science["data"]["normalization"]["source"]
            or metadata.drop_last != science["training"]["drop_last"]):
        raise ComponentError("Prepared loader scientific identities do not match the training configuration")
    if (not isinstance(data.train.dataset, TrainingView) or data.train.sampler is not data._sampler
            or data.train.drop_last != metadata.drop_last or data.train.batch_size != BATCH_SIZE
            or counts.batch_size != BATCH_SIZE or counts.official_training != TRAIN_SIZE + VALIDATION_SIZE
            or counts.training_partition != TRAIN_SIZE or len(data.train.dataset) != TRAIN_SIZE
            or counts.validation_partition != VALIDATION_SIZE or len(data.validation.dataset) != VALIDATION_SIZE
            or counts.test_partition != TEST_SIZE or len(data.test.dataset) != TEST_SIZE
            or counts.training_batches != len(data.train)
            or counts.validation_batches != len(data.validation) or counts.test_batches != len(data.test)):
        raise ComponentError("Loader coverage/batch metadata violates the full CIFAR contract")
    processed = counts.training_batches * BATCH_SIZE if metadata.drop_last else TRAIN_SIZE
    if counts.training_processed_per_epoch != processed:
        raise ComponentError("Loader processed-example count disagrees with declared drop_last")


def _validate_boundary(counters: Counters, best: BestSelection | None, data: DataLoaders,
                       scheduler: WarmupCosineScheduler) -> None:
    counters.validate(complete_boundary=True)
    if (counters.attempted_batches != counters.completed_epochs * data.metadata.counts.training_batches
            or counters.processed_examples != counters.completed_epochs * data.metadata.counts.training_processed_per_epoch):
        raise ComponentError("Completed-epoch counters disagree with full loader coverage")
    expected_position = counters.completed_epochs if scheduler.interval == "epoch" else counters.successful_updates
    if scheduler.completed_units != expected_position:
        raise ComponentError("Scheduler position must match its own completed-epoch/successful-update counter")
    if best is None:
        if counters.completed_epochs:
            raise ComponentError("Completed epochs require best-validation metadata")
    else:
        best.validate(counters.completed_epochs)


def _backward_and_update(loss: torch.Tensor, model: nn.Module, optimizer: Optimizer,
                         runtime: RuntimePolicy) -> tuple[bool, tuple[str, ...], float | None, float | None]:
    scaler = runtime.scaler
    if getattr(optimizer, "_step_supports_amp_scaling", False):
        raise ComponentError("Only the pinned native unfused optimizer/GradScaler call contract is supported")
    scale_before = None if scaler is None else scaler.get_scale()
    if scaler is None:
        loss.backward()
    else:
        if not math.isfinite(scale_before) or scale_before <= 0:
            raise NumericalFailure("grad_scaler", "Invalid gradient scale before backward")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    bad = gradient_nonfinite_names(model)
    if bad and scaler is None:
        raise NumericalFailure("gradients", "Nonfinite unscaled gradients without float16 GradScaler", bad)
    calls = 0

    def observed_step(_optimizer, _args, _kwargs) -> None:
        nonlocal calls
        calls += 1

    hook = optimizer.register_step_post_hook(observed_step)
    try:
        if scaler is None:
            optimizer.step()
        else:
            # Native SGD/RMSprop/AdamW steps return None even when they update.
            # The public post-hook fires only if GradScaler invoked that step.
            scaler.step(optimizer)
            scaler.update()
    finally:
        hook.remove()
    if calls != (0 if bad else 1):
        raise NumericalFailure("optimizer_update", "Optimizer execution disagrees with finite-gradient/AMP-skip checks", bad)
    scale_after = None if scaler is None else scaler.get_scale()
    if scale_after is not None and (not math.isfinite(scale_after) or scale_after <= 0):
        raise NumericalFailure("grad_scaler", "Gradient scale became nonfinite or zero")
    require_clean_model(model, runtime.device, stage="post_batch_clean_state")
    return calls == 1, bad, scale_before, scale_after


def _snapshot(config: RunConfig, run: RunHandle, data: DataLoaders, model: nn.Module,
              optimizer: Optimizer, scheduler: WarmupCosineScheduler, runtime: RuntimePolicy,
              streams: RuntimeStreams, counters: Counters, best: BestSelection) -> TrainingSnapshot:
    run.require_active(config)
    require_clean_model(model, runtime.device, stage="snapshot_clean_state")
    require_finite_optimizer(optimizer, model)
    # Detached CPU copies must be complete before training may mutate anything.
    return TrainingSnapshot(
        config=run.config.to_dict(), provenance=deepcopy(run.provenance),
        clean_model=owned_cpu_snapshot(model.state_dict()),
        optimizer=owned_cpu_snapshot(optimizer.state_dict()), scheduler=deepcopy(scheduler.state_dict()),
        scaler=None if runtime.scaler is None else deepcopy(runtime.scaler.state_dict()),
        rng=streams.state_dict(), data_continuation=data.continuation_metadata(counters.completed_epochs),
        counters=counters, best=best,
        component_metadata={"model": describe_model(model).to_dict(), "optimizer": describe_optimizer(optimizer).to_dict(),
                            "runtime": runtime.metadata(), "randomness": streams.metadata(),
                            "data": data.metadata.to_dict(), "scheduler": scheduler.metadata()})


def train_run(config: RunConfig, data: DataLoaders, model: nn.Module, optimizer: Optimizer,
              scheduler: WarmupCosineScheduler, run: RunHandle, *, runtime: RuntimePolicy,
              streams: RuntimeStreams, persistence: TrainingPersistence,
              continuation: ContinuationState | None = None) -> TrainingResult:
    """Train remaining original epochs, then require persisted-best full test.

    No command wraps this function and no default persistence service exists.
    Step 5 must restore all live components before passing ContinuationState.
    Returning success requires durable epoch/final/test/summary acknowledgements.
    """
    require_persistence(persistence)  # Fail before computation if Step 5 is absent.
    config.require_research_ready()
    run.require_active(config)
    if continuation is not None and not isinstance(continuation, ContinuationState):
        raise ComponentError("Continuation requires Step 5's validated ContinuationState")
    if (run.operation == "continue") != (continuation is not None):
        raise ComponentError("Continue requires validated restored state; new/overwrite requires fresh components")
    if not isinstance(data, DataLoaders):
        raise ComponentError("Training requires the existing build_loaders DataLoaders")
    context = RecordContext.from_run(run, data.metadata)
    counters = Counters() if continuation is None else continuation.counters
    best = None if continuation is None else continuation.best
    last_commit = None if continuation is None else continuation.receipt
    # Establish the last acknowledged boundary before any live-component check.
    # A later preflight failure must still report the saved durable progress.
    if last_commit is not None:
        if not isinstance(last_commit, CommitReceipt):
            raise PersistenceContractError("Continuation requires a durable CommitReceipt")
        last_commit.validate(config, counters, best)
    durable_epochs = 0 if last_commit is None else last_commit.completed_epochs
    epoch_index: int | None = None
    batch_index: int | None = None
    phase = "component_preflight"
    commit_outcome = "no_commit_in_flight"
    modes = tuple((module, module.training) for module in model.modules())
    try:
        _validate_data(data, config)
        runtime.require_active(config)
        if validate_optimizer(optimizer, config).model() is not model:
            raise ComponentError("Trainer and optimizer must share the same original clean model")
        scheduler.require_binding(optimizer, config, len(data.train))
        require_clean_model(model, runtime.device, stage="training_entry")
        require_finite_optimizer(optimizer, model)
        _validate_boundary(counters, best, data, scheduler)
        if continuation is None:
            if optimizer.state or scheduler.completed_units != 0:
                raise ComponentError("A fresh run requires fresh optimizer state and scheduler position zero")
            if runtime.scaler is not None and (runtime.scaler.state_dict()["_growth_tracker"] != 0
                                               or runtime.scaler.get_scale() != 65536.0):
                raise ComponentError("A fresh run requires the pinned fresh GradScaler state")
        else:
            if data.restore_next_epoch(continuation.data_continuation) != counters.completed_epochs:
                raise ComponentError("Restored data epoch differs from the validated training boundary")
        streams.claim(config, runtime.device, continuation=continuation is not None)
        with streams.preserve():
            append_train_log(run, context, "training_entry", {
                "status": "trained_pending_test" if durable_epochs == EPOCHS else "training",
                "durable_completed_epochs": durable_epochs, "runtime": runtime.metadata(),
                "randomness": streams.metadata(), "counters": asdict(counters)})
        science = config.scientific
        for epoch_index in range(counters.completed_epochs, EPOCHS):
            phase, batch_index = "epoch_start", None
            runtime.require_active(config)
            run.require_active()
            data.set_epoch(epoch_index)
            model.train()
            started = time.perf_counter()
            before = counters
            scheduler_before = scheduler.completed_units
            first_lr = scheduler.current_learning_rates[0]
            last_lr = first_lr
            accumulator = MetricAccumulator(runtime.device)
            skips = []
            for batch_index, batch in enumerate(data.train):
                phase = "training_batch"
                if batch_index >= data.metadata.counts.training_batches:
                    raise ComponentError("Training loader exceeded its declared full-epoch batch count")
                counters = replace(counters, attempted_batches=counters.attempted_batches + 1)
                optimizer.zero_grad(set_to_none=True)
                images, labels = runtime.batch(batch, science["model"]["num_classes"])
                remaining = data.metadata.counts.training_processed_per_epoch - accumulator.examples
                if labels.numel() != min(BATCH_SIZE, remaining):
                    raise ComponentError("Training batch size disagrees with full coverage and declared drop_last")
                last_lr = float(optimizer.param_groups[0]["lr"])
                with runtime.autocast():
                    logits = training_forward(model, images, alpha=science["alpha"],
                                              noise_generator=streams.noise_generator)
                loss = classification_loss(logits, labels, science["model"]["num_classes"])
                accumulator.add(loss, logits, labels)
                counters = replace(counters, processed_examples=counters.processed_examples + labels.numel())
                phase = "backward_optimizer_update"
                updated, bad, scale_before, scale_after = _backward_and_update(loss, model, optimizer, runtime)
                if updated:
                    counters = replace(counters, successful_updates=counters.successful_updates + 1)
                    if scheduler.interval == "optimizer_step":
                        scheduler.step()
                else:
                    counters = replace(counters, skipped_updates=counters.skipped_updates + 1)
                    skip = AmpSkip(batch_index, bad, scale_before, scale_after)
                    skips.append(skip)
                    # Retain the diagnostic even if this epoch later fails. Only
                    # the eventual epoch transaction makes its counters durable.
                    with streams.preserve():
                        append_train_log(run, context, "amp_skipped_update", {
                            "status": "training", "epoch_index": epoch_index,
                            "durable_completed_epochs": durable_epochs,
                            "counters": asdict(counters), **asdict(skip)})
                optimizer.zero_grad(set_to_none=True)
                del loss, logits, images, labels
            phase, batch_index = "epoch_coverage", None
            training_metrics = accumulator.finish()
            if (training_metrics.examples != data.metadata.counts.training_processed_per_epoch
                    or training_metrics.batches != data.metadata.counts.training_batches):
                raise ComponentError("Training epoch did not process its full declared loader coverage")
            if scheduler.interval == "epoch":
                scheduler.step()
            phase = "clean_layer_statistics"
            require_clean_model(model, runtime.device, stage=phase)
            require_finite_optimizer(optimizer, model)
            statistics = clean_layer_statistics(model)
            phase = "clean_validation"
            validation = evaluate_clean(model, data.validation, split="validation", config=config, runtime=runtime)
            selected = best is None or validation.correct > best.correct
            if selected:
                best = BestSelection(epoch_index, validation.correct)
            counters = replace(counters, completed_epochs=epoch_index + 1)
            _validate_boundary(counters, best, data, scheduler)
            record = EpochRecord(
                epoch_index, counters.completed_epochs,
                "clean_training" if science["alpha"] == 0 else "perturbed_training",
                training_metrics, validation, counters.attempted_batches - before.attempted_batches,
                counters.successful_updates - before.successful_updates,
                counters.skipped_updates - before.skipped_updates, counters,
                first_lr, last_lr, scheduler.current_learning_rates[0], scheduler_before,
                scheduler.completed_units, scheduler.interval, time.perf_counter() - started,
                selected, best, tuple(skips), "trained_pending_test" if counters.completed_epochs == EPOCHS else "training")
            phase = "epoch_snapshot"
            with streams.preserve():
                snapshot = _snapshot(config, run, data, model, optimizer, scheduler, runtime, streams, counters, best)
                boundary = EpochBoundary(context, record, statistics, snapshot,
                                         None if last_commit is None else last_commit.commit_id)
                phase, commit_outcome = "epoch_commit", "unknown_until_acknowledged"
                run.require_active()
                receipt = persistence.commit_epoch(run, boundary)
                if not isinstance(receipt, CommitReceipt):
                    raise PersistenceContractError("commit_epoch must return a durable CommitReceipt")
                receipt.validate(config, counters, best)
                if last_commit is not None and (receipt.commit_id == last_commit.commit_id
                        or (not selected and receipt.best != last_commit.best)):
                    raise PersistenceContractError("Commit must advance its ID and preserve earlier best on validation ties/non-improvement")
                last_commit, durable_epochs = receipt, counters.completed_epochs
                commit_outcome = "acknowledged"
                del boundary, snapshot
                phase = "epoch_diagnostic_log"
                append_train_log(run, context, "epoch_committed", {"commit_id": receipt.commit_id, **record.to_dict()})
            commit_outcome = "no_commit_in_flight"

        # A separate evaluation model prevents best weights from ever being paired
        # with final optimizer state in the live continuation model or latest.
        phase, epoch_index, batch_index = "load_persisted_best", None, None
        if last_commit is None or best is None or last_commit.best is None:
            raise PersistenceContractError("Full-budget clean test requires a durable final boundary and persisted best")
        last_commit.validate(config, counters, best)
        with streams.preserve():
            append_train_log(run, context, "best_test_entry", {
                "status": "evaluating", "durable_completed_epochs": durable_epochs,
                "final_commit_id": last_commit.commit_id,
                "best_checkpoint": asdict(last_commit.best), "best": asdict(best)})
            evaluation_model = build_model(config).to(device=runtime.device, dtype=torch.float32)
            original_parameters = dict(evaluation_model.named_parameters())
            original_buffers = dict(evaluation_model.named_buffers())
            run.require_active()
            loaded = persistence.load_best(run, evaluation_model, last_commit.best, best)
            if (not isinstance(loaded, BestLoadReceipt) or loaded.identity != identity_metadata(config)
                    or loaded.checkpoint != last_commit.best):
                raise PersistenceContractError("load_best did not acknowledge the persisted validation-selected checkpoint")
            current = dict(evaluation_model.named_parameters())
            buffers = dict(evaluation_model.named_buffers())
            if (set(current) != set(original_parameters) or any(current[name] is not p for name, p in original_parameters.items())
                    or set(buffers) != set(original_buffers) or any(buffers[name] is not b for name, b in original_buffers.items())):
                raise PersistenceContractError("Best loading must copy clean values into existing parameter/buffer bindings")
        phase = "clean_test"
        runtime.require_active(config)
        started = time.perf_counter()
        test_metrics = evaluate_clean(evaluation_model, data.test, split="test", config=config, runtime=runtime)
        test = TestRecord(test_metrics, best, asdict(last_commit.best), last_commit.commit_id,
                          time.perf_counter() - started)
        del evaluation_model, original_parameters, original_buffers, current, buffers
        phase, commit_outcome = "completion_commit", "unknown_until_acknowledged"
        with streams.preserve():
            run.require_active()
            completion = persistence.publish_completion(run, context, test, last_commit)
            if not isinstance(completion, CompletionReceipt):
                raise PersistenceContractError("publish_completion must return a durable CompletionReceipt")
            completion.validate(config, last_commit)
        # No further fallible logging after this durable completion acknowledgement.
        return TrainingResult("completed", counters, best, test, last_commit, completion)
    except BaseException as exc:
        # Avoid CUDA or foreign optimizer calls while handling an execution
        # failure: gradients never belong to a checkpoint or failure record.
        for parameter in model.parameters():
            parameter.grad = None
        cause = exc.cause if isinstance(exc, EvaluationFailure) else exc
        failed_batch = exc.batch_index if isinstance(exc, EvaluationFailure) else batch_index
        status = ("failed_numerical" if isinstance(cause, NumericalFailure) else
                  "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed_execution")
        failure = FailureRecord(status, phase,
                                str(exc), type(cause).__name__, epoch_index, failed_batch, counters,
                                durable_epochs, None if last_commit is None else last_commit.commit_id,
                                cause.tensors if isinstance(cause, NumericalFailure) else (),
                                numerical_stage=cause.stage if isinstance(cause, NumericalFailure) else None,
                                commit_outcome=commit_outcome)
        published, publication_error = False, None
        try:
            run.require_active()
            # CPU-only failure record publication must remain possible after a CUDA
            # failure. Training has stopped; never capture/save the failed live state.
            persistence.publish_failure(run, context, failure)
            published = True
            append_train_log(run, context, "failure", asdict(failure))
        except BaseException as publication_exc:
            publication_error = f"{type(publication_exc).__name__}: {publication_exc}"
        raise TrainingRunError(failure, failure_published=published, publication_error=publication_error) from exc
    finally:
        for module, training in modes:
            module.training = training
