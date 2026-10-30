"""Isolated alpha-zero diagnostics, never a production trainer or checkpoint writer."""

from copy import deepcopy
import json

from .diagnostic_protocol import DiagnosticError, DiagnosticOutput, diagnostic_identity, validate_protocol, require_replay_source
from .serialization import atomic_write_json, strict_json_text
from .diagnostic_failures import DiagnosticStageFailure, diagnostic_stage, failure_details


def _matched_runtime(runtime, audit):
    saved = audit["latest_metadata"]["component_metadata"]["runtime"]
    if any(runtime.device_metadata[k] != saved["device_properties"][k] for k in ("name", "capability")):
        raise DiagnosticError("Match the original GPU model/capability before a one-factor comparison")
    if strict_json_text(runtime.effective_backend) != strict_json_text(saved["effective_backend"]):
        raise DiagnosticError("Match the recorded backend/environment settings, including CUBLAS_WORKSPACE_CONFIG")


def _matched_data(data, audit):
    from .checkpoints import data_identity
    if strict_json_text(data_identity(data.metadata.to_dict())) != strict_json_text(audit["latest_metadata"]["data_identity"]):
        raise DiagnosticError("Actual prepared data/order/normalization bindings differ from the original checkpoint")


def run_diagnostic(config, protocol, audit, *, base_dir, output):
    with diagnostic_stage("setup", "diagnostic_setup"):
        return _run_diagnostic(config, protocol, audit, base_dir=base_dir, output=output)


def _run_diagnostic(config, protocol, audit, *, base_dir, output):
    if not isinstance(output, DiagnosticOutput):
        raise DiagnosticError("Diagnostic training requires an exclusively claimed guarded output, never a raw research path")
    with diagnostic_stage("output", "output_ownership"):
        output.guard()
    import torch
    from .config import RunConfig
    from .data import build_loaders
    from .models import build_model, describe_model
    from .optimizers import build_optimizer, describe_optimizer
    from .scheduler import _Schedule
    from .runtime import build_runtime
    from .randomness import RuntimeStreams
    from .checkpoints import copy_clean_model, read_checkpoint, decode_rng
    from .training_contracts import CheckpointReference
    from .artifacts import effective_storage_config, resolve_run_paths
    from .diagnostic_audit import state_digest
    from .diagnostic_observer import BatchObserver
    from .evaluation import classification_loss, evaluate_clean
    from .perturbation import training_forward
    from .metrics import MetricAccumulator, clean_layer_statistics
    from .numerical import require_clean_model, require_finite_optimizer, gradient_nonfinite_names, NumericalFailure
    with diagnostic_stage("preflight", "protocol_validation"):
        protocol = validate_protocol(protocol, config)
    comparison = audit["source_comparison"]
    with diagnostic_stage("preflight", "source_compatibility"):
        require_replay_source(audit, protocol["source_difference_review"])
    document = config.to_dict()
    document["scientific"]["precision"]["mode"] = protocol["precision_mode"]
    effective = RunConfig(document)  # Only an allowed precision declaration changes; never rewrite saved config.
    with diagnostic_stage("setup", "runtime_setup"):
        runtime = build_runtime(effective)
        _matched_runtime(runtime, audit)
    with diagnostic_stage("data", "data_build"):
        data = build_loaders(config, base_dir=base_dir)
        _matched_data(data, audit)
    limit = protocol["max_batches_per_epoch"]
    if limit is not None and (limit > len(data.train) or protocol["epochs"] != 1):
        raise DiagnosticError("A truncated diagnostic is one epoch prefix only; never advance across omitted original batches")
    with diagnostic_stage("setup", "model_construction"):
        model = build_model(config).to(runtime.device)
    saved_components = audit["latest_metadata"]["component_metadata"]
    if strict_json_text(describe_model(model).to_dict()) != strict_json_text(saved_components["model"]):
        raise DiagnosticError("Instantiated model differs from the archived native structure/regularization/initialization descriptor")
    original_optimizer = build_optimizer(model, config)
    original_options = describe_optimizer(original_optimizer).to_dict()
    if strict_json_text({k: v for k, v in original_options.items() if k != "current_learning_rates"}) != strict_json_text(
            {k: v for k, v in saved_components["optimizer"].items() if k != "current_learning_rates"}):
        raise DiagnosticError("Original native optimizer defaults/ownership differ from the archived component metadata")
    options = {k: deepcopy(v) for k, v in original_optimizer.param_groups[0].items() if k != "params"}
    # Do not deepcopy Parameter objects into an optimizer: retain original ownership.
    parameters = list(original_optimizer.param_groups[0]["params"])
    options.update(lr=protocol["learning_rate"], eps=protocol["epsilon"])
    optimizer = torch.optim.RMSprop(parameters, **options)
    del original_optimizer
    if (len(optimizer.param_groups) != 1 or len(optimizer.param_groups[0]["params"]) != len(parameters)
            or any(a is not b for a, b in zip(optimizer.param_groups[0]["params"], parameters)) or optimizer.state):
        raise DiagnosticError("Diagnostic optimizer must own the same clean parameters with fresh state")
    expected_options = {**original_options["effective_options"], "lr": protocol["learning_rate"], "eps": protocol["epsilon"]}
    actual_options = {k: v for k, v in optimizer.param_groups[0].items() if k != "params"}
    if strict_json_text(actual_options) != strict_json_text(expected_options):
        raise DiagnosticError("Instantiated optimizer differs from the single-factor diagnostic declaration")
    training = config.scientific["training"]
    schedule = _Schedule(config.scientific["architecture"], training["scheduler_interval"], 200, training["warmup_epochs"],
        len(data.train) if training["scheduler_interval"] == "optimizer_step" else None,
        (protocol["learning_rate"],), training["warmup_start_factor"], training["cosine_min_lr"])
    initial = protocol["initial_checkpoint"]
    if initial is not None:
        paths = resolve_run_paths(effective_storage_config(config, base_dir), base_dir)
        reference = CheckpointReference("latest", initial["relative_path"], initial["sha256"], 0)
        payload = read_checkpoint(paths, reference, config, audit["saved_run_metadata"]["provenance"])
        if strict_json_text(payload["metadata"]["data_identity"]) != strict_json_text(audit["latest_metadata"]["data_identity"]):
            raise DiagnosticError("Initial and latest prepared-data identities disagree")
        copy_clean_model(model, payload)
        data.restore_next_epoch(payload["continuation"]["data_continuation"])
        streams = RuntimeStreams(effective, runtime.device, initialize=False)
        streams.load_state_dict(decode_rng(payload["continuation"]["rng"]))
        del payload
    else:
        streams = RuntimeStreams(effective, runtime.device, initialize=True)
    streams.claim(effective, runtime.device, continuation=initial is not None)
    position, updates, completed_epochs = 0, 0, 0
    epochs = []
    outcome = {"kind": "optimization_diagnostic_v1", "diagnostic_id": diagnostic_identity(protocol),
        "protocol": protocol, "research_authority": False, "main_v1_completion": False,
        "original_run_identity": audit["identity"], "original_training_transaction": audit["transaction_reference"],
        "actual_diagnostic_provenance": comparison["actual_provenance"], "runtime": runtime.metadata(),
        "diagnostic_process_threads": audit["diagnostic_process_threads"],
        "data": data.metadata.to_dict(), "randomness": streams.metadata(), "model": describe_model(model).to_dict(),
        "initial_state_sha256": state_digest(model.state_dict()), "initial_checkpoint": initial,
        "initialization_evidence": "supplied_epoch_zero" if initial is not None else "seeded_reconstruction; original initial bytes unavailable",
        "original_optimizer_metadata": original_options,
        "effective_optimizer_options": {k: v for k, v in optimizer.param_groups[0].items() if k != "params"},
        "schedule": schedule.to_dict(), "epochs": epochs, "status": "diagnostic_started",
        "failure": None, "secondary_failures": [], "validation_trace_policy": "full_original_epochs_only"}
    epoch, batch_index, observer, iterator = None, None, None, None

    def stage(origin, name, *, batch=None):
        return diagnostic_stage(origin, name, epoch_index=epoch, batch_index=batch)

    def failed(exc):
        record = failure_details(exc, origin="model_computation", stage="diagnostic_loop",
            epoch_index=epoch, batch_index=batch_index)
        if outcome["failure"] is None:
            outcome.update(status="diagnostic_failed", failure=record, error=f"{type(exc).__name__}: {exc}")
        else:
            outcome["secondary_failures"].append(record)

    try:
        with stage("output", "protocol_publication"):
            atomic_write_json(output / "diagnostic_protocol.json", protocol)
        with stage("output", "trace_file"), (output / "batches.jsonl").open("x", encoding="utf-8", newline="\n") as trace:
            def emit(row):
                with diagnostic_stage("output", "batch_trace_publication", epoch_index=row["epoch_index"], batch_index=row["batch_index"]):
                    output.guard()
                    trace.write(json.dumps(row, allow_nan=False, ensure_ascii=False) + "\n")
                    trace.flush()
            with stage("observation", "observer_setup"), BatchObserver(model, optimizer, every=protocol["trace_every"],
                               activation_modules=protocol["activation_modules"], emit=emit) as observer:
                try:
                    for epoch in range(protocol["epochs"]):
                        batch_index = None
                        with stage("model_computation", "epoch_entry"):
                            runtime.require_active(effective)
                        with stage("data", "epoch_order_setup"):
                            data.set_epoch(epoch)
                        with stage("model_computation", "epoch_model_setup"):
                            model.train()
                            metrics = MetricAccumulator(runtime.device)
                        full_epoch = limit is None or limit == len(data.train)
                        with stage("data", "training_iterator"):
                            iterator = iter(data.train)
                        for batch_index in range(len(data.train) if limit is None else limit):
                            with stage("data", "training_batch_read", batch=batch_index):
                                batch = next(iterator)
                            with stage("model_computation", "optimizer_prepare", batch=batch_index):
                                optimizer.zero_grad(set_to_none=True)
                                optimizer.param_groups[0]["lr"] = schedule.rates_at(position)[0]
                            observer.start(epoch, batch_index)
                            with stage("model_computation", "input_transfer", batch=batch_index):
                                images, labels = runtime.batch(batch, 10)
                            with stage("model_computation", "training_forward", batch=batch_index):
                                with runtime.autocast():
                                    logits = training_forward(model, images, alpha=0., noise_generator=streams.noise_generator)
                            # Capture bad logits/BN state even if the real finite-loss
                            # guard rejects this batch before backward.
                            observer.forward(logits, labels, None)
                            with stage("model_computation", "training_loss", batch=batch_index):
                                loss = classification_loss(logits, labels, 10)
                            observer.mean_loss(loss)
                            with stage("model_computation", "backward", batch=batch_index):
                                metrics.add(loss, logits, labels)
                                loss.backward()  # Both approved modes are unscaled; no float16 arm.
                            observer.gradients()
                            with stage("model_computation", "gradient_check", batch=batch_index):
                                bad = gradient_nonfinite_names(model)
                                if bad:
                                    raise NumericalFailure("gradients", "Nonfinite diagnostic gradients", bad)
                            with stage("model_computation", "optimizer_step", batch=batch_index):
                                optimizer.step()
                            with stage("model_computation", "post_update_clean_check", batch=batch_index):
                                require_clean_model(model, runtime.device, stage="diagnostic_post_batch_clean_state")
                            updates += 1
                            if schedule.interval == "optimizer_step":
                                position += 1
                            observer.finish()
                            with stage("model_computation", "optimizer_clear", batch=batch_index):
                                optimizer.zero_grad(set_to_none=True)
                        batch_index = None
                        with stage("data", "training_epoch_exhaustion"):
                            if full_epoch:
                                try:
                                    next(iterator)
                                except StopIteration:
                                    pass
                                else:
                                    raise DiagnosticError("Training iterator exceeded its declared original epoch length")
                            iterator = None  # Release prefix/full-epoch worker iterator before validation.
                        if full_epoch:
                            completed_epochs += 1
                            if schedule.interval == "epoch":
                                position += 1
                        with stage("model_computation", "epoch_exit_check"):
                            require_clean_model(model, runtime.device, stage="diagnostic_epoch_exit")
                            require_finite_optimizer(optimizer, model)
                        with stage("observation", "layer_statistics"):
                            statistics = clean_layer_statistics(model)
                        with stage("model_computation", "clean_validation"):
                            with observer.validation_runtime(runtime, epoch, observe=full_epoch) as validation_runtime:
                                validation = evaluate_clean(model, data.validation, split="validation", config=effective, runtime=validation_runtime)
                        with stage("model_computation", "epoch_metrics"):
                            training_metrics = metrics.finish().to_dict()
                            next_learning_rate = schedule.rates_at(position)[0]
                        epochs.append({"epoch_label": epoch + 1, "full_original_epoch": full_epoch,
                            "training": training_metrics, "clean_validation": validation.to_dict(),
                            "eligible_weight_statistics": [vars(stat) for stat in statistics],
                            "successful_updates": updates, "scheduler_position": position,
                            "next_learning_rate": next_learning_rate})
                        with stage("output", "epoch_publication"):
                            atomic_write_json(output / f"diagnostic_epoch_{epoch + 1:03}.json", epochs[-1])
                    outcome["status"] = "diagnostic_budget_finished"
                except Exception as exc:
                    failed(exc)
                    try:
                        observer.finish(failure=outcome["failure"])
                    except Exception as trace_error:
                        failed(trace_error)
    except Exception as exc:
        failed(exc)
    finally:
        iterator = None
    try:
        with stage("observation", "final_state_digest"):
            outcome["final_state_sha256"] = state_digest(model.state_dict())
    except Exception as exc:
        outcome["final_state_sha256"] = None
        failed(exc)
    outcome.update(completed_full_original_epochs=completed_epochs, successful_updates=updates,
                   scheduler_position=position, observation_errors=[] if observer is None else observer.observation_errors)
    try:
        with stage("output", "diagnostic_publication"):
            atomic_write_json(output / "diagnostic.json", outcome)
    except DiagnosticStageFailure as exc:
        exc.record["prior_failure"] = outcome["failure"]
        exc.record["secondary_failures"] = outcome["secondary_failures"]
        raise
    return outcome


def recheck_validation(config, audit, *, base_dir, source_review):
    with diagnostic_stage("setup", "validation_recheck_setup"):
        return _recheck_validation(config, audit, base_dir=base_dir, source_review=source_review)


def _recheck_validation(config, audit, *, base_dir, source_review):
    """Load an independent best model and reproduce full original validation counts."""
    from .artifacts import effective_storage_config, resolve_run_paths
    from .checkpoints import read_checkpoint, copy_clean_model
    from .training_contracts import CheckpointReference
    from .runtime import build_runtime
    from .models import build_model
    from .data import build_loaders
    from .evaluation import evaluate_clean
    from .evaluation_state import CleanStateBaseline
    require_replay_source(audit, source_review)
    if audit["transaction"]["status"] not in ("completed", "training_completed") or audit["receipt"]["completed_epochs"] != 200:
        raise DiagnosticError("Published-validation reproduction requires verified full-budget training completion")
    ref = audit["checkpoints"]["best"]
    if ref is None:
        raise DiagnosticError("No immutable selected-best checkpoint")
    paths = resolve_run_paths(effective_storage_config(config, base_dir), base_dir)
    with diagnostic_stage("checkpoint", "best_checkpoint_read"):
        payload = read_checkpoint(paths, CheckpointReference(**ref), config, audit["saved_run_metadata"]["provenance"])
    if strict_json_text(payload["metadata"]["data_identity"]) != strict_json_text(audit["latest_metadata"]["data_identity"]):
        raise DiagnosticError("Best and latest prepared-data identities disagree")
    with diagnostic_stage("setup", "runtime_setup"):
        runtime = build_runtime(config, evaluation_only=True)
        _matched_runtime(runtime, audit)
    with diagnostic_stage("setup", "best_model_construction"):
        model = build_model(config).to(runtime.device)
        copy_clean_model(model, payload)
    with diagnostic_stage("observation", "best_state_baseline"):
        baseline = CleanStateBaseline(model)
    with diagnostic_stage("data", "data_build"):
        data = build_loaders(config, base_dir=base_dir)
        _matched_data(data, audit)
    selected = audit["transaction"]["best"]
    with diagnostic_stage("model_computation", "best_clean_validation", epoch_index=selected["epoch_index"]):
        with baseline.realization():
            observed = evaluate_clean(model, data.validation, split="validation", config=config, runtime=runtime)
    expected = audit["epochs"][selected["epoch_index"]]["document"]["record"]["validation"]
    return {"kind": "original_best_validation_recheck", "research_authority": False,
        "original_run_identity": audit["identity"], "training_transaction": audit["transaction_reference"],
        "source_difference_review": source_review,
        "actual_diagnostic_provenance": audit["source_comparison"]["actual_provenance"],
        "checkpoint": ref, "expected": expected, "observed": observed.to_dict(),
        "exact_correct_and_examples_match": (observed.correct, observed.examples) == (expected["correct"], expected["examples"]),
        "loss_difference": observed.loss - expected["loss"], "runtime": runtime.metadata(),
        "original_artifact_writes": False, "test_used": False}
