"""Exactly one task per invocation. No signals, retry loops or job orchestration."""

from dataclasses import asdict, dataclass
from enum import IntEnum

from .artifacts import RunDirectoryError, effective_storage_config, open_run
from .config import ConfigError
from .coverage import TaskObservation, inspect_task
from .manifests import Manifest, Task, ManifestError, _same, manifest_from_dict, select_task, task_config


class ExitCode(IntEnum):
    TRAINING_COMPLETED = 0
    INTERRUPTED = 10
    NUMERICAL_FAILURE = 20
    EXECUTION_FAILURE = 21
    REJECTED = 22


@dataclass(frozen=True)
class TaskOutcome:
    run_id: str
    scientific_fingerprint: str
    outcome: str
    exit_code: ExitCode
    operation: str
    observation: TaskObservation | None = None
    completed_epochs: int | None = None
    diagnostic: str | None = None
    failure: dict | None = None
    failure_published: bool | None = None
    lifecycle: str = "training"

    def to_dict(self):
        return asdict(self)


def dispatch(observation: TaskObservation, *, operation: str = "auto") -> str:
    """Resolve one operation without changing anything; locks always block."""
    if operation not in ("auto", "new", "continue", "recover", "overwrite"):
        raise ManifestError("Unknown task operation")
    if observation.condition in ("locked", "unstable", "inspection_error"):
        raise RunDirectoryError(f"Task is {observation.condition}: {observation.diagnostic}")
    if operation == "overwrite":
        if observation.condition == "missing":
            raise RunDirectoryError("Wholly missing run needs new, not overwrite")
        return "overwrite"
    if observation.condition == "missing":
        if operation not in ("auto", "new"):
            raise RunDirectoryError("Wholly missing task is only eligible for new")
        return "new"
    if observation.condition != "ok":
        raise RunDirectoryError(f"Task inspection rejected {observation.condition}: {observation.diagnostic}")
    if operation == "recover":
        return "recover"
    if observation.training_complete:
        if operation in ("auto", "continue"):
            return "already_training_completed"
        raise RunDirectoryError("Task is already complete; new cannot replace it")
    if observation.full_budget_committed and operation in ("auto", "continue"):
        return "finalize_training"
    if observation.terminal_numerical:
        return "blocked_numerical"
    if operation == "new":
        raise RunDirectoryError("Existing task cannot be initialized as new")
    if observation.execution_failed and operation != "continue":
        return "blocked_execution"
    if observation.stored_status in ("initialized", "training", "interrupted", "trained_pending_test", "evaluating", "failed_execution"):
        return "continue"
    raise RunDirectoryError("Unsupported stored status; preserve evidence")


def execute_task(manifest: Manifest, task: Task, *, base_dir, source_root, overrides=None,
                 operation: str = "auto", overwrite_run_id: str | None = None,
                 numerical_reference: dict | None = None, stop=None) -> TaskOutcome:
    """Validate membership/dispatch, then construct and run exactly one selected task.

    ``continue`` explicitly authorizes retrying an execution failure; it cannot
    override terminal numerical evidence. ``recover`` only repairs publication
    (optionally reconciles one exact numerical reference), never calls train_run.
    Step 7 may pass StopRequest; this module registers no signal handlers.
    """
    selected_operation, observation = operation, None
    stage, acknowledged_completion = "declaration", None
    common = {"run_id": task.run_id, "scientific_fingerprint": task.fingerprint}
    try:
        manifest = manifest_from_dict(manifest.to_dict())
        selected = select_task(manifest, run_id=task.run_id)
        _same(task.to_dict(), selected.to_dict(), "selected task/parent manifest membership")
        config = effective_storage_config(task_config(selected, overrides), base_dir)
        if type(operation) is not str or operation not in ("auto", "new", "continue", "recover", "overwrite"):
            raise ManifestError("Unknown task operation")
        if operation == "overwrite":
            if overwrite_run_id != task.run_id:
                raise ManifestError("Overwrite requires --overwrite-run-id equal to the exact selected run ID")
        elif overwrite_run_id is not None:
            raise ManifestError("overwrite_run_id only applies to overwrite")
        if numerical_reference is not None and operation != "recover":
            raise ManifestError("Numerical evidence selection only applies to publication recovery")
        from .interruption import StopRequest
        if stop is not None and not isinstance(stop, StopRequest):
            raise ManifestError("stop must be the existing StopRequest")
        if operation == "recover" and stop is not None:
            raise ManifestError("StopRequest applies to scientific execution, not publication repair")
        stage = "provenance"
        from .execution_provenance import collect_provenance
        provenance = collect_provenance(source_root, attempt={"grid_id": manifest.grid_id,
            "manifest_document_sha256": manifest.document_sha256, "global_index": task.global_index,
            "requested_operation": operation})
        # Stored-state preflight occurs before loaders/model/runtime construction.
        observation = inspect_task(selected, base_dir=base_dir, overrides=overrides, provenance=provenance)
        selected_operation = dispatch(observation, operation=operation)
        if selected_operation == "already_training_completed":
            return TaskOutcome(**common, outcome="already_training_completed", exit_code=ExitCode.TRAINING_COMPLETED,
                               operation=selected_operation, observation=observation,
                               completed_epochs=observation.completed_epochs)
        if selected_operation == "finalize_training" or (selected_operation == "recover" and observation.full_budget_committed):
            if numerical_reference is not None:
                raise ManifestError("Full-budget publication recovery does not reconcile historical test failures; omit --numerical-reference")
            from .continuation import TrainingFinalizationPlan
            from .persistence import DurableTrainingPersistence
            from .metrics import RecordContext
            plan = TrainingFinalizationPlan()
            stage = "ownership_preflight"
            with open_run(config, base_dir=base_dir, provenance=provenance,
                          operation=selected_operation, continuation_plan=plan) as run:
                if selected_operation == "recover":
                    DurableTrainingPersistence(run).repair_publication()
                    return TaskOutcome(**common, outcome="training_publication_repaired", exit_code=ExitCode.TRAINING_COMPLETED,
                        operation="recover", observation=observation, completed_epochs=observation.completed_epochs)
                completion = DurableTrainingPersistence(run).publish_training_completion(
                    run, RecordContext(plan.current.document["context"]), plan.current.receipt)
                completion.validate(config, plan.current.receipt)
                acknowledged_completion = plan.current.receipt.completed_epochs
                return TaskOutcome(**common, outcome="training_completed", exit_code=ExitCode.TRAINING_COMPLETED,
                    operation="finalize_training", completed_epochs=acknowledged_completion)
        if selected_operation in ("blocked_numerical", "blocked_execution"):
            numeric = selected_operation == "blocked_numerical"
            return TaskOutcome(**common, outcome=selected_operation,
                exit_code=ExitCode.NUMERICAL_FAILURE if numeric else ExitCode.EXECUTION_FAILURE,
                operation=selected_operation, observation=observation, completed_epochs=observation.completed_epochs,
                failure=observation.failure,
                diagnostic="Terminal numerical failure; recover evidence only" if numeric else
                           "Execution failure requires explicit --operation continue or recover; no automatic retry")
        if selected_operation == "recover" and observation.numerical_evidence and not observation.stored_status == "failed_numerical":
            if numerical_reference not in observation.numerical_evidence:
                raise ManifestError("Recover requires an exact --numerical-reference from the inspected terminal evidence")
        if numerical_reference is not None and numerical_reference not in observation.numerical_evidence:
            raise ManifestError("Numerical reference is not current validated failure evidence")

        stage = "construction"
        from .data import build_loaders
        from .runtime import build_runtime
        from .models import build_model
        from .optimizers import build_optimizer
        from .scheduler import build_scheduler
        from .randomness import RuntimeStreams
        from .continuation import ContinuationPlan
        from .persistence import DurableTrainingPersistence
        from .training import TrainingRunError, train_run
        import torch

        data = build_loaders(config, base_dir=base_dir)
        if any(data.metadata.to_dict()[key] != value for key, value in selected.prepared_data.items()):
            raise ManifestError("Actual prepared loaders disagree with manifest prepared identities")
        runtime = build_runtime(config)
        model = build_model(config)
        model.to(device=runtime.device, dtype=torch.float32)
        optimizer = build_optimizer(model, config)
        continuing = selected_operation in ("continue", "recover")
        scheduler = build_scheduler(optimizer, config, len(data.train), initialize=not continuing)
        streams = RuntimeStreams(config, runtime.device, initialize=not continuing)
        plan = None if not continuing else ContinuationPlan(data=data, model=model, optimizer=optimizer,
            scheduler=scheduler, runtime=runtime, streams=streams,
            allow_execution_failure=operation == "continue")
        provenance["allocation"].update(runtime=runtime.metadata(), effective_execution=config.execution)
        stage = "ownership_preflight"
        with open_run(config, base_dir=base_dir, provenance=provenance, operation=selected_operation,
                      overwrite_run_id=overwrite_run_id, continuation_plan=plan) as run:
            service = DurableTrainingPersistence(run)
            if selected_operation == "recover":
                if numerical_reference is not None:
                    from .checkpoint_io import ArtifactReference
                    service.reconcile_numerical_failure(ArtifactReference.from_dict(numerical_reference))
                service.repair_publication()
                return TaskOutcome(**common, outcome="publication_repaired", exit_code=ExitCode.TRAINING_COMPLETED,
                    operation=selected_operation, observation=observation,
                    completed_epochs=observation.completed_epochs,
                    diagnostic="Publication repaired; stored scientific status is unchanged unless numerical evidence was explicitly reconciled")
            stage = "restoration"
            continuation = None if plan is None else plan.restore(run)
            stage = "training"
            try:
                result = train_run(config, data, model, optimizer, scheduler, run, runtime=runtime,
                    streams=streams, persistence=service, continuation=continuation,
                    stop=stop if stop is not None else StopRequest())
            except TrainingRunError as exc:
                # While holding the lease, completion dominates a lost ack. No
                # repair/retraining is attempted; retain any reconciliation error.
                from .checkpoint_journal import read_committed
                reconciliation = None
                count = exc.failure.durable_completed_epochs
                try:
                    current = read_committed(run.paths, run.config, run.provenance)
                    count = current.receipt.completed_epochs
                    if current.status in ("completed", "training_completed"):
                        acknowledged_completion = count
                        return TaskOutcome(**common, outcome="training_completed_after_lost_ack", exit_code=ExitCode.TRAINING_COMPLETED,
                            operation=selected_operation, completed_epochs=count,
                            diagnostic=str(exc), failure=asdict(exc.failure), failure_published=exc.failure_published)
                except Exception as inspection_error:
                    reconciliation = f"; authoritative reconciliation unavailable: {type(inspection_error).__name__}: {inspection_error}"
                code = {"interrupted": ExitCode.INTERRUPTED, "failed_numerical": ExitCode.NUMERICAL_FAILURE,
                        "failed_execution": ExitCode.EXECUTION_FAILURE}[exc.failure.status]
                return TaskOutcome(**common, outcome=exc.failure.status, exit_code=code, operation=selected_operation,
                    completed_epochs=count, diagnostic=str(exc) + (reconciliation or ""),
                    failure=asdict(exc.failure), failure_published=exc.failure_published)
            acknowledged_completion = result.counters.completed_epochs
            return TaskOutcome(**common, outcome="training_completed", exit_code=ExitCode.TRAINING_COMPLETED,
                operation=selected_operation, completed_epochs=result.counters.completed_epochs)
    except Exception as exc:
        # Construction/preflight/restoration/lease-exit errors must never produce
        # a fresh fallback. Report the condition even when no failure could be
        # durably published because there was no valid initialized boundary.
        if acknowledged_completion is not None:
            return TaskOutcome(**common, outcome="training_completed_after_diagnostic_error", exit_code=ExitCode.TRAINING_COMPLETED,
                operation=selected_operation, completed_epochs=acknowledged_completion,
                diagnostic=f"Committed completion; subsequent {type(exc).__name__}: {exc}")
        if stage in ("ownership_preflight", "restoration") and selected_operation in ("continue", "finalize_training"):
            # Another cooperating writer may have committed completion/failure
            # while components were being constructed. The plan repeats all
            # gates under locks; a stable reinspection classifies that rejection.
            refreshed = inspect_task(selected, base_dir=base_dir, overrides=overrides, provenance=provenance)
            if refreshed.condition == "ok":
                if refreshed.training_complete:
                    return TaskOutcome(**common, outcome="already_training_completed", exit_code=ExitCode.TRAINING_COMPLETED,
                        operation=selected_operation, observation=refreshed,
                        completed_epochs=refreshed.completed_epochs, diagnostic=str(exc))
                if refreshed.terminal_numerical or (refreshed.execution_failed and operation == "auto"):
                    numeric = refreshed.terminal_numerical
                    return TaskOutcome(**common, outcome="blocked_numerical" if numeric else "blocked_execution",
                        exit_code=ExitCode.NUMERICAL_FAILURE if numeric else ExitCode.EXECUTION_FAILURE,
                        operation=selected_operation, observation=refreshed, completed_epochs=refreshed.completed_epochs,
                        diagnostic=str(exc), failure=refreshed.failure)
        rejected = isinstance(exc, (ConfigError, RunDirectoryError)) or stage in (
            "declaration", "ownership_preflight", "restoration")
        return TaskOutcome(**common, outcome="rejected" if rejected else "failed_execution",
            exit_code=ExitCode.REJECTED if rejected else ExitCode.EXECUTION_FAILURE,
            operation=selected_operation, observation=observation,
            diagnostic=f"{type(exc).__name__}: {exc}", failure_published=False)
