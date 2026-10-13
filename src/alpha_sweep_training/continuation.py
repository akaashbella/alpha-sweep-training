"""Complete read-only preflight and ordered restoration under existing run locks."""

from copy import deepcopy

from .artifacts import RunHandle, RunPaths
from .config import RunConfig
from .data import DataLoaders
from .randomness import RuntimeStreams
from .runtime import RuntimePolicy
from .scheduler import WarmupCosineScheduler
from torch import nn
from torch.optim import Optimizer
from .checkpoint_io import fail, same
from .checkpoint_journal import CommittedState, numerical_failure_evidence, read_committed
from .checkpoints import copy_clean_model, data_identity, decode_rng, model_inventory
from .metrics import BestSelection, Counters
from .models import describe_model
from .numerical import require_clean_model, require_finite_optimizer
from .optimizers import describe_optimizer, validate_optimizer
from .training_contracts import ContinuationState


class ContinuationPlan:
    """Prepared live components, still fresh/unrestored. No launcher or builder.

    First use read_committed for stored preflight, then construct model/device,
    optimizer and initialize=False scheduler. Construct RuntimeStreams with
    initialize=False. Pass this plan to open_run; it is rechecked under locks
    BEFORE attempt/config writes. Call restore only inside that run context.
    """
    def __init__(self, *, data: DataLoaders, model: nn.Module, optimizer: Optimizer,
                 scheduler: WarmupCosineScheduler, runtime: RuntimePolicy, streams: RuntimeStreams):
        self.data, self.model, self.optimizer = data, model, optimizer
        self.scheduler, self.runtime, self.streams = scheduler, runtime, streams
        self._validated = None
        self._restored = False

    def preflight(self, paths: RunPaths, config: RunConfig, provenance: dict, *, recovery: bool = False) -> CommittedState:
        if self._restored:
            fail("Do not reuse a restored continuation plan")
        current = read_committed(paths, config, provenance)
        if not recovery and current.status == "completed":
            fail("Task is already durably completed; use read-only inspect_completion or explicit publication recovery")
        if not recovery and current.status == "failed_numerical":
            fail("Numerical failure is terminal; do not automatically continue/retry it")
        if not recovery and numerical_failure_evidence(paths, config, provenance):
            fail("Numerical failure evidence remains terminal even without acknowledgement; use explicit recovery to reconcile it, never automatic retry")
        payload = current.latest
        meta, state = payload["metadata"], payload["continuation"]
        self.runtime.require_active(config)
        require_clean_model(self.model, self.runtime.device, stage="continuation_component_preflight")
        info = validate_optimizer(self.optimizer, config)
        if info.model() is not self.model or self.optimizer.state:
            fail("Continuation preflight requires a fresh native optimizer on this clean model")
        if self.scheduler._optimizer is not self.optimizer or self.scheduler._completed_units is not None:
            fail("Continuation scheduler must be constructed on this optimizer with initialize=False")
        self.streams.require_unrestored()
        same(describe_model(self.model).to_dict(), meta["component_metadata"]["model"], "actual model structure")
        inventory = model_inventory(self.model.state_dict(), list(info.names))
        same(inventory, meta["inventory"], "actual clean parameter/buffer inventory")
        same(getattr(self.model.state_dict(), "_metadata", {}), meta["state_metadata"], "model state component versions")
        current_optimizer = describe_optimizer(self.optimizer).to_dict()
        saved_optimizer = meta["component_metadata"]["optimizer"]
        same({key: value for key, value in current_optimizer.items() if key != "current_learning_rates"},
             {key: value for key, value in saved_optimizer.items() if key != "current_learning_rates"}, "actual optimizer recipe/order/options")
        position = self.scheduler.validate_state_dict(state["scheduler"])
        same(list(self.scheduler.learning_rates_at(position)), saved_optimizer["current_learning_rates"], "restored optimizer LR")
        same(data_identity(self.data.metadata.to_dict()), meta["data_identity"], "actual prepared dataset identities")
        counters = Counters(**meta["counters"])
        same(self.data.continuation_metadata(counters.completed_epochs), state["data_continuation"], "actual data continuation")
        self.streams.validate_state_dict(decode_rng(state["rng"]))
        # Device names/indices and allocation metadata are operational. The
        # actual assigned GPU already passed build_runtime's native capability
        # checks; conservative software/source compatibility is checked above.
        saved_runtime = meta["component_metadata"]["runtime"]
        actual_runtime = self.runtime.metadata()
        for key in ("engine_version", "precision", "master_dtype", "evaluation_precision", "grad_scaler"):
            same(actual_runtime[key], saved_runtime[key], f"runtime {key}")
        for key, value in actual_runtime["effective_backend"].items():
            if key != "cublas_workspace_config":
                same(saved_runtime["effective_backend"][key], value, f"effective backend {key}")
        self._validated = current
        return current

    def restore(self, run: RunHandle) -> ContinuationState:
        run.require_active()
        if run.operation != "continue" or self._validated is None:
            fail("Restore requires an open_run continuation that performed full preflight")
        # Detect replacement since the under-lock preflight, then repeat every
        # validation before changing ANY live state. No partial restore fallback.
        prior = self._validated.reference
        current = self.preflight(run.paths, run.config, run.provenance)
        same(current.reference.to_dict(), prior.to_dict(), "preflight/restoration transaction")
        payload, state = current.latest, current.latest["continuation"]
        copy_clean_model(self.model, payload)
        # Pinned native load_state_dict maps parameter-shaped floating states to
        # their parameters, while preserving noncapturable step scalars on CPU.
        # Never apply an indiscriminate .to(cuda) to the optimizer state tree.
        self.optimizer.load_state_dict(deepcopy(state["optimizer"]))
        validate_optimizer(self.optimizer, run.config)
        self.scheduler.load_state_dict(state["scheduler"])
        if self.runtime.scaler is not None:
            self.runtime.scaler.load_state_dict(deepcopy(state["scaler"]))
        counters = Counters(**payload["metadata"]["counters"])
        best = None if payload["metadata"]["best"] is None else BestSelection(**payload["metadata"]["best"])
        self.data.restore_next_epoch(state["data_continuation"])
        require_clean_model(self.model, self.runtime.device, stage="restored_clean_state")
        require_finite_optimizer(self.optimizer, self.model)
        self.runtime.require_active(run.config)
        for values in self.optimizer.state.values():
            for key, value in values.items():
                if key == "step" and value.device.type != "cpu":
                    fail("Native noncapturable optimizer step scalar was moved off CPU")
        # Restore runtime/noise randomness LAST, after all constructors/copies.
        self.streams.load_state_dict(decode_rng(state["rng"]))
        self._restored = True
        return ContinuationState(counters, best, current.receipt, deepcopy(state["data_continuation"]))
