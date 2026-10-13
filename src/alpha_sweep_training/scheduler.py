"""Original-budget linear warmup/cosine indexed by completed cadence units.

No PyTorch scheduler constructor side effects and no checkpoint file handling.
The trainer must distinguish completed epochs from successful optimizer updates.
"""

from dataclasses import dataclass
import math
from typing import Any, Mapping

from torch.optim import Optimizer

from .component_contracts import ComponentError, finite_number, integer_count, require_config
from .config import RunConfig
from .optimizers import validate_optimizer
from .spec import SCHEDULER_VERSION


class SchedulerError(ComponentError):
    """Missing schedule decisions, invalid position, or incompatible continuation."""


@dataclass(frozen=True)
class _Schedule:
    architecture: str
    interval: str
    original_epochs: int
    warmup_epochs: int
    steps_per_epoch: int | None
    base_learning_rates: tuple[float, ...]
    warmup_start_factor: float
    cosine_min_lr: float

    @property
    def total_units(self) -> int:
        return self.original_epochs * (self.steps_per_epoch if self.interval == "optimizer_step" else 1)

    @property
    def warmup_units(self) -> int:
        return self.warmup_epochs * (self.steps_per_epoch if self.interval == "optimizer_step" else 1)

    def to_dict(self) -> dict:
        return {"format_version": 1, "scheduler_version": SCHEDULER_VERSION,
                "architecture": self.architecture, "interval": self.interval,
                "original_epochs": self.original_epochs, "warmup_epochs": self.warmup_epochs,
                "steps_per_epoch": self.steps_per_epoch,
                "base_learning_rates": list(self.base_learning_rates),
                "warmup_start_factor": self.warmup_start_factor, "cosine_min_lr": self.cosine_min_lr}


class WarmupCosineScheduler:
    """The next training unit uses lr(n); step() follows its completion.

    initialize=False is mandatory for continuation. It leaves optimizer LRs
    untouched and requires load_state_dict before logging state or advancing.
    This object is owned by one training process/thread, like its optimizer.
    """

    def __init__(self, optimizer: Optimizer, config: RunConfig,
                 steps_per_epoch: int | None, *, initialize: bool = True):
        science = require_config(config, "scheduler").scientific
        training = science["training"]
        missing = [f"scientific.training.{name}" for name in
                   ("warmup_start_factor", "cosine_min_lr", "scheduler_interval")
                   if training[name] is None]
        if missing:
            raise SchedulerError("Scheduler construction requires explicit decisions for " + ", ".join(missing))
        if type(initialize) is not bool:
            raise SchedulerError("initialize must be bool; use False for checkpoint restoration")
        info = validate_optimizer(optimizer, config)
        interval = training["scheduler_interval"]
        if interval not in ("epoch", "optimizer_step"):
            raise SchedulerError("Scheduler interval must be epoch or optimizer_step")
        if steps_per_epoch is not None:
            integer_count(steps_per_epoch, "scheduler steps_per_epoch (actual len(loaders.train))", minimum=1)
        if interval == "optimizer_step" and steps_per_epoch is None:
            raise SchedulerError("optimizer_step cadence requires the actual positive len(loaders.train), respecting drop_last")
        if steps_per_epoch is not None:
            size, batch = science["data"]["train_size"], training["batch_size"]
            floor_count, ceiling_count = size // batch, (size + batch - 1) // batch
            allowed = {floor_count, ceiling_count} if training["drop_last"] is None else {
                floor_count if training["drop_last"] else ceiling_count}
            if steps_per_epoch not in allowed:
                raise SchedulerError("steps_per_epoch disagrees with the fixed training split, batch size, and declared drop_last; pass the actual loader count")
        factor = finite_number(training["warmup_start_factor"], "scheduler warmup_start_factor")
        minimum = finite_number(training["cosine_min_lr"], "scheduler cosine_min_lr")
        if not 0 <= factor <= 1 or not 0 <= minimum <= info.base_learning_rate:
            raise SchedulerError("Scheduler requires f in [0,1] and m between zero and the original base LR")
        self._schedule = _Schedule(info.architecture, interval, training["epochs"],
                                   training["warmup_epochs"],
                                   steps_per_epoch if interval == "optimizer_step" else None,
                                   (info.base_learning_rate,), factor, minimum)
        self._optimizer = optimizer
        self._completed_units: int | None = None
        if initialize:
            if optimizer.state:
                raise SchedulerError("Fresh scheduler initialization requires a fresh optimizer; use initialize=False for continuation")
            self._require_lr(self._schedule.base_learning_rates)
            rates = self._rates_at(0)
            self._apply(rates)
            self._completed_units = 0

    @property
    def interval(self) -> str:
        return self._schedule.interval

    @property
    def completed_units(self) -> int:
        if self._completed_units is None:
            raise SchedulerError("Continuation scheduler is awaiting load_state_dict; do not restart warmup")
        return self._completed_units

    @property
    def total_units(self) -> int:
        return self._schedule.total_units

    @property
    def warmup_units(self) -> int:
        return self._schedule.warmup_units

    @property
    def base_learning_rates(self) -> tuple[float, ...]:
        return self._schedule.base_learning_rates

    @property
    def current_learning_rates(self) -> tuple[float, ...]:
        position = self.completed_units
        validate_optimizer(self._optimizer)
        rates = self._rates_at(position)
        self._require_lr(rates)
        return tuple(float(group["lr"]) for group in self._optimizer.param_groups)

    def _rates_at(self, position: int) -> tuple[float, ...]:
        integer_count(position, "scheduler completed_units")
        if position > self.total_units:
            raise SchedulerError("Schedule position exceeds the original training horizon")
        schedule = self._schedule
        # Return declared endpoints exactly, avoiding subtraction/addition
        # roundoff at these mathematically identical cosine boundaries.
        if position == self.warmup_units:
            return schedule.base_learning_rates
        if position == self.total_units:
            return tuple(schedule.cosine_min_lr for _ in schedule.base_learning_rates)
        if position < self.warmup_units:
            fraction = schedule.warmup_start_factor + (1 - schedule.warmup_start_factor) * position / self.warmup_units
            return tuple(base * fraction for base in schedule.base_learning_rates)
        fraction = (1 + math.cos(math.pi * (position - self.warmup_units)
                                 / (self.total_units - self.warmup_units))) / 2
        return tuple(schedule.cosine_min_lr + (base - schedule.cosine_min_lr) * fraction
                     for base in schedule.base_learning_rates)

    def _require_lr(self, expected: tuple[float, ...]) -> None:
        actual = tuple(finite_number(group["lr"], "optimizer LR") for group in self._optimizer.param_groups)
        if len(actual) != len(expected) or any(not math.isclose(a, b, rel_tol=1e-12, abs_tol=0.0)
                                               for a, b in zip(actual, expected)):
            raise SchedulerError("Optimizer LR disagrees with the scheduler position; restore optimizer state before scheduler state and do not change LR externally")

    def _apply(self, rates: tuple[float, ...]) -> None:
        for group, rate in zip(self._optimizer.param_groups, rates):
            group["lr"] = rate

    def step(self) -> None:
        """Call once after a completed epoch OR a successful optimizer update.

        AMP-skipped updates must not call this for optimizer_step cadence. Never
        catch up using attempted batches or manufacture updates at epoch 200.
        """
        position = self.completed_units
        validate_optimizer(self._optimizer)
        self._require_lr(self._rates_at(position))
        if position == self.total_units:
            raise SchedulerError("Cannot advance beyond the original training horizon")
        rates = self._rates_at(position + 1)
        self._apply(rates)
        self._completed_units = position + 1

    def state_dict(self) -> dict:
        self.current_learning_rates  # Reject inconsistent optimizer state before serialization.
        return {**self._schedule.to_dict(), "completed_units": self.completed_units}

    def validate_state_dict(self, state: Mapping[str, Any]) -> int:
        """Preflight compatibility/position without changing state or LR.

        May run before optimizer restoration: it does not compare current LR to
        checkpoint position. Step 5 must additionally validate the full run's
        epoch, successful-update, model, optimizer, precision and RNG state.
        """
        validate_optimizer(self._optimizer)
        expected = self._schedule.to_dict()
        if not isinstance(state, Mapping) or set(state) != set(expected) | {"completed_units"}:
            raise SchedulerError("Invalid scheduler state keys/format")
        for key, value in expected.items():
            actual = state[key]
            if type(value) is float:
                matches = finite_number(actual, f"scheduler state {key}") == value
            elif type(value) is list:
                matches = type(actual) is list and len(actual) == len(value)
                if matches:
                    matches = all(finite_number(item, f"scheduler state {key}") == reference
                                  for item, reference in zip(actual, value))
            else:
                matches = type(actual) is type(value) and actual == value
            if not matches:
                raise SchedulerError(f"Incompatible scheduler state {key}; retain the original complete schedule descriptor")
        position = integer_count(state["completed_units"], "scheduler state completed_units")
        if position > self.total_units:
            raise SchedulerError("Saved scheduler position exceeds the original horizon")
        return position

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        position = self.validate_state_dict(state)
        rates = self._rates_at(position)
        self._require_lr(rates)  # All compatibility checks precede either mutation.
        self._apply(rates)
        self._completed_units = position

    def metadata(self) -> dict:
        """Current LR/position are observations, separate from original base LRs."""
        return {**self.state_dict(), "total_units": self.total_units,
                "warmup_units": self.warmup_units,
                "current_learning_rates": list(self.current_learning_rates),
                "position_semantics": "completed_cadence_units; next_unit_uses_lr_at_position"}


def build_scheduler(optimizer: Optimizer, config: RunConfig, steps_per_epoch: int | None,
                    *, initialize: bool = True) -> WarmupCosineScheduler:
    """Pass len(loaders.train); use initialize=False before restoring optimizer."""
    return WarmupCosineScheduler(optimizer, config, steps_per_epoch, initialize=initialize)
