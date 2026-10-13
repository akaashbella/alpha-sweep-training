"""Small stop requests; future handlers request, never serialize checkpoints."""

from typing import Literal


class TrainingInterrupted(InterruptedError):
    pass


class StopRequest:
    """One process owner. Discard requests take precedence over finish-epoch.

    Attribute assignments need no checkpoint I/O or locks in a future Python
    signal handler. Orchestration still owns signal registration and delivery.
    """
    def __init__(self) -> None:
        self.policy: Literal["finish_epoch", "discard_epoch"] | None = None
        self.reason = ""

    def request(self, policy: Literal["finish_epoch", "discard_epoch"], reason: str) -> None:
        if policy not in ("finish_epoch", "discard_epoch") or type(reason) is not str or not reason:
            raise ValueError("Stop requires finish_epoch/discard_epoch and an explicit reason")
        self.reason = reason
        if self.policy != "discard_epoch":
            self.policy = policy

    def check(self, *, at_boundary: bool, phase: str) -> None:
        if self.policy == "discard_epoch" or (at_boundary and self.policy == "finish_epoch"):
            raise TrainingInterrupted(f"Stop requested ({self.policy}) at {phase}: {self.reason}")
