"""Owning Python process signal delivery; handlers only request a stop."""

from contextlib import contextmanager
import os
import signal
import threading

from .hpc_policy import WorkflowError
from .interruption import StopRequest

OWNER_ENV = "ALPHA_SWEEP_ALLOCATION_OWNER_PID"


@contextmanager
def allocation_signal_handlers(stop, *, warning_policy="discard_epoch", termination_policy="discard_epoch"):
    if not isinstance(stop, StopRequest) or threading.current_thread() is not threading.main_thread():
        raise WorkflowError("Install allocation handlers in the owning main thread with StopRequest")
    if not hasattr(signal, "SIGUSR1") or warning_policy not in ("discard_epoch", "finish_epoch") or termination_policy != "discard_epoch":
        raise WorkflowError("The production runner requires POSIX USR1 and explicit discard/finish policy")
    policies = {signal.SIGUSR1: warning_policy, signal.SIGTERM: termination_policy, signal.SIGINT: termination_policy}
    reasons = {signal.SIGUSR1: "Slurm advance warning", signal.SIGTERM: "Allocation termination", signal.SIGINT: "Allocation interrupt"}
    previous = {number: signal.getsignal(number) for number in policies}
    previous_owner = os.environ.get(OWNER_ENV)
    os.environ[OWNER_ENV] = str(os.getpid())

    def handler(number, _frame):
        # Attribute-only StopRequest update: no I/O, locks, logging, subprocess,
        # checkpoint, scheduler or CUDA calls. The batch shell execs this PID.
        stop.request(policies[number], reasons[number])

    installed = []
    try:
        for number in policies:
            signal.signal(number, handler)
            installed.append(number)
        yield
    finally:
        for number in installed:
            signal.signal(number, previous[number])
        if previous_owner is None:
            os.environ.pop(OWNER_ENV, None)
        else:
            os.environ[OWNER_ENV] = previous_owner


def configure_data_worker_signals():
    """Workers never own the parent's StopRequest or advance-warning delivery.

    B:USR1 targets only the batch PID, replaced by the Python owner via exec.
    Ignore accidental worker USR1/INT; TERM keeps normal child termination.
    SIGKILL/node failure remains abrupt and is never a safe-stop assertion.
    """
    owner = os.environ.get(OWNER_ENV)
    if owner is not None and owner != str(os.getpid()) and hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
