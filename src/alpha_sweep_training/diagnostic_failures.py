"""Structured diagnostic failures without importing compute modules at CLI startup."""

from contextlib import contextmanager
from copy import deepcopy
import sys


class DiagnosticStageFailure(RuntimeError):
    def __init__(self, record):
        self.record = record
        super().__init__(f"{record['origin']}/{record['stage']}: {record['message']}")


def failure_details(exc, *, origin, stage, epoch_index=None, batch_index=None):
    chain, seen, current = [], set(), exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        # Implicit context may just be an output/observer error while handling
        # another failure. Only explicit causes carry its origin forward.
        current = current.__cause__
    annotated = next((error for error in chain if isinstance(error, DiagnosticStageFailure)), None)
    record = deepcopy(annotated.record) if annotated else {
        "origin": origin, "stage": stage, "epoch_index": epoch_index, "batch_index": batch_index,
        "exception_type": type(exc).__name__, "message": str(exc), "numerical_failure": None}
    # These typed classes can only have produced an error if their modules are
    # already loaded. Do not import Torch/CUDA just to format a setup/I/O error.
    numerical_module = sys.modules.get(f"{__package__}.numerical")
    numerical_type = getattr(numerical_module, "NumericalFailure", ())
    evaluation_module = sys.modules.get(f"{__package__}.evaluation")
    evaluation_type = getattr(evaluation_module, "EvaluationFailure", ())
    for error in chain:
        if isinstance(error, numerical_type):
            record["numerical_failure"] = {"stage": error.stage, "reason": error.reason, "tensors": list(error.tensors)}
        if isinstance(error, evaluation_type):
            record["split"] = error.split
            if record["batch_index"] is None:
                record["batch_index"] = error.batch_index
    record["exception_chain"] = [{"type": type(error).__name__, "message": str(error)} for error in chain]
    return record


@contextmanager
def diagnostic_stage(origin, stage, *, epoch_index=None, batch_index=None):
    try:
        yield
    except DiagnosticStageFailure:
        raise
    except Exception as exc:
        raise DiagnosticStageFailure(failure_details(exc, origin=origin, stage=stage,
            epoch_index=epoch_index, batch_index=batch_index)) from exc
