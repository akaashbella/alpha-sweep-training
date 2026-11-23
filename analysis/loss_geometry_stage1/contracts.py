"""Stage 1 options and lightweight reuse of the experiment's frozen contracts."""

from alpha_sweep_training.identity import canonical_alpha
from alpha_sweep_training.spec import ALPHAS, ARCHITECTURES, DATASETS, EVALUATION_ALPHAS

WINDOWS = {"primary": (0.01, 0.02, 0.05), "secondary": (0.01, 0.02, 0.05, 0.10)}
IDENTITY = ("dataset", "architecture", "training_alpha", "canonical_training_alpha",
            "training_seed", "training_run_id", "best_checkpoint_identity")
LIMITATIONS = (
    "The second-order Taylor approximation is local; 2c is an empirical proxy, not a measured Hessian trace.",
    "Fits use three or four noisy checkpoint means and share an estimated clean baseline; no inferential uncertainty is asserted.",
    "Three training seeds are training replicates; three positive-alpha realizations are within-checkpoint replicates.",
    "Loss robustness does not guarantee accuracy robustness; absolute clean and perturbed accuracy must also be considered.",
    "Architecture comparisons are descriptive and confounded with architecture-specific optimization recipes, including recorded recipe revisions.",
    "In-sample fit quality does not establish prediction; larger-alpha residuals are descriptive with no formal breakdown threshold.",
    "Correlation is descriptive; repeated training seeds and checkpoints do not become independent experiments by pooling evaluation alphas.",
    "Publication integrity is not scientific completeness; this analysis validates frozen summaries and does not reread live authorities.",
)


class AnalysisError(ValueError):
    """Incompatible input, unsafe output or invalid numerical result."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)
