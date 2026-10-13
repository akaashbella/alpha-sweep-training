"""Compute-free independent evaluation declaration, slots, identity and RNG keys."""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json

from .config import ConfigError, RunConfig
from .data_contract import content_identity
from .identity import canonical_alpha
from .spec import (BATCH_SIZE, TEST_SIZE, EVALUATION_ALPHAS, EVALUATION_REPLICATES,
                   EVALUATION_PROTOCOL_VERSION, EVALUATION_RANDOMNESS_VERSION)


class EvaluationProtocolError(ConfigError):
    pass


@dataclass(frozen=True)
class EvaluationSlot:
    evaluation_alpha: float
    evaluation_replicate: int

    def __post_init__(self):
        if type(self.evaluation_alpha) not in (int, float) or self.evaluation_alpha not in EVALUATION_ALPHAS:
            raise EvaluationProtocolError("Select an approved finite numeric evaluation_alpha")
        expected = (0,) if self.evaluation_alpha == 0 else EVALUATION_REPLICATES
        if type(self.evaluation_replicate) is not int or self.evaluation_replicate not in expected:
            raise EvaluationProtocolError("Clean evaluation has replicate 0; positive alphas have replicates 1–3")

    @property
    def slot_id(self):
        return f"evaluation-alpha-{canonical_alpha(self.evaluation_alpha)}__replicate-{self.evaluation_replicate}"

    def to_dict(self):
        return {"evaluation_alpha": float(self.evaluation_alpha),
                "canonical_evaluation_alpha": canonical_alpha(self.evaluation_alpha),
                "evaluation_replicate": self.evaluation_replicate, "slot_id": self.slot_id}


def prescribed_slots() -> tuple[EvaluationSlot, ...]:
    return tuple(EvaluationSlot(alpha, replicate) for alpha in EVALUATION_ALPHAS
                 for replicate in ((0,) if alpha == 0 else EVALUATION_REPLICATES))


def select_evaluation_slots(*, evaluation_alphas=None, evaluation_replicates=None):
    alphas = list(EVALUATION_ALPHAS) if evaluation_alphas is None else evaluation_alphas
    if type(alphas) is not list or not alphas or any(type(a) not in (int, float) or a not in EVALUATION_ALPHAS for a in alphas):
        raise EvaluationProtocolError("Supply a nonempty list of approved numeric evaluation alphas")
    if len(set(alphas)) != len(alphas):
        raise EvaluationProtocolError("Duplicate evaluation alphas are not selectors")
    if evaluation_replicates is not None:
        if (type(evaluation_replicates) is not list or not evaluation_replicates
                or any(type(r) is not int or r not in (0, *EVALUATION_REPLICATES) for r in evaluation_replicates)
                or len(set(evaluation_replicates)) != len(evaluation_replicates)):
            raise EvaluationProtocolError("Supply unique integer evaluation replicate indices")
        if 0 in alphas and 0 not in evaluation_replicates:
            raise EvaluationProtocolError("A selected clean alpha requires its single replicate 0")
        if 0 not in alphas and 0 in evaluation_replicates:
            raise EvaluationProtocolError("Replicate 0 only selects the clean alpha")
        if any(a > 0 for a in alphas) and not any(r in EVALUATION_REPLICATES for r in evaluation_replicates):
            raise EvaluationProtocolError("Positive alphas require at least one replicate in 1–3")
        if all(a == 0 for a in alphas) and evaluation_replicates != [0]:
            raise EvaluationProtocolError("Do not create extra clean replicates")
    return tuple(slot for slot in prescribed_slots() if slot.evaluation_alpha in alphas
                 and (evaluation_replicates is None or slot.evaluation_replicate in evaluation_replicates))


@dataclass(frozen=True)
class EvaluationProtocol:
    _document: dict

    def to_dict(self):
        return deepcopy(self._document)

    @property
    def fingerprint(self):
        return content_identity(self._document)[7:]


def evaluation_protocol(config: RunConfig) -> EvaluationProtocol:
    precision = config.scientific["precision"]
    pending = [f"scientific.precision.{k}" for k, v in precision.items() if v is None]
    data = config.scientific["data"]
    if any(v is None for v in data["normalization"].values()):
        pending.append("scientific.data.normalization (finalized preparation)")
    if pending:
        raise EvaluationProtocolError("Finalize evaluation inputs: " + ", ".join(pending))
    return EvaluationProtocol({"protocol_version": EVALUATION_PROTOCOL_VERSION, "format_version": 1,
        "evaluation_alphas": list(EVALUATION_ALPHAS), "clean_replicates": [0],
        "positive_replicates": list(EVALUATION_REPLICATES), "checkpoint_policy": "clean_validation_selected_best",
        "selection_policy": "strict_improvement_earliest_tie; never_test",
        "perturbation": {"formula": "W_clean + evaluation_alpha * population_std(W_clean) * epsilon",
            "scope": "all_owned_Conv2d_and_Linear_weights_simultaneously", "std_correction": 0,
            "detach_scale_and_noise": True, "scale_noise_dtype": "float32", "lifetime": "one_full_test_pass",
            "fresh_clean_origin_each_slot": True, "bias_normalization_buffer_noise": False,
            "zero_bypass": True, "functional": True, "gradients": False, "module_mode": "eval",
            "clean_state_preservation": "exact_all_parameters_buffers_bindings_and_prior_modes"},
        "randomness_version": EVALUATION_RANDOMNESS_VERSION,
        "rng_keys": ["best_checkpoint_identity", "protocol_version", "canonical_evaluation_alpha", "evaluation_replicate"],
        "preprocessing": {"version": data["preprocessing_version"], "transforms": data["evaluation_transforms"],
                          "normalization": data["normalization"], "image_size": data["image_size"]},
        "test_examples": TEST_SIZE, "batch_size": BATCH_SIZE, "precision": precision,
        "master_dtype": "float32", "loss_dtype": "float32", "scaler": False,
        "evaluation_compatibility": "pinned_model_contracts_same_full_torch_vision_versions; source_digest_independent"})


def validate_protocol(document, config) -> EvaluationProtocol:
    expected = evaluation_protocol(config)
    if content_identity(document) != content_identity(expected.to_dict()):
        raise EvaluationProtocolError("Unknown or incompatible evaluation protocol fields/version/settings")
    return expected


def perturbation_seed(best_checkpoint_identity: dict, slot: EvaluationSlot) -> int | None:
    if slot.evaluation_alpha == 0:
        return None
    # Identity includes immutable generation and actual digest, not its path.
    if (type(best_checkpoint_identity) is not dict or set(best_checkpoint_identity) != {"generation", "sha256"}
            or any(type(best_checkpoint_identity[k]) is not str or len(best_checkpoint_identity[k]) != n
                   or any(c not in "0123456789abcdef" for c in best_checkpoint_identity[k])
                   for k, n in (("generation", 32), ("sha256", 64)))):
        raise EvaluationProtocolError("Seed derivation requires verified immutable best-checkpoint identity")
    payload = {"randomness_version": EVALUATION_RANDOMNESS_VERSION,
        "best_checkpoint_identity": best_checkpoint_identity, "protocol_version": EVALUATION_PROTOCOL_VERSION,
        "canonical_evaluation_alpha": canonical_alpha(slot.evaluation_alpha), "evaluation_replicate": slot.evaluation_replicate}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little")
