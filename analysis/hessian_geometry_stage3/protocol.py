"""Frozen design, partition validation and append-only probe budgets."""

from collections import Counter
from itertools import product
import math
import re

from .storage import code_digest, identity, plain, publication, require, seal

RANDOMNESS = "stage3-hutchinson-rademacher-sha256-v1"
PROBE_COUNTS = (16, 32, 64)
PILOT_ALPHAS = (0., .10, .30)
EVAL_ALPHAS = (0., .01, .02, .05, .10, .20)
DATASETS = ("cifar10", "cifar100")
ARCHITECTURES = ("resnet50", "mobilenetv3_large", "efficientnet_b0", "convnext_tiny")
EXAMPLES = {"validation": 5000, "test": 10000}
HISTORICAL_COMMIT = "8010d07d9d1f0886ab1e180523f316623ef622a4"
HISTORICAL_SOURCE = "a5726bac3f449455d36e2f53840f3e0fd77768a7ab2b989d2884da20f920005c"
PRESERVATION = "stage3_parameters_buffers_modules_grad_flags_grad_fields_and_modes_verified"


def is_digest(value, size=64):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{%d}" % size, value) is not None


def probe_seed(checkpoint, probe, tensor):
    value = identity({"checkpoint": checkpoint, "version": RANDOMNESS, "probe": probe, "tensor": tensor})
    return int.from_bytes(bytes.fromhex(value[:16]), "little")


def validate_layout(layout):
    entries = layout["tensors"]
    names = [row["name"] for row in entries]
    require(names and len(set(names)) == len(names) and all(row["module_type"] in ("Conv2d", "Linear")
            and row["name"] == (row["module_name"] + "." if row["module_name"] else "") + "weight"
            and row["requires_grad"] is True and type(row["numel"]) is int and row["numel"] > 0
            and math.prod(row["shape"]) == row["numel"] for row in entries), "Malformed eligible tensor inventory")
    stages = layout["stages"]
    require(len(layout["stage_order"]) == len(stages) and set(layout["stage_order"]) == set(stages) and
            all(stages.values()) and Counter(name for group in stages.values() for name in group) == Counter(names),
            "Disjoint stages must cover each eligible tensor exactly once")
    require(all(group and len(set(group)) == len(group) and set(group) <= set(names)
                for group in layout["diagnostics"].values()), "Invalid overlapping diagnostic group")


def decomposition(layout, tensors):
    require(set(tensors) == {row["name"] for row in layout["tensors"]} and
            all(type(x) in (int, float) and math.isfinite(x) for x in tensors.values()), "Invalid tensor quadratic forms")
    stages = {name: math.fsum(tensors[t] for t in names) for name, names in layout["stages"].items()}
    diagnostics = {name: math.fsum(tensors[t] for t in names) for name, names in layout["diagnostics"].items()}
    total = math.fsum(tensors.values())
    # Scale the tolerance by absolute mass to accommodate signed cancellation.
    error = abs(math.fsum(stages.values())-total)
    require(error <= 1e-12*max(1., math.fsum(abs(x) for x in tensors.values())), "Per-probe stage sums disagree with the full network")
    return {"whole": total, "tensors": tensors, "stages": stages, "diagnostics": diagnostics,
            "partition_absolute_error": error}


def corrected(row):
    evidence = row["evidence"]
    key, science, binding = row["key"], evidence["scientific"], evidence["binding"]
    require((science["dataset"], science["architecture"], science["alpha"], science["training_seed"]) ==
            tuple(key[k] for k in ("dataset", "architecture", "training_alpha", "training_seed")),
            "Stage 2 scientific selection/checkpoint identity mismatch")
    train, best = binding["training_identity"], binding["best_checkpoint_identity"]
    require(all(binding[k] == key[k] for k in key) and train["run_id"] == row["run_id"] and
            is_digest(train["scientific_fingerprint"]) and is_digest(best["sha256"]) and is_digest(best["generation"], 32)
            and binding["best_checkpoint"]["sha256"] == best["sha256"] and
            binding["best_checkpoint"]["role"] == "best" and
            all(binding["data_identity"].get(k) == v for k, v in evidence["prepared_data"].items()),
            "Stage 2 best checkpoint fingerprint, generation, digest or prepared identity mismatch")
    require(train["specification_version"] == science["specification_version"] and
            evidence["format_version"] == 1 and is_digest(evidence["best_training_provenance"]["code"]["source_sha256"]),
            "Unsupported checkpoint schema or absent historical source digest")
    if key["architecture"] == "mobilenetv3_large":
        optimum = evidence["best_optimizer"]
        require(science["optimizer"]["name"] == "rmsprop" and science["optimizer"]["epsilon"] == .0316 and
                optimum["declared_recipe"] == science["optimizer"] and optimum["effective_options"]["eps"] == .0316,
                "Corrected MobileNet RMSprop epsilon=0.0316 is required in the declaration and native best metadata")
        if key["dataset"] == "cifar10":
            code = evidence["best_training_provenance"]["code"]
            require(code.get("commit") == HISTORICAL_COMMIT and code["source_sha256"] == HISTORICAL_SOURCE,
                    "Corrected CIFAR-10 MobileNet requires the pinned historical training source")
    validate_layout(row["layout"])


def pilot_membership(stage2):
    from analysis.layer_sensitivity_stage2.groups import targets

    require(stage2["split"] == "validation" and stage2["examples_per_pass"] == 5000 and
            stage2["evaluation_alphas"] == list(EVAL_ALPHAS) and stage2["realizations_per_positive_alpha"] == 8 and
            stage2["scopes"] == ["stages"] and len(stage2["selected_seeds"]) == 1 and type(stage2["selected_seeds"][0]) is int,
            "Require the completed Stage 2 exploratory validation pilot protocol")
    rows = stage2["checkpoints"]
    seed = stage2["selected_seeds"][0]
    expected = set(product(DATASETS, ARCHITECTURES, PILOT_ALPHAS, [seed]))
    keys = [tuple(r["key"][k] for k in ("dataset", "architecture", "training_alpha", "training_seed")) for r in rows]
    identifiers = [r["checkpoint_id"] for r in rows]
    require(len(rows) == 24 and set(keys) == expected and len(set(identifiers)) == 24
            and identifiers == [identity(row["key"])[:20] for row in rows], "Stage 2 pilot checkpoint membership or identifiers differ")
    require(stage2["estimates"]["passes"] == 8904 and sum(r.get("passes", 0) for r in rows) == 8904,
            "Expected the complete 8,904-pass Stage 2 stage-only pilot")
    for row in rows:
        require(row["status"] == "eligible" and row["targets"] == targets(row["layout"], ("stages",)) and
                row["passes"] == 1+40*len(row["targets"]), "Stage 2 targets must be the original stage partition plus full network")
    return rows


def numerical_policy(backend):
    require(backend in ("cudnn", "native"), "Select cudnn or native explicitly")
    return {"dtype": "float32", "loss_dtype": "float32", "scalar_accumulation": "float64",
            "autocast": False, "tf32": False, "backend": backend, "deterministic": True}


def workload(core, count, cap):
    n = len(core["checkpoints"])
    require(n == 24 and count in PROBE_COUNTS and type(cap) is int and n*count <= cap,
            f"Requested {n*count} complete dataset probes exceeds the cap or pilot membership is incomplete")
    batches = (core["examples"]+core["batch_size"]-1)//core["batch_size"]
    return {"checkpoints_requested": n, "checkpoints_eligible": n, "probes_per_checkpoint": count,
            "dataset_probes": n*count, "examples": n*count*core["examples"], "batches_per_probe": batches,
            "batch_hvps": n*count*batches, "max_total_probes": cap}


def make_plan(audit, audit_dir, *, split, batch_size, backend, cap):
    require(audit["code_sha256"] == code_digest(), "Audit uses different analysis source")
    require(all(row["status"] == "eligible" for row in audit["checkpoints"]),
            "The exact 24-checkpoint pilot requires 24 eligible audit rows. Inspect audit exclusions")
    require(type(batch_size) is int and 1 <= batch_size <= 128, "Batch size must be 1 through 128")
    core = {k: audit[k] for k in ("checkpoints", "stage2_plan_dir", "stage2_results", "stage2_plan_sha256",
                                  "stage2_plan_marker_sha256", "code_sha256")}
    core.update(audit_sha256=audit["document_sha256"], protected_roots=sorted({*audit["protected_roots"], str(plain(audit_dir))}),
                split=split, examples=EXAMPLES[split], batch_size=batch_size, numerics=numerical_policy(backend),
                randomness=RANDOMNESS, distribution="Rademacher", first_probe_id=1)
    return plan_document(core, 16, cap, [])


def plan_document(core, count, cap, revisions):
    return seal("plan", {"core": core, "work_sha256": identity(core), "count": count,
                         "workload": workload(core, count, cap), "extensions": revisions})


def load_plan(directory, *, verify_source=False):
    plan = publication(directory, "plan")
    core = plan["core"]
    require(identity(core) == plan["work_sha256"] and plan["workload"] == workload(core, plan["count"], plan["workload"]["max_total_probes"]),
            "Plan workload or immutable family identity mismatch")
    counts = [r["new_count"] for r in plan["extensions"]]
    require([16, *counts] == list(PROBE_COUNTS[:PROBE_COUNTS.index(plan["count"])+1]), "Invalid append-only probe budget history")
    for index, rev in enumerate(plan["extensions"]):
        require(rev["old_count"] == PROBE_COUNTS[index] and rev["justification"].strip() and is_digest(rev["parent_sha256"]),
                "Missing extension ancestry or justification")
        parent = publication(rev["parent_dir"], "plan")
        require(parent["document_sha256"] == rev["parent_sha256"] and parent["work_sha256"] == plan["work_sha256"]
                and parent["core"] == core and parent["count"] == rev["old_count"] and
                parent["extensions"] == plan["extensions"][:index], "Extension parent does not match the immutable plan family")
    if verify_source:
        require(core["code_sha256"] == code_digest(), "Plan uses different analysis source")
    return plan


def extend(parent, parent_dir, *, count, cap, justification):
    require(justification.strip() and parent["count"] in (16, 32) and count == 2*parent["count"],
            "Extend 16 to 32 or 32 to 64 with a documented sampling justification")
    return plan_document(parent["core"], count, cap, [*parent["extensions"],
        {"parent_sha256": parent["document_sha256"], "parent_dir": str(plain(parent_dir)), "old_count": parent["count"],
         "new_count": count, "justification": justification}])


def verify_seeds(plan):
    for row in plan["core"]["checkpoints"]:
        checkpoint = row["stage2_row"]["evidence"]["binding"]["best_checkpoint_identity"]
        names = [t["name"] for t in row["stage2_row"]["layout"]["tensors"]]
        seeds = {probe_seed(checkpoint, p, name) for p, name in product(range(1, plan["count"]+1), names)}
        require(len(seeds) == plan["count"]*len(names), "Hutchinson seed collision")
