"""Read-only provenance audit and explicit, deterministic pass accounting."""

from itertools import product
from pathlib import Path

from alpha_sweep_training.config import RunConfig
from alpha_sweep_training.identity import canonical_alpha, identity_metadata
from alpha_sweep_training.manifests import load_manifest, select_task, task_config
from alpha_sweep_training.serialization import read_json
from alpha_sweep_training.spec import ALPHAS, ARCHITECTURES, DATASETS, EVALUATION_ALPHAS, TEST_SIZE, VALIDATION_SIZE

from .groups import model_layout, targets
from .storage import identity, plain, require, seal, sha_file, source_digest

MOBILE = "mobilenetv3_large"
CORRECTED_EPSILON = 0.0316
PILOT_TRAIN = (0., .10, .30)
PILOT_EVAL = (.01, .02, .05, .10, .20)
RANDOMNESS = "stage2-sha256-checkpoint-alpha-realization-tensor-v1"


def tensor_seed(checkpoint, alpha, realization, tensor):
    digest = identity({"randomness_version": RANDOMNESS, "checkpoint": checkpoint,
                       "evaluation_alpha": canonical_alpha(alpha), "realization": realization, "tensor": tensor})
    return int.from_bytes(bytes.fromhex(digest[:16]), "little")


def candidate_key(dataset, architecture, alpha, seed):
    return {"dataset": dataset, "architecture": architecture, "training_alpha": alpha, "training_seed": seed}


def scientific_key(science):
    return candidate_key(science["dataset"], science["architecture"], science["alpha"], science["training_seed"])


def require_corrected(science, optimizer=None):
    if science["architecture"] != MOBILE:
        return
    recipe = science["optimizer"]
    diagnostic = "Stage 2 requires corrected MobileNetV3-Large RMSprop epsilon=0.0316"
    if science["dataset"] == "cifar10":
        diagnostic += "; this checkout has no validated CIFAR-10 corrected specification or run identity"
    require(recipe.get("name") == "rmsprop" and recipe.get("epsilon") == CORRECTED_EPSILON, diagnostic)
    if optimizer is not None:
        require(optimizer.get("declared_recipe") == recipe and optimizer.get("effective_options", {}).get("eps") == CORRECTED_EPSILON,
                diagnostic + "; saved best optimizer metadata does not prove the corrected recipe")


def evidence(task, bound):
    metadata = bound.payload["metadata"]
    science = RunConfig(metadata["config"]).scientific
    optimizer = metadata["component_metadata"]["optimizer"]
    require(science == task.config.scientific and metadata["identity"] == identity_metadata(task.config)
            and bound.binding["training_identity"] == metadata["identity"], "Best checkpoint scientific identity mismatch")
    require(metadata["role"] == "best" and metadata["clean_weights"] is True and type(metadata["format_version"]) is int
            and metadata["format_version"] == 1, "Expected validated clean best checkpoint format 1")
    binding = bound.binding
    require(metadata["generation"] == binding["best_checkpoint_identity"]["generation"]
            and binding["best_checkpoint"]["sha256"] == binding["best_checkpoint_identity"]["sha256"]
            and metadata["best"] == binding["best_validation"], "Best selection/generation/digest binding mismatch")
    require(metadata["provenance"]["code"].get("source_sha256"), "Missing historical training source digest")
    require(optimizer["declared_recipe"] == science["optimizer"], "Best optimizer recipe disagrees with scientific configuration")
    require_corrected(science, optimizer)
    return {"binding": binding, "scientific": science, "format_version": metadata["format_version"],
            "component_versions": metadata["component_versions"], "best_training_provenance": metadata["provenance"],
            "best_optimizer": optimizer, "training_transaction": bound.training_transaction,
            "prepared_data": task.prepared_data}


def load_source(path, base, overrides):
    from alpha_sweep_training.artifacts import effective_storage_config

    manifest_file = plain(path)
    base_dir = plain(base)
    manifest = load_manifest(manifest_file)
    protected = {str(base_dir), str(manifest_file)}
    for task in manifest.tasks:
        cfg = effective_storage_config(task_config(task, overrides), base_dir)
        for key, value in cfg.execution["storage"].items():
            if value is not None:
                protected.add(str(plain(Path(value).parent if key == "split_file" else value)))
    reference = {"manifest": str(manifest_file), "manifest_sha256": manifest.document_sha256,
                 "base_dir": str(base_dir), "overrides": overrides}
    return manifest, reference, protected


def audited_binding(task, reference, *, metadata_only):
    from alpha_sweep_training.evaluation_inputs import bind_evaluation_checkpoint
    from alpha_sweep_training.planning_inspection import PlanningInspection

    require_corrected(task.config.scientific)
    # The original binding validates journal history, source/digest, optimizer
    # state and restricted checkpoint formats. No compatibility is waived.
    with PlanningInspection().activate():
        bound = bind_evaluation_checkpoint(task, base_dir=reference["base_dir"], overrides=reference["overrides"],
                                           metadata_only=metadata_only)
        proof = evidence(task, bound)
    return proof, bound


def audit(inputs_file, training_alphas, seed_policy):
    from alpha_sweep_training.models import build_model, describe_model

    inputs_file = plain(inputs_file)
    code = source_digest()
    input_digest = sha_file(inputs_file)
    inputs = read_json(inputs_file)
    require(set(inputs) <= {"sources", "protected_roots", "training_seed_plan"} and
            type(inputs["sources"]) is list and inputs["sources"] and
            type(inputs["protected_roots"]) is list and inputs["protected_roots"],
            "Inputs require sources and all external workflow/evaluation/report protected roots")
    protected = {str(inputs_file), *(str(plain(p)) for p in inputs["protected_roots"])}
    sources, source_errors = [], []
    for declaration in inputs["sources"]:
        require(set(declaration) <= {"manifest", "base_dir", "overrides", "run_ids"}, "Unexpected source declaration keys")
        path, base = plain(declaration["manifest"]), plain(declaration["base_dir"])
        protected.update((str(path), str(base)))
        # Resolve declared override roots even if the manifest is unreadable.
        overrides = declaration.get("overrides", {})
        for key, value in overrides.items():
            if key.startswith("execution.storage.") and value is not None:
                override = plain(base / Path(value).expanduser())
                protected.add(str(override.parent if key.endswith(".split_file") else override))
        try:
            manifest, reference, roots = load_source(path, base, overrides)
            protected.update(roots)
            run_ids = declaration.get("run_ids")
            chosen = manifest.tasks if run_ids is None else [select_task(manifest, run_id=r) for r in run_ids]
            require(len({t.run_id for t in chosen}) == len(chosen), "Duplicate declared run IDs")
            sources.append((manifest, reference, chosen))
        except Exception as exc:
            source_errors.append({"manifest": str(path), "reason": f"{type(exc).__name__}: {exc}"})

    declared_seeds = inputs.get("training_seed_plan")
    from alpha_sweep_training.manifests import seed_plan

    if declared_seeds is not None:
        seeds = tuple(seed_plan(declared_seeds))
    else:
        require(sources, "No readable manifest supplies a seed plan. Declare training_seed_plan with three predetermined seeds to publish exclusions")
        seeds = sources[0][0].seeds
    require(all(manifest.seeds == seeds for manifest, _, _ in sources), "Source manifests disagree with the predetermined ordered seed plan")
    selected_seeds = seeds[:1] if seed_policy == "first" else seeds
    rows = []
    for dataset, arch, alpha, seed in product(DATASETS, ARCHITECTURES, training_alphas, selected_seeds):
        key = candidate_key(dataset, arch, alpha, seed)
        candidates, rejected = {}, []
        for _, ref, chosen in sources:
            for task in chosen:
                if scientific_key(task.config.scientific) != key:
                    continue
                try:
                    require_corrected(task.config.scientific)
                    candidates[identity({"source": ref, "run_id": task.run_id})] = (task, ref)
                except ValueError as exc:
                    rejected.append({"run_id": task.run_id, "source": ref, "reason": str(exc)})
        row = {"key": key, "status": "excluded", "rejected_candidates": rejected}
        if len(candidates) != 1:
            row["reason"] = "Ambiguous accepted declarations; select exact run IDs and roots" if candidates else "No compatible candidate in supplied manifests"
            if dataset == "cifar10" and arch == MOBILE and not candidates:
                row["reason"] += ". This checkout has no validated corrected CIFAR-10 MobileNet epsilon=0.0316 contract"
        else:
            task, ref = next(iter(candidates.values()))
            row.update(run_id=task.run_id, source=ref)
            try:
                proof, bound = audited_binding(task, ref, metadata_only=True)
                model = build_model(bound.config)
                require(describe_model(model).to_dict() == bound.payload["metadata"]["component_metadata"]["model"],
                        "Constructed model metadata differs from the original best checkpoint")
                layout = model_layout(model, arch)
                require(layout["tensors"] == bound.payload["metadata"]["component_metadata"]["model"]["eligible_weights"],
                        "Constructed eligible inventory differs from checkpoint")
                row.update(status="eligible", evidence=proof, layout=layout)
            except Exception as exc:
                row["reason"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    # Within one dataset, comparisons must refer to one prepared data identity.
    for dataset in DATASETS:
        own = [r for r in rows if r["key"]["dataset"] == dataset and r["status"] == "eligible"]
        if len({identity(r["evidence"]["prepared_data"]) for r in own}) > 1:
            for row in own:
                row.update(status="excluded", reason="Dataset candidates disagree on prepared data identities")
    require(source_digest() == code and sha_file(inputs_file) == input_digest, "Source or input selection changed during audit")
    return seal("audit", {"inputs_file": str(inputs_file), "inputs_sha256": input_digest, "source_errors": source_errors,
                         "source_sha256": code, "protected_roots": sorted(protected),
                         "training_seed_plan": list(seeds), "selected_seeds": list(selected_seeds), "checkpoints": rows})


def make_plan(audit_doc, *, split, alphas, realizations, scopes, max_passes, audit_dir=None):
    require(audit_doc["source_sha256"] == source_digest(), "Audit source differs from this checkout; repeat audit")
    size = VALIDATION_SIZE if split == "validation" else TEST_SIZE
    rows = []
    for row in audit_doc["checkpoints"]:
        item = dict(row, checkpoint_id=identity(row["key"])[:20])
        if row["status"] == "eligible":
            chosen = targets(row["layout"], scopes)
            item.update(targets=chosen, passes=1+len(chosen)*len(alphas)*realizations)
        rows.append(item)
    expected = sum(item.get("passes", 0) for item in rows)
    require(expected <= max_passes, f"Plan exceeds explicit cap: {expected} passes > {max_passes}")
    for row in rows:
        if row["status"] != "eligible":
            continue
        checkpoint = row["evidence"]["binding"]["best_checkpoint_identity"]
        seen = set()
        for tensor, alpha, rep in product(row["layout"]["tensors"], alphas, range(1, realizations+1)):
            seed = tensor_seed(checkpoint, alpha, rep, tensor["name"])
            require(seed not in seen, "Seed collision across the prescribed tensor/alpha/realization keys")
            seen.add(seed)
    protected = sorted({*audit_doc["protected_roots"], *([str(plain(audit_dir))] if audit_dir is not None else [])})
    return seal("plan", {"audit_sha256": audit_doc["document_sha256"], "source_sha256": audit_doc["source_sha256"],
                         "protected_roots": protected, "split": split, "examples_per_pass": size,
                         "evaluation_alphas": [0., *alphas], "realizations_per_positive_alpha": realizations,
                         "scopes": list(scopes), "randomness_version": RANDOMNESS, "selected_seeds": audit_doc["selected_seeds"],
                         "estimates": {"checkpoints_requested": len(rows), "checkpoints_eligible": sum("passes" in r for r in rows),
                                       "targets": sum(len(r.get("targets", ())) for r in rows), "passes": expected,
                                       "expected_examples": size*expected, "max_passes": max_passes},
                         "checkpoints": rows})


def planned_slots(plan, row):
    yield {"target_id": "clean", "evaluation_alpha": 0., "realization": 0}
    for target, alpha, rep in product(row["targets"], plan["evaluation_alphas"][1:],
                                      range(1, plan["realizations_per_positive_alpha"]+1)):
        yield {"target_id": target["id"], "evaluation_alpha": alpha, "realization": rep}


def rebind(row):
    source = row["source"]
    manifest = load_manifest(plain(source["manifest"]))
    require(manifest.document_sha256 == source["manifest_sha256"], "Manifest changed since provenance audit")
    task = select_task(manifest, run_id=row["run_id"])
    require(scientific_key(task.config.scientific) == row["key"], "Planned checkpoint identity mismatch")
    proof, bound = audited_binding(task, source, metadata_only=False)
    require(proof == row["evidence"], "Best checkpoint/training transaction changed since audit")
    return bound
