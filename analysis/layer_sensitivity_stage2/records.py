"""Plan-bound raw records, resumable passes and read-only inspection."""

from contextlib import contextmanager
import math
import os
import platform
import time
import uuid

from alpha_sweep_training.spec import BATCH_SIZE

from .planning import planned_slots, tensor_seed
from .storage import disjoint, finish, identity, install, lease, plain, published, read, require, seal, source_digest


def slot_id(slot):
    return identity(slot)[:24]


def checkpoint_root(root, row):
    return plain(root / "checkpoints" / row["checkpoint_id"])


def run_root(plan, plan_dir, path, *, create):
    root = disjoint(path, [*plan["protected_roots"], plan_dir], existing=True)
    expected = seal("run_owner", {"plan_sha256": plan["document_sha256"]})
    def recoverable():
        return all(p.name == ".initialize.lock" or (p.name.startswith(".OWNER.json.") and p.name.endswith(".staging"))
                   for p in root.iterdir())

    if create:
        root.parent.mkdir(parents=True, exist_ok=True)
        try:
            root.mkdir()
        except FileExistsError:
            pass
        require((root / "OWNER.json").exists() or recoverable(), "Existing output has no Stage 2 owner")
        with lease(root / ".initialize.lock"):
            if not (root / "OWNER.json").exists():
                require(recoverable(), "Existing output has no Stage 2 owner")
                install(root / "OWNER.json", expected)
    require(read(root / "OWNER.json", "run_owner") == expected, "Output namespace belongs to another plan")
    return root


def linkage(plan, row):
    return {"plan_sha256": plan["document_sha256"], "checkpoint_id": row["checkpoint_id"],
            "key": row["key"], "run_id": row["run_id"], "source_sha256": plan["source_sha256"],
            "checkpoint": row["evidence"]["binding"]["best_checkpoint_identity"],
            "training_identity": row["evidence"]["binding"]["training_identity"],
            "optimizer_recipe": row["evidence"]["scientific"]["optimizer"]}


def read_inventory(plan, row, path):
    doc = read(path, "inventory")
    require(doc["binding"] == linkage(plan, row), "Inventory checkpoint identity mismatch")
    layout = doc["inventory"]
    require({k: v for k, v in layout.items() if k != "tensors"} == {k: v for k, v in row["layout"].items() if k != "tensors"}
            and [{k: v for k, v in t.items() if k != "population_std"} for t in layout["tensors"]] == row["layout"]["tensors"],
            "Inventory differs from the audited target layout")
    require(all(type(t["population_std"]) in (int, float) and math.isfinite(t["population_std"]) and t["population_std"] >= 0
                for t in layout["tensors"]), "Invalid clean population standard deviation")
    return doc


def selected_names(row, slot):
    if slot["target_id"] == "clean":
        return []
    return next(t["tensors"] for t in row["targets"] if t["id"] == slot["target_id"])


def validate_record(doc, plan, row, slot, inventory_doc, clean):
    from .measurement import changes, normalization

    require(doc["binding"] == linkage(plan, row) and doc["slot"] == slot and
            doc["inventory_sha256"] == inventory_doc["document_sha256"] and doc["status"] == "success",
            "Realization identity/inventory mismatch")
    data = doc["data"]
    n = plan["examples_per_pass"]
    require(data["prepared"] == row["evidence"]["prepared_data"] and data["split"] == plan["split"] and data["examples"] == n,
            "Realization data identity/coverage mismatch")
    require(type(data["ordered_indices_sha256"]) is str and len(data["ordered_indices_sha256"]) == 64 and
            all(c in "0123456789abcdef" for c in data["ordered_indices_sha256"]), "Missing ordered example identity")
    require(doc["preservation"] == "exact_clean_parameters_buffers_bindings_and_modes", "Missing preservation evidence")
    require(type(doc["duration_seconds"]) in (int, float) and math.isfinite(doc["duration_seconds"]) and doc["duration_seconds"] >= 0,
            "Invalid pass duration")
    metrics = doc["metrics"]
    require(type(metrics["loss"]) in (int, float) and math.isfinite(metrics["loss"]) and metrics["loss"] >= 0 and
            type(metrics["correct"]) is int and 0 <= metrics["correct"] <= n and metrics["examples"] == n and
            metrics["batches"] == (n+BATCH_SIZE-1)//BATCH_SIZE and metrics["accuracy"] == metrics["correct"]/n and
            metrics["accuracy_unit"] == "fraction" and metrics["loss_unit"] == "mean_cross_entropy_nats_per_example",
            "Malformed or incomplete measured metrics")
    preds = bytes.fromhex(doc["predictions"])
    labels = bytes.fromhex(doc["labels"] if clean is None else clean["labels"])
    classes = row["evidence"]["scientific"]["model"]["num_classes"]
    require(len(preds) == len(labels) == n and max(preds) < classes and max(labels) < classes, "Incomplete or invalid class predictions")
    if clean is None:
        require(slot["target_id"] == "clean" and doc["clean_sha256"] is None, "Unexpected clean reference")
        clean_metrics, clean_preds = metrics, preds
    else:
        require(doc["clean_sha256"] == clean["document_sha256"] and doc["labels"] is None and
                data == clean["data"] and doc["runtime"] == clean["runtime"], "Baseline/data/runtime pairing mismatch")
        clean_metrics, clean_preds = clean["metrics"], bytes.fromhex(clean["predictions"])
    delta = changes(clean_metrics, metrics, labels, clean_preds, preds)
    targets = selected_names(row, slot)
    seeds = {name: tensor_seed(doc["binding"]["checkpoint"], slot["evaluation_alpha"], slot["realization"], name) for name in targets}
    require(doc["tensor_seeds"] == seeds and len(set(seeds.values())) == len(seeds) and doc["randomness_version"] == plan["randomness_version"],
            "Realization seed derivation mismatch")
    require(doc["clean_metrics"] == clean_metrics and doc["changes"] == delta and
            doc["normalization"] == normalization(inventory_doc["inventory"], targets, slot["evaluation_alpha"], delta["delta_loss"]),
            "Paired changes/normalization mismatch")


def inspect_checkpoint(plan, row, root):
    directory = checkpoint_root(root, row)
    inv = read_inventory(plan, row, directory / "inventory.json") if (directory / "inventory.json").exists() else None
    records, coverage, clean = {}, [], None
    slots = list(planned_slots(plan, row))
    expected_files = {f"{slot_id(slot)}.json" for slot in slots}
    if (directory / "slots").exists():
        require(all(p.name in expected_files or (p.name.startswith(".") and p.name.endswith(".staging"))
                    for p in plain(directory / "slots").iterdir()), "Unexpected realization file")
    for slot in slots:
        item = slot_id(slot)
        path = directory / "slots" / f"{item}.json"
        detail = {"checkpoint_id": row["checkpoint_id"], **slot, "status": "pending", "attempts": []}
        attempt_dir = directory / "attempts" / item
        if attempt_dir.exists():
            for entry in sorted(plain(attempt_dir).iterdir()):
                if entry.name.startswith(".") and entry.name.endswith(".staging"):
                    detail["attempts"].append({"unconfirmed_staging_file": entry.name})
                    continue
                attempt = read(entry, "attempt")
                require(attempt["binding"] == linkage(plan, row) and attempt["slot"] == slot and
                        attempt["status"] in ("started", "failed"), "Attempt identity mismatch")
                detail["attempts"].append(attempt)
            if detail["attempts"]:
                detail["status"] = "failed" if any(a.get("status") == "failed" for a in detail["attempts"]) else "partial"
        if path.exists():
            require(inv is not None and (slot["target_id"] == "clean" or clean is not None), "Missing inventory or clean authority")
            doc = read(path, "realization")
            validate_record(doc, plan, row, slot, inv, clean)
            if slot["target_id"] == "clean":
                clean = doc
            records[item] = doc
            detail["status"] = "success"
        coverage.append(detail)
    marker = directory / "COMPLETE.json"
    if marker.exists():
        doc = published(directory, "checkpoint_complete", "checkpoint_complete.json")
        refs = read(marker, "publication")["files"]
        require(set(refs) == {"inventory.json", "checkpoint_complete.json", *(f"slots/{key}.json" for key in records)},
                "Checkpoint completion marker must bind inventory and all prescribed files")
        require(len(records) == row["passes"] and doc["binding"] == linkage(plan, row) and
                doc["realizations"] == {key: value["document_sha256"] for key, value in records.items()},
                "Checkpoint completion marker disagrees with prescribed validated slots")
    evidence = {"staging_files": [str(p.relative_to(directory)) for p in sorted(directory.glob(".*.staging"))],
                "worker_attempts": []}
    jobs = directory / "workers"
    if jobs.exists():
        for path in sorted(plain(jobs).iterdir()):
            if path.name.startswith(".") and path.name.endswith(".staging"):
                evidence["staging_files"].append(f"workers/{path.name}")
                continue
            job = read(path, "worker_attempt")
            require(job["binding"] == linkage(plan, row) and job["status"] in ("started", "failed", "finished"),
                    "Worker attempt identity mismatch")
            evidence["worker_attempts"].append(job)
    return records, coverage, inv, marker.exists(), evidence


@contextmanager
def worker_attempt(directory, plan, row, slots):
    token = uuid.uuid4().hex
    path = directory / "workers"
    allocation = {key: os.environ[key] for key in ("SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_RESTART_COUNT",
                                                  "CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG") if key in os.environ}
    attempt = {"binding": linkage(plan, row), "attempt": token, "slots": slots, "allocation": allocation}
    install(path / f"{token}.started.json", seal("worker_attempt", {**attempt, "status": "started"}))
    try:
        yield token
    except BaseException as exc:
        install(path / f"{token}.failed.json", seal("worker_attempt", {**attempt, "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}"}))
        raise
    else:
        install(path / f"{token}.finished.json", seal("worker_attempt", {**attempt, "status": "finished"}))


def run_selected(plan, plan_dir, row, root_path, *, target_ids, max_passes, retry_failed,
                 evaluation_alphas=None, realizations=None):
    from .components import prepared_worker
    from .measurement import changes, measure, normalization

    require(plan["source_sha256"] == source_digest(), "Plan source differs from current checkout")
    require(row["status"] == "eligible", f"Checkpoint excluded by provenance audit: {row.get('reason')}")
    allowed = {t["id"] for t in row["targets"]}
    require(not target_ids or set(target_ids) <= allowed, f"Unknown planned target. Available: {', '.join(sorted(allowed))}")
    require(not evaluation_alphas or set(evaluation_alphas) <= set(plan["evaluation_alphas"][1:]), "Select only planned positive evaluation alphas")
    require(not realizations or all(1 <= r <= plan["realizations_per_positive_alpha"] for r in realizations), "Select only planned positive realization IDs")
    root = run_root(plan, plan_dir, root_path, create=True)
    directory = checkpoint_root(root, row)
    with lease(directory / ".worker.lock"):
        existing, coverage, inv_doc, complete, _ = inspect_checkpoint(plan, row, root)
        selected = [s for s in planned_slots(plan, row) if slot_id(s) not in existing and
                    (s["target_id"] == "clean" or
                     ((not target_ids or s["target_id"] in target_ids) and
                      (not evaluation_alphas or s["evaluation_alpha"] in evaluation_alphas) and
                      (not realizations or s["realization"] in realizations)))]
        failed = {slot_id({key: c[key] for key in ("target_id", "evaluation_alpha", "realization")})
                  for c in coverage if c["status"] == "failed"}
        selected = selected[:max_passes]
        if not retry_failed:
            require(not any(slot_id(s) in failed for s in selected), "Failed selected pass requires --retry-failed")
        if selected:
            with worker_attempt(directory, plan, row, selected) as worker_id, prepared_worker(row, plan["split"], plan["examples_per_pass"]) as worker:
                model, loader, runtime, baseline, inv, scales, data, runtime_info = worker
                runtime_info["python"] = platform.python_version()
                candidate = seal("inventory", {"binding": linkage(plan, row), "inventory": inv})
                if inv_doc is None:
                    install(directory / "inventory.json", candidate)
                    inv_doc = candidate
                require(inv_doc == candidate, "Current loaded clean inventory differs from stored inventory")
                clean_slot = next(planned_slots(plan, row))
                clean = existing.get(slot_id(clean_slot))
                if clean is not None:
                    require(clean["data"] == data and clean["runtime"] == runtime_info, "Resume requires compatible clean baseline data and runtime")
                for slot in selected:
                    item, token = slot_id(slot), uuid.uuid4().hex
                    attempt_dir = directory / "attempts" / item
                    attempt = {"binding": linkage(plan, row), "slot": slot, "attempt": token}
                    install(attempt_dir / f"{token}.started.json", seal("attempt", {**attempt, "status": "started"}))
                    started = time.perf_counter()
                    try:
                        runtime.require_active()
                        names = selected_names(row, slot)
                        num_classes = row["evidence"]["scientific"]["model"]["num_classes"]
                        metrics, labels, predictions, seeds = measure(
                            model, loader, baseline=baseline, device=runtime.device, classes=num_classes, expected=data["examples"],
                            scales=scales, selected=names, checkpoint=attempt["binding"]["checkpoint"],
                            alpha=slot["evaluation_alpha"], realization=slot["realization"],
                            prepare_batch=lambda batch: runtime.batch(batch, num_classes), autocast=runtime.autocast,
                        )
                        require(clean is None or labels.hex() == clean["labels"], "Ordered labels differ from clean pass")
                        clean_metrics = metrics if clean is None else clean["metrics"]
                        clean_predictions = predictions if clean is None else bytes.fromhex(clean["predictions"])
                        delta = changes(clean_metrics, metrics, labels, clean_predictions, predictions)
                        doc = seal("realization", {"binding": attempt["binding"], "slot": slot, "status": "success",
                            "metrics": metrics, "clean_metrics": clean_metrics, "data": data, "runtime": runtime_info,
                            "labels": labels.hex() if clean is None else None, "predictions": predictions.hex(),
                            "clean_sha256": None if clean is None else clean["document_sha256"],
                            "inventory_sha256": inv_doc["document_sha256"], "changes": delta,
                            "normalization": normalization(inv, names, slot["evaluation_alpha"], delta["delta_loss"]),
                            "preservation": "exact_clean_parameters_buffers_bindings_and_modes",
                            "worker_attempt": worker_id, "duration_seconds": time.perf_counter()-started,
                            "randomness_version": plan["randomness_version"], "tensor_seeds": seeds})
                        validate_record(doc, plan, row, slot, inv_doc, clean)
                        install(directory / "slots" / f"{item}.json", doc)
                        existing[item] = doc
                        if slot["target_id"] == "clean":
                            clean = doc
                    except BaseException as exc:
                        install(attempt_dir / f"{token}.failed.json", seal("attempt", {**attempt, "status": "failed",
                                "reason": f"{type(exc).__name__}: {exc}"}))
                        raise
        if len(existing) == row["passes"] and not complete:
            marker = seal("checkpoint_complete", {"binding": linkage(plan, row),
                          "realizations": {key: doc["document_sha256"] for key, doc in existing.items()}})
            path = directory / "checkpoint_complete.json"
            if path.exists():
                require(read(path, "checkpoint_complete") == marker, "Interrupted checkpoint marker does not match verified results")
            else:
                install(path, marker)
            finish(directory, "checkpoint_complete", ["checkpoint_complete.json", "inventory.json",
                   *(f"slots/{key}.json" for key in existing)])
        return {"checkpoint_id": row["checkpoint_id"], "executed_passes": len(selected), "accepted_passes": len(existing),
                "planned_passes": row["passes"], "checkpoint_complete": len(existing) == row["passes"]}
