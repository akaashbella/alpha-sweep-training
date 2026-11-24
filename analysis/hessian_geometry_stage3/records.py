"""Immutable per-probe records and bounded resumption across plan extensions."""

from contextlib import contextmanager
import math
import os
import time
import uuid

from .protocol import PRESERVATION, decomposition, is_digest, probe_seed
from .storage import disjoint, identity, install, lease, plain, read, require, seal, sha_file
from .worker import validate_runtime


def namespace(plan, directory, results, *, create=False):
    core = plan["core"]
    roots = [*core["protected_roots"], directory, *(revision["parent_dir"] for revision in plan["extensions"])]
    root = disjoint(results, roots, existing=True)
    owner = seal("owner", {"work_sha256": plan["work_sha256"]})
    if create:
        root.mkdir(parents=True, exist_ok=True)
        def recoverable():
            return all(p.name == ".initialize.lock" or p.name.startswith(".OWNER.json.") and p.name.endswith(".staging")
                       for p in root.iterdir())
        require((root/"OWNER.json").exists() or recoverable(), "Existing output is not a Stage 3 run namespace")
        with lease(root/".initialize.lock"):
            if not (root/"OWNER.json").exists():
                require(recoverable(), "Run namespace has no ownership authority")
                install(root/"OWNER.json", owner)
    require(read(root/"OWNER.json", "owner") == owner, "Run namespace belongs to another plan family")
    return root


def binding(plan, row):
    previous = row["stage2_row"]["evidence"]["binding"]
    return {"work_sha256": plan["work_sha256"], "checkpoint_id": row["checkpoint_id"], "key": row["key"],
            "checkpoint": previous["best_checkpoint_identity"], "training_identity": previous["training_identity"],
            "manifest_sha256": row["stage2_row"]["source"]["manifest_sha256"], "reader": row["reader"],
            "training_source_sha256": row["stage2_row"]["evidence"]["best_training_provenance"]["code"]["source_sha256"],
            "stage2_plan_sha256": plan["core"]["stage2_plan_sha256"]}


def validate_inventory(inv, plan, row):
    require(inv["binding"] == binding(plan, row) and inv["layout"] == row["stage2_row"]["layout"], "Curvature inventory binding mismatch")
    expected = {t["name"]: t["population_std"] for t in row["stage2"]["inventory"]["tensors"]}
    require(set(inv["scales"]) == set(expected) and all(type(v) in (float, int) and math.isfinite(v) and v >= 0 and
            math.isclose(v, expected[k], rel_tol=2e-6, abs_tol=1e-12) for k, v in inv["scales"].items()), "Invalid audited weight scales")


def validate_probe(record, plan, row, probe, inv):
    core = plan["core"]
    require(probe in range(1, plan["count"]+1) and type(record["probe"]) is int and record["probe"] == probe and
            record["binding"] == binding(plan, row) and record["inventory_sha256"] == inv["document_sha256"],
            "Probe/checkpoint/plan identity mismatch")
    require(record["status"] == "success" and record["preservation"] == PRESERVATION and
            record["randomness"] == core["randomness"] and record["distribution"] == core["distribution"],
            "Missing numerical preservation or randomness evidence")
    require(record["seeds"] == {t: probe_seed(record["binding"]["checkpoint"], probe, t) for t in inv["scales"]},
            "Probe seeds do not match the immutable identity")
    data = record["data"]
    result = record["measurement"]
    n = core["examples"]
    require(data["prepared"] == row["stage2"]["data"]["prepared"] and data["split"] == core["split"] and
            data["examples"] == n and is_digest(data["ordered_indices_sha256"]), "Probe data identity mismatch")
    if core["split"] == "validation":
        require(data == row["stage2"]["data"], "Stage 2 ordered validation identity mismatch")
    else:
        require(data["ordered_indices_sha256"] == identity(list(range(n))), "Unexpected official test order")
    require(type(result["examples"]) is int and result["examples"] == n and
            type(result["batches"]) is int and result["batches"] == plan["workload"]["batches_per_probe"], "Incomplete probe pass")
    require(type(result["clean_correct"]) is int and 0 <= result["clean_correct"] <= n and
            result["clean_accuracy"] == result["clean_correct"]/n and
            type(result["clean_loss"]) in (float, int) and math.isfinite(result["clean_loss"]) and result["clean_loss"] >= 0,
            "Invalid clean loss/accuracy")
    require(decomposition(inv["layout"], result["contributions"]["tensors"]) == result["contributions"],
            "Probe tensor, stage or network contributions disagree")
    require(all(result["contributions"]["tensors"][name] == 0 for name, scale in inv["scales"].items() if scale == 0),
            "Zero-scale tensor has a nonzero contribution")
    validate_runtime(record["runtime"], core["numerics"])
    require(type(record["duration_seconds"]) in (float, int) and math.isfinite(record["duration_seconds"]) and
            record["duration_seconds"] >= 0 and all(type(record[k]) is int and record[k] >= 0 for k in
            ("peak_allocated_bytes", "peak_reserved_bytes")) and record["peak_reserved_bytes"] >= record["peak_allocated_bytes"],
            "Invalid resource measurements")


def cohort(record):
    return (record["runtime"], record["data"], *(record["measurement"][k] for k in ("clean_loss", "clean_correct", "clean_accuracy")))


def probe_file(directory, probe):
    return directory/"probes"/f"{probe:03d}.json"


def observe(plan, row, root):
    directory = plain(root/"checkpoints"/row["checkpoint_id"])
    inv = read(directory/"inventory.json", "inventory") if (directory/"inventory.json").exists() else None
    if inv is not None:
        validate_inventory(inv, plan, row)
    records, slots = {}, []
    for probe in range(1, plan["count"]+1):
        attempts = []
        path = directory/"attempts"/f"{probe:03d}"
        if path.exists():
            for name in sorted(plain(path).iterdir()):
                if name.name.startswith(".") and name.name.endswith(".staging"):
                    attempts.append({"status": "orphan_staging", "name": name.name})
                    continue
                doc = read(name, "attempt")
                require(doc["binding"] == binding(plan, row) and doc["probe"] == probe and
                        doc["status"] in ("started", "failed"), "Attempt identity mismatch")
                attempts.append(doc)
        status = "failed" if any(a["status"] == "failed" for a in attempts) else "partial" if attempts else "pending"
        path = probe_file(directory, probe)
        if path.exists():
            require(inv is not None, "Probe has no verified inventory")
            record = read(path, "probe")
            validate_probe(record, plan, row, probe, inv)
            require(not records or cohort(record) == cohort(next(iter(records.values()))),
                    "Probe cohort differs in effective runtime, ordered data or deterministic clean baseline")
            records[probe], status = record, "success"
        slots.append({"probe": probe, "status": status, "attempts": attempts})
    completed_counts = set()
    for count in (16, 32, 64):
        marker_path = directory/f"COMPLETE-{count:03d}.json"
        if marker_path.exists() and count <= plan["count"]:
            marker = read(marker_path, "checkpoint_complete")
            require(all(p in records for p in range(1, count+1)) and marker == completion(plan, row, directory, count),
                    "Checkpoint completion checksum or probe coverage mismatch")
            completed_counts.add(count)
    marker = plan["count"] in completed_counts
    staging = sorted(str(p.relative_to(directory)) for p in directory.glob("**/.*.staging") if p.is_file())
    return records, inv, {"checkpoint_id": row["checkpoint_id"], **row["key"], "publication_complete": marker,
                          "slots": slots, "staging": staging}


def completion(plan, row, directory, count):
    return seal("checkpoint_complete", {"binding": binding(plan, row), "count": count,
                "inventory_file_sha256": sha_file(directory/"inventory.json"),
                "probe_file_sha256": {str(p): sha_file(probe_file(directory, p)) for p in range(1, count+1)}})


@contextmanager
def attempt(directory, plan, row, probe):
    token = uuid.uuid4().hex
    attempt_dir = directory/"attempts"/f"{probe:03d}"
    doc = {"binding": binding(plan, row), "probe": probe, "attempt_id": token,
           "allocation": {k: os.environ[k] for k in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
                                                      "SLURM_RESTART_COUNT", "CUDA_VISIBLE_DEVICES") if k in os.environ}}
    install(attempt_dir/f"{token}.started.json", seal("attempt", {**doc, "status": "started"}))
    started = time.perf_counter()
    try:
        yield doc
    except BaseException as exc:
        install(attempt_dir/f"{token}.failed.json", seal("attempt", {**doc, "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}", "duration_seconds": time.perf_counter()-started}))
        raise


def execute(request, reader):
    from contextlib import ExitStack
    import torch
    from .estimator import measure_probe
    from .protocol import load_plan
    from .worker import components, effective_backend

    plan_dir = request["plan_dir"]
    plan = load_plan(plan_dir, verify_source=True)
    require(plan["document_sha256"] == request["plan_sha256"], "Plan changed before isolated execution")
    row = next(r for r in plan["core"]["checkpoints"] if r["checkpoint_id"] == request["checkpoint_id"])
    require(row["reader"] == reader and row["status"] == "eligible", "Isolated reader does not match the audited checkpoint")
    desired = request["probes"]
    require(len(set(desired)) == len(desired) and all(type(p) is int and 1 <= p <= plan["count"] for p in desired)
            and 0 < len(desired) <= request["max_probes"], "Use unique planned probe IDs within the explicit run cap")
    root = namespace(plan, plan_dir, request["results"], create=True)
    directory = plain(root/"checkpoints"/row["checkpoint_id"])
    with lease(directory/".worker.lock"), ExitStack() as lifetime:
        existing, inv, coverage = observe(plan, row, root)
        failed = {slot["probe"] for slot in coverage["slots"] if slot["status"] == "failed"}
        pending = sorted(set(desired)-existing.keys())
        require(request["retry_failed"] or not (failed & set(pending)), "Failed probe requires explicit --retry-failed")
        worker = None
        for probe in pending:
            with attempt(directory, plan, row, probe) as log:
                if worker is None:
                    worker = lifetime.enter_context(components(plan["core"], row, request["device"]))
                    candidate = seal("inventory", {"binding": binding(plan, row), "layout": worker["layout"], "scales": worker["floats"]})
                    validate_inventory(candidate, plan, row)
                    if inv is None:
                        install(directory/"inventory.json", candidate)
                        inv = candidate
                    require(inv == candidate, "Resume changed clean population scales or inventory")
                require(effective_backend() == worker["runtime"]["effective"], "Effective curvature backend changed before probe")
                device = worker["device"]
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                result = measure_probe(worker["model"], worker["loader"], worker["prepare"], worker["entries"],
                                       worker["scales"], worker["layout"], log["binding"]["checkpoint"], probe,
                                       classes=worker["classes"], expected=plan["core"]["examples"],
                                       batches=plan["workload"]["batches_per_probe"], guard=worker["guard"])
                torch.cuda.synchronize(device)
                require(effective_backend() == worker["runtime"]["effective"], "Effective curvature backend changed during probe")
                record = seal("probe", {**log, "status": "success", "preservation": PRESERVATION,
                             "randomness": plan["core"]["randomness"], "distribution": plan["core"]["distribution"],
                             "seeds": {t: probe_seed(log["binding"]["checkpoint"], probe, t) for t in inv["scales"]},
                             "measurement": result, "inventory_sha256": inv["document_sha256"],
                             "data": worker["data"], "runtime": worker["runtime"], "authorized_plan_sha256": plan["document_sha256"],
                             "duration_seconds": time.perf_counter()-started, "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                             "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)})
                validate_probe(record, plan, row, probe, inv)
                require(not existing or cohort(record) == cohort(next(iter(existing.values()))),
                        "Resume would mix different runtime settings or clean baseline metrics")
                install(probe_file(directory, probe), record)
                existing[probe] = record
        if len(existing) == plan["count"] and not coverage["publication_complete"]:
            install(directory/f"COMPLETE-{plan['count']:03d}.json", completion(plan, row, directory, plan["count"]))
    return {"checkpoint_id": row["checkpoint_id"], "executed": pending, "verified_probes": len(existing),
            "expected_probes": plan["count"], "checkpoint_publication_complete": len(existing) == plan["count"]}
