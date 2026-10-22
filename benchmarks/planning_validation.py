"""Synthetic CPU/filesystem benchmark, not training or an experiment artifact.

Run with src on PYTHONPATH: python -B benchmarks/planning_validation.py
All temporary bytes are synthetic and removed. No torch, datasets or scheduler.
Reports validation reads separately from filesystem-cache/physical I/O; Windows
conservatively hashes snapshots and cannot measure POSIX stat-only I/O savings.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from time import perf_counter
from types import SimpleNamespace

from alpha_sweep_training import metrics
from alpha_sweep_training.planning_inspection import PlanningInspection


def legacy_layers(context, record, statistics, *, include_header):
    rows = [{**metrics._identity_columns(context), "epoch_index": record.epoch_index,
             "completed_epochs": record.completed_epochs, "status": record.status,
             "boundary": "after_training_updates_before_clean_validation", **asdict(statistic),
             "shape": metrics._json_cell(statistic.shape), "std_correction": 0, "weights": "clean"}
            for statistic in statistics]
    return metrics._csv(rows, include_header=include_header)


def csv_inputs(args):
    context = metrics.RecordContext({"run_id": "synthetic-only", "attempt_id": "synthetic",
        "provenance": {"code": {"inventory": [{"path": f"src/synthetic/component_{i:04}.py",
            "sha256": f"{i:064x}"} for i in range(args.inventory_entries)]},
            "software": {"python": "synthetic", "torch": "not loaded"}}})
    statistics = tuple(metrics.LayerStatistic(f"layer.{i}.weight", "Conv2d", (16, 16, 3, 3),
                        2304, i * 0.001) for i in range(args.layers))
    return context, statistics


def csv_benchmark(args):
    context, statistics = csv_inputs(args)

    def run(render):
        digest, size = hashlib.sha256(), 0
        start = perf_counter()
        for epoch in range(args.epochs):
            record = SimpleNamespace(epoch_index=epoch, completed_epochs=epoch + 1, status="training")
            text = render(context, record, statistics, include_header=epoch == 0).encode("utf-8")
            digest.update(text)
            size += len(text)
        return {"seconds": perf_counter() - start, "bytes": size, "sha256": digest.hexdigest()}

    print("Benchmark: legacy CSV reconstruction", file=sys.stderr, flush=True)
    before = run(legacy_layers)
    print("Benchmark: optimized CSV reconstruction", file=sys.stderr, flush=True)
    after = run(metrics.layer_statistics_csv)
    if (before["bytes"], before["sha256"]) != (after["bytes"], after["sha256"]):
        raise AssertionError("CSV byte equivalence failed")
    return {"baseline": before, "optimized": after,
            "speedup": before["seconds"] / after["seconds"],
            "identity_serializations_before": args.epochs * args.layers,
            "identity_serializations_after": args.epochs}


def read_benchmark(args, *, reconstruct_csv=False):
    with tempfile.TemporaryDirectory(prefix="alpha-planning-benchmark-") as temp:
        roots, expected = [], {}
        block = b"synthetic-checkpoint-bytes-only\n"
        payload = (block * (args.checkpoint_bytes // len(block) + 1))[:args.checkpoint_bytes]
        print("Benchmark: creating temporary synthetic read workload", file=sys.stderr, flush=True)
        for task in range(args.tasks):
            root = Path(temp) / str(task)
            root.mkdir()
            roots.append(root)
            for role in ("latest", "best", "final"):
                path = root / f"{role}.bin"
                path.write_bytes(payload)
                expected[path] = hashlib.sha256(payload).hexdigest()
            for epoch in range(args.epochs):
                path = root / f"epoch-{epoch}.json"
                value = json.dumps({"synthetic_epoch": epoch, "synthetic_identity": "not research data"}).encode()
                path.write_bytes(value)
                expected[path] = hashlib.sha256(value).hexdigest()

        def run(cached):
            cache = PlanningInspection()
            counters = {"validations": 0, "validator_read_bytes": 0, "csv_reconstructions": 0}
            context, statistics = csv_inputs(args)
            render = metrics.layer_statistics_csv if cached else legacy_layers

            def validate(root):
                counters["validations"] += 1
                digest = hashlib.sha256()
                for path in sorted(root.iterdir()):
                    raw = path.read_bytes()
                    counters["validator_read_bytes"] += len(raw)
                    if hashlib.sha256(raw).hexdigest() != expected[path]:
                        raise AssertionError("Synthetic artifact digest mismatch")
                    if path.suffix == ".json":
                        json.loads(raw)
                    digest.update(raw)
                if reconstruct_csv:
                    counters["csv_reconstructions"] += 1
                    for epoch in range(args.epochs):
                        record = SimpleNamespace(epoch_index=epoch, completed_epochs=epoch + 1, status="training")
                        digest.update(render(context, record, statistics, include_header=epoch == 0).encode("utf-8"))
                return digest.hexdigest()

            start = perf_counter()
            results = []
            for index in range(args.passes):
                print(f"Benchmark: {'cached' if cached else 'baseline'} pass {index + 1}/{args.passes}, "
                      f"CSV reconstruction={reconstruct_csv}", file=sys.stderr, flush=True)
                values = []
                for root in roots:
                    values.append(cache.reuse(str(root), (root,), lambda: validate(root)) if cached else validate(root))
                results.append(values)
            if cached:
                cache.assert_unchanged()
            return {**counters, "seconds": perf_counter() - start, "cache_hits": cache.hits}, results

        print("Benchmark: repeated uncached deep reads", file=sys.stderr, flush=True)
        before, before_values = run(False)
        print("Benchmark: operation-local reuse plus final freshness guard", file=sys.stderr, flush=True)
        after, after_values = run(True)
        if before_values != after_values:
            raise AssertionError("Read observation equivalence failed")
        return {"baseline": before, "optimized": after,
                "speedup": before["seconds"] / after["seconds"],
                "windows_snapshot_content_hash_fallback": os.name == "nt",
                "read_bytes_scope": "Deep validator only; excludes snapshot fallback reads, OS cache and physical I/O"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--layers", type=int, default=64)
    parser.add_argument("--inventory-entries", type=int, default=96)
    parser.add_argument("--tasks", type=int, default=8)
    parser.add_argument("--passes", type=int, default=5)
    parser.add_argument("--checkpoint-bytes", type=int, default=2 * 1024 * 1024)
    args = parser.parse_args()
    if any(value < 1 for value in vars(args).values()):
        parser.error("All workload sizes must be positive")
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "workload": vars(args), "csv": csv_benchmark(args), "reads": read_benchmark(args),
              "planning_with_csv": read_benchmark(args, reconstruct_csv=True)}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
