"""Frozen MobileNet epsilon study and direct three-seed checkpointed pilot.

No scheduler calls, test-set iteration, diagnostic training loop, or automatic
replacement. Build/report are explicit later CPU operations; run is GPU training.
"""

import argparse
from dataclasses import asdict
import os
from pathlib import Path
import sys

from .config import ConfigError, RunConfig, load_config
from .identity import scientific_fingerprint
from .manifests import (DATA_KEYS, MOBILE_GRID_VERSION, Manifest, ManifestError, Task,
                        _grid_id, _sealed, _check_seal, _same, load_manifest,
                        manifest_from_dict, save_manifest_document, select_task)
from .serialization import atomic_write_json, fsync_directory, read_json, strict_json_text
from .spec import ALPHAS, MOBILE_EPSILON_PROFILE, SPECIFICATION_VERSION, optimizer_recipe

# Exact seed-2 alpha-zero scientific identity supplied by the paired evidence.
# Execution paths are deliberately absent from this hash.
AUDITED_BASELINE_FINGERPRINT = "89deb14382b735fd8a8f59cb790886e93f48eb19560993c7e43cb4d6bf564099"
ORIGINAL_ENV = "/nfs/hpc/share/bellaak/research/alpha-sweep/alpha-sweep-training/.venv"
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def candidate_config(original, candidate_root, *, seed=0, alpha=0.0):
    """Copy audited science, changing only profile/epsilon and task selectors."""
    original.require_research_ready()
    if (original.scientific["specification_version"] != SPECIFICATION_VERSION
            or scientific_fingerprint(original) != AUDITED_BASELINE_FINGERPRINT):
        raise ConfigError("Use the exact audited original seed-2 alpha-zero resolved configuration; no inferred data/backend defaults")
    document = original.to_dict()
    science = document["scientific"]
    science.update(specification_version=MOBILE_EPSILON_PROFILE, alpha=alpha,
                   training_seed=seed, training_seeds=[0, 1, 2])
    science["optimizer"] = optimizer_recipe("mobilenetv3_large", MOBILE_EPSILON_PROFILE)
    storage = document["execution"]["storage"]
    storage.update(output_root=str(Path(candidate_root) / "runs"),
                   checkpoint_root=None, result_root=None, log_root=None)
    # Match the paired full-data loader/process policy, outside scientific hash.
    document["execution"].update(num_workers=4, device="cuda:0")
    return RunConfig(document)


def build_study(original, candidate_root, *, base_dir):
    """Read existing preparation only; all thirty native tasks exist from day one."""
    from .artifacts import effective_storage_config
    from .prepared import load_prepared_dataset
    original = effective_storage_config(original, base_dir)
    first = candidate_config(original, candidate_root)
    bundle = load_prepared_dataset(first, base_dir=base_dir, metadata_only=True)
    prepared = {key: bundle.manifest[key] for key in DATA_KEYS
                if key not in ("official_training_identity", "official_test_identity", "preparation_identity")}
    prepared.update(preparation_identity=bundle.manifest["identity"],
                    official_training_identity=bundle.sources.training_record["identity"],
                    official_test_identity=bundle.sources.test_record["identity"])
    tasks = [Task("", 1, index, index, candidate_config(original, candidate_root, seed=seed, alpha=alpha), prepared)
             for index, (alpha, seed) in enumerate((a, s) for a in ALPHAS for s in (0, 1, 2))]
    grid = _grid_id(tasks, MOBILE_GRID_VERSION)
    tasks = tuple(Task(grid, t.phase, t.global_index, t.phase_index, t.config, t.prepared_data) for t in tasks)
    return manifest_from_dict(Manifest(tasks, (0, 1, 2), grid, MOBILE_GRID_VERSION).to_dict())


def pilot_tasks(manifest):
    manifest = manifest_from_dict(manifest.to_dict())
    if manifest.grid_version != MOBILE_GRID_VERSION:
        raise ManifestError("The pilot requires the separate thirty-run MobileNet candidate manifest")
    tasks = tuple(select_task(manifest, global_index=i) for i in (0, 1, 2))
    _same([(t.config.training_alpha, t.config.scientific["training_seed"]) for t in tasks],
          [(0.0, 0), (0.0, 1), (0.0, 2)], "exact pilot selection")
    return tasks


def _plain(path):
    from .artifacts import _is_link
    if any(_is_link(p) for p in (path, *path.parents)):
        raise ConfigError(f"Linked candidate input/output forbidden: {path}")


def _claim(root, protected, *, kind):
    from .serialization import durable_mkdir
    root = Path(root).expanduser().absolute()
    _plain(root)
    root = root.resolve()
    for item in protected:
        item = Path(item).expanduser().resolve()
        if root == item or root in item.parents or item in root.parents:
            raise ConfigError(f"New namespace overlaps protected storage: {item}")
    if root.exists():
        raise ConfigError("Namespace already exists; preserve partial/completed evidence and choose a new path")
    durable_mkdir(root.parent)
    root.mkdir()
    fsync_directory(root.parent)
    atomic_write_json(root / "OWNER.json", _sealed({"kind": kind, "profile": MOBILE_EPSILON_PROFILE}))
    return root


def _launch_inputs(activation_script):
    from .data_contract import file_digest
    configured = Path(sys.executable).absolute()
    resolved = configured.resolve(strict=True)
    activation = Path(activation_script).expanduser().absolute()
    _plain(resolved)
    _plain(activation)
    if not resolved.is_file() or not activation.is_file():
        raise ConfigError("Existing interpreter and activation script required; install nothing")
    checkout = Path(__file__).resolve().parents[2]
    batch = checkout / "scripts" / "mobile-epsilon-pilot.sh"
    declaration = checkout / "configs" / "candidates" / "mobile-eps0316.yaml"
    _plain(batch)
    _plain(declaration)
    return {"python_configured": str(configured), "python_resolved": str(resolved),
            "python_sha256": file_digest(resolved), "activation_script": str(activation),
            "activation_sha256": file_digest(activation), "modules": [],
            "batch_template": str(batch), "batch_sha256": file_digest(batch),
            "candidate_declaration": str(declaration), "candidate_declaration_sha256": file_digest(declaration)}


def _build(args):
    from .artifacts import effective_storage_config
    from .diagnostic_protocol import protected_config_roots
    from .execution_provenance import collect_provenance
    original = effective_storage_config(load_config(args.saved_config), args.base_dir)
    # All scientific, data, source and launch checks precede namespace publication.
    manifest = build_study(original, Path(args.candidate_root).expanduser().resolve(), base_dir=args.base_dir)
    pin = collect_provenance(args.source_root, attempt={"lifecycle": "candidate_pilot_plan"})
    _same(pin["software"], {"python": "3.11.15", "torch": "2.5.1+cu124", "torchvision": "0.20.1+cu124",
                           "numpy": "1.26.4", "PyYAML": "6.0.3", "cuda": "12.4", "cudnn": "90100"},
          "audited existing software environment")
    inputs = _launch_inputs(args.activation_script)
    protected = protected_config_roots(original, args.base_dir, args.source_root, args.protected_root)
    protected.append(Path(args.saved_config))
    root = _claim(args.candidate_root, protected, kind="mobile_candidate_study_v1")
    path = root / "study.json"
    save_manifest_document(path, manifest.to_dict(), parent=manifest)
    tasks = pilot_tasks(manifest)
    for task in tasks:
        atomic_write_json(root / f"seed-{task.config.scientific['training_seed']}.config.json", task.config.to_dict())
    plan = _sealed({"pilot_version": 1, "profile": MOBILE_EPSILON_PROFILE,
        "candidate_root": str(root), "manifest_path": str(path),
        "grid_id": manifest.grid_id, "manifest_document_sha256": manifest.document_sha256,
        "source_root": str(Path(args.source_root).resolve()), "base_dir": str(Path(args.base_dir).resolve()),
        "source_pin": pin["code"], "software_pin": pin["software"], "launch_inputs": inputs,
        "original_scientific_fingerprint": scientific_fingerprint(original),
        "protected_roots": [str(Path(p).resolve()) for p in protected],
        "pilot_run_ids": [t.run_id for t in tasks], "pilot_indices": [0, 1, 2],
        "study_runs": 30, "pilot_runs": 3, "epochs_per_run": 200,
        "adoption": "pending training/validation review; no automatic remaining-27 execution",
        "thread_policy": {"torch_intraop": 8, "torch_interop": 112, "environment": {k: None for k in THREAD_ENV}}})
    (root / "allocation-logs").mkdir()
    fsync_directory(root)
    atomic_write_json(root / "pilot.json", plan)  # Publish the runnable plan last.
    return plan


def load_pilot(path):
    path = Path(path).expanduser().absolute()
    _plain(path)
    plan = read_json(path)
    _check_seal(plan)
    root = Path(plan["candidate_root"])
    _plain(root / "OWNER.json")
    _plain(root / "study.json")
    _same(str(path), str(root / "pilot.json"), "pilot ownership path")
    _same(read_json(root / "OWNER.json"), _sealed({"kind": "mobile_candidate_study_v1", "profile": MOBILE_EPSILON_PROFILE}), "candidate owner")
    _same(plan["pilot_version"], 1, "pilot version")
    _same(plan["profile"], MOBILE_EPSILON_PROFILE, "pilot profile")
    _same(plan["manifest_path"], str(root / "study.json"), "owned manifest path")
    manifest = load_manifest(plan["manifest_path"])
    _same(plan["grid_id"], manifest.grid_id, "pilot grid")
    _same(plan["manifest_document_sha256"], manifest.document_sha256, "pilot manifest seal")
    _same(plan["pilot_run_ids"], [t.run_id for t in pilot_tasks(manifest)], "pilot membership")
    _same(plan["pilot_indices"], [0, 1, 2], "pilot indices")
    _same(plan["original_scientific_fingerprint"], AUDITED_BASELINE_FINGERPRINT, "audited original identity")
    _same([plan["study_runs"], plan["pilot_runs"], plan["epochs_per_run"]], [30, 3, 200], "frozen pilot budget")
    _same(plan["thread_policy"], {"torch_intraop": 8, "torch_interop": 112,
                                  "environment": {k: None for k in THREAD_ENV}}, "audited thread policy")
    # Validate all task storage bindings, including the unselected 27, rather
    # than allowing a sealed plan to redirect a continuation to original roots.
    for task in manifest.tasks:
        storage = task.config.execution["storage"]
        _same(storage["output_root"], str(root / "runs"), "owned run root")
        for key in ("checkpoint_root", "result_root", "log_root"):
            _same(storage[key], None, "candidate nested artifact roots")
    return plan, manifest


def _run(args):
    from .allocation_signals import allocation_signal_handlers
    from .interruption import StopRequest
    stop = StopRequest()
    with allocation_signal_handlers(stop):
        plan, manifest = load_pilot(args.plan)
        from .execution_provenance import collect_provenance
        _same(_launch_inputs(plan["launch_inputs"]["activation_script"]), plan["launch_inputs"], "frozen launch inputs")
        actual = collect_provenance(plan["source_root"], attempt={"lifecycle": "candidate_pilot_allocation"})
        _same(actual["code"]["source_sha256"], plan["source_pin"]["source_sha256"], "frozen pilot source")
        _same(actual["software"], plan["software_pin"], "frozen pilot software")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ConfigError("Preserve the audited CUBLAS_WORKSPACE_CONFIG=:4096:8")
        for key in THREAD_ENV:
            if os.environ.get(key) is not None:
                raise ConfigError(f"Audited thread environment requires {key} unset")
        for key, expected in {"SLURM_NTASKS": "1", "SLURM_CPUS_PER_TASK": "8",
                              "SLURM_JOB_PARTITION": "dgxh", "SLURM_JOB_NUM_NODES": "1"}.items():
            _same(os.environ.get(key), expected, f"actual pilot allocation {key}")
        if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_RESTART_COUNT", "0") != "0":
            raise ConfigError("Use a fresh explicit allocation, never Slurm automatic requeue")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not visible or "," in visible:
            raise ConfigError("Exactly one Slurm-assigned visible GPU required")
        array_index = os.environ.get("SLURM_ARRAY_TASK_ID")
        if array_index is not None:
            _same(array_index, str(args.seed), "pilot array index/actual seed")
        stop.check(at_boundary=True, phase="pilot_before_gpu")
        import torch
        torch.set_num_threads(8)
        torch.set_num_interop_threads(112)
        if torch.cuda.device_count() != 1:
            raise ConfigError("Exactly one assigned GPU required")
        props = torch.cuda.get_device_properties(0)
        if (props.name, props.major, props.minor, props.total_memory) != ("NVIDIA H100 80GB HBM3", 9, 0, 85009367040):
            raise ConfigError("Pilot requires the audited full H100; MIG slices are a different hardware comparison")
        stop.check(at_boundary=True, phase="pilot_before_dispatch")
        from .task_execution import execute_task
        task = pilot_tasks(manifest)[args.seed]
        result = execute_task(manifest, task, base_dir=plan["base_dir"], source_root=plan["source_root"],
                              operation=args.operation, stop=stop)
        return result.to_dict(), int(result.exit_code)


def validation_report(plan, manifest):
    """Read historical journal authority only, retaining all three full trajectories."""
    from .artifacts import resolve_run_paths
    from .coverage import inspect_task, _stamp, _lock_paths
    from .checkpoint_journal import read_committed
    from .checkpoint_io import ArtifactReference, read_document
    rows = []
    for task in pilot_tasks(manifest):
        observation = inspect_task(task, base_dir=plan["base_dir"])
        row = {"training_seed": task.config.scientific["training_seed"], "run_id": task.run_id,
               "scientific_fingerprint": task.fingerprint, "observation": observation.to_dict(),
               "trajectory": [], "final": None, "best": None, "selected_epoch": None,
               "training_completion_verified": False, "failures": observation.to_dict()["evidence_failures"],
               "health": {"assessment": "unreviewed", "thresholds": None,
                          "meaning": "Completion/finiteness never establishes healthy learning"}}
        try:
            if observation.condition != "ok":
                raise ConfigError(f"No stable journal authority: {observation.condition}")
            paths = resolve_run_paths(task.config, plan["base_dir"])
            before = _stamp(paths)
            saved = read_json(paths.metadata / "run_metadata.json")["provenance"]
            current = read_committed(paths, task.config, saved)
            _same(current.reference.to_dict(), observation.transaction, "report observation/transaction")
            trajectory = [{"reference": ref, "epoch": record.epoch_index + 1,
                           "record": record.to_dict(), "weight_statistics": [asdict(s) for s in statistics]}
                          for ref, (_, record, statistics) in zip(current.document["records"], current.records)]
            failure_ref = current.document["artifacts"]["failure"]
            failures = list(row["failures"])
            if failure_ref is not None:
                failures.append({"reference": failure_ref,
                                 "document": read_document(paths, ArtifactReference.from_dict(failure_ref))})
            if before != _stamp(paths) or any(p.exists() for p in _lock_paths(paths)):
                raise ConfigError("Journal/locks changed during report; no authority claimed")
            row.update(trajectory=trajectory, transaction=current.reference.to_dict(),
                       receipt=asdict(current.receipt), failures=failures, provenance=saved,
                       training_completion_verified=observation.training_complete)
            if trajectory:
                row["last_committed"] = trajectory[-1]
                row["health"]["observations"] = {
                    "validated_committed_epochs": len(trajectory),
                    "maximum_training_loss": max(e["record"]["training"]["loss"] for e in trajectory),
                    "maximum_validation_loss": max(e["record"]["validation"]["loss"] for e in trajectory),
                    "weight_statistics": "every eligible tensor at every recorded epoch, preserved in trajectory",
                    "uncommitted_or_failed_epoch_metrics": "not synthesized; preserved failure evidence is separate"}
            if current.receipt.completed_epochs == 200:
                row["final"] = trajectory[-1]
            best = current.document["best"]
            if best is not None:
                row["selected_epoch"] = best["epoch_index"] + 1
                row["best"] = trajectory[best["epoch_index"]]
        except Exception as exc:
            row["training_completion_verified"] = False
            row["inspection_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return _sealed({"kind": "mobile_candidate_validation_report_v1", "research_authority": False,
        "grid_id": manifest.grid_id, "manifest_document_sha256": manifest.document_sha256,
        "pilot_seeds": [0, 1, 2], "seeds": rows,
        "pilot_training_complete": all(r["training_completion_verified"] for r in rows),
        "study_training_complete": None, "study_expected_runs": 30,
        "adoption_decision": "pending human review of every seed's full training/validation trajectory",
        "official_test_used": False, "health_thresholds": None})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Later CPU stage: freeze thirty-run study and exactly three pilot tasks")
    build.add_argument("--saved-config", required=True)
    build.add_argument("--base-dir", required=True)
    build.add_argument("--source-root", required=True)
    build.add_argument("--candidate-root", required=True)
    build.add_argument("--activation-script", default=ORIGINAL_ENV + "/bin/activate")
    build.add_argument("--protected-root", action="append", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--plan", required=True)
    run = commands.add_parser("run", help="Later explicit GPU stage: one alpha-zero seed, native persistence/continuation")
    run.add_argument("--plan", required=True)
    run.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    run.add_argument("--operation", choices=("auto", "new", "continue", "recover"), default="auto")
    report = commands.add_parser("report", help="Later CPU stage: every seed, full trajectories, no test data")
    report.add_argument("--plan", required=True)
    report.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            result, code = _build(args), 0
        elif args.command == "run":
            result, code = _run(args)
        else:
            plan, manifest = load_pilot(args.plan)
            if args.command == "inspect":
                result = {"plan": plan, "tasks": [t.to_dict() for t in pilot_tasks(manifest)]}
            else:
                result = validation_report(plan, manifest)
                root = _claim(args.output_dir, [plan["candidate_root"], plan["source_root"], *plan["protected_roots"]],
                              kind="derived_mobile_candidate_validation_v1")
                atomic_write_json(root / "validation.json", result)
            code = 0
        print(strict_json_text(result), end="")
        return code
    except Exception as exc:
        from .interruption import TrainingInterrupted
        code = 10 if isinstance(exc, TrainingInterrupted) else 22
        print(strict_json_text({"outcome": "rejected", "diagnostic": f"{type(exc).__name__}: {exc}", "exit_code": code}), file=sys.stderr)
        return code


if __name__ == "__main__":
    raise SystemExit(main())
