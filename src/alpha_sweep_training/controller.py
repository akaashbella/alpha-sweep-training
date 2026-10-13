"""Production manifest/selection/status/one-task CLI; computation is lazy."""

import argparse
from pathlib import Path
import sys

from .config import _read_yaml, load_config
from .manifests import (ManifestError, build_manifest, load_manifest, load_phase, manifest_from_dict,
                        save_manifest_document, select_task, task_config, validate_phase_document)
from .serialization import read_json, strict_json_text
from .spec import ARCHITECTURES, DATASETS


def _selectors(parser):
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--global-index", type=int)
    group.add_argument("--phase-index", type=int)
    group.add_argument("--run-id")
    parser.add_argument("--phase", type=int)


def _execution_options(parser):
    parser.add_argument("--base-dir", default=".", help="Base for selected storage paths; use persistent shared storage")
    parser.add_argument("--execution-overrides", help="Strict JSON file of typed allowed dotted execution fields")
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--output-root")
    parser.add_argument("--checkpoint-root")
    parser.add_argument("--result-root")
    parser.add_argument("--log-root")
    parser.add_argument("--data-root")
    parser.add_argument("--split-file")


def _overrides(args):
    result = {} if args.execution_overrides is None else read_json(args.execution_overrides)
    if type(result) is not dict:
        raise ManifestError("Execution override JSON must be an object of typed dotted fields")
    for attr, key in (("device", "execution.device"), ("num_workers", "execution.num_workers"),
        *((name, f"execution.storage.{name}") for name in
          ("output_root", "checkpoint_root", "result_root", "log_root", "data_root", "split_file"))):
        value = getattr(args, attr)
        if value is not None:
            if key in result:
                raise ManifestError(f"Conflicting execution override sources for {key}")
            result[key] = value
    return result


def _selection(manifest, args):
    return select_task(manifest, global_index=args.global_index, phase=args.phase,
                       phase_index=args.phase_index, run_id=args.run_id)


def _build(args):
    bases = {name: load_config(getattr(args, f"{name}_config")) for name in DATASETS}
    models = {name: _read_yaml(Path(args.models_dir) / f"{name}.yaml") for name in ARCHITECTURES}
    manifest = build_manifest(bases, models, seeds=args.seeds, base_dir=args.base_dir)
    output = Path(args.output).expanduser().absolute()
    phase_paths = [Path(args.phase1_output).expanduser().absolute() if args.phase1_output else
                   output.with_name(output.stem + ".phase1.json"),
                   Path(args.phase2_output).expanduser().absolute() if args.phase2_output else
                   output.with_name(output.stem + ".phase2.json")]
    paths = [output, *phase_paths]
    if len({path.resolve() for path in paths}) != len(paths):
        raise ManifestError("Full/phase output paths must be distinct")
    previous = load_manifest(output) if output.exists() else None
    documents = [manifest.to_dict(), manifest.phase_view(1), manifest.phase_view(2)]
    # Check the whole existing bundle before any file publication. Invalid old
    # documents remain evidence even when replacement was explicitly requested.
    for index, path in enumerate(paths):
        if any(p.is_symlink() or getattr(p, "is_junction", lambda: False)() for p in (path, *path.parents)):
            raise ManifestError(f"Linked manifest output: {path}")
        if path.exists():
            if index == 0:
                manifest_from_dict(read_json(path))
            else:
                try:
                    validate_phase_document(read_json(path), manifest)
                except ManifestError:
                    if previous is None:
                        raise
                    validate_phase_document(read_json(path), previous)
            if strict_json_text(read_json(path)) != strict_json_text(documents[index]) and not args.replace_manifests:
                raise ManifestError(f"Valid conflicting output {path}; explicit --replace-manifests required")
    # Publish phases first and the full parent last. Interrupted replacement
    # retains the old parent until both views exist, and accepts each old/target
    # compatible view on reinvocation without inventing an independent parent.
    publication = [None] * len(paths)
    for index in (1, 2, 0):
        publication[index] = save_manifest_document(paths[index], documents[index], parent=manifest,
            previous_parent=previous, replace=args.replace_manifests)
    return {"grid_id": manifest.grid_id, "document_sha256": manifest.document_sha256,
            "tasks": len(manifest.tasks), "outputs": [
                {"path": str(path), "operation": result} for path, result in zip(paths, publication)]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Validate finalized inputs and publish full + linked phase documents")
    for dataset in DATASETS:
        build.add_argument(f"--{dataset}-config", required=True)
    build.add_argument("--models-dir", default="configs/models")
    build.add_argument("--seeds", type=int, nargs=3, required=True, help="Explicit approved seed plan in order")
    build.add_argument("--base-dir", default=".")
    build.add_argument("--output", required=True)
    build.add_argument("--phase1-output")
    build.add_argument("--phase2-output")
    build.add_argument("--replace-manifests", action="store_true")
    inspect = commands.add_parser("inspect", help="Validate full document and optionally its phase view")
    inspect.add_argument("--manifest", required=True)
    inspect.add_argument("--phase-view")
    select = commands.add_parser("select", help="Return exactly one validated task")
    select.add_argument("--manifest", required=True)
    _selectors(select)
    _execution_options(select)
    run = commands.add_parser("run", help="Execute exactly one task; no Slurm/signals/retries")
    run.add_argument("--manifest", required=True)
    _selectors(run)
    _execution_options(run)
    run.add_argument("--source-root", required=True, help="Checkout actually imported by this interpreter")
    run.add_argument("--operation", choices=("auto", "new", "continue", "recover", "overwrite"), default="auto")
    run.add_argument("--overwrite-run-id")
    run.add_argument("--numerical-reference", help="Strict JSON ArtifactReference from coverage; recover only")
    status = commands.add_parser("status", help="Read-only observations joined to all 240 expected tasks")
    status.add_argument("--manifest", required=True)
    _execution_options(status)
    args = parser.parse_args(argv)
    try:
        code = 0
        if args.command == "build":
            result = _build(args)
        else:
            manifest = load_manifest(args.manifest)
            if args.command == "inspect":
                phase = None if args.phase_view is None else load_phase(args.phase_view, manifest)
                result = {"grid_id": manifest.grid_id, "document_sha256": manifest.document_sha256,
                          "tasks": len(manifest.tasks), "seeds": list(manifest.seeds),
                          "phase_tasks": None if phase is None else len(phase)}
            elif args.command == "status":
                from .coverage import collect_coverage
                result = collect_coverage(manifest, base_dir=args.base_dir, overrides=_overrides(args))
            else:
                task = _selection(manifest, args)
                overrides = _overrides(args)
                task_config(task, overrides)  # Reject science/invalid types before execution imports.
                if args.command == "select":
                    result = {"task": task.to_dict(), "effective_config": task_config(task, overrides).to_dict()}
                else:
                    from .task_execution import execute_task
                    outcome = execute_task(manifest, task, base_dir=args.base_dir, source_root=args.source_root,
                        overrides=overrides, operation=args.operation, overwrite_run_id=args.overwrite_run_id,
                        numerical_reference=None if args.numerical_reference is None else read_json(args.numerical_reference))
                    result, code = outcome.to_dict(), int(outcome.exit_code)
        sys.stdout.write(strict_json_text(result))
        return code
    except Exception as exc:
        sys.stderr.write(strict_json_text({"outcome": "rejected", "exit_code": 22,
                                          "diagnostic": f"{type(exc).__name__}: {exc}"}))
        return 22


if __name__ == "__main__":
    raise SystemExit(main())
