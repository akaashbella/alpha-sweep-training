"""Validate the completed Stage 2 pilot and rebind through pinned native readers."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from analysis.layer_sensitivity_stage2.records import inspect_checkpoint, run_root
from analysis.layer_sensitivity_stage2.storage import published as stage2_publication

from .protocol import HISTORICAL_COMMIT, HISTORICAL_SOURCE, corrected, pilot_membership
from .storage import CHECKOUT, code_digest, identity, plain, require, seal, sha_file


def stage2_plan(directory):
    try:
        plan = stage2_publication(directory, "plan", "plan.json")
        pilot_membership(plan)
        return plan
    except (ValueError, OSError, KeyError, TypeError) as exc:
        raise ValueError(f"Need the actual completed Stage 2 pilot plan in a supported publication schema: {exc}") from exc


def stage2_checkpoint(plan, row, root):
    corrected(row)
    require(row["status"] == "eligible", "Stage 2 row was excluded")
    path = plain(root / "checkpoints" / row["checkpoint_id"])
    before = sha_file(path / "COMPLETE.json")
    completion = stage2_publication(path, "checkpoint_complete", "checkpoint_complete.json")
    responses, _, inventory, complete, _ = inspect_checkpoint(plan, row, root)
    require(complete and len(responses) == row["passes"] and
            completion["realizations"] == {key: value["document_sha256"] for key, value in responses.items()} and
            before == sha_file(path / "COMPLETE.json"), "Stage 2 checkpoint lacks a stable verified complete publication")
    slots = {}
    for result in responses.values():
        slot = result["slot"]
        slots[slot["target_id"], slot["evaluation_alpha"], slot["realization"]] = result
    return slots, inventory


def git_head(root):
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "--verify", "HEAD"], check=True,
                            capture_output=True, text=True, timeout=30)
    return result.stdout.strip()


def reader_pin(root, *, historical):
    root = plain(root)
    require((root/"src"/"alpha_sweep_training"/"evaluation_inputs.py").is_file(), "Missing native checkpoint reader checkout")
    commit = git_head(root)
    if historical:
        require(commit == HISTORICAL_COMMIT, f"Historical CIFAR-10 MobileNet reader requires commit {HISTORICAL_COMMIT}")
    executable = Path(sys.executable).expanduser().absolute()
    require(executable.is_file(), "Missing configured Python interpreter")
    return {"root": str(root), "commit": commit, "historical": historical,
            "python": str(executable), "python_sha256": sha_file(executable.resolve())}


def pin_in_process(pin):
    from alpha_sweep_training.execution_provenance import SOURCE_INVENTORY_VERSION, source_inventory
    from alpha_sweep_training.serialization import strict_json_text

    require(reader_pin(pin["root"], historical=pin["historical"]) == {k: v for k, v in pin.items() if k != "source_sha256"},
            "Reader checkout or interpreter changed")
    files = source_inventory(pin["root"])
    actual = hashlib.sha256(strict_json_text({"version": SOURCE_INVENTORY_VERSION, "files": files}).encode()).hexdigest()
    expected = HISTORICAL_SOURCE if pin["historical"] else pin.get("source_sha256", actual)
    require(actual == expected and pin.get("source_sha256", actual) == actual, "Pinned reader source inventory differs")
    return {**pin, "source_sha256": actual}


def call_reader(pin, operation, directive):
    """Separate interpreter prevents historical and current alpha_sweep imports mixing."""
    configured = Path(sys.executable).expanduser().absolute()
    require(str(configured) == pin["python"] and sha_file(configured.resolve()) == pin["python_sha256"],
            "Launch with the exact interpreter pinned by the audit")
    native = plain(pin["root"])
    require(native == CHECKOUT or pin["historical"] is True, "Current reader must use this checkout")
    require(git_head(native) == pin["commit"] and (not pin["historical"] or pin["commit"] == HISTORICAL_COMMIT),
            "Reader Git revision changed before launch")
    bootstrap = ("import sys; sys.path[:0] = [sys.argv[1], sys.argv[2]]; "
                 "from analysis.hessian_geometry_stage3.worker import serve; serve()")
    result = subprocess.run([pin["python"], "-I", "-B", "-c", bootstrap,
                             str(native/"src"), str(CHECKOUT)],
                            input=json.dumps({"pin": pin, "operation": operation, **directive}),
                            stdout=subprocess.PIPE, encoding="utf-8")
    require(result.returncode == 0, f"Reader process exited {result.returncode} for {operation}")
    payload = json.loads(result.stdout)
    require(payload["ok"], payload.get("reason", "Historical reader rejected the request"))
    return payload["result"]


def bind(row, pin, *, metadata_only):
    from alpha_sweep_training.manifests import load_manifest, select_task
    from alpha_sweep_training.artifacts import effective_storage_config
    from analysis.layer_sensitivity_stage2.planning import audited_binding, scientific_key

    corrected(row)
    source = row["source"]
    manifest = load_manifest(plain(source["manifest"]))
    require(manifest.document_sha256 == source["manifest_sha256"], "Native manifest digest differs from Stage 2")
    task = select_task(manifest, run_id=row["run_id"])
    require(scientific_key(task.config.scientific) == row["key"], "Native manifest task differs from Stage 2")
    proof, bound = audited_binding(task, source, metadata_only=metadata_only)
    expected = row["evidence"]
    require(identity(proof) == identity({key: expected[key] for key in proof}),
            "Revalidated best binding, recipe, source, prepared data or training transaction differs from Stage 2")
    actual_cfg = effective_storage_config(bound.config, source["base_dir"])
    protected = [str(plain(source["manifest"])), str(plain(source["base_dir"])), str(plain(pin["root"]))]
    for key, path in actual_cfg.execution["storage"].items():
        if path is not None:
            protected.append(str(plain(Path(path).parent if key == "split_file" else path)))
    return bound, sorted(set(protected))


def audit(plan_dir, results, historical_checkout=None, protected=()):
    start = code_digest()
    plan_dir = plain(plan_dir)
    plan = stage2_plan(plan_dir)
    root = run_root(plan, plan_dir, results, create=False)
    pinned_stage2 = sha_file(plan_dir / "COMPLETE.json")
    roots = {str(plan_dir), str(root), *plan["protected_roots"], *(str(plain(p)) for p in protected)}
    if historical_checkout:
        roots.add(str(plain(historical_checkout)))
    pins, audits = {}, []
    for row in pilot_membership(plan):
        audit_row = {"stage2_row": row, "status": "excluded", "checkpoint_id": row["checkpoint_id"], "key": row["key"]}
        try:
            responses, inv = stage2_checkpoint(plan, row, root)
            clean = responses[("clean", 0., 0)]
            reference = root / "checkpoints" / row["checkpoint_id"] / "COMPLETE.json"
            audit_row["stage2"] = {"clean_sha256": clean["document_sha256"], "complete_file_sha256": sha_file(reference),
                                   "inventory_sha256": inv["document_sha256"], "data": clean["data"],
                                   "metrics": clean["metrics"], "runtime": clean["runtime"], "inventory": inv["inventory"]}
            historical = row["key"]["dataset"] == "cifar10" and row["key"]["architecture"] == "mobilenetv3_large"
            require(not historical or historical_checkout, f"Supply the isolated historical checkout at {HISTORICAL_COMMIT}")
            if historical not in pins:
                pins[historical] = reader_pin(historical_checkout if historical else CHECKOUT, historical=historical)
            answer = call_reader(pins[historical], "audit", {"row": row})
            if historical:
                require(answer["reader"]["source_sha256"] == HISTORICAL_SOURCE, "Historical reader source pin mismatch")
            pins[historical] = answer["reader"]
            audit_row.update(status="eligible", reader=answer["reader"], audit_software=answer["software"])
            roots.update(answer["protected_roots"])
        except (ValueError, OSError, RuntimeError, KeyError, TypeError, subprocess.SubprocessError) as exc:
            audit_row["reason"] = f"{type(exc).__name__}: {exc}"
        audits.append(audit_row)
    require(start == code_digest() and pinned_stage2 == sha_file(plan_dir / "COMPLETE.json"),
            "Analysis source or Stage 2 plan marker changed during audit")
    return seal("audit", {"code_sha256": start, "stage2_plan_dir": str(plan_dir), "stage2_results": str(root),
                         "stage2_plan_sha256": plan["document_sha256"], "stage2_plan_marker_sha256": pinned_stage2,
                         "protected_roots": sorted(roots), "checkpoints": audits})
