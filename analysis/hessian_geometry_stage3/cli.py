"""Stage 3 command-line entry point for separately authorized execution."""

import argparse
import json
import subprocess
import sys

from .protocol import EXAMPLES, extend, load_plan, make_plan, verify_seeds
from .storage import disjoint, plain, publication, publish, require


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def parser():
    app = argparse.ArgumentParser(prog="python -m analysis.hessian_geometry_stage3", description=__doc__)
    commands = app.add_subparsers(dest="command", required=True)
    cmd = commands.add_parser("audit", help="Validate the completed Stage 2 pilot and rebind each checkpoint read-only")
    cmd.add_argument("--stage2-plan-dir", required=True)
    cmd.add_argument("--stage2-results", required=True)
    cmd.add_argument("--historical-checkout", help="Native corrected CIFAR-10 MobileNet checkout")
    cmd.add_argument("--protected-root", action="append", required=True, help="Stage 1 results and external original storage roots. Repeat as needed")
    cmd.add_argument("--output", required=True)
    cmd = commands.add_parser("plan", help="Freeze the pilot or append a documented probe budget extension")
    inputs = cmd.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--audit-dir")
    inputs.add_argument("--parent-plan")
    cmd.add_argument("--output", required=True)
    cmd.add_argument("--max-total-probes", type=positive, required=True)
    cmd.add_argument("--backend", choices=("cudnn", "native"))
    cmd.add_argument("--batch-size", type=positive)
    cmd.add_argument("--split", choices=tuple(EXAMPLES))
    cmd.add_argument("--probes", choices=(16, 32, 64), type=int)
    cmd.add_argument("--justification")
    cmd = commands.add_parser("run", help="Execute one audited checkpoint and explicitly selected probes")
    cmd.add_argument("--plan-dir", required=True)
    cmd.add_argument("--results", required=True)
    cmd.add_argument("--checkpoint", required=True)
    cmd.add_argument("--probe", type=positive, nargs="+", required=True, help="Exact probe IDs")
    cmd.add_argument("--max-probes", type=positive, required=True)
    cmd.add_argument("--device", default="cuda:0")
    cmd.add_argument("--retry-failed", action="store_true")
    cmd = commands.add_parser("summarize", help="Publish complete-only traces and coverage")
    cmd.add_argument("--plan-dir", required=True)
    cmd.add_argument("--results", required=True)
    cmd.add_argument("--output", required=True)
    cmd = commands.add_parser("compare", help="Join audited Stage 2 responses and optional original Stage 1 reporting")
    cmd.add_argument("--plan-dir", required=True)
    cmd.add_argument("--summary-dir", required=True)
    cmd.add_argument("--stage1-report")
    cmd.add_argument("--stage1-collection")
    cmd.add_argument("--historical-evaluation-root", help="Existing corrected native test sweep, read only")
    cmd.add_argument("--output", required=True)
    cmd = commands.add_parser("plot", help="Render reproducible PNG and PDF figures")
    cmd.add_argument("--summary-dir", required=True)
    cmd.add_argument("--comparison-dir")
    cmd.add_argument("--output", required=True)
    cmd.add_argument("--dpi", type=positive, default=240)
    cmd = commands.add_parser("inspect", help="Check checksums and optional raw probe coverage")
    selected = cmd.add_mutually_exclusive_group(required=True)
    selected.add_argument("--plan-dir")
    selected.add_argument("--publication-dir")
    cmd.add_argument("--kind", choices=("audit", "plan", "summary", "comparison", "figures"))
    cmd.add_argument("--results")
    return app


def dispatch(args):
    if args.command == "audit":
        from .provenance import audit

        protected = [*args.protected_root, args.stage2_plan_dir, args.stage2_results]
        disjoint(args.output, [*protected, *([args.historical_checkout] if args.historical_checkout else [])])
        report = audit(args.stage2_plan_dir, args.stage2_results, args.historical_checkout, args.protected_root)
        root = publish(args.output, report["protected_roots"], "audit", report)
        rows = report["checkpoints"]
        return {"output": str(root), "requested": len(rows), "eligible": sum(r["status"] == "eligible" for r in rows),
                "exclusions": [{"checkpoint_id": r["checkpoint_id"], "key": r["key"], "reason": r["reason"]} for r in rows if r["status"] != "eligible"]}
    if args.command == "plan":
        if args.parent_plan:
            require(args.split is None and args.batch_size is None and args.backend is None, "An extension preserves split, batch size and backend")
            require(args.probes is not None and args.justification, "An extension needs --probes and --justification")
            parent = load_plan(args.parent_plan, verify_source=True)
            plan = extend(parent, args.parent_plan, count=args.probes, cap=args.max_total_probes, justification=args.justification)
        else:
            require(args.probes in (None, 16) and args.justification is None, "The initial budget is 16 probes per checkpoint")
            plan = make_plan(publication(args.audit_dir, "audit"), args.audit_dir, split=args.split or "validation",
                             batch_size=args.batch_size or 32, backend=args.backend or "cudnn", cap=args.max_total_probes)
        verify_seeds(plan)
        protect = [*plan["core"]["protected_roots"], *(r["parent_dir"] for r in plan["extensions"])]
        root = publish(args.output, protect, "plan", plan)
        return {"output": str(root), **plan["workload"]}
    if args.command == "plot":
        from .figures import plot

        require(72 <= args.dpi <= 1200, "DPI must be 72–1200")
        return plot(args.summary_dir, args.comparison_dir, args.output, args.dpi)
    if args.command == "inspect" and args.publication_dir:
        require(args.kind and not args.results, "Publication inspection needs --kind. Probe inspection needs --plan-dir")
        doc = publication(args.publication_dir, args.kind)
        rows = doc.get("checkpoints", []) if args.kind == "audit" else []
        return {"publication_complete": True, "document_sha256": doc["document_sha256"],
                "audit_checkpoints": [{k: row[k] for k in ("checkpoint_id", "key", "status", "reason") if k in row} for row in rows]}
    require(args.command != "inspect" or args.kind is None, "--kind applies to --publication-dir")
    plan = load_plan(args.plan_dir, verify_source=args.command == "run")
    if args.command == "run":
        from .provenance import call_reader

        rows = [r for r in plan["core"]["checkpoints"] if r["checkpoint_id"] == args.checkpoint]
        require(len(rows) == 1, "Select one exact audited checkpoint ID")
        require(len(set(args.probe)) == len(args.probe) <= args.max_probes and max(args.probe) <= plan["count"],
                "Select unique planned probe IDs within --max-probes")
        return call_reader(rows[0]["reader"], "run", {"checkpoint_id": args.checkpoint,
                           "plan_dir": str(plain(args.plan_dir)), "plan_sha256": plan["document_sha256"],
                           "results": str(plain(args.results)), "probes": args.probe, "max_probes": args.max_probes,
                           "device": args.device, "retry_failed": args.retry_failed})
    if args.command == "summarize":
        from .summary import summarize

        return summarize(plan, plain(args.plan_dir), plain(args.results), args.output)
    if args.command == "compare":
        from .comparison import compare

        return compare(plan, args.summary_dir, args.output, stage1_report=args.stage1_report,
                       stage1_collection=args.stage1_collection, historical_evaluation_root=args.historical_evaluation_root)
    report = {"workload": plan["workload"], "plan_sha256": plan["document_sha256"],
              "work_sha256": plan["work_sha256"], "checkpoints": []}
    if args.results:
        from .records import namespace, observe

        root = namespace(plan, args.plan_dir, args.results)
        for row in plan["core"]["checkpoints"]:
            try:
                _, _, status = observe(plan, row, root)
            except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
                status = {"checkpoint_id": row["checkpoint_id"], **row["key"], "publication_complete": False, "rejected": str(exc)}
            report["checkpoints"].append(status)
        report["scientific_coverage_complete"] = all(r["publication_complete"] for r in report["checkpoints"])
    else:
        report["checkpoints"] = [{k: row[k] for k in ("checkpoint_id", "key", "status")} for row in plan["core"]["checkpoints"]]
    return report


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        print(json.dumps(dispatch(args), indent=2, sort_keys=True, allow_nan=False))
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, ImportError, subprocess.SubprocessError) as exc:
        print(f"Stage 3 {args.command}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0
