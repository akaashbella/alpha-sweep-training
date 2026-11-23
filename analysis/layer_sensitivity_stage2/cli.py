"""Independent commands for a reviewed, bounded Stage 2 study."""

import argparse
import json
import sys

from alpha_sweep_training.spec import ALPHAS, EVALUATION_ALPHAS

from .planning import PILOT_EVAL, PILOT_TRAIN, audit, make_plan
from .storage import plain, publish_document, published, require


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Use a positive integer")
    return number


def grid(values, approved, label):
    require(len(set(values)) == len(values) and set(values) <= set(approved), f"Use unique approved {label}: {approved}")
    return tuple(value for value in approved if value in values)


def parser():
    root = argparse.ArgumentParser(prog="python -m analysis.layer_sensitivity_stage2", description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    p = commands.add_parser("audit", help="Verify read-only historical checkpoint authority on CPU")
    p.add_argument("--inputs", required=True)
    p.add_argument("--output", required=True)
    choice = p.add_mutually_exclusive_group()
    choice.add_argument("--training-alphas", type=float, nargs="+", default=PILOT_TRAIN)
    choice.add_argument("--all-training-alphas", action="store_true")
    p.add_argument("--seeds", choices=("first", "all"), default="first")
    p = commands.add_parser("plan", help="Plan exact pass counts from a completed audit")
    p.add_argument("--audit-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--evaluation-alphas", type=float, nargs="+", default=PILOT_EVAL, help="Positive approved values. A clean pass is implicit")
    p.add_argument("--realizations", type=positive, default=8)
    p.add_argument("--scopes", nargs="+", choices=("stages", "tensors", "diagnostics"), default=("stages",))
    p.add_argument("--max-plan-passes", type=positive, required=True)
    p = commands.add_parser("run", help="Execute a bounded selection for one audited checkpoint on an assigned GPU")
    p.add_argument("--plan-dir", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--checkpoint", required=True, help="Exact checkpoint_id in plan.json")
    p.add_argument("--target", action="append", help="Exact target id, repeatable. Missing clean pass runs first")
    p.add_argument("--evaluation-alpha", type=float, action="append", help="Repeat to select planned positive alphas")
    p.add_argument("--realization", type=positive, action="append", help="Repeat to select planned positive realization IDs")
    p.add_argument("--max-passes", required=True, type=positive)
    p.add_argument("--retry-failed", action="store_true")
    p = commands.add_parser("summarize", help="Aggregate validated independent result records")
    p.add_argument("--plan-dir", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("plot", help="Export PNG and PDF figures from a completed summary")
    p.add_argument("--summary-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--evaluation-alpha", type=float, default=.05)
    p.add_argument("--profile-scope", choices=("stage", "tensor", "diagnostic"), default="stage")
    p.add_argument("--profile-target", action="append", default=[], help="Exact target name within the plotted scope")
    p.add_argument("--dpi", type=positive, default=240)
    p = commands.add_parser("inspect", help="Check publication integrity or all prescribed raw slots")
    selection = p.add_mutually_exclusive_group(required=True)
    selection.add_argument("--plan-dir")
    selection.add_argument("--publication-dir")
    p.add_argument("--kind", choices=("audit", "plan", "summary", "figures"))
    p.add_argument("--results", help="Optional run namespace to verify against the plan")
    p.add_argument("--include-slots", action="store_true")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "audit":
            alphas = ALPHAS if args.all_training_alphas else grid(args.training_alphas, ALPHAS, "training alphas")
            document = audit(args.inputs, alphas, args.seeds)
            publish_document(args.output, document["protected_roots"], "audit", document)
            rows = document["checkpoints"]
            outcome = {"output": str(plain(args.output)), "requested": len(rows), "eligible": sum(r["status"] == "eligible" for r in rows),
                       "source_errors": document["source_errors"]}
        elif args.command == "plan":
            audited = published(args.audit_dir, "audit", "audit.json")
            alphas = grid(args.evaluation_alphas, EVALUATION_ALPHAS[1:], "positive evaluation alphas")
            require(len(set(args.scopes)) == len(args.scopes), "Duplicate target scope")
            scopes = [s for s in ("stages", "tensors", "diagnostics") if s in args.scopes]
            plan = make_plan(audited, split=args.split, alphas=alphas, realizations=args.realizations,
                             scopes=scopes, max_passes=args.max_plan_passes, audit_dir=args.audit_dir)
            publish_document(args.output, [args.audit_dir, *plan["protected_roots"]], "plan", plan)
            outcome = {"output": str(plain(args.output)), **plan["estimates"]}
        elif args.command == "plot":
            from .figures import plot

            require(72 <= args.dpi <= 1200 and args.evaluation_alpha in EVALUATION_ALPHAS[1:], "Use DPI from 72 to 1200 and an approved positive alpha")
            outcome = plot(args.summary_dir, args.output, alpha=args.evaluation_alpha, dpi=args.dpi,
                           scope=args.profile_scope, profile_targets=args.profile_target)
        elif args.command == "inspect" and args.publication_dir:
            require(args.kind is not None and not args.results and not args.include_slots,
                    "Publication inspection requires --kind. Raw slot inspection requires --plan-dir")
            doc = published(args.publication_dir, args.kind, f"{args.kind}.json")
            outcome = {"publication_complete": True, "kind": args.kind, "document_sha256": doc["document_sha256"]}
        else:
            plan = published(args.plan_dir, "plan", "plan.json")
            if args.command == "run":
                from .records import run_selected

                selected = [row for row in plan["checkpoints"] if row["checkpoint_id"] == args.checkpoint]
                require(len(selected) == 1, "Select one exact checkpoint_id from plan.json")
                outcome = run_selected(plan, args.plan_dir, selected[0], args.results, target_ids=args.target,
                                       max_passes=args.max_passes, retry_failed=args.retry_failed,
                                       evaluation_alphas=args.evaluation_alpha, realizations=args.realization)
            elif args.command == "summarize":
                from .report import summarize

                outcome = summarize(plan, args.plan_dir, args.results, args.output)
            else:
                outcome = {"plan_sha256": plan["document_sha256"], **plan["estimates"]}
                if args.results:
                    from .report import collect

                    coverage = collect(plan, args.plan_dir, args.results)["coverage"]
                    outcome["run_complete"] = all(r["status"] == "complete" for r in coverage)
                    outcome["executable_passes_complete"] = all(r["status"] in ("complete", "excluded") for r in coverage)
                    outcome["checkpoints"] = coverage if args.include_slots else [{k: v for k, v in r.items() if k != "slots"} for r in coverage]
                else:
                    outcome["checkpoints"] = [{k: r[k] for k in ("checkpoint_id", "key", "status", "reason", "passes") if k in r}
                                              for r in plan["checkpoints"]]
        print(json.dumps(outcome, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError) as exc:
        print(f"Stage 2 {args.command} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
