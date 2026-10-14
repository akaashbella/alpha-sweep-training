"""Production reporting CLI; no research execution, repairs or scheduler calls."""

import argparse
import sys

from .controller import _execution_options, _overrides
from .manifests import load_manifest
from .serialization import strict_json_text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect", help="Read/export every manifest task and prescribed slot, including unavailable observations")
    collect.add_argument("--manifest", required=True)
    collect.add_argument("--source-root", required=True)
    collect.add_argument("--workflow-root", required=True, help="Reserved workflow store, even before execution; protects derived output")
    collect.add_argument("--evaluation-root")
    collect.add_argument("--protected-root", action="append", default=[], help="Additional operational/Slurm log/preparation roots")
    collect.add_argument("--output", required=True, help="New derived directory outside all managed stores and the base/checkout")
    _execution_options(collect)
    for name in ("summarize", "plot"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--collection-dir", required=True)
        cmd.add_argument("--output", required=True)
        cmd.add_argument("--source-root", required=True)
        if name == "plot":
            cmd.add_argument("--formats", nargs="+", default=["png", "pdf", "svg"], choices=("png", "pdf", "svg"))
            cmd.add_argument("--dpi", type=int, default=200)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--report-dir", required=True)
    args = parser.parse_args(argv)
    try:
        from .report_publication import publish_collection, publish_summaries, load_collection, inspect_report
        if args.command == "collect":
            from .results import collect_results
            collection = collect_results(load_manifest(args.manifest), base_dir=args.base_dir, source_root=args.source_root,
                overrides=_overrides(args), evaluation_root=args.evaluation_root, workflow_root=args.workflow_root,
                extra_protected=args.protected_root)
            result = publish_collection(collection, output_dir=args.output, source_root=args.source_root)
        elif args.command == "inspect":
            result = inspect_report(args.report_dir)
        else:
            collection = load_collection(args.collection_dir)
            if args.command == "summarize":
                result = publish_summaries(collection, output_dir=args.output, source_root=args.source_root, collection_dir=args.collection_dir)
            else:
                from .plotting import plot_results
                result = plot_results(collection, output_dir=args.output, source_root=args.source_root,
                                      collection_dir=args.collection_dir, formats=tuple(args.formats), dpi=args.dpi)
        print(strict_json_text(result), end="")
        return 0
    except Exception as exc:
        print(strict_json_text({"reporting_error": f"{type(exc).__name__}: {exc}", "exit_code": 22,
                               "scientific_completion_asserted": False}), end="", file=sys.stderr)
        return 22


if __name__ == "__main__":
    raise SystemExit(main())
