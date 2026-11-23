"""Independent Stage 1 command; never invokes the experiment's execution CLI."""

import argparse
import sys

from .analysis import analyze
from .contracts import AnalysisError
from .loading import load_summaries
from .output import publish, validate_output_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Analyze existing completed reporting-v1 summaries; no live scientific storage reads")
    parser.add_argument("--report-dir", required=True, help="Completed summaries publication directory containing summaries.json")
    parser.add_argument("--output", required=True, help="New directory outside input, checkout and managed stores")
    parser.add_argument("--protected-root", action="append", required=True,
                        help="Repeat for every research/storage/workflow/original-report base; summaries do not embed these paths")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)
    try:
        if not 72 <= args.dpi <= 1200:
            raise AnalysisError("DPI must be in 72..1200")
        output = validate_output_path(args.output, args.report_dir, args.protected_root)
        source = load_summaries(args.report_dir)
        result = analyze(source.document, source.seeds)
        outcome = publish(source, result, output, args.protected_root, dpi=args.dpi)
        import json
        print(json.dumps(outcome, sort_keys=True))
        return 0
    except (AnalysisError, OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
        print(f"Stage 1 analysis rejected/failed: {exc}", file=sys.stderr)
        return 2
