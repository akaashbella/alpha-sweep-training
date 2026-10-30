"""Opt-in audit/diagnostic CLI. Every execution requires a separate explicit flag."""

import argparse
from pathlib import Path
import sys

from .config import load_config
from .data_contract import file_digest
from .diagnostic_protocol import DiagnosticError, new_output, protected_config_roots, validate_protocol
from .serialization import atomic_write_json, read_json, strict_json_text
from .diagnostic_failures import diagnostic_stage, failure_details


def _publish(output, filename, document):
    with diagnostic_stage("output", filename):
        atomic_write_json(output / filename, document)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("audit-run", "recheck-validation", "train", "health"):
        command = commands.add_parser(name)
        command.add_argument("--execute", action="store_true", help="Explicit future execution stage; no implicit runtime action")
        command.add_argument("--source-root", required=True)
        command.add_argument("--output-dir", required=True)
        command.add_argument("--protected-root", action="append", required=True,
                             help="All additional workflow/evaluation/research roots; repeat as needed")
        if name != "health":
            command.add_argument("--saved-config", required=True)
            command.add_argument("--base-dir", required=True)
            command.add_argument("--initial-reference", help="Optional JSON with verified epoch-zero relative_path and sha256")
            command.add_argument("--source-review", help="Exact reviewed auxiliary saved/actual hashes and reasons, diagnostic-only")
        if name == "train":
            command.add_argument("--protocol", required=True)
        if name == "health":
            source = command.add_mutually_exclusive_group(required=True)
            source.add_argument("--collection", help="Complete sealed collection only; rejects truncated JSON")
            source.add_argument("--report-root", help="Use collection/ CSVs and summaries/ only; never read collection.json")
            command.add_argument("--health-policy", required=True)
    args = parser.parse_args(argv)
    if not args.execute:
        parser.error("No runtime operation authorized by default. Review source/commands first; a later execution stage requires --execute")
    output = None
    result = None
    try:
        if args.command == "health":
            print("[diagnostic] Reading descriptive health evidence; no scientific completion is asserted", file=sys.stderr, flush=True)
            from .optimization_health import health_report
            policy = read_json(args.health_policy)
            if args.report_root:
                from .diagnostic_csv import report_csv_health
                result = report_csv_health(args.report_root, policy, source_root=args.source_root)
                protected = [args.report_root, args.source_root, *args.protected_root]
            else:
                collection = read_json(args.collection)
                result = health_report(collection, policy)
                protected = [*collection["protected_roots"], args.source_root,
                             Path(args.collection).parent, *args.protected_root]
            with diagnostic_stage("output", "output_claim"):
                output = new_output(args.output_dir, protected)
            filename = "health.json"
        else:
            config_sha = file_digest(Path(args.saved_config))
            config = load_config(args.saved_config)
            from .diagnostic_audit import audit_run
            initial = read_json(args.initial_reference) if args.initial_reference else None
            source_review = read_json(args.source_review) if args.source_review else {}
            protocol = None
            if args.command == "train":
                protocol = validate_protocol(read_json(args.protocol), config)
                if initial is not None and initial != protocol["initial_checkpoint"]:
                    raise DiagnosticError("Protocol and explicit initial reference disagree")
                initial = protocol["initial_checkpoint"]
                if args.source_review and source_review != protocol["source_difference_review"]:
                    raise DiagnosticError("Protocol and explicit source review disagree")
            print("[diagnostic] Validating original journal/checkpoint/config/source evidence", file=sys.stderr, flush=True)
            audit = audit_run(config, base_dir=args.base_dir, source_root=args.source_root, initial_reference=initial)
            print("[diagnostic] Stored-state audit finished; compute commands still require matched/reviewed inputs", file=sys.stderr, flush=True)
            if file_digest(Path(args.saved_config)) != config_sha:
                raise DiagnosticError("Selected config file changed during audit")
            audit["selected_config_input"] = {"path": str(Path(args.saved_config).absolute()), "sha256": config_sha}
            protected = protected_config_roots(config, args.base_dir, args.source_root,
                [*args.protected_root, Path(args.saved_config).parent])
            with diagnostic_stage("output", "output_claim"):
                output = new_output(args.output_dir, protected)
            _publish(output, "audit.json", audit)
            if args.command == "audit-run":
                result, filename = audit, "audit.json"
                with diagnostic_stage("output", "export_list"), (output / "export-files.txt").open("x", encoding="utf-8", newline="\n") as stream:
                    for entry in audit["export_files"]:
                        path = entry["path"]
                        if "\n" in path or "\r" in path:
                            raise DiagnosticError("Export path cannot be represented by a newline-delimited tar list")
                        stream.write(path + "\n")
            elif args.command == "recheck-validation":
                from .diagnostic_training import recheck_validation
                result, filename = recheck_validation(config, audit, base_dir=args.base_dir, source_review=source_review), "validation_recheck.json"
                if not result["exact_correct_and_examples_match"]:
                    result["status"] = "validation_count_mismatch"
            else:
                from .diagnostic_training import run_diagnostic
                result, filename = run_diagnostic(config, protocol, audit, base_dir=args.base_dir, output=output), "diagnostic.json"
        _publish(output, filename, result)
        _publish(output, "diagnostic_execution.json", {"kind": "isolated_diagnostic_execution",
            "command": args.command, "research_authority": False,
            "diagnostic_id": result.get("diagnostic_id"), "source_root": str(Path(args.source_root).absolute())})
        print(strict_json_text({"output_dir": str(output), "kind": result["kind"], "research_authority": False,
                               "status": result.get("status", "diagnostic_evidence_written")}), end="")
        if result.get("status") == "diagnostic_failed" or result.get("exact_correct_and_examples_match") is False:
            return 2
        return 0
    except Exception as exc:
        failure = {"kind": "diagnostic_execution_error", "research_authority": False,
                   "error": f"{type(exc).__name__}: {exc}",
                   "failure": failure_details(exc, origin="inspection", stage=f"{args.command}_preflight")}
        if result is not None and result.get("failure") is not None:
            failure["prior_diagnostic_failure"] = result["failure"]
        if output is not None:
            try:
                _publish(output, "DIAGNOSTIC_FAILURE.json", failure)
            except Exception as publication:
                failure["publication_failure"] = failure_details(publication, origin="output", stage="failure_publication")
        print(strict_json_text(failure), end="", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
