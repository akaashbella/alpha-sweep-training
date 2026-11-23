"""Temporary-file publication integrity, path protection and output-only checks."""

from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stderr, redirect_stdout

from ..analysis import analyze
from ..cli import main
from ..contracts import AnalysisError
from ..loading import load_summaries, validate_summaries
from ..output import publish, validate_output_path
from .fixtures import SEEDS, fixture, publication, reseal
from ..contracts import EVALUATION_ALPHAS


class PublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = fixture()

    def test_load_preserves_all_input_bytes_and_provenance(self):
        with tempfile.TemporaryDirectory(prefix="stage1-software-fixture-") as directory:
            root = Path(directory)/"summaries"
            publication(root, self.document)
            before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.iterdir()}
            loaded = load_summaries(root)
            loaded.assert_unchanged()
            self.assertEqual(loaded.seeds, SEEDS)
            self.assertEqual(loaded.provenance["summary_document_sha256"], self.document["document_sha256"])
            self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.iterdir()})
            (root/"checkpoint_robustness.csv").write_text("changed", encoding="utf-8")
            with self.assertRaises(AnalysisError):
                loaded.assert_unchanged()

    def test_corrupt_incomplete_and_incompatible_publications_rejected(self):
        def replace_json(root, name, change):
            document = json.loads((root/name).read_text(encoding="utf-8"))
            change(document)
            (root/name).write_text(json.dumps(reseal(document)), encoding="utf-8")
        mutations = {
            "missing completion": lambda p: (p/"COMPLETE.json").unlink(),
            "failure evidence": lambda p: (p/"PUBLICATION_FAILURE.json").write_text("{}"),
            "unconfirmed": lambda p: (p/"UNCONFIRMED_COMPLETE.json").write_text("{}"),
            "corrupt summary bytes": lambda p: (p/"summaries.json").write_text("{}"),
            "wrong owner version": lambda p: replace_json(p, "OWNER.json", lambda d: d.update(reporting_version=2)),
            "wrong summary owner linkage": lambda p: replace_json(p, "OWNER.json", lambda d: d.update(collection_sha256="d"*64)),
            "bad completion seal": lambda p: (p/"COMPLETE.json").write_text("{}"),
            "missing required output": lambda p: (p/"coverage.json").unlink(),
            "incomplete file declaration": lambda p: replace_json(p, "COMPLETE.json", lambda d: d["files"].pop()),
            "duplicate file declaration": lambda p: replace_json(p, "COMPLETE.json", lambda d: d["files"].append(d["files"][0])),
            "parent reference": lambda p: replace_json(p, "COMPLETE.json", lambda d: d["files"][0].update(relative_path="../outside")),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory(prefix="stage1-software-fixture-") as directory:
                root = Path(directory)/"summaries"
                publication(root, self.document)
                mutate(root)
                with self.assertRaises(AnalysisError):
                    load_summaries(root)

    def test_incompatible_summary_version_and_incomplete_official_mean(self):
        for mutate in (lambda d: d.update(reporting_version=2),
                       lambda d: d["checkpoint_robustness"][0]["loss"].update(complete=False),
                       lambda d: d["checkpoint_robustness"][0]["loss"]["official"].update(mean=999)):
            doc = deepcopy(self.document)
            mutate(doc)
            with self.assertRaises(AnalysisError):
                validate_summaries(reseal(doc))

    def test_output_overlap_ancestors_existing_and_source_tree_rejected(self):
        with tempfile.TemporaryDirectory(prefix="stage1-path-fixture-") as directory:
            base = Path(directory)
            report, research = base/"summaries", base/"research"
            report.mkdir()
            research.mkdir()
            for output in (report, report/"new", base, research, research/"checkpoints"/"new", Path(__file__).resolve().parents[1]/"new"):
                with self.subTest(output=output), self.assertRaises(AnalysisError):
                    validate_output_path(output, report, [research])
            self.assertEqual(validate_output_path(base/"separate", report, [research]), base/"separate")
            existing = base/"existing"
            existing.mkdir()
            with self.assertRaises(AnalysisError):
                validate_output_path(existing, report, [research])
            with self.assertRaises(AnalysisError):
                validate_output_path(base/"separate", report, [])

    def test_linked_paths_rejected_when_platform_permits(self):
        with tempfile.TemporaryDirectory(prefix="stage1-path-fixture-") as directory:
            base = Path(directory)
            real, link = base/"real", base/"link"
            real.mkdir()
            try:
                link.symlink_to(real, target_is_directory=True)
            except OSError:
                self.skipTest("Platform does not permit directory symlinks for this user")
            with self.assertRaises(AnalysisError):
                validate_output_path(link/"new", base/"report", [base/"research"])

    def test_cli_rejects_missing_input_without_creating_output(self):
        with tempfile.TemporaryDirectory(prefix="stage1-path-fixture-") as directory:
            base = Path(directory)
            with redirect_stderr(io.StringIO()):
                code = main(["--report-dir", str(base/"missing"), "--output", str(base/"new"), "--protected-root", str(base/"research")])
            self.assertEqual(code, 2)
            self.assertFalse((base/"new").exists())

    def test_loading_and_analysis_import_no_execution_or_checkpoint_modules(self):
        self.assertNotIn("torch", sys.modules)
        for name in ("training", "evaluation_execution", "evaluation_inputs", "checkpoint_io", "results", "hpc_workflow"):
            self.assertNotIn("alpha_sweep_training." + name, sys.modules)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Matplotlib is not installed; no package installation authorized")
    def test_exports_and_cli_without_modifying_input(self):
        with tempfile.TemporaryDirectory(prefix="stage1-software-fixture-") as directory:
            base = Path(directory)
            report, output = base/"report", base/"analysis-output"
            publication(report, self.document)
            before = {p.name: p.read_bytes() for p in report.iterdir()}
            # Rendering is independently tested below; keep this serialization
            # check focused and inexpensive while preserving real CLI loading.
            with patch("analysis.loss_geometry_stage1.plotting.plot_analysis", return_value=[]), redirect_stdout(io.StringIO()):
                code = main(["--report-dir", str(report), "--output", str(output), "--protected-root", str(base/"research")])
            self.assertEqual(code, 0)
            self.assertTrue((output/"COMPLETE.json").is_file())
            self.assertIn("synthetic-software-fixture-only", (output/"analysis_metadata.json").read_text(encoding="utf-8"))
            self.assertTrue((output/"findings.md").is_file())
            self.assertEqual(before, {p.name: p.read_bytes() for p in report.iterdir()})
            with redirect_stderr(io.StringIO()):
                self.assertEqual(main(["--report-dir", str(report), "--output", str(output), "--protected-root", str(base/"research")]), 2)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Matplotlib is not installed; no package installation authorized")
    def test_render_all_six_families_in_temporary_software_fixture_storage(self):
        from ..plotting import plot_analysis
        result = analyze(self.document, SEEDS)
        with tempfile.TemporaryDirectory(prefix="stage1-software-fixture-figures-") as directory:
            with patch("analysis.loss_geometry_stage1.plotting.DATASETS", {"cifar10": 10}), patch("analysis.loss_geometry_stage1.plotting.ARCHITECTURES", ("resnet50",)):
                index = plot_analysis(result, Path(directory), SEEDS, dpi=72)
            self.assertEqual({i["family"] for i in index}, {1, 2, 3, 4, 5, 6})
            self.assertEqual(len(index), 7)  # Figure 2 has full and local views.
            for item in index:
                self.assertEqual(item["status"], "generated")
                self.assertEqual(len(item["files"]), 2)
                for name in item["files"]:
                    self.assertGreater((Path(directory)/name).stat().st_size, 100)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Matplotlib is not installed; no package installation authorized")
    def test_all_missing_data_record_figure_exclusions_without_fabrication(self):
        from ..plotting import plot_analysis
        missing = {(run, alpha, rep) for run in range(240) for alpha in EVALUATION_ALPHAS
                   for rep in ((0,) if alpha == 0 else (1, 2, 3))}
        doc = fixture(missing=missing)
        # Match entirely unbound exported slots: no invented checkpoint or raw ref.
        for checkpoint in doc["checkpoint_robustness"]:
            checkpoint["best_checkpoint_identity"] = None
            for metric in ("loss", "accuracy"):
                for individual in checkpoint[metric]["individuals"]:
                    individual["raw_reference"] = None
        doc = reseal(doc)
        validate_summaries(doc)
        result = analyze(doc, SEEDS)
        self.assertEqual(result["coverage"]["eligible_loss_responses"], 0)
        self.assertEqual(result["coverage"]["eligible_fits"], 0)
        self.assertTrue(all(s["best_checkpoint_identity"] is None for s in result["coverage"]["realization_slots"]))
        with tempfile.TemporaryDirectory(prefix="stage1-software-fixture-no-data-") as directory:
            with patch("analysis.loss_geometry_stage1.plotting.DATASETS", {"cifar10": 10}), patch("analysis.loss_geometry_stage1.plotting.ARCHITECTURES", ("resnet50",)):
                index = plot_analysis(result, Path(directory), SEEDS, dpi=72)
            self.assertTrue(all(i["status"] == "unavailable_no_eligible_data" for i in index))
            self.assertEqual(list((Path(directory)/"figures").iterdir()), [])

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Matplotlib is not installed; no package installation authorized")
    def test_failure_retains_evidence_without_completion(self):
        with tempfile.TemporaryDirectory(prefix="stage1-software-fixture-") as directory:
            base = Path(directory)
            report = base/"report"
            publication(report, self.document)
            source = load_summaries(report)
            with patch("analysis.loss_geometry_stage1.plotting.plot_analysis", side_effect=ValueError("fixture render failure")):
                with self.assertRaises(ValueError):
                    publish(source, analyze(source.document, SEEDS), base/"output", [str(base/"research")], dpi=72)
            self.assertTrue((base/"output"/"ANALYSIS_FAILURE.json").is_file())
            self.assertFalse((base/"output"/"COMPLETE.json").exists())
