"""Focused pure-CPU serialization/read-reuse checks; no research execution."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import hashlib
import io
import os
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from alpha_sweep_training import metrics
from alpha_sweep_training.planning_inspection import (
    InspectionChanged, PlanningInspection, active_inspection, compatibility_key,
    evidence_stamp, stderr_progress, training_paths,
)


def context_fixture(entries=8):
    return metrics.RecordContext({"run_id": "synthetic,quoted\"\nidentity", "alpha": 0.1,
        "nullable": None, "seed": 7, "unicode": "\u03b1", "attempt_id": "attempt",
        "provenance": {"code": {"inventory": [
            {"path": f"src/synthetic/component_{i:04}.py", "sha256": f"{i:064x}"}
            for i in range(entries)]}, "allocation": {"node": "synthetic"}}})


def statistics_fixture(count=8):
    return tuple(metrics.LayerStatistic(f"layer.{i},weight", "Conv2d", (16, 16, 3, 3),
                                       2304, i * 0.001) for i in range(count))


def legacy_layers(context, record, statistics, *, include_header):
    """Pre-change implementation kept as a byte-equivalence oracle."""
    rows = [{**metrics._identity_columns(context), "epoch_index": record.epoch_index,
             "completed_epochs": record.completed_epochs, "status": record.status,
             "boundary": "after_training_updates_before_clean_validation", **asdict(statistic),
             "shape": metrics._json_cell(statistic.shape), "std_correction": 0, "weights": "clean"}
            for statistic in statistics]
    return metrics._csv(rows, include_header=include_header)


def provenance_fixture(source="0" * 64):
    return {"recorded_at_utc": "synthetic", "code": {"commit": None, "dirty": True,
        "source_sha256": source}, "software": {"python": "synthetic", "torch": "synthetic",
        "torchvision": "synthetic", "cuda": None, "cudnn": None}, "allocation": {}}


class SerializationChecks(unittest.TestCase):
    def test_exact_csv_bytes_and_one_identity_serialization_per_context(self):
        for count in (1, 8, 64):
            for header in (False, True):
                with self.subTest(count=count, header=header):
                    context = context_fixture(16)
                    original = deepcopy(context.identity)
                    record = SimpleNamespace(epoch_index=199, completed_epochs=200, status="training")
                    stats = statistics_fixture(count)
                    expected = legacy_layers(context, record, stats, include_header=header).encode("utf-8")
                    with patch.object(metrics, "_identity_columns", wraps=metrics._identity_columns) as render:
                        actual = metrics.layer_statistics_csv(context, record, stats, include_header=header)
                    self.assertEqual(expected, actual.encode("utf-8"))
                    self.assertEqual(render.call_count, 1)
                    self.assertEqual(context.identity, original)

    def test_changed_context_and_nonfinite_statistics_are_not_hidden(self):
        context = context_fixture()
        record = SimpleNamespace(epoch_index=0, completed_epochs=1, status="training")
        first = metrics.layer_statistics_csv(context, record, statistics_fixture(), include_header=True)
        context.identity["attempt_id"] = "other"
        second = metrics.layer_statistics_csv(context, record, statistics_fixture(), include_header=True)
        self.assertNotEqual(first, second)
        with self.assertRaises(ValueError):
            metrics.layer_statistics_csv(context, record,
                (metrics.LayerStatistic("weight", "Linear", (1,), 1, float("nan")),), include_header=False)


class InspectionChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="alpha-planning-unit-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "run"
        self.root.mkdir()
        self.file = self.root / "immutable.bin"
        self.file.write_bytes(b"original")
        self.lock = self.root.parent / ".run.lock"
        self.paths = (self.root, self.lock)
        self.cache = PlanningInspection()
        self.calls = 0

    def validate(self):
        self.calls += 1
        if self.lock.exists():
            raise RuntimeError("owned")
        return hashlib.sha256(self.file.read_bytes()).hexdigest()

    def read(self):
        return self.cache.reuse("authority", self.paths, self.validate)

    def test_unchanged_read_equivalence_and_operation_isolation(self):
        first = self.read()
        self.assertEqual(first, self.read())
        self.assertEqual(self.calls, 1)
        self.cache.assert_unchanged()
        other = PlanningInspection()
        self.assertEqual(first, other.reuse("authority", self.paths, self.validate))
        self.assertEqual(self.calls, 2)
        self.assertIsNone(active_inspection())
        with self.cache.activate():
            self.assertIs(active_inspection(), self.cache)
            with other.activate():
                self.assertIs(active_inspection(), other)
            self.assertIs(active_inspection(), self.cache)
        self.assertIsNone(active_inspection())

    def test_artifact_edits_replacement_and_deletion_invalidate(self):
        self.read()
        # Filesystem timestamps must visibly advance even on coarse local clocks.
        self.file.write_bytes(b"modified payload")
        os.utime(self.file, ns=(1_500_000_000_000_000_000, 1_500_000_000_000_000_000))
        with self.assertRaises(InspectionChanged):
            self.cache.assert_unchanged()
        second = self.read()
        self.assertEqual(self.calls, 2)
        self.assertNotEqual(second, hashlib.sha256(b"original").hexdigest())
        replacement = self.root / "replacement"
        replacement.write_bytes(b"modified payload")
        os.replace(replacement, self.file)
        self.read()
        self.assertEqual(self.calls, 3)
        self.file.unlink()
        with self.assertRaises(FileNotFoundError):
            self.read()
        self.file.write_bytes(b"recovered")
        self.read()
        self.assertEqual(self.calls, 5)

    def test_owner_lock_and_new_failure_evidence_invalidate(self):
        self.read()
        for name in ("HEAD.json", ".run_owner.json", "resolved_config.yaml", "orphan_failure.json"):
            (self.root / name).write_text(name, encoding="utf-8")
            with self.assertRaises(InspectionChanged):
                self.cache.assert_unchanged()
            self.read()
        self.lock.write_text("another owner", encoding="utf-8")
        with self.assertRaises(InspectionChanged):
            self.cache.assert_unchanged()
        with self.assertRaises(RuntimeError):
            self.read()
        self.assertTrue(self.lock.exists())

    def test_mutation_during_validation_never_enters_cache(self):
        def changing():
            result = self.validate()
            (self.root / "new-evidence").write_text("changed", encoding="utf-8")
            return result
        with self.assertRaises(InspectionChanged):
            self.cache.reuse("authority", self.paths, changing)
        self.read()
        self.assertEqual(self.calls, 2)

    def test_same_size_edit_with_restored_mtime_invalidates(self):
        self.read()
        before = self.file.stat()
        self.file.write_bytes(b"modified")
        os.utime(self.file, ns=(before.st_atime_ns, before.st_mtime_ns))
        with self.assertRaises(InspectionChanged):
            self.cache.assert_unchanged()
        self.read()
        self.assertEqual(self.calls, 2)

    def test_missing_becomes_present(self):
        missing = self.root / "absent"
        first = self.cache.reuse("missing", (missing,), lambda: "missing")
        missing.write_text("new", encoding="utf-8")
        self.assertEqual(first, "missing")
        self.assertEqual(self.cache.reuse("missing", (missing,), lambda: "present"), "present")

    def test_link_evidence_invalidates(self):
        before = evidence_stamp(self.paths)
        link = self.root / "linked"
        try:
            link.symlink_to(self.file)
        except OSError:
            self.skipTest("Local Windows account lacks symlink privileges")
        self.assertNotEqual(before, evidence_stamp(self.paths))

    def test_linked_ancestor_is_not_traversed(self):
        original_lstat = Path.lstat
        parent = self.root.parent
        original = parent.lstat()
        redirected = SimpleNamespace(st_dev=original.st_dev, st_ino=original.st_ino,
            st_mode=stat.S_IFLNK | 0o777, st_file_attributes=0x400,
            st_size=original.st_size, st_mtime_ns=original.st_mtime_ns, st_ctime_ns=original.st_ctime_ns)

        def lstat(path):
            return redirected if path == parent else original_lstat(path)

        # Metadata boundary mock works without local symlink privileges. A real
        # linked-file change is exercised separately when the OS permits it.
        with patch.object(Path, "lstat", lstat), \
             patch.object(Path, "iterdir", side_effect=AssertionError("Traversed a redirected root")):
            token = evidence_stamp((self.root,))
        self.assertIn((str(self.root), "linked-ancestor"), token)

    def test_production_observation_wrapper_reuses_and_invalidates(self):
        from alpha_sweep_training import coverage
        paths = SimpleNamespace(managed_roots=lambda: (self.root,))
        config = SimpleNamespace(to_dict=lambda: {"synthetic": "config"})
        task = SimpleNamespace(run_id="synthetic", prepared_data={"split": "synthetic"})
        observation = SimpleNamespace(condition="ok", transaction={"digest": "synthetic"})
        with patch.object(coverage, "task_config", return_value=config), \
             patch.object(coverage, "effective_storage_config", return_value=config), \
             patch.object(coverage, "resolve_run_paths", return_value=paths), \
             patch.object(coverage, "_inspect_task", return_value=observation) as deep:
            # The deep scientific reader is a boundary mock here; this check
            # exercises production cache wiring, not checkpoint format validity.
            with self.cache.activate():
                first = coverage.inspect_task(task, base_dir=self.temp.name)
                self.assertIs(first, coverage.inspect_task(task, base_dir=self.temp.name))
                self.assertEqual(deep.call_count, 1)
                (self.root / "HEAD.json").write_text("new transaction", encoding="utf-8")
                self.assertIs(first, coverage.inspect_task(task, base_dir=self.temp.name))
                self.assertEqual(deep.call_count, 2)
                coverage.inspect_task(task, base_dir=self.temp.name, provenance=provenance_fixture())
                self.assertEqual(deep.call_count, 3)
                same_source = provenance_fixture()
                same_source["recorded_at_utc"] = "new inspection"
                coverage.inspect_task(task, base_dir=self.temp.name, provenance=same_source)
                self.assertEqual(deep.call_count, 3)
                coverage.inspect_task(task, base_dir=self.temp.name, provenance=provenance_fixture("1" * 64))
                self.assertEqual(deep.call_count, 4)

    def test_provenance_partition_and_malformed_inputs(self):
        first = provenance_fixture()
        changed_attempt = deepcopy(first)
        changed_attempt["recorded_at_utc"] = "another inspection"
        changed_attempt["allocation"] = {"pid": 123}
        self.assertEqual(compatibility_key(first), compatibility_key(changed_attempt))
        self.assertNotEqual(compatibility_key(first), compatibility_key(provenance_fixture("1" * 64)))
        changed_attempt["software"]["python"] = "different"
        self.assertNotEqual(compatibility_key(first), compatibility_key(changed_attempt))
        changed_attempt["code"]["dirty"] = "invalid"
        with self.assertRaises(ValueError):
            compatibility_key(changed_attempt)

    def test_progress_is_stderr_only(self):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            stderr_progress("synthetic preflight")
        self.assertEqual(out.getvalue(), "")
        self.assertIn("synthetic preflight", err.getvalue())


if __name__ == "__main__":
    unittest.main()
