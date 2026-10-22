"""Focused pure-CPU serialization/read-reuse checks; no research execution."""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
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
    record_dependency,
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


@contextmanager
def frozen_metadata(path):
    """Freeze target/ancestors in stat AND lstat, independent of pathlib internals."""
    saved = {candidate: candidate.lstat() for candidate in (path, *path.parents)}
    real_stat, real_lstat = Path.stat, Path.lstat

    def unchanged_stat(candidate, *args, **kwargs):
        return saved[candidate] if candidate in saved else real_stat(candidate, *args, **kwargs)

    def unchanged_lstat(candidate, *args, **kwargs):
        return saved[candidate] if candidate in saved else real_lstat(candidate, *args, **kwargs)

    with patch.object(Path, "stat", unchanged_stat), patch.object(Path, "lstat", unchanged_lstat):
        yield saved[path]


class SerializationChecks(unittest.TestCase):
    def test_json_reader_binds_exact_utf8_crlf_bytes_without_probe_reread(self):
        from alpha_sweep_training.serialization import read_json
        with tempfile.TemporaryDirectory(prefix="alpha-json-unit-") as temporary:
            path = Path(temporary) / "record.json"
            content = '{\r\n"value":"\u03b1"\r\n}\r\n'.encode("utf-8")
            path.write_bytes(content)
            cache = PlanningInspection()
            value = cache.reuse("json", (path,), lambda: read_json(path), track_dependencies=True)
            self.assertEqual(value, {"value": "\u03b1"})
            self.assertEqual(cache.read_metrics["fingerprint_read_bytes"], len(content))
            self.assertEqual(cache.read_metrics["dependency_probe_read_bytes"], 0)
            cache.assert_unchanged()

    def test_json_reader_retains_duplicate_nonfinite_and_encoding_rejection(self):
        from alpha_sweep_training.serialization import read_json
        with tempfile.TemporaryDirectory(prefix="alpha-json-unit-") as temporary:
            path = Path(temporary) / "record.json"
            for content in (b'{"a":0,"a":1}', b'{"a":NaN}', b'{"a":1e999}',
                            b'\xef\xbb\xbf{}', '{}'.encode("utf-16")):
                with self.subTest(content=content):
                    path.write_bytes(content)
                    with self.assertRaises(ValueError):
                        read_json(path)

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
        self.file.write_bytes(b"modified payload")
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

    def test_changed_bytes_with_identical_stat_metadata_invalidates(self):
        # Deterministic on every OS/filesystem: all exposed stat fields for the
        # dependency remain identical. No sleep or timestamp-resolution premise.
        for tracked in (False, True):
            with self.subTest(tracked=tracked):
                self.file.write_bytes(b"original")
                self.cache = PlanningInspection()
                def read():
                    return self.cache.reuse("authority", self.paths,
                        self.tracked_validate if tracked else self.validate, track_dependencies=tracked)

                before_calls = self.calls
                with frozen_metadata(self.file) as saved:
                    metadata_before = evidence_stamp(self.paths, content_paths=())
                    read()
                    self.cache.assert_unchanged()
                    self.file.write_bytes(b"modified")
                    self.assertEqual(self.file.stat(), saved)
                    self.assertEqual(self.file.lstat(), saved)
                    self.assertEqual(metadata_before, evidence_stamp(self.paths, content_paths=()))
                    with self.assertRaises(InspectionChanged):
                        self.cache.assert_unchanged()
                    self.assertEqual(read(), hashlib.sha256(b"modified").hexdigest())
                self.assertEqual(self.calls - before_calls, 2)

    def tracked_validate(self):
        value = self.validate()
        record_dependency(self.file, value)
        return value

    def test_tracked_dependencies_exclude_unused_history_and_count_reads(self):
        history = self.root / "unreferenced-history.bin"
        history.write_bytes(b"h" * (1024 * 1024))
        read = lambda: self.cache.reuse("tracked", self.paths, self.tracked_validate, track_dependencies=True)
        with frozen_metadata(history):
            first = read()
            history.write_bytes(b"H" * (1024 * 1024))
            self.assertEqual(first, read())
            self.cache.assert_unchanged()
        self.assertEqual(self.calls, 1)
        # One after-validation hash, one hit hash, one final hash. The 1 MiB
        # history was enumerated but never content-read for these observations.
        self.assertEqual(self.cache.read_metrics["fingerprint_read_bytes"], 3 * len(b"original"))
        self.assertEqual(self.cache.read_metrics["fingerprint_files_hashed"], 3)

    def test_tracked_mutation_during_validation_with_frozen_stat_is_rejected(self):
        def changing():
            value = self.tracked_validate()
            self.file.write_bytes(b"modified")
            return value

        with frozen_metadata(self.file):
            with self.assertRaises(InspectionChanged):
                self.cache.reuse("tracked", self.paths, changing, track_dependencies=True)
            result = self.cache.reuse("tracked", self.paths, self.tracked_validate, track_dependencies=True)
        self.assertEqual(result, hashlib.sha256(b"modified").hexdigest())
        self.assertEqual(self.calls, 2)

    def test_nested_cache_hit_propagates_dependency_and_final_guard_deduplicates(self):
        inner = PlanningInspection()
        read_inner = lambda: inner.reuse("inner", self.paths, self.tracked_validate, track_dependencies=True)
        read_inner()
        self.cache.reuse("outer", self.paths, read_inner, track_dependencies=True)
        # A second entry references the same file. The final guard must hash it
        # once, not once per observation/journal/cache key.
        self.cache.reuse("another", self.paths, read_inner, track_dependencies=True)
        before = self.cache.read_metrics["fingerprint_read_bytes"]
        self.cache.assert_unchanged()
        self.assertEqual(self.cache.read_metrics["fingerprint_read_bytes"] - before, len(b"original"))
        self.assertEqual(self.calls, 1)
        self.file.write_bytes(b"modified")
        with self.assertRaises(InspectionChanged):
            self.cache.assert_unchanged()

    def test_tracked_json_reader_invalidates_under_frozen_stat(self):
        from alpha_sweep_training.serialization import read_json
        record = self.root / "epoch.json"
        record.write_bytes(b'{"value":0}')
        with frozen_metadata(record):
            first = self.cache.reuse("json", self.paths, lambda: read_json(record), track_dependencies=True)
            self.assertEqual(first, {"value": 0})
            record.write_bytes(b'{"value":1}')
            with self.assertRaises(InspectionChanged):
                self.cache.assert_unchanged()
            self.assertEqual(self.cache.reuse("json", self.paths, lambda: read_json(record),
                                             track_dependencies=True), {"value": 1})

    def test_new_membership_invalidates_even_when_directory_stat_is_frozen(self):
        with frozen_metadata(self.root):
            self.cache.reuse("tracked", self.paths, self.tracked_validate, track_dependencies=True)
            (self.root / "new-failure.json").write_bytes(b'{}')
            with self.assertRaises(InspectionChanged):
                self.cache.assert_unchanged()

    def test_unregistered_json_control_is_always_hashed(self):
        owner = self.root / ".run_owner.json"
        owner.write_bytes(b'{"owner":0}')
        with frozen_metadata(owner):
            self.cache.reuse("tracked", self.paths, self.tracked_validate, track_dependencies=True)
            owner.write_bytes(b'{"owner":1}')
            with self.assertRaises(InspectionChanged):
                self.cache.assert_unchanged()

    def test_conflicting_reads_or_escaped_dependencies_never_enter_cache(self):
        def changing():
            self.tracked_validate()
            self.file.write_bytes(b"modified")
            self.tracked_validate()

        with self.assertRaises(InspectionChanged):
            self.cache.reuse("tracked", self.paths, changing, track_dependencies=True)
        outside = self.root.parent / "outside.bin"
        outside.write_bytes(b"outside")
        with self.assertRaises(InspectionChanged):
            self.cache.reuse("tracked", self.paths,
                lambda: record_dependency(outside, hashlib.sha256(b"outside").hexdigest()), track_dependencies=True)

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
