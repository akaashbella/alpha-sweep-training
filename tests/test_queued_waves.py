"""Operational-only unit checks: no models, data, files, GPUs or scheduler calls.

Manifest membership and launch-file reading are mocked at their boundaries in
render tests; scientific authority is outside this focused queue-policy suite.
"""

from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from alpha_sweep_training.data_contract import content_identity
from alpha_sweep_training.hpc_policy import (
    OperationalPolicy, PATH_KEYS, PUBLISHED_POLICY, WorkflowError, array_argument,
    resource_budget, validate_array_arguments, validate_array_evidence, wave_capacity,
)
from alpha_sweep_training import work_render
from alpha_sweep_training.work_plans import seal, validate_plan, wave_reservations, workflow_identity


def declaration(version=2):
    """Operational fixture only; numbers are test bounds, never account defaults."""
    paths = {name: f"/unit/{name}" for name in PATH_KEYS}
    paths["split_file"] = None
    resources = dict(partition="dgxh", account="unit_account", qos=None, constraint=None,
        nodes=1, gpus=1, cpus=8, memory_gb=32, wall_time="12:00:00",
        max_concurrent=4, num_workers=4, native_threads=4)
    limits = {key: PUBLISHED_POLICY[key] for key in (
        "max_wall_hours", "max_gpus", "max_cpus", "max_memory_gb",
        "gpu_hours", "cpu_hours", "memory_gb_hours")}
    limits.update(max_allocations=4, inspection_report_path="/unit/evidence.json",
                  inspection_report_sha256="0" * 64)
    if version == 2:
        resources["max_queued"] = 40
        limits.update(max_submitted_tasks=40, max_array_size=1001, max_array_tasks=40)
    return dict(policy_version=version, resources=resources, paths=paths,
        environment=dict(python_executable="/unit/python", activation_script=None, modules=[], variables={}),
        signals=dict(lead_seconds=600, warning_policy="discard_epoch", termination_policy="discard_epoch"),
        continuation=dict(max_allocations_per_work=20, controller_wait_seconds=120),
        commands=dict(sbatch="sbatch", squeue="squeue", sacct="sacct", scontrol="scontrol",
                      sacctmgr="sacctmgr", timeout_seconds=30), effective_limits=limits)


def config_evidence(size="1001", parameters="max_array_tasks=40"):
    return {"reports": [dict(query_id="configuration", ok=True,
        stdout=f"MaxArraySize = {size}\nSchedulerParameters = {parameters}\n")]}


def items(count):
    return [seal(dict(unit_key=f"unit-{i}", training_run_id=f"run-{i}", training_global_index=100 + i))
            for i in range(count)]


def bundle_fixture(policy, count=4):
    """An in-memory operational mapping, without a scientific manifest fixture."""
    manifest = SimpleNamespace(grid_id="grid", document_sha256="parent")
    selected = items(count)
    identity = workflow_identity(manifest)
    wave = work_render.wave_identity(identity, "plan", policy.fingerprint, selected)
    bundle = seal(dict(bundle_version=policy.to_dict()["policy_version"], kind="allocation_wave",
        wave_id=wave, workflow_id=identity, plan_sha256="plan", manifest_path="/unit/manifest.json",
        manifest_grid_id="grid", manifest_document_sha256="parent", policy=policy.to_dict(),
        policy_fingerprint=policy.fingerprint, execution_provenance={"historical": True}, items=selected,
        array_mapping=[dict(array_index=i, work_item_sha256=item["document_sha256"],
            training_global_index=item["training_global_index"]) for i, item in enumerate(selected)],
        scheduler_comment=f"alpha-sweep:{identity[:12]}:{wave}", resource_policy=resource_budget(policy, count),
        script_sha256="0" * 64))
    return manifest, bundle


class QueuedWaveChecks(unittest.TestCase):
    def test_forty_queued_four_running_and_distinct_totals(self):
        policy = OperationalPolicy(declaration())
        budget = resource_budget(policy, 40)
        self.assertEqual((budget["queued_tasks"], budget["running_tasks"]), (40, 4))
        self.assertEqual(budget["requested_overlap"], dict(max_wall_hours=12, max_gpus=4,
            max_cpus=32, max_memory_gb=128, gpu_hours=48, cpu_hours=384, memory_gb_hours=1536))
        self.assertEqual(budget["requested_wave_totals"]["gpu_hours"], 480)
        self.assertEqual(array_argument(policy, 40), "--array=0-39%4")
        self.assertEqual(array_argument(policy, 2), "--array=0-1%2")
        self.assertEqual(resource_budget(policy, 0)["running_tasks"], 0)
        for count in (-1, True, 0, 41):
            with self.subTest(count=count), self.assertRaises(WorkflowError):
                array_argument(policy, count)

    def test_queue_bounds_do_not_use_running_allowance(self):
        for field in ("max_submitted_tasks", "max_array_size", "max_array_tasks"):
            doc = declaration()
            doc["effective_limits"][field] = 39
            with self.subTest(field=field), self.assertRaises(WorkflowError):
                OperationalPolicy(doc)
        doc = declaration()
        doc["resources"]["max_queued"] = 8
        doc["effective_limits"]["max_submitted_tasks"] = 8
        self.assertEqual(resource_budget(OperationalPolicy(doc), 8)["running_tasks"], 4)

    def test_running_resource_and_time_limits_remain_enforced(self):
        changes = (
            ("max_allocations", 3), ("max_gpus", 3), ("max_cpus", 31),
            ("max_memory_gb", 127), ("gpu_hours", 47), ("cpu_hours", 383),
            ("memory_gb_hours", 1535), ("max_wall_hours", 11),
        )
        for field, bound in changes:
            doc = declaration()
            doc["effective_limits"][field] = bound
            with self.subTest(field=field), self.assertRaises(WorkflowError):
                OperationalPolicy(doc)
        doc = declaration()
        doc["resources"]["max_concurrent"] = 5  # 5 * 12 GPU-hours still exceeds published overlap.
        doc["effective_limits"]["max_allocations"] = 5
        with self.assertRaises(WorkflowError):
            OperationalPolicy(doc)

    def test_legacy_policy_fingerprint_and_budget_are_unchanged(self):
        original = declaration(1)
        policy = OperationalPolicy(original)
        self.assertEqual(policy.to_dict(), original)
        self.assertEqual(policy.fingerprint, content_identity(original)[7:])
        self.assertEqual(wave_capacity(policy), 4)
        self.assertEqual(resource_budget(policy, 4), dict(allocations=4,
            requested_overlap=dict(max_wall_hours=12, max_gpus=4, max_cpus=32, max_memory_gb=128,
                gpu_hours=48, cpu_hours=384, memory_gb_hours=1536),
            limits={key: PUBLISHED_POLICY[key] for key in original["effective_limits"] if key in PUBLISHED_POLICY},
            published_policy=PUBLISHED_POLICY, effective_restrictions_reviewed=True,
            unmeasured_planning_requests=True, unrelated_jobs_controlled=False))
        self.assertEqual(array_argument(policy, 4), "--array=0-3%4")
        with self.assertRaises(WorkflowError):
            resource_budget(policy, 5)

    def test_versions_and_unreviewed_limit_types_are_explicit(self):
        for version in (True, 0, 3):
            doc = declaration()
            doc["policy_version"] = version
            with self.subTest(version=version), self.assertRaises(WorkflowError):
                OperationalPolicy(doc)
        for field in ("max_submitted_tasks", "max_array_size", "max_array_tasks"):
            doc = declaration()
            doc["effective_limits"].pop(field)
            with self.subTest(field=field), self.assertRaises(WorkflowError):
                OperationalPolicy(doc)
        doc = declaration(1)
        doc["resources"]["max_queued"] = 40
        with self.assertRaises(WorkflowError):
            OperationalPolicy(doc)  # No silent v1 migration.

    def test_recorded_submission_must_enforce_sealed_throttle(self):
        policy = OperationalPolicy(declaration())
        validate_array_arguments(policy, 40, ["--parsable", "--array=0-39%4"])
        for bad in (["--array=0-39%40"], ["--array=0-39"], ["--array=0-39%4", "-a0-39%40"],
                    ["--array=0-38%4"], ["--array", "0-39%4"]):
            with self.subTest(arguments=bad), self.assertRaises(WorkflowError):
                validate_array_arguments(policy, 40, bad)

    def test_cluster_array_evidence_and_documented_default(self):
        policy = OperationalPolicy(declaration())
        validate_array_evidence(config_evidence(), policy)
        validate_array_evidence(config_evidence(parameters="bf_continue"), policy)
        validate_array_evidence(config_evidence(parameters="(null)"), policy)
        for size, parameters in (("39", "max_array_tasks=40"), ("1001", "max_array_tasks=39"),
            ("1001", ""), ("1001", "max_array_tasks=bad"), ("", "(null)"),
            ("1001", "max_array_tasks=40,max_array_tasks=40"), ("1001", "MAX_ARRAY_TASKS=39")):
            with self.subTest(size=size, parameters=parameters), self.assertRaises(WorkflowError):
                validate_array_evidence(config_evidence(size, parameters), policy)
        # v1 report compatibility has no retroactive new evidence requirement.
        validate_array_evidence({}, OperationalPolicy(declaration(1)))

    def test_uncertain_waves_reserve_throttled_running_capacity(self):
        current = OperationalPolicy(declaration())
        waves = [dict(status="submitted", items=items(40), policy=declaration()),
                 dict(status="submission_uncertain", items=items(4), policy=declaration(1)),
                 dict(status="terminal", items=items(40), policy=declaration())]
        self.assertEqual(wave_reservations({"waves": waves}, current),
            dict(reserved_queued_tasks=44, reserved_running_tasks=8))
        self.assertEqual(wave_reservations({"waves": waves[1:]}, OperationalPolicy(declaration(1))),
            dict(reserved_allocations=4))

    def test_render_caps_queue_keeps_original_indices_and_blocks_unresolved_wave(self):
        policy = OperationalPolicy(declaration())
        manifest = SimpleNamespace(grid_id="grid", document_sha256="parent")
        plan = dict(unresolved_waves=[], items=items(50), document_sha256="plan",
                    manifest_path="/unit/manifest.json", execution_provenance={})
        with patch.object(OperationalPolicy, "require_ready"), patch.object(work_render, "validate_plan"), \
                patch.object(work_render, "batch_source", return_value="reviewed script\n"):
            bundle, _ = work_render.render_work(plan, manifest, policy)
            self.assertEqual(len(bundle["items"]), 40)
            self.assertEqual(bundle["resource_policy"]["running_tasks"], 4)
            self.assertEqual([r["training_global_index"] for r in bundle["array_mapping"]], list(range(100, 140)))
            args = work_render.submission_arguments(bundle, policy)
            self.assertIn("--array=0-39%4", args)
            with self.assertRaises(WorkflowError):
                work_render.render_work(plan, manifest, policy, work_ids=[f"unit-{i}" for i in range(41)])
            legacy, _ = work_render.render_work(plan, manifest, OperationalPolicy(declaration(1)))
            self.assertEqual(len(legacy["items"]), 4)
            self.assertEqual(legacy["bundle_version"], 1)
            plan["unresolved_waves"] = ["active"]
            with self.assertRaises(WorkflowError):
                work_render.render_work(plan, manifest, policy)

    def test_existing_four_task_and_new_forty_task_bundles_read_without_rewriting(self):
        for version, count in ((1, 4), (2, 40)):
            policy = OperationalPolicy(declaration(version))
            manifest, bundle = bundle_fixture(policy, count)
            before = deepcopy(bundle)
            with patch.object(work_render, "validate_item"):
                self.assertIs(work_render.validate_bundle(bundle, manifest, policy), bundle)
            self.assertEqual(bundle, before)
            # Actual legacy record intentionally has no new launch-input pin.
            if version == 1:
                with patch("alpha_sweep_training.execution_provenance.collect_launch_inputs"):
                    from alpha_sweep_training.execution_provenance import require_launch_inputs
                    with self.assertRaises(WorkflowError):
                        require_launch_inputs(bundle["execution_provenance"], policy)
            other = OperationalPolicy(declaration(2 if version == 1 else 1))
            with patch.object(work_render, "validate_item"), self.assertRaises(WorkflowError):
                work_render.validate_bundle(bundle, manifest, other)

    def test_legacy_plan_reads_with_original_fields(self):
        policy = OperationalPolicy(declaration(1))
        manifest = SimpleNamespace(grid_id="grid", document_sha256="parent")
        plan = seal(dict(plan_version=1, kind="workflow_work_plan", workflow_id=workflow_identity(manifest),
            manifest_path="/unit/manifest.json", manifest_grid_id="grid", manifest_document_sha256="parent",
            policy=policy.to_dict(), policy_fingerprint=policy.fingerprint, registry_revision=0,
            phase2_training_eligible=False, unresolved_waves=[], resource_policy=resource_budget(policy, 4),
            reserved_allocations=0, capacity_available_for_submission=True, execution_provenance={},
            items=[], observations=[]))
        before = deepcopy(plan)
        self.assertIs(validate_plan(plan, manifest, policy), plan)
        self.assertEqual(plan, before)

    def test_new_plan_separates_reservations_and_rejects_legacy_shape(self):
        policy = OperationalPolicy(declaration())
        manifest = SimpleNamespace(grid_id="grid", document_sha256="parent")
        plan = seal(dict(plan_version=2, kind="workflow_work_plan", workflow_id=workflow_identity(manifest),
            manifest_path="/unit/manifest.json", manifest_grid_id="grid", manifest_document_sha256="parent",
            policy=policy.to_dict(), policy_fingerprint=policy.fingerprint, registry_revision=1,
            phase2_training_eligible=False, unresolved_waves=["uncertain"], resource_policy=resource_budget(policy, 0),
            reserved_queued_tasks=40, reserved_running_tasks=4, capacity_available_for_submission=False,
            execution_provenance={}, items=[], observations=[]))
        self.assertIs(validate_plan(plan, manifest, policy), plan)
        bad = {k: v for k, v in plan.items() if k != "document_sha256"}
        bad.pop("reserved_queued_tasks")
        bad.pop("reserved_running_tasks")
        bad["reserved_allocations"] = 40
        with self.assertRaises(WorkflowError):
            validate_plan(seal(bad), manifest, policy)


if __name__ == "__main__":
    unittest.main()
