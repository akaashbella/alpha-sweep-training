"""Structured Slurm subprocesses, invoked only by explicit future operations."""

from datetime import datetime, timezone
import os
import re
import subprocess

from .hpc_policy import WorkflowError, exact, same
from .work_plans import check_seal, seal

TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY",
            "PREEMPTED", "BOOT_FAIL", "DEADLINE"}

RESTRICTION_REPORT_VERSION = 2
# Documented LIST/SHOW format names in Slurm 24.11. The deployed version is
# collected separately; unsupported fields are errors, never silently dropped.
ASSOCIATION_FIELDS = ("Cluster", "Account", "User", "Partition", "ID", "ParentID", "ParentName",
    "QOS", "DefaultQOS", "Flags", "GrpTRES", "GrpTRESMins", "GrpTRESRunMins",
    "MaxTRESPJ", "MaxTRESPN", "MaxTRESMinsPJ", "GrpWall", "MaxWall",
    "GrpJobs", "GrpSubmitJobs", "MaxJobs", "MaxSubmitJobs", "GrpJobsAccrue", "MaxJobsAccrue")
QOS_FIELDS = ("Name", "ID", "Flags", "LimitFactor", "UsageFactor", "UsageThreshold",
    "GrpTRES", "GrpTRESMins", "GrpTRESRunMins", "MaxTRESPJ", "MaxTRESPN", "MaxTRESPU", "MaxTRESPA",
    "MaxTRESMinsPJ", "MaxTRESRunMinsPU", "MaxTRESRunMinsPA", "MinTRES", "GrpWall", "MaxWall",
    "GrpJobs", "GrpSubmitJobs", "MaxJobsPU", "MaxJobsPA", "MaxSubmitJobsPU", "MaxSubmitJobsPA",
    "GrpJobsAccrue", "MaxJobsAccruePU", "MaxJobsAccruePA", "Preempt", "PreemptMode", "PreemptExemptTime")
RESTRICTION_COLLECTION = {"contract": "association-qos-allocation-limits-v2", "documented_slurm_release": "24.11",
    "format_reference": "https://raw.githubusercontent.com/SchedMD/slurm/slurm-24-11-0-1/doc/man/man1/sacctmgr.1",
    "cumulative_usage_fields": ["GrpTRESMins", "GrpWall"],
    "simultaneous_requested_minutes_fields": ["GrpTRESRunMins", "MaxTRESRunMinsPU", "MaxTRESRunMinsPA"],
    "per_job_minutes_fields": ["MaxTRESMinsPJ"],
    "blank_value_semantics": "unresolved for operator review; never inferred zero or unlimited"}


def restriction_requests(resources):
    """Fixed coverage contract, including visible ancestors and actual tool versions."""
    def request(query_id, command, arguments, fields=()):
        return {"query_id": query_id, "command": command, "arguments": arguments, "field_names": list(fields)}

    def table(query_id, entity, fields, options=()):
        # Explicit widths avoid normal display truncation. Raw output and all
        # column names survive; an at-width value is still reported as suspect.
        return request(query_id, "sacctmgr", ["--noheader", "--parsable2", "show", entity, *options,
            "format=" + ",".join(f"{name}%4096" for name in fields)], fields)

    return [request("sacctmgr_version", "sacctmgr", ["--version"]),
        request("scontrol_version", "scontrol", ["--version"]),
        request("partition", "scontrol", ["show", "partition", "dgxh", "--oneliner"]),
        request("configuration", "scontrol", ["show", "config"]),
        table("account_associations", "association", ASSOCIATION_FIELDS, [f"account={resources['account']}"]),
        # Do not filter away parent accounts. WOPLimits preserves where each
        # limit is set; the account query above retains inherited output too.
        table("association_hierarchy", "association", ASSOCIATION_FIELDS, ["WOPLimits"]),
        table("qos", "qos", QOS_FIELDS)]


def _restriction_rows(fields, stdout):
    """Parse strings for collection coverage only, never numerical entitlements."""
    if not fields:
        return None, []
    rows, issues = [], []
    for number, line in enumerate(stdout.splitlines(), start=1):
        if not line.strip():
            continue
        values = line.split("|")
        if len(values) != len(fields):
            issues.append(f"row {number}: expected {len(fields)} columns, received {len(values)}")
            continue
        row = dict(zip(fields, (value.strip() for value in values)))
        if any(len(value) >= 4096 for value in row.values()):
            issues.append(f"row {number}: field reached display width; obtain untruncated evidence")
        rows.append(row)
    if not rows:
        issues.append("No association/QOS rows were returned; absent evidence is not unlimited")
    return rows, issues


def restriction_collection_issues(reports, resources):
    """Recompute collection completeness from exact queries and preserved output."""
    expected = restriction_requests(resources)
    if type(reports) is not list or len(reports) != len(expected):
        raise WorkflowError("Restriction report lacks the required version-2 query coverage; recollect evidence")
    issues, tables = [], {}
    for actual, spec in zip(reports, expected):
        exact(actual, set(spec) | {"ok", "stdout", "error", "rows", "blank_fields"}, "restriction query evidence")
        for key, value in spec.items():
            same(actual[key], value, f"restriction query coverage {spec['query_id']}/{key}")
        if type(actual["ok"]) is not bool:
            raise WorkflowError("Malformed restriction query success flag")
        if not actual["ok"]:
            if type(actual["error"]) is not str or not actual["error"]:
                raise WorkflowError("Failed restriction query must retain an explicit diagnostic")
            same(actual["rows"], None, "failed query has no accepted rows")
            same(actual["blank_fields"], [], "failed query blank-field evidence")
            issues.append(f"{spec['query_id']}: query failed (unsupported fields, denied access or I/O); {actual['error']}")
            continue
        same(actual["error"], None, "successful restriction query error")
        stdout = actual["stdout"]
        if type(stdout) is not str:
            raise WorkflowError("Successful restriction query must retain raw output")
        if not stdout.strip():
            issues.append(f"{spec['query_id']}: empty output; evidence missing")
        if spec["query_id"] in ("sacctmgr_version", "scontrol_version") and not re.fullmatch(
                r"slurm \d+\.\d+(?:\.\d+)?(?:[-+][A-Za-z0-9_.-]+)?", stdout.strip(), re.IGNORECASE):
            issues.append(f"{spec['query_id']}: unrecognized version output; installed-version evidence missing")
        rows, row_issues = _restriction_rows(spec["field_names"], stdout)
        same(actual["rows"], rows, "restriction rows bound to raw output")
        blanks = [] if rows is None else [{"row_index": i, "field_names": [k for k, v in row.items() if v == ""]}
                                         for i, row in enumerate(rows) if any(v == "" for v in row.values())]
        same(actual["blank_fields"], blanks, "explicit blank restriction values")
        issues.extend(f"{spec['query_id']}: {issue}" for issue in row_issues)
        if rows is not None:
            tables[spec["query_id"]] = rows
    account_rows = tables.get("account_associations", [])
    if not any(row["Account"] == resources["account"] for row in account_rows):
        issues.append("No evidence for the explicitly selected account")
    hierarchy = {}
    for row in tables.get("association_hierarchy", []):
        key = (row["Cluster"], row["ID"])
        if not row["Cluster"] or not re.fullmatch(r"[1-9]\d*", row["ID"]) or key in hierarchy:
            issues.append("Association hierarchy has missing/duplicate cluster or association identities")
        hierarchy[key] = row
    for row in account_rows:
        key, visited = (row["Cluster"], row["ID"]), set()
        while True:
            if key in visited or key not in hierarchy:
                issues.append(f"Account association {row['Cluster']}/{row['ID']}: ancestor evidence missing/cyclic at {key}")
                break
            visited.add(key)
            parent = hierarchy[key]
            parent_id = parent["ParentID"]
            if parent_id in ("", "0"):
                if parent["ParentName"]:
                    issues.append(f"Association {key}: parent name present without parent identity")
                break
            key = (parent["Cluster"], parent_id)
    qos_rows = tables.get("qos", [])
    qos_names = set()
    for row in qos_rows:
        if not row["Name"] or row["Name"] in qos_names or not re.fullmatch(r"[1-9]\d*", row["ID"]):
            issues.append("QOS evidence has missing/duplicate names or invalid QOS identities")
        qos_names.add(row["Name"])
    if resources["qos"] is not None and not any(row["Name"] == resources["qos"] for row in qos_rows):
        issues.append("Explicitly selected QOS is absent from evidence")
    return issues


def validate_restriction_report(report, resources):
    check_seal(report)
    version = report.get("restriction_report_version")
    if type(version) is not int or version != RESTRICTION_REPORT_VERSION:
        raise WorkflowError(f"Restriction report version {version!r} is not submission-compatible; recollect version 2 evidence and review its digest/bounds (no automatic migration)")
    exact(report, {"restriction_report_version", "collection_contract", "account", "qos", "partition",
        "observed_at_utc", "reports", "all_queries_succeeded", "collection_issues", "collection_complete",
        "automatic_entitlement_interpretation", "document_sha256"}, "restriction report")
    same(report["collection_contract"], RESTRICTION_COLLECTION, "required restriction collection contract")
    for key in ("account", "qos", "partition"):
        same(report[key], resources[key], f"effective restriction report {key}")
    same(report["automatic_entitlement_interpretation"], False, "explicit operator entitlement review")
    issues = restriction_collection_issues(report["reports"], resources)
    same(report["all_queries_succeeded"], all(row["ok"] for row in report["reports"]), "actual query success coverage")
    same(report["collection_issues"], issues, "actual restriction collection issues")
    same(report["collection_complete"], not issues, "actual restriction collection completeness")
    if issues:
        raise WorkflowError("Incomplete restriction evidence; resolve unsupported fields, permissions or missing scope before review/submission: " + "; ".join(issues))
    return report


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class SlurmClient:
    def __init__(self, policy):
        self.policy = policy
        self.commands = policy.to_dict()["commands"]

    def invoke(self, name, arguments):
        # No shell. SBATCH_* must not silently add directives to managed requests.
        environment = {k: v for k, v in os.environ.items() if not k.startswith("SBATCH_")}
        environment["TZ"] = "UTC"
        completed = subprocess.run([self.commands[name], *arguments], shell=False,
            check=False, capture_output=True, text=True, timeout=self.commands["timeout_seconds"], env=environment)
        if completed.returncode:
            raise WorkflowError(f"{name} exit {completed.returncode}: {completed.stderr.strip()}; stdout={completed.stdout.strip()}; scheduler state remains uncertain")
        return completed.stdout

    def submit(self, arguments, *, workflow, wave_id):
        """Only an active shared controller and durable exact intent can submit."""
        from .workflow_registry import WorkflowLease
        if not isinstance(workflow, WorkflowLease):
            raise WorkflowError("Submission requires the concrete shared workflow authority")
        workflow.guard()
        wave = workflow.wave(wave_id)
        same(wave["policy_fingerprint"], self.policy.fingerprint, "submission client policy")
        same(wave["submission_arguments"], list(arguments), "durable exact submission intent")
        if wave["status"] != "submission_uncertain" or wave["scheduler"] is not None:
            raise WorkflowError("Do not resubmit acknowledged or resolved work")
        raw = self.invoke("sbatch", arguments).strip()
        match = re.fullmatch(r"([1-9]\d*)(?:;([A-Za-z0-9_.-]+))?", raw)
        if match is None:
            raise WorkflowError("Unrecognized sbatch acknowledgement; preserve submission uncertainty")
        return {"job_id": match[1], "cluster": match[2], "acknowledgement": raw}

    def observe_wave(self, wave):
        """Exact array IDs only. Absent queue rows + complete accounting prove terminal.

        Missing/delayed accounting, duplicate/requeued records, unfamiliar states,
        query failures and a parent-only row never free managed capacity.
        """
        job = wave["scheduler"]
        if job is None:
            return {"terminal": False, "condition": "submission_uncertain", "allocations": [], "diagnostic": "Attach verified exact scheduler identity or explicitly resolve non-submission"}
        job_id = job["job_id"]
        if not re.fullmatch(r"[1-9]\d*", job_id):
            raise WorkflowError("Malformed recorded scheduler job identity")
        cluster_args = [] if job["cluster"] is None else ["--clusters", job["cluster"]]
        queue, parent_present, queue_observed = {}, False, False
        try:
            queue_text = self.invoke("squeue", [*cluster_args, "--noheader", "--array", "--jobs", job_id,
                                                "--format=%i|%T"])
            since = datetime.fromisoformat(wave["intent_at_utc"]).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
            accounts = {}
            expected = {f"{job_id}_{i}": i for i in range(len(wave["items"]))}
            parent_present = False
            for line in queue_text.splitlines():
                if not line.strip():
                    continue
                fields = line.strip().split("|")
                if len(fields) != 2:
                    raise WorkflowError("Malformed squeue response")
                ident, state = fields
                if ident == job_id:
                    parent_present = True
                elif ident not in expected or ident in queue:
                    raise WorkflowError("Unexpected/duplicate array identity in squeue; do not infer free capacity")
                else:
                    queue[ident] = state
            queue_observed = True
            accounting_text = self.invoke("sacct", [*cluster_args, "--noheader", "--parsable2", "--allocations", "--duplicates",
                "--jobs", job_id, "--starttime", since,
                "--format=JobID%64,State%40,ExitCode,Comment%128,Account%128,Partition%128,End"])
            for line in accounting_text.splitlines():
                if not line.strip():
                    continue
                fields = line.strip().split("|")
                if len(fields) != 7:
                    raise WorkflowError("Malformed sacct response")
                ident, state, exit_code, comment, account, partition, ended = fields
                if ident == job_id:  # Aggregate parent alone cannot prove any element terminal.
                    continue
                if ident not in expected:
                    raise WorkflowError("Unexpected scheduler array/step identity in accounting")
                same(comment, wave["scheduler_comment"], "exact submission comment")
                same(account, wave["policy"]["resources"]["account"], "allocation account")
                same(partition, "dgxh", "allocation partition")
                accounts.setdefault(ident, []).append({"state": state, "exit_code": exit_code, "end": ended})
            rows = []
            for ident, index in expected.items():
                records = accounts.get(ident, [])
                terminal = False
                state = queue.get(ident, "UNKNOWN")
                if ident not in queue and not parent_present and len(records) == 1:
                    record = records[0]
                    state = record["state"].split(" ", 1)[0]
                    terminal = (state in TERMINAL and record["end"] not in ("", "Unknown", "None")
                                and re.fullmatch(r"\d+:\d+", record["exit_code"]) is not None)
                    if terminal:
                        datetime.fromisoformat(record["end"])
                rows.append({"array_index": index, "scheduler_id": ident, "state": state,
                             "terminal": terminal, "queue_present": ident in queue or parent_present,
                             "accounting_records": records})
            return {"terminal": all(r["terminal"] for r in rows), "condition": "observed",
                    "observed_at_utc": utc_now(), "allocations": rows, "diagnostic": None}
        except (OSError, subprocess.SubprocessError, WorkflowError, ValueError) as exc:
            return {"terminal": False, "condition": "uncertain", "observed_at_utc": utc_now(),
                    "allocations": [{"array_index": i, "scheduler_id": f"{job_id}_{i}",
                        "state": queue.get(f"{job_id}_{i}", "UNKNOWN"), "terminal": False,
                        "queue_present": True if parent_present or f"{job_id}_{i}" in queue else (False if queue_observed else None),
                        "accounting_records": []} for i in range(len(wave["items"]))],
                    "diagnostic": f"{type(exc).__name__}: {exc}"}

    def verify_submission_identity(self, wave, job_id, cluster=None):
        """Explicit operator attachment; exact unique comment AND array/account binding.

        The operator supplies the ID. No search through unrelated loosely named
        jobs and no inference of non-submission from an empty squeue response.
        """
        if type(job_id) is not str or not re.fullmatch(r"[1-9]\d*", job_id):
            raise WorkflowError("Supply the exact positive scheduler array ID")
        if cluster is not None and (type(cluster) is not str or not re.fullmatch(r"[A-Za-z0-9_.-]+", cluster)):
            raise WorkflowError("Invalid explicit scheduler cluster")
        cluster_args = [] if cluster is None else ["--clusters", cluster]
        raw = self.invoke("scontrol", [*cluster_args, "show", "job", "--oneliner", job_id])
        records = []
        for line in raw.splitlines():
            fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]*)", line))
            if not fields:
                continue
            same(fields.get("Comment"), wave["scheduler_comment"], "submission token in scheduler")
            same(fields.get("Account"), wave["policy"]["resources"]["account"], "scheduler account")
            same(fields.get("Partition"), "dgxh", "scheduler partition")
            same(fields.get("ArrayJobId"), job_id, "scheduler array root")
            selector = fields.get("ArrayTaskId", "").split("%", 1)[0]
            for part in selector.split(","):
                match = re.fullmatch(r"(\d+)(?:-(\d+)(?::([1-9]\d*))?)?", part)
                if match is None:
                    raise WorkflowError("Malformed scheduler array selector")
                first, last = int(match[1]), int(match[2] or match[1])
                if not 0 <= first <= last < len(wave["items"]):
                    raise WorkflowError("Scheduler array mapping escaped the reserved wave")
            records.append(fields)
        if not records:
            raise WorkflowError("Scheduler identity unavailable; retain unresolved intent and obtain administrator evidence")
        return {"job_id": job_id, "cluster": cluster, "acknowledgement": "operator_verified_scontrol",
                "identity_evidence": records}

    def inspect_restrictions(self):
        r = self.policy.to_dict()["resources"]
        if r["account"] is None:
            raise WorkflowError("Supply the actual Slurm account before inspecting effective restrictions")
        reports = []
        for spec in restriction_requests(r):
            try:
                stdout = self.invoke(spec["command"], spec["arguments"])
                rows, _ = _restriction_rows(spec["field_names"], stdout)
                blanks = [] if rows is None else [{"row_index": i, "field_names": [k for k, v in row.items() if v == ""]}
                                                 for i, row in enumerate(rows) if any(v == "" for v in row.values())]
                reports.append({**spec, "ok": True, "stdout": stdout, "error": None, "rows": rows, "blank_fields": blanks})
            except (OSError, subprocess.SubprocessError, WorkflowError) as exc:
                reports.append({**spec, "ok": False, "stdout": None, "error": str(exc), "rows": None, "blank_fields": []})
        issues = restriction_collection_issues(reports, r)
        return seal({"restriction_report_version": RESTRICTION_REPORT_VERSION, "collection_contract": RESTRICTION_COLLECTION,
            "account": r["account"], "qos": r["qos"], "partition": "dgxh",
            "observed_at_utc": utc_now(), "reports": reports, "all_queries_succeeded": all(r["ok"] for r in reports),
            "collection_issues": issues, "collection_complete": not issues,
            "automatic_entitlement_interpretation": False})
