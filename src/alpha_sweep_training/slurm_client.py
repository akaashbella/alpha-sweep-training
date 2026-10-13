"""Structured Slurm subprocesses, invoked only by explicit future operations."""

from datetime import datetime, timezone
import os
import re
import subprocess

from .hpc_policy import WorkflowError, same
from .work_plans import seal

TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY",
            "PREEMPTED", "BOOT_FAIL", "DEADLINE"}


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
        requests = [("scontrol", ["show", "partition", "dgxh", "--oneliner"]),
                    ("scontrol", ["show", "config"]),
                    ("sacctmgr", ["--noheader", "--parsable2", "show", "association", f"account={r['account']}",
                        "format=Cluster,Account,User,Partition,QOS,DefaultQOS,GrpTRES,GrpTRESMins,MaxTRESMins,MaxWall,MaxJobs,MaxSubmit"]),
                    ("sacctmgr", ["--noheader", "--parsable2", "show", "qos",
                        "format=Name,GrpTRES,GrpTRESMins,MaxTRESMins,MaxWall,MaxJobsPU,MaxSubmitJobsPU"])]
        reports = []
        for command, args in requests:
            try:
                reports.append({"command": command, "arguments": args, "ok": True, "stdout": self.invoke(command, args), "error": None})
            except (OSError, subprocess.SubprocessError, WorkflowError) as exc:
                reports.append({"command": command, "arguments": args, "ok": False, "stdout": None, "error": str(exc)})
        return seal({"restriction_report_version": 1, "account": r["account"], "qos": r["qos"], "partition": "dgxh",
            "observed_at_utc": utc_now(), "reports": reports, "all_queries_succeeded": all(r["ok"] for r in reports),
            "automatic_entitlement_interpretation": False})
