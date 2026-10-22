# Pre-submission validation performance

This correction targets repeated validation during `plan`, `submit` and
`advance --submit`. It retains 40 queued tasks and the running throttle of four;
policy versions, resource checks, mappings and registry formats do not change.
No policy schema change is required. Switching to a different execution checkout
path still requires the existing explicit quiescent policy-adoption procedure.

The HPC regression in `dcddd03` invalidates its POSIX timestamp fast path:
`test_same_size_edit_with_restored_mtime_invalidates` failed after `original` was
replaced by `modified` and mtime restored. Unchanged size/mtime/ctime is **not**
content evidence, regardless of operating system, mount type or apparent precision.
The correction below supersedes the timestamp assumptions in that commit.

## Authority and freshness

Each planning operation owns one in-memory `PlanningInspection`. `advance` shares
it with its subsequent submission preflight; separate commands start empty.
The first observation still performs restricted checkpoint loading, content hashes,
current/previous journal validation, epoch/layer validation, immutable completion
CSV byte comparison, prepared-data/ownership checks and continuation compatibility.
Validated observations and compact journal metadata can then be reused. The cache
retains neither tensors nor a persistent index.

Every platform now uses SHA-256 content evidence. Generic `reuse` callers hash
every regular file in their declared scope, conservatively. Production observation,
journal, checkpoint and evaluation-coverage readers opt into dependency tracking:
their content readers register the exact bytes/digests used by validation. Every
reuse rehashes those dependencies. Successful scientific validation is still
required; hashes alone cannot establish completion or compatibility.

`read_checkpoint` records the digest verified from its restricted-load stream;
artifact/data digest readers register their observed hashes. `read_json` registers
the exact raw UTF-8 bytes it parses, including CRLF, without an extra probe read.
Nested cache hits propagate their freshly checked dependencies to their callers.
Conflicting reads or dependencies outside the declared scope reject caching.
Future validators using uninstrumented readers must retain the whole-scope fallback.

Tracked content includes the current/previous transaction dependencies, referenced
latest/best/final checkpoints, epoch records, completion CSV/summary bytes, scanned
failure evidence and evaluation records/ancestry. Unreferenced old checkpoint
payloads are not content-read simply because they share a directory. Recursive
directory membership/type/ancestor evidence still covers all descendants, including
new orphan failures and linked paths. HEADs, resolved config, owners and locks are
always content-hashed, even if a validator did not read them. Timestamps remain
conservative change hints only. Linked paths retain existing rejection/diagnostic
rules; snapshots never follow child links or release locks.

On a miss, structure/control evidence brackets validation and newly discovered
dependencies must still match the bytes actually read before entering the cache.
On a hit, content and structure are freshly compared before returning the result.
The final pre-intent guard freshly hashes the union of retained dependencies once,
deduplicating shared files across observation/journal keys. That digest map exists
only inside that single guard; it is never reused by later checks. A content edit
with every stat field unchanged therefore invalidates both reuse and finalization.
These independent read observations remain subject to concurrent filesystem reads;
they do not provide a writer lease or an atomic grid snapshot.

Within one journal read, identical checkpoints shared by current/previous
transactions are validated once, with file freshness checks before reuse. Temporary
tensors are released before retaining compact planning evidence. `inspect_task`
no longer calls `inspect_completion` after the journal has already validated
completion. Planning evaluation bindings use already validated best metadata;
actual evaluation still loads the full immutable best payload.

Historical and actual-source eligibility remain distinct. Keys include effective
configuration, prepared identities and relevant source/software compatibility.
Inspection timestamps do not affect compatibility; complete supplied provenance
is still validated, and persisted provenance is never rewritten. Evaluation
coverage has its own binding/provenance/root key and snapshots its namespace,
locks, raw slots and sweep authority.

Submission still reconciles scheduler/registry ownership, rebuilds eligibility
against current state, checks selected sealed work and launch/source/software pins,
verifies restriction evidence and compares registered mapping/script content.
Immediately before durable reservation, it freshly checks **all** retained evidence,
including CIFAR-10 observations supporting the CIFAR-100 gate. A change rejects the
attempt before intent or sbatch; replan after writers exit. Mutation during
validation also rejects reuse. Completion never derives from existence, aliases
or scheduler outcomes. Independent observations do not lock the scientific grid.

Progress goes only to stderr: reconciliation, training task index/run ID,
immutable completion CSV verification, evaluation eligibility, reuse counts,
final freshness guard and durable-intent/submission boundary. JSON stdout remains
machine-readable. Counts describe cache operations across observation/journal/
coverage categories, not completed research tasks.

## Focused verification and measured limits

The user explicitly authorized these isolated tests and synthetic benchmark after
the source-only implementation phase. No model, torch, dataset, training,
evaluation, checkpoint creation, scheduler command or HPC execution was involved.
With `src` on `PYTHONPATH`:

```text
python -B -m unittest discover -s tests -p test_planning_inspection.py -v
python -B benchmarks/planning_validation.py --tasks 4 --passes 3 --history-files 12
```

Tests cover exact legacy/new UTF-8 CSV bytes and serialization counts; quoted,
newline, Unicode and null fields; headers; context changes; nonfinite rejection;
operation isolation; production observation-cache wiring; source/software keys;
malformed provenance; changed/deleted/replaced artifacts; same-size edits with
restored mtime; new HEAD/owner/config/failure evidence; locks; mutation during
validation; final freshness rejection; and stderr isolation. The deep scientific
reader is mocked only at the observation wiring boundary. These tests do not
certify checkpoint formats or real submission. The symlink case is skipped if
Windows denies symlink creation.

The deterministic regression explicitly freezes both `Path.stat` and `Path.lstat`
for the dependency and its ancestors, and asserts that the entire structural
snapshot stays identical while replacing eight bytes with different eight bytes. Both generic and
tracked cache modes reject the edit. It has no sleeps, skips or forced timestamp
changes. The original restored-mtime test also passes. Tests additionally cover
tracked JSON under frozen stats, exact CRLF/UTF-8 binding, unchanged strict JSON
rejections, nested dependency propagation, final-guard deduplication, unused-history
read counts, changed directory membership under frozen directory stats, ownership
content changes and conflicting/out-of-scope reads.

Final local discovery (`python -B -m unittest discover -s tests -v`) ran 35 cases:
34 passed; the pre-existing real-symlink test was skipped for Windows privileges.
The original HPC-failing test, new deterministic regressions, mocked ancestor-link
test and all 12 unchanged queue/legacy-wave checks passed. No invalidation regression
is skipped.

Measured on Windows 11, Python 3.13.12 (standard-library verification only;
supported research Python remains 3.10–3.12):

| Synthetic workload | Before | After | Result |
| --- | ---: | ---: | --- |
| CSV: 200 epochs, 64 layers/epoch, 96 provenance inventory entries | 3.244 s | 1.811 s | 1.79× faster |
| Identity serializations for that CSV | 12,800 | 200 | Once per context/render |
| Four runs, three repeated passes, three 2 MiB binaries and 200 JSON records/run, including full CSV reconstruction and final guard | 38.402 s | 8.855 s | 4.34× faster; equivalent observations |
| Deep validations / CSV reconstructions | 12 / 12 | 4 / 4 | Eight unchanged reads reused |
| Deep-validator read bytes | 75,656,952 | 25,218,984 | Content-fingerprint reads reported separately below |
| Same filesystem workload without CSV reconstruction | 0.735 s | 1.902 s | Content-safe reuse costs more than this cheap synthetic reader |

The workload also includes 12 unreferenced 2 MiB history files per run: 96 MiB
of unused history alongside 25,218,984 bytes of validated dependencies. All three
modes produce equivalent observations. Read overhead is measured explicitly:

| Content-safe mode | Validator bytes | Fingerprint bytes | Total instrumented bytes | Unused-history hash bytes |
| --- | ---: | ---: | ---: | ---: |
| Whole-scope conservative fallback | 25,218,984 | 629,411,400 | 654,630,384 | 503,316,480 |
| Production-style dependency tracking | 25,218,984 | 100,875,936 | 126,094,920 | 0 |

Dependency tracking avoided 528,535,464 read bytes versus whole-history hashing in
this workload. Whole-scope fallback took 2.973 s for reads alone / 9.814 s with CSV;
tracked reuse took 1.902 s / 8.855 s. These counters include the final guard.
Probe reads were zero because this benchmark supplies observed digests, as the
production hash readers do; JSON now supplies its actual parsed bytes as well.

Correctness has a real I/O cost: tracked total reads are 126,094,920 bytes versus
75,656,952 for the uncached synthetic reader, despite fewer deep validations.
With three passes and one final guard, tracked fingerprint volume here is four
times the live dependency byte set, independent of unused-history size. There is
no claim of zero-read reuse or lower physical I/O. Current referenced large
checkpoints still must be rehashed when no trusted storage immutability proof is
available; deserialization and byte-identical CSV reconstruction remain reusable.

Both CSV renderers produced exactly 159,151,994 bytes with SHA-256
`2699f64d808e859748005ec8bc1bb00aa5073a3f7c82bb5f695ef3979094d101`.
The benchmark preserves the old per-row renderer as its equivalence oracle and
compares repeated deep reads against reuse plus the final guard. It removes its
temporary synthetic files and writes nothing to scientific namespaces.

These are single local wall-clock measurements, not HPC throughput estimates.
Read counters measure actual bytes passed through the synthetic validator and
content fingerprint/probe hash readers, not `/proc`, OS cache misses or storage
traffic. The synthetic reader hashes/parses bytes and renders CSV; it
does not deserialize PyTorch checkpoints or exercise live submission, registry
recovery, all 240 tasks or concurrent cross-node writers. POSIX metadata latency,
large checkpoints, shared-filesystem coherence and end-to-end HPC improvement
remain unmeasured. Cold scans still deeply validate all selected authority.

## Exact HPC update procedure

Transfer source through a separate staged checkout/three-way merge, preserving the
active checkout, policies, registry history, bundles, results and raw checkpoints.
This patch does **not** edit `work_render.py` or `slurm_client.py`. Retain and review
HPC email notifications, full-queue squeue with exact job-ID filtering, and blank
sacct Comment identity fallback using exact recorded SubmitLine arguments. Do not
replace those files wholesale with workstation copies. Private HPC patches were
not available here; their merged behavior remains unverified.

The following are future operator instructions; none were executed here. Set all
variables to reviewed absolute paths, with new patch/staging paths outside research
namespaces. Do not substitute an active checkout for `STAGE` or overwrite a patch.

1. Export this reviewed tracked diff against `dcddd03` from the workstation. For
   PowerShell, with `FIX_PATCH` set to a new destination:

   ```powershell
   $fixPaths = @(
     'src/alpha_sweep_training/planning_inspection.py',
     'src/alpha_sweep_training/checkpoint_io.py',
     'src/alpha_sweep_training/checkpoint_journal.py',
     'src/alpha_sweep_training/checkpoints.py',
     'src/alpha_sweep_training/coverage.py',
     'src/alpha_sweep_training/data_contract.py',
     'src/alpha_sweep_training/evaluation_coverage.py',
     'src/alpha_sweep_training/serialization.py',
     'tests/test_planning_inspection.py', 'benchmarks/planning_validation.py',
     'docs/planning-validation-performance.md', 'docs/operating-guide.md',
     'docs/step9-progress.md'
   )
   git diff --binary --output="$FIX_PATCH" dcddd03 -- $fixPaths
   Get-FileHash -LiteralPath "$FIX_PATCH" -Algorithm SHA256
   ```

   Transfer the patch through the operator's approved channel and compare its
   SHA-256 with `sha256sum "$FIX_PATCH"` on HPC. This export excludes the local
   untracked `AGENTS.md`; no policy, registry, bundle or result is included.

2. Preserve the actual HPC baseline and all private changes before staging. Bash:

   ```bash
   umask 077
   git -C "$ACTIVE" rev-parse HEAD
   git -C "$ACTIVE" status --short
   git -C "$ACTIVE" diff --binary --output="$HPC_PATCH" HEAD
   git -C "$ACTIVE" ls-files --others --exclude-standard
   ```

   Set `ACTIVE_HEAD` to the recorded exact commit. Keep the private patch secure.
   Preserve reviewed untracked source/operational inputs separately by exact path;
   Git's diff does not contain them. Do not copy run/checkpoint/data namespaces.
   Leave the active checkout, original policy and environment intact.

3. Create a separate staging clone from that actual baseline and merge both diffs:

   ```bash
   git clone --no-hardlinks --no-checkout "$ACTIVE" "$STAGE"
   git -C "$STAGE" checkout --detach "$ACTIVE_HEAD"
   git -C "$STAGE" apply --check "$HPC_PATCH"
   git -C "$STAGE" apply --index "$HPC_PATCH"
   git -C "$STAGE" apply --3way "$FIX_PATCH"
   git -C "$STAGE" diff HEAD --stat
   git -C "$STAGE" diff HEAD
   cmp "$ACTIVE/src/alpha_sweep_training/work_render.py" "$STAGE/src/alpha_sweep_training/work_render.py"
   cmp "$ACTIVE/src/alpha_sweep_training/slurm_client.py" "$STAGE/src/alpha_sweep_training/slurm_client.py"
   ```

   If the private diff is empty, omit its two `apply` commands. Any apply conflict
   must be resolved/reviewed in staging; never discard the HPC side wholesale.
   Both `cmp` checks should succeed: this fix does not edit either module. Review
   email options, full-queue/exact-ID filtering and exact SubmitLine fallback in
   the combined source. Restore only separately reviewed untracked source inputs
   needed by staging. Do not reset, clean, pull over or patch the active checkout.

4. Use the existing approved interpreter, without installing anything, to verify
   staging. These commands use only isolated temporary synthetic bytes:

   ```bash
   cd "$STAGE"
   PYTHONPATH="$STAGE/src" "$PYTHON" -B -m unittest discover -s tests -v
   PYTHONPATH="$STAGE/src" "$PYTHON" -B benchmarks/planning_validation.py --tasks 4 --passes 3 --history-files 12
   ```

   Require the restored-mtime and unchanged-stat regressions to pass; do not
   accept a skip for either. Record actual HPC measurements and review the staged
   diff, then freeze the reviewed combined source. No Slurm or research operation
   is part of these checks.

5. **Do not switch existing allocations or unfinished training to the new source.**
   This fix changes the inventory digest. Keep the original frozen checkout for
   compatible continuation; finish unfinished work under its original source.
   A terminal current wave alone does not make unfinished checkpoints compatible.
   Use the original controller and sealed policy for workflow-only inspection,
   reconciliation and ownership/submission resolution under the
   [existing recovery procedure](operating-guide.md#historical-policy-inspection-recovery-and-subsequent-adoption).
   Never rewrite provenance, old bundles or checkpoints, and never waive source
   equality to install this fix.

6. Once all recorded allocations are conclusively inactive, ownership/submission
   ambiguity is resolved and the frozen-source continuation constraint is satisfied,
   a separately approved execution switch may use the staged checkout. Keep the
   original policy unchanged. If `paths.checkout` changes, create a reviewed new
   policy copy retaining the same registry/data/result roots and the exact 40/4
   settings, then use explicit adoption with both policies:

   ```bash
   PYTHONPATH="$STAGE/src" "$PYTHON" -B -m alpha_sweep_training.hpc_workflow adopt-policy \
     --manifest "$FULL_MANIFEST" --policy "$REVIEWED_POLICY" \
     --previous-policy "$ORIGINAL_POLICY" --reason "$APPROVED_UPDATE_REASON"
   ```

   Adoption is a later authorized operational action, not a command run by this
   source correction. It adds authority only through existing quiescence checks;
   it never rewrites historical policy/snapshots. Use fresh plans/bundles with new
   pins for future work, not resubmission of historical bundles. Stop before any
   submission; this handoff authorizes no job or research execution.
