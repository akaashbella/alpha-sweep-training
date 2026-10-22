# Queued waves with a separate running throttle

The operational update of 2026-10-21 supports up to 40 eligible tasks in one
array, with four running together by default. It changes no scientific setting,
task identity, checkpoint, seed, result or continuation policy. No HPC checkout,
job or experiment storage was changed while implementing it.

## Policy and resource contract

New policies use `policy_version: 2`. Keep `resources.max_concurrent: 4` and
set `resources.max_queued: 40`. `max_queued` counts all elements in the wave,
including pending and running elements; the supported ceiling is 40. Rendering
selects up to that many eligible distinct training runs/checkpoints, preserving
their original manifest indices. Short waves remain valid. Forty tasks render
`--array=0-39%4`; two tasks render `--array=0-1%2`.

Slurm's percent suffix bounds simultaneous running elements. Each array element
still counts as a job for job/submission limits; array size also has independent
cluster limits. See the [official array guide](https://slurm.schedmd.com/job_array.html)
and [versioned configuration reference](https://raw.githubusercontent.com/SchedMD/slurm/slurm-24-11-0-1/doc/man/man5/slurm.conf.5).

| Setting | Version-2 meaning |
| --- | --- |
| `resources.max_queued` | Maximum elements rendered/queued in a wave; default 40 |
| `resources.max_concurrent` | Running throttle; default 4, shared across both datasets/lifecycles through one unresolved wave |
| `effective_limits.max_allocations` | Reviewed allowance for simultaneously running jobs; explicitly re-review its meaning when migrating from v1 |
| `effective_limits.max_submitted_tasks` | Reviewed available submission allowance for all pending plus running elements, considering account/user/QOS/group limits and other jobs |
| `effective_limits.max_array_size` | Reviewed bound on array index span; zero-based contiguous arrays require element count <= this value |
| `effective_limits.max_array_tasks` | Reviewed bound on number of tasks in a single array |
| Existing GPU/CPU/memory and resource-hour bounds | Apply to simultaneous requested resources/time at the enforced running throttle |

Every reviewed bound must be explicit and positive. None of the new entitlement
fields has a production default; `effective_limits: null` remains an unresolved
declaration and cannot authorize submission. Setting `max_concurrent` to 40 is
rejected by the original resource checks. A configured queue that exceeds reviewed
limits is rejected rather than silently shrinking or weakening the limits.

For 40 tasks at one GPU, eight CPUs, 32 GB and 12 hours each:

| Quantity | Queued / whole wave | Simultaneous reservation |
| --- | --- | --- |
| Tasks | 40 | 4 |
| GPUs | 40 requests | 4 |
| CPUs | 320 requests | 32 |
| Host RAM | 1,280 GB requested across tasks | 128 GB |
| Requested GPU-hours | 480 | 48 |
| Requested CPU-hours | 3,840 | 384 |
| Requested GB-hours | 15,360 | 1,536 |

Whole-wave totals are recorded separately in `requested_wave_totals`; they are
neither measured consumption nor a reservation of 40 simultaneous GPUs. They
must still be considered when reviewing cumulative quotas/usage and enforcement.
Do not confuse `GrpTRESMins`/decayed cumulative usage with `GrpTRESRunMins` or
per-user/account running-minute limits. This code does not guess cumulative
entitlements or available headroom from blank accounting fields.

The existing sealed restriction report version 2 remains required. In addition
to all prior coverage checks, a version-2 policy's submission gate reads the
report's raw `configuration` output for exactly one `MaxArraySize` and complete
`SchedulerParameters` value. Explicit `max_array_tasks` limits must parse correctly.
Only when the complete recorded parameter value omits that option does its
documented MaxArraySize default apply. Missing/blank/ambiguous configuration is
an error. Reviewed array bounds cannot exceed the recorded cluster caps.
Recollect/review evidence if the existing report does not establish those facts.
The account/QOS submission allowance remains an explicit operator review, never
an automatic inference from a successful query or cluster array cap.

## Seals, ownership and legacy compatibility

Policy v1 is loaded as originally written: no fields are inserted, no policy
fingerprints change and its budget still treats every wave element as simultaneous.
Existing four-task waves therefore retain their original budget/seal semantics.
They remain readable/reconcilable, including old bundles without launch pins.
The existing launch-pin gate still rejects those unpinned bundles for new execution.

Policy v2 produces plan and bundle version 2. Its budget names `queued_tasks`,
`running_tasks`, `running_throttle`, `queued_limits` and `requested_wave_totals`.
Plans separate `reserved_queued_tasks` from `reserved_running_tasks`; v1 plans
retain the historical `reserved_allocations` field. Registry/owner/state/outcome
and individual work-item formats stay version 1; each historical wave is validated
against its own sealed policy/bundle version. Historical snapshots/bundles are
never rewritten or resealed.

Before new durable intent, the exact argument list must contain the expected
single array range/throttle. Version-2 registry reads repeat that check, so a
budget for four running tasks cannot authorize a `...%40` submission. The rendered
mapping, policy, budget and durable submission arguments remain bound together.

Keep **one unresolved wave**. Pending, unknown and residual-owner elements reserve
the entire wave's queue capacity and its full possible running throttle. An opening
is filled by Slurm within that same array; it does not authorize a second wave.
Reconciliation still requires every expected array index to be terminal and owning
leases gone. If accounting reopens multiple old waves, all their reservations remain
visible and new submission stays blocked. Safe continuation, attempt bounds,
numerical/execution failure distinctions and exact acknowledgement recovery are
unchanged. CIFAR-100 training remains ineligible until all 120 CIFAR-10 training
completions validate; queueing does not pre-authorize later-phase work.

## Safe transfer and explicit migration on HPC

This checkout does not contain the HPC-local email notifications, full-queue squeue
query/exact-ID filtering, or blank-Comment SubmitLine identity fallback. The update
intentionally leaves `slurm_client.py` unchanged and does not edit the renderer's
batch-source body. Those private HPC changes have not been merged or runtime-tested
here. Never copy whole local files over their HPC counterparts or discard conflicts.

1. Leave the active HPC checkout/policy/environment untouched while the current
   wave can run or continue. Record its exact source state and preserve all HPC
   tracked diffs and relevant untracked operational inputs privately, including email
   settings. This work does not perform that transfer or access the HPC filesystem.
2. Export the reviewed change as a patch/commit and apply it in a **separate staging
   checkout**, based on the actual HPC revision. Use a three-way merge of the feature
   and the saved HPC-local changes. Preserve conflicts for manual review; do not use
   reset/clean, checkout-theirs/ours for entire files, or edit the active checkout.
   If using `git am --3way`, first arrange a clean staging index and retain a private
   copy/commit of site changes; do not publish private email/configuration accidentally.
3. Review the staged diff specifically: email options must survive in rendering;
   the HPC squeue query must still inspect the full queue and filter exact managed
   IDs; blank sacct Comment must still require the **exact recorded original
   SubmitLine arguments**, not a loose job-name match. The new `--array=...%4` is
   stored in those original arguments. Re-run the focused checks in staging and
   perform the site's separately authorized compatibility review before promotion.
   Use the existing compatible shared interpreter with the staged `src` directory
   explicitly on PYTHONPATH when invoking staged module entry points. Do not retarget
   an active editable installation or assume changing the shell directory selects
   the staged package; provenance requires the checkout actually imported.
4. Finish/reconcile the current wave with its original sealed policy. If submission,
   accounting or ownership is uncertain, use the existing
   [historical-policy recovery sequence](operating-guide.md#historical-policy-inspection-recovery-and-subsequent-adoption).
   A terminal allocation is not proof of scientific completion. **Unfinished
   training still requires the identical frozen source/software digest.** This
   update changes the inventory digest. Complete/continue any such work under its
   original frozen checkout before changing its execution source; do not waive the
   compatibility gate or rewrite saved provenance to make migration appear valid.
   Use explicit eligible `--work-id` selections under the original policy to drain
   unfinished tasks without starting more tasks under the old source when needed.
   Existing complete results remain readable. Quiescence alone is insufficient to
   make unfinished checkpoints compatible with a new source tree.
5. Create a separate reviewed v2 policy, retaining the same workflow root/full
   manifest, setting the independent queue/throttle values and all three new
   reviewed limits. Select the staged frozen checkout for **new** execution. Do
   not edit the old policy or raw/saved scientific data. Re-review restriction
   evidence, submission headroom, cluster array caps and cumulative quotas.
6. After every prior allocation is conclusively terminal or explicitly proven not
   submitted, and all recorded ownership/submission ambiguity is resolved, use the
   existing explicit adoption operation (future instructions only):

   ```bash
   alpha-sweep-workflow status --manifest "$FULL_MANIFEST" --policy "$ORIGINAL_POLICY" --workflow-only
   alpha-sweep-workflow reconcile --manifest "$FULL_MANIFEST" --policy "$ORIGINAL_POLICY"
   alpha-sweep-workflow adopt-policy --manifest "$FULL_MANIFEST" --policy "$REVIEWED_V2_POLICY" \
     --previous-policy "$ORIGINAL_POLICY" --reason "$REVIEWED_MIGRATION_REASON"
   ```

   Run the appropriate original/staged command from the checkout that actually
   supplies it. Adoption validates original policy identity and authoritative
   quiescence, adding only a new registry snapshot; old wave policies/arguments,
   bundles, receipts and attempt ordinals remain unchanged.
7. Plan/render a new reviewed wave separately from any explicit submission:

   ```bash
   alpha-sweep-workflow plan --manifest "$FULL_MANIFEST" --policy "$REVIEWED_V2_POLICY" > "$NEW_PLAN"
   alpha-sweep-workflow render --manifest "$FULL_MANIFEST" --policy "$REVIEWED_V2_POLICY" --plan "$NEW_PLAN"
   ```

   Inspect the actual selected indices, bindings, 40/4 budget, original HPC options
   and exact argument vector. No submission is authorized or performed by this
   implementation/transfer handoff. Keep the original checkout available for audit
   and compatible old work; use no second registry to evade the unresolved-wave rule.

## Focused verification

`tests/test_queued_waves.py` exercises only operational declarations and in-memory
wave metadata. Run from the repository with `src` on PYTHONPATH:

```text
python -B -m unittest discover -s tests -p test_queued_waves.py -v
```

Checks cover 40 queued/4 running, small/oversized arrays, independent job/submit/
array limits, simultaneous GPU/CPU/memory and requested-time bounds, raw cluster
evidence, durable throttle mismatches, uncertain-wave reservations, unchanged v1
fingerprints/budgets/plans/bundles and current render caps. Scientific manifest
membership and launch-file I/O are mocked in focused renderer tests; no models,
datasets, checkpoints, Slurm calls or research/artifact generation occur.

Local verification used the available Python 3.13 interpreter. These standard-library
operational checks do not expand the supported research Python range (3.10–3.12),
validate the private HPC merge or establish live scheduler/CUDA/filesystem behavior.
All 12 focused tests passed locally. No scheduler query/submission, model/data
computation, HPC checkout modification or scientific artifact generation occurred.
