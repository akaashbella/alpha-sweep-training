# Checkpointed MobileNet epsilon candidate

Prepared October 31, 2026, from audited HEAD
`71922f32b165d6a84b1786cbd858768f7e46c52d`. This implements the supplied
`mobile-epsilon-findings/CODEX_NEXT_STEPS.md` handoff. The original checkout,
untracked `AGENTS.md`, and HPC/private patches were not changed. Local review
checkout: `/private/tmp/alpha-sweep-mobile-epsilon-candidate`, branch
`candidate/mobile-epsilon-00316`. No commit or push was made. Copy this temporary
checkout to durable development storage before relying on it long term.

## Evidence and decision boundary

The supplied `FINDINGS.md`, `data/verification.json`, and raw paired protocol and
diagnostic JSONs were read. Their comparison reports epoch-20 training/validation
accuracy 85.64%/78.80%, best validation 82.90% at epoch 19, and final losses
0.41835/0.65030 for epsilon .0316. The matched .001 baseline reports 10.14%/11.20%,
losses 2.30992/2.30389 and maximum validation loss 8.59271e20. This session did not
rerun or independently reconstruct those experiments. They justify a candidate,
not a Gaussian-perturbation defect claim or full-budget stability claim.

The baseline protocol's exact scientific fingerprint is
`89deb14382b735fd8a8f59cb790886e93f48eb19560993c7e43cb4d6bf564099`.
Build requires the actual original **seed-2 alpha-zero resolved config** with
this fingerprint, then reads its existing prepared-data manifest. There is no
guessed normalization, split seed, relocated dataset, placeholder completion or
dataset generation. If that saved config/data are unavailable, build fails before
publishing the candidate namespace. Preserve relative-path meaning by supplying
the original run's actual base directory; absolute saved paths remain absolute.

## Frozen science and compatibility

`scientific.specification_version: cifar10-mobile-eps0316-v1` is a new opt-in
profile restricted to CIFAR-10 MobileNetV3-Large. Epsilon is exactly .0316.
RMSprop LR .01, momentum .9, smoothing alpha .9, decay .00001 and pinned native
defaults remain unchanged. All clean parameters, including bias/normalization,
remain in one group. EfficientNet and all main-v1 declarations retain .001.

Schema 4, fingerprint 4 and shared builder/checkpoint/journal/evaluation versions
are unchanged. The new profile and declared epsilon are included in the existing
scientific hash. Main-v1 effective documents, hash payload, readable run names,
path layout and historical readers retain their prior meaning. Storage, planning,
source/software/activation pins remain outside scientific hashes. This changed
source is **not compatible permission to continue old main-v1 managed work**;
historical source/software gates are retained.

The candidate builder copies the audited original science, changing only profile,
optimizer epsilon and alpha/seed task selectors. Config validation additionally
fixes seeds [0,1,2], 200 epochs/batch 128, five-epoch warmup start .01, epoch
cadence, cosine minimum 0, drop_last false, bfloat16, highest matmul precision,
TF32 false, deterministic algorithms/cuDNN true and benchmark false. Native data,
initialization, RNG keys, architecture, loss, augmentation, perturbation, clean
validation, earliest best ties, and persistence are reused unchanged.

`configs/candidates/mobile-eps0316.yaml` is the review declaration, to layer only
over that saved baseline, not `configs/research.yaml`. `mobile_candidate build`
constructs the equivalent resolved declarations and changes execution paths to
the owned root; it also preserves the paired loader setting of four workers.

## Study, roots and pilot

The distinct manifest grid version `cifar10-mobile-eps0316-30-tasks-v1` requires
exactly ten approved alphas × actual seeds 0,1,2, in alpha-major/seed-minor order.
The old grid version still requires exactly 240 main-v1 tasks. No truncated main
manifest is accepted. Candidate indices 0–2 are permanently part of this same
30-run identity; accepting the recipe requires no relabeling or pilot repetition.
Changing any scientific setting requires a fresh compatibility assessment and,
for another recipe, a new profile. Pilot completion does not complete the study.

Proposed new HPC source checkout:
`/nfs/hpc/share/bellaak/research/alpha-sweep/alpha-sweep-mobile-eps0316`.
Proposed new study root:
`/nfs/hpc/share/bellaak/research/alpha-sweep/mobile-eps0316-v1`.
These are selected handoff paths, not claims that directories exist on HPC.
Build refuses an existing root, linked ancestors or overlap with original base,
source/data/artifact roots and all explicitly supplied protected roots. Supply
every original workflow/evaluation/report namespace as additional protected roots.
An interrupted build leaves evidence; choose a new root instead of replacing it.

The study root owns `OWNER.json`, `study.json`, `pilot.json`,
`seed-0.config.json`, `seed-1.config.json`, `seed-2.config.json`, and
`allocation-logs/`. All thirty resolved configs live in `study.json`.
`pilot.json` pins that complete manifest, exact three-member selection, source
inventory/software, configured/resolved existing interpreter and its bytes,
activation bytes, candidate declaration and batch-template bytes, and thread
policy. No main workflow registry is reused or created. The direct pilot is
explicitly operator-managed; it has no scheduler client, automatic replacement,
requeue, submission ambiguity reconciliation, or adoption decision.

Native run metadata is below
`$CANDIDATE_ROOT/runs/cifar10-mobile-eps0316-v1/<run_id>/`, with nested
`checkpoints/`, `results/`, `logs/` owned by the existing run lease. Each candidate
run ID also includes the profile prefix. This protects writers even when a root
override is supplied through a generic native service. Copied foreign profile
owners reject continuation and archival overwrite. The pilot runner accepts no
storage overrides or overwrite operation and rechecks all thirty storage bindings.

| Index / array index | Actual seed | Training alpha | Epochs | Run ID |
| --- | ---: | ---: | ---: | --- |
| 0 | 0 | 0 | 200 | cifar10-mobile-eps0316-v1__cifar10__mobilenetv3_large__alpha-0__seed-0 |
| 1 | 1 | 0 | 200 | cifar10-mobile-eps0316-v1__cifar10__mobilenetv3_large__alpha-0__seed-1 |
| 2 | 2 | 0 | 200 | cifar10-mobile-eps0316-v1__cifar10__mobilenetv3_large__alpha-0__seed-2 |

`mobile_candidate run` registers existing attribute-only signal handlers and
delegates to `execute_task`, `train_run`, `DurableTrainingPersistence` and full
`ContinuationPlan`. Latest restores progress; best is validation selection; final
requires the original 200-epoch budget. An advance warning discards the current
uncommitted epoch and replays from the durable boundary in a new allocation.
Native automatic dispatch blocks numerical/execution failures. An explicit
`--operation continue` can authorize only execution-failure retry, after review.
Count-200 finalization uses the existing publication-only plan, without extra
epochs or tensor checkpoint writes. No official test pass is needed for training.

The main-v1 managed workflow and official 240/6720 reporting pipeline explicitly
reject candidate studies. Their contracts have not been generalized. Native
one-task execution, coverage and independent evaluation accept the narrow manifest.
Candidate coverage has thirty expected training runs and 840 independent evaluation
slots (28 per checkpoint); original counts remain 240/6720. No independent test
evaluation belongs to this pilot or the epsilon adoption decision.

## Later CPU verification and plan preparation — unrun

The only interpreter found locally was `/usr/bin/python3`; no compatible existing
venv was found in the checkout, standard user venv/Conda/pyenv/uv directories or
checked Python 3.11 locations. System Python was not used. No packages were
installed and no tests, project imports, compilation, linting, preparation,
training, evaluation, reporting or Slurm commands were executed here.

Run the focused CPU suite later in the **original existing environment**, from
the new candidate checkout. The runner gates Python/dependency versions before
project/Torch imports, hides CUDA, and selects only CPU classes. It now includes
nine candidate checks in addition to the established twenty CPU checks. The
mixed GPU suite is never discovered by this runner. The supplied H100 test
fixture correction is included (`@t.inference_mode()` on nested `whole_pass`);
production evaluation is unchanged. Historical H100/CPU passes reported in the
handoff are prior evidence, not passes for this new source.

```bash
export CANDIDATE_CHECKOUT=/nfs/hpc/share/bellaak/research/alpha-sweep/alpha-sweep-mobile-eps0316
export CANDIDATE_ROOT=/nfs/hpc/share/bellaak/research/alpha-sweep/mobile-eps0316-v1
EXISTING_ENV=/nfs/hpc/share/bellaak/research/alpha-sweep/alpha-sweep-training/.venv
source "$EXISTING_ENV/bin/activate"
cd "$CANDIDATE_CHECKOUT"
export PYTHONPATH="$CANDIDATE_CHECKOUT/src"
"$EXISTING_ENV/bin/python" -B tests/run_diagnostic_cpu_checks.py
```

The candidate checks cover audited legacy document/fingerprint/native recipes,
native candidate epsilon/group ownership and schedule restore, rejected profile
combinations/options, exact thirty-member grid and three-member pilot, foreign
resume/overwrite owners and disjoint roots, best/final/latest receipt authority,
audited-baseline binding, earliest exact ties, and full per-seed reporting.
Tiny native CPU modules and record doubles do not establish MobileNet/CUDA/runtime
checkpoint correctness. `tests/fixtures/main_v1_spec.py` contains the exact
audited original specification as a frozen compatibility oracle.

After those checks, identify `SAVED_SEED2_CONFIG`, `ORIGINAL_BASE`,
`ORIGINAL_WORKFLOW_ROOT`, and `ORIGINAL_EVALUATION_ROOT` from the actual saved
run/config and storage evidence. These values are not present in the supplied
package; do not substitute inferred paths. The exact fingerprint gate verifies
the saved scientific config. Add more `--protected-root` flags for other stores.

```bash
: "${SAVED_SEED2_CONFIG:?Actual audited original resolved_config.yaml}"
: "${ORIGINAL_BASE:?Original relative-path base directory}"
: "${ORIGINAL_WORKFLOW_ROOT:?Original workflow registry root}"
: "${ORIGINAL_EVALUATION_ROOT:?Original independent evaluation root}"
"$EXISTING_ENV/bin/python" -B -m alpha_sweep_training.mobile_candidate build \
  --saved-config "$SAVED_SEED2_CONFIG" --base-dir "$ORIGINAL_BASE" \
  --source-root "$CANDIDATE_CHECKOUT" --candidate-root "$CANDIDATE_ROOT" \
  --activation-script "$EXISTING_ENV/bin/activate" \
  --protected-root "$ORIGINAL_WORKFLOW_ROOT" --protected-root "$ORIGINAL_EVALUATION_ROOT"
export CANDIDATE_PLAN="$CANDIDATE_ROOT/pilot.json"
"$EXISTING_ENV/bin/python" -B -m alpha_sweep_training.mobile_candidate inspect --plan "$CANDIDATE_PLAN"
"$EXISTING_ENV/bin/python" -B -m alpha_sweep_training.controller inspect --manifest "$CANDIDATE_ROOT/study.json"
```

Build pins the recorded Python 3.11.15, Torch 2.5.1+cu124, TorchVision
0.20.1+cu124, NumPy 1.26.4, PyYAML 6.0.3, CUDA 12.4 and cuDNN 90100. A different
environment rejects this plan instead of silently substituting software. No new
modules are requested. Any necessary site module change needs review before
freezing, with actual environment and external input evidence retained.

## Separate future GPU and submission stages — unrun

Inspect current scheduler resources/account/QOS before requesting an allocation:

```bash
sinfo -p dgxh -N -o '%N %t %G %c %m'
scontrol show node dgxh-1
scontrol show node dgxh-2
sacctmgr show assoc where user="$USER" format=Cluster,Account,Partition,QOS
```

Historical dgxh-1 devices were H100 MIG slices. The successful paired diagnostics
used a full NVIDIA H100 80GB HBM3 on dgxh-2 (capability 9.0, 85,009,367,040 device
bytes). The runner checks these actual full-device properties, rather than
assuming a node or GPU label proves equivalence. Historical account `eecs` is
evidence only; this guide grants no current account/QOS entitlement.

An operator may separately authorize the existing GPU reference suite in a
matching assigned full-H100 allocation, before pilot training:

```bash
cd "$CANDIDATE_CHECKOUT"
ALPHA_SWEEP_DIAGNOSTIC_TESTS=1 ALPHA_SWEEP_CUDA_TESTS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTHONPATH="$CANDIDATE_CHECKOUT/src" "$EXISTING_ENV/bin/python" -B -m unittest \
  discover -s tests -p 'test_computation_contracts.py' -v
```

The batch template is `scripts/mobile-epsilon-pilot.sh`. Reviewed launch path:
Slurm -> Bash sources existing activation -> candidate `cd`/PYTHONPATH -> `exec`
owning Python PID -> `mobile_candidate run` -> native task/trainer/persistence.
It imports no `work_render` or `slurm_client` path. Historical renderer bytes are
still unverified; no HPC patch is overwritten or approved by this scoped path.
Candidate source/software and external launch bytes are independently pinned.
The runner preserves audited torch intraop 8/inter-op 112 with thread environment
variables unset, plus CUBLAS `:4096:8`; native runtime records actual GPU/backend.

For a **later explicitly approved** pilot submission, set `VERIFIED_ACCOUNT`,
`VERIFIED_FULL_H100_GRES` and `VERIFIED_FULL_H100_NODE` from current evidence.
GRES is the complete operator-verified request (including count 1), not a guessed
typed label. Supply any required verified `--qos` separately. One node, task and
GPU per run; eight CPUs, 32 GB and twelve hours are unmeasured planning requests.
Three concurrent tasks fit the planning maximum of four only if no other workflow
work consumes that capacity; coordinate original/new work explicitly.

```bash
: "${VERIFIED_ACCOUNT:?Current authorized account}"
: "${VERIFIED_FULL_H100_GRES:?Current exact one-full-H100 GRES request}"
: "${VERIFIED_FULL_H100_NODE:?Current node exposing the required full GPU}"
sbatch --partition=dgxh --account="$VERIFIED_ACCOUNT" \
  --nodelist="$VERIFIED_FULL_H100_NODE" --gres="$VERIFIED_FULL_H100_GRES" \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=12:00:00 \
  --array=0-2%3 --signal=B:USR1@300 --no-requeue \
  --output="$CANDIDATE_ROOT/allocation-logs/%A_%a.out" \
  --error="$CANDIDATE_ROOT/allocation-logs/%A_%a.err" \
  --export=ALL,CANDIDATE_CHECKOUT,CANDIDATE_PLAN \
  "$CANDIDATE_CHECKOUT/scripts/mobile-epsilon-pilot.sh"
```

No submission command has been invoked. Record the exact intent and returned job
ID in a new operator ledger. If acknowledgement is lost, inspect accounting
before any resubmission. Do not resume an old managed wave under these pins.
After an allocation ends, inspect journal coverage, not scheduler exit alone:

```bash
"$EXISTING_ENV/bin/python" -B -m alpha_sweep_training.controller status \
  --manifest "$CANDIDATE_ROOT/study.json" --base-dir "$ORIGINAL_BASE"
```

Only explicitly verified clean interrupted members may receive a new bounded
allocation automatically compatible with `auto`; select their exact array
indices, not the whole study. Preserve numerical terminal failures. Execution
or unclassified failures need an audited operator decision before explicit
`PILOT_OPERATION=continue`. No replacement is implemented by this direct path.
Keep source/environment frozen across all allocations. Native continuation restores
optimizer before scheduler, then data/runtime/noise state without reseeding.
At a committed count-200 boundary, publication-only recovery can also use native
`controller run --operation recover --manifest ... --run-id <exact candidate ID>
--base-dir "$ORIGINAL_BASE" --source-root "$CANDIDATE_CHECKOUT"` on CPU; it does
not need this pilot runner's full-GPU preflight.

## Per-seed validation report and later decision — unrun

After quiescence, choose a new derived report directory disjoint from the study,
original stores, source and prepared data. The report joins exactly all three
pilot identities to historical native journal authority. It keeps missing/locked/
partial/corrupt conditions, committed trajectories, every layer SD at every epoch,
last committed/final/best training and validation metrics, earliest selected epoch,
latest/best/final receipt, raw provenance and explicit failure evidence. Health
assessment remains unreviewed, without new thresholds or an accuracy target.
Full-budget completion and healthy learning remain separate observations.

```bash
: "${NEW_VALIDATION_REPORT:?New separate derived output directory}"
"$EXISTING_ENV/bin/python" -B -m alpha_sweep_training.mobile_candidate report \
  --plan "$CANDIDATE_PLAN" --output-dir "$NEW_VALIDATION_REPORT"
```

`validation.json` is finite, sealed and derived; it never repairs storage or
publishes research completion. Review every seed's complete 200-epoch losses,
training accuracy, final/best validation and SD trajectories and all failures.
Do not discard weak seeds or choose a winner from repeats. There are no new
200-epoch results here, so adoption is **pending**. Do not select epsilon or other
settings using official test/robustness results.

If accepted with frozen settings, retain these exact three runs and plan only
indices 3–29 (remaining 27) from this same sealed study. This task neither selects
nor schedules them. A separate approved launch plan is required. Independent
robustness can later use native `controller evaluate` with the same manifest/run
IDs and a new candidate evaluation root; its existing slot leases, fixed-network
realizations and resumability remain unchanged. Training selection and independent
robustness completion continue to be separate lifecycles.

## Changed-file inventory and validation status

There are 22 changed files: sixteen modified files and six new files.

Modified source: `spec.py`, `config.py`, `optimizers.py`, `identity.py`,
`artifacts.py`, `manifests.py`, `coverage.py`, `evaluation_execution.py`,
`work_plans.py`, `results.py` under `src/alpha_sweep_training/`.
New source: `src/alpha_sweep_training/mobile_candidate.py`.
New configuration/script: `configs/candidates/mobile-eps0316.yaml`,
`scripts/mobile-epsilon-pilot.sh`.
Checks: modified `tests/run_diagnostic_cpu_checks.py` and
`tests/test_computation_contracts.py`; new `tests/test_mobile_candidate.py` and
frozen `tests/fixtures/main_v1_spec.py`.
Documentation: this guide, plus links in `docs/scientific-specification.md`,
`docs/configuration.md`, `docs/models-and-optimization.md`, and
`docs/manifests-and-controller.md`.

Validation performed: Git working-tree/branch/HEAD review, supplied evidence reads,
source/diff inspection, native launch/continuation/selection/identity reader review,
and CLI argument comparison against the implemented parsers. **Runtime validation
results: none; focused CPU checks unrun because no compatible local environment
was found.** No claim of runtime correctness, complete convergence, adopted recipe,
full scientific fix or completed pilot is made. No dataset/model artifact, research
manifest/plan, report, Slurm job, commit or push was generated/executed here.
