# CIFAR-100 revised MobileNet recipe and HPC handoff

The existing main-branch preparation, models, manifest/controller, continuation,
independent evaluation, reporting and Slurm paths are reused. No training algorithm
or workflow is replaced. This revision adopts RMSprop epsilon **0.0316** for new
CIFAR-100 MobileNetV3-Large runs, following the reported CIFAR-10 result. It neither
establishes a forward/backward defect nor guarantees CIFAR-100 stability.

## Recipe, identity and source review

`main-v1` remains readable with its original epsilon 0.001 and schema/fingerprint
format 4. `main-v1-cifar100-mobile-eps0316` is accepted only for CIFAR-100
MobileNetV3-Large. It changes epsilon alone: LR 0.01, momentum 0.9, RMSprop smoothing
0.9 and weight decay 0.00001 remain fixed. The native optimizer builder selects
this versioned central recipe and passes its epsilon to native RMSprop.

`alpha-sweep build --cifar100-mobile-eps0316` revises exactly 30 tasks (global
indices 150 through 179 in the prescribed ordering). Those run names append
`__recipe-mobile-eps0316-v1`; their scientific fingerprints also change. Other
210 scientific configurations, fingerprints and run names stay identical to an
otherwise identical historical build. ResNet50, EfficientNet-B0 and ConvNeXt-Tiny
recipes are unchanged. Without the flag, the builder preserves the historical grid.

Reviewed existing source provides:

- `cifar_source._dataset_class` selects TorchVision CIFAR100; data/model declarations
  use 100 classes. All four native constructors receive `num_classes=100` and the
  existing classifier/stem guards inspect that structure.
- Preparation persists one 45,000/5,000 official-training split per dataset, with
  an explicit independent preparation seed. `derive_training` computes normalization
  from the 45,000 training indices only. Loaders remain read-only; no seed or
  normalization default is introduced.
- Manifest validation requires the full 240-task Cartesian product, three explicit
  seeds, shared per-dataset preparation and linked 120-task phase views. The revised
  parent grid/document identities change, including phase-view parent linkage;
  even the scientifically unchanged phase-1 view must come from the revised parent.
- Native 32x32 inputs, augmentation, precision, 200 epochs, batch 128, scheduler,
  perturbation/clean-master/BatchNorm/RNG semantics, clean-validation selection and
  earliest exact ties are unchanged. Each best checkpoint retains 28 prescribed
  independent evaluation slots: CIFAR-100 has **120 training runs / 3,360 passes**.

Use new manifest bundle paths and a separate workflow registry/policy copy for the
revised parent; never replace a parent in an owned registry. This uses the existing
workflow implementation. Keep research output/checkpoint/result/log roots pointed
at the original shared stores when those contain the phase-1 authorities: original
tasks reuse their exact identities, and revised MobileNet tasks get distinct child
paths from their suffix. A wholly new empty output root will not satisfy the gate.
Keep `paths.split_file: null` in workflow policy, inheriting each dataset's own split.
Preserve original manifests, registries, raw research artifacts and frozen checkouts.
Unfinished historical training still requires its exact original source/software;
this revision does not waive continuation or launch-pin checks.

## Completion gate to inspect on HPC

Both managed planning and manual one-task training still require **all 120 original
CIFAR-10 tasks** to be successfully training-complete. These are the exact phase-1
IDs/fingerprints in the revised full manifest, unchanged from an otherwise identical
historical build, across all four architectures, ten alphas and seeds 0, 1, 2.
In particular all 30 original CIFAR-10 MobileNet tasks still use epsilon 0.001;
revised CIFAR-10 MobileNet IDs from another experiment cannot substitute for them.

Inspect each original metadata directory's `resolved_config.yaml`, `run_metadata.json`
and `state/HEAD.json` through existing read-only status. The journal reader verifies
referenced immutable full-budget checkpoints, best/final selection, metrics/layer
statistics, completion transaction and training summary using stored provenance.
Require phase 1 `training_successfully_completed=true` and 120 completions. A missing,
failed, locked, partial, incompatible, corrupt or unfinalized authority leaves the
gate closed. Derived summary aliases, scheduler exits, or the reported success of
a different recipe do not prove this condition. Independent robustness completion
is separate. No gate override is added; any unmet original completion must be
resolved under its existing scientific/source/failure rules before the pilot.

## Available verification

Local focused checks use the existing Python 3.11.9 interpreter, without installs.
Nine new declaration/mocked checks and 34 existing regressions passed. One native
check was skipped for missing dependencies and one existing Windows symlink check
was skipped for missing privileges. All seven changed Python files parsed, and
`git diff --check` passed.
`test_cifar100_recipe` covers frozen pre-change fingerprint fixtures, strict
version/recipe pairing, exactly 30 changed tasks, distinct child storage paths,
coverage/indices/phase linkage, baseline mapping, 3,360 phase-2 slots, CLI parsing
and the original completion gate. Preparation is explicitly mocked: these are
configuration/manifest checks, not checks of actual dataset files or model execution.
The golden hashes were derived from the pre-change main source, not revised defaults.

Native model/optimizer/forward checks are pending locally: Torch 2.5.1,
TorchVision 0.20.1, NumPy 1.26.4 and PyYAML >=6 are absent in that interpreter.
No version guard is bypassed. The existing diagnostic CPU runner stops at that
dependency gate. Existing queue-policy and planning/inspection regressions pass
apart from the existing symlink-privilege skip on Windows. Syntax/source checks
do not establish research runtime correctness. The optional reference commit
`8010d07d9d1f0886ab1e180523f316623ef622a4` was unavailable in the local object database;
no fetch, merge or operational code import was performed.

On a separately authorized CPU allocation with an **existing pinned environment**,
from the intended revised checkout:

```bash
export PYTHONPATH="$PWD/src:$PWD/tests${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES='' python -m unittest -v test_cifar100_recipe
python -m unittest -v test_queued_waves test_planning_inspection
CUDA_VISIBLE_DEVICES='' python tests/run_diagnostic_cpu_checks.py
```

Require the native test to execute rather than skip. It constructs the real four
CIFAR-100 architectures (both MobileNet recipe versions), checks native optimizer
types/groups/options, performs CPU inference on 32x32 inputs and checks 100 logits.
It uses no CIFAR data, research training or optimizer updates. The diagnostic runner
uses tiny/mocked boundaries documented in its existing tests. Real CIFAR-100
full-budget stability, CUDA/precision, persistence and Slurm remain HPC runtime work.

## Later preparation, build and twelve-run baseline

All commands below are a handoff, not actions performed in this task. Supply approved
paths/runtime settings/account/evidence per [operating-guide.md](operating-guide.md).
Reuse compatible existing preparation. If CIFAR-100 preparation is absent, with
the approved independent split seed and train-partition normalization policy:

```bash
alpha-sweep-prepare --config "$PREPARATION_BASE_CONFIG" --dataset cifar100 \
  --data-root "$SHARED_CIFAR_ROOT" --split-seed "$APPROVED_CIFAR100_SPLIT_SEED" \
  --normalization derive_training
```

Official files must already exist; no download flag is included. Finish separate
research inputs based on each prepared `dataset_config.yaml`, preserving its
checksummed data identities. Resolve existing scheduler, precision and runtime
choices explicitly; do not overlay pending dataset declarations over preparation.
Keep the phase-1 input scientifically identical to the original completion authorities.

```bash
alpha-sweep build --cifar10-config "$FINALIZED_CIFAR10_CONFIG" \
  --cifar100-config "$FINALIZED_CIFAR100_CONFIG" --models-dir configs/models \
  --seeds 0 1 2 --base-dir "$SHARED_BASE" --cifar100-mobile-eps0316 \
  --output "$REVISED_FULL_MANIFEST"
alpha-sweep inspect --manifest "$REVISED_FULL_MANIFEST" \
  --phase-view "${REVISED_FULL_MANIFEST%.json}.phase2.json"
alpha-sweep status --manifest "$REVISED_FULL_MANIFEST" --base-dir "$SHARED_BASE"
```

Derive the baseline mapping from the validated manifest, including exact run IDs
and fingerprints; do not select the first twelve phase indices:

```bash
python - "$REVISED_FULL_MANIFEST" <<'PY'
import sys
from alpha_sweep_training.manifests import load_manifest
m = load_manifest(sys.argv[1])
rows = [t for t in m.tasks if t.phase == 2 and t.config.training_alpha == 0]
assert m.seeds == (0, 1, 2) and len(rows) == 12
for t in rows:
    print(t.global_index, t.phase_index, t.config.scientific['architecture'],
          t.config.scientific['training_seed'], t.run_id, t.fingerprint)
PY
```

For this ordered manifest the mapping is:

| Architecture | Seeds in order | Global indices | Phase-2 indices |
| --- | --- | --- | --- |
| ResNet50 | 0, 1, 2 | 120, 121, 122 | 0, 1, 2 |
| MobileNetV3-Large (epsilon 0.0316) | 0, 1, 2 | 150, 151, 152 | 30, 31, 32 |
| EfficientNet-B0 | 0, 1, 2 | 180, 181, 182 | 60, 61, 62 |
| ConvNeXt-Tiny | 0, 1, 2 | 210, 211, 212 | 90, 91, 92 |

These are twelve ordinary **200-epoch** phase tasks, counting toward the 120, not
a reduced-budget experiment. Once the original phase-1 gate and existing workflow
policy/source/environment/restriction requirements validate:

```bash
alpha-sweep-workflow plan --manifest "$REVISED_FULL_MANIFEST" \
  --policy "$REVISED_POLICY" > "$NEW_WORK_PLAN"
```

Select the twelve eligible **training** `unit_key` values from that plan by matching
the manifest-derived global indices above (exclude evaluation work). Use the existing
renderer's explicit selection, preserving sealed plans and manifests:

```bash
alpha-sweep-workflow render --manifest "$REVISED_FULL_MANIFEST" \
  --policy "$REVISED_POLICY" --plan "$NEW_WORK_PLAN" --work-id "${BASELINE_WORK_IDS[@]}"
```

The Bash array `BASELINE_WORK_IDS` must contain those exact eligible plan keys.
Review the returned mapping and existing shared concurrency limits before a later
explicit submission. No submission is part of this task. Automatic advance may
select additional grid work; continue the pilot with explicit selections until
all twelve full-budget authorities can be reviewed. Manual allocated-node selection
also remains available through `alpha-sweep run --phase 2 --phase-index INDEX` with
the same manifest, storage, source and completion gate; see the operating guide.
