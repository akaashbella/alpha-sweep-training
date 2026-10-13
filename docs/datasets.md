# CIFAR preparation and reproducible loaders

Step 2 implements production source for official CIFAR-10 and CIFAR-100 only.
It has not been executed or runtime-verified. Do not execute the commands below
during implementation. Package installation, downloads, artifact generation,
tests, import checks, compilation, training, and preliminary experiments remain
deferred to authorized HPC work after all nine implementation steps.

## Interfaces and dependencies

- `prepare.prepare_dataset(config, base_dir=..., normalization=NormalizationRequest(...),
  split_seed=..., download=False)` returns `PreparationResult(config, config_path,
  manifest_path, reused)`. It does not require research run identity, three
  training seeds, precision, batch handling, or scheduler decisions.
- `alpha-sweep-prepare` and `python -m alpha_sweep_training.prepare` expose the
  same explicit preparation CLI for one selected dataset.
- `prepared.load_prepared_dataset(config, base_dir=...)` validates completed
  artifacts and current official sources without writing or downloading.
- Its Step 6 `metadata_only=True` mode validates the same prepared bundle and
  official raw-file checksums/descriptors without constructing image-array sources.
  Manifest construction uses it; full loaders retain decoded identity validation.
- `data.build_loaders(config, base_dir=..., pin_memory=False,
  persistent_workers=False)` returns `DataLoaders` with `train`, `validation`,
  `test`, typed `metadata`, `set_epoch`, `continuation_metadata`, and
  `restore_next_epoch`. Full research readiness remains a later launcher gate.
- `data.build_test_loader(config, base_dir=...)` returns `TestData(loader, metadata)`
  for independent robustness evaluation. It validates the same prepared bundle
  read-only and builds only the deterministic full official 10,000-image test
  loader, with actual prepared identities. It creates no training view/sampler/
  iterator and performs no download, split or normalization derivation. Its private
  worker generator consumes no training model/noise RNG streams. See
  [weight-robustness-evaluation.md](weight-robustness-evaluation.md).

Dependencies are declared, not installed: PyYAML, torch 2.5.1, torchvision 0.20.1,
and NumPy 1.26.4, using Python 3.10–3.12. The torch/torchvision pair is documented
by [PyTorch's official version instructions](https://pytorch.org/get-started/previous-versions/#v251).
Later HPC setup must select a CUDA-enabled build appropriate for the allocated
GPU and driver; availability and GPU compatibility have not been verified.
NumPy's version is pinned for explicit split generation and normalization
accumulation. Preparation records actual Python/torch/torchvision/NumPy versions
and hashes of the source implementing preparation. Later run provenance and
data continuation checks also retain the actual torch RNG implementation version.

## Preparation inputs and later HPC commands

Supply a main-experiment base config, selected dataset, persistent shared data
root, and **one explicitly approved normalization policy**. Supply an approved
independent preparation seed when creating a new split. Neither policy nor seed
has a default. A valid existing split can be reused without supplying a seed;
if supplied, that seed must match the persisted generation record.

The following shell variables are placeholders for decisions and paths supplied
at execution time. They do not finalize a seed, normalization policy, or constants.
Use the derivation command only if training-partition derivation is approved:

```bash
alpha-sweep-prepare \
  --config configs/research.yaml \
  --dataset cifar10 \
  --data-root "$SHARED_CIFAR_ROOT" \
  --split-seed "$APPROVED_PREPARATION_SEED" \
  --normalization derive_training
```

For approved explicit constants instead:

```bash
alpha-sweep-prepare \
  --config configs/research.yaml \
  --dataset cifar10 \
  --data-root "$SHARED_CIFAR_ROOT" \
  --split-seed "$APPROVED_PREPARATION_SEED" \
  --normalization explicit \
  --mean "$APPROVED_MEAN_R" "$APPROVED_MEAN_G" "$APPROVED_MEAN_B" \
  --std "$APPROVED_STD_R" "$APPROVED_STD_G" "$APPROVED_STD_B" \
  --normalization-source "$DOCUMENTED_CONSTANTS_SOURCE"
```

Without `--download`, all official files must already exist and pass integrity
checks. Add `--download` only to explicitly acquire/extract the official source.
It never repairs an existing partial/corrupt extracted directory or replaces an
invalid archive. A checksum-valid archive may be extracted if its destination
does not exist. Existing valid files are reused even when acquisition is allowed.
For CIFAR-100, select `--dataset cifar100` and its own approved normalization
inputs; the CLI also explicitly selects its 100-class classifier declaration.
Do not reuse a finalized CIFAR-10 split/normalization config as a CIFAR-100 base.

The module entry point has the same arguments once dependencies exist and the
package is installed or `<repository>/src` is on PYTHONPATH. Environment setup is
later HPC operational work, not a check performed in Step 2.

## Shared storage, locking, and finalization

Raw sources and preparation artifacts live under the supplied data root:

```text
<data_root>/
  cifar-10-batches-py/                 # official CIFAR-10 source
  cifar-100-python/                   # official CIFAR-100 source
  .alpha_sweep_prepared/
    .cifar10.lock                     # only while preparation owns it
    cifar10/
      split.json
      normalization.json
      dataset_config.yaml
      manifest.json
    cifar100/                         # same four artifact names
```

Preparation holds a dataset-level exclusive cooperative lock through official
acquisition/extraction, source validation, generation, and commit. Its token,
hostname, PID, and timestamp identify the owner. The preparation lock is separate
from run locks; `open_run` is never used. A lock is never expired/stolen or
automatically removed as stale. Readers reject an active/stale preparation lock.

Compute the documents first, then atomically serialize each file in a unique
same-parent staging directory. Write the final manifest last and publish the
complete directory by rename. Manifest checksums cover split.json,
normalization.json, and the original complete dataset_config.yaml. POSIX
directory fsync follows commit. A missing manifest, leftover stage, corrupted
file, or incompatible preparation stops automatic reuse and requires manual
recovery; there is no overwrite/repair mode. No cleanup silently discards an
existing split. Complete compatible preparation is reused without rewriting
any shared file, re-deriving statistics, or generating another split.

The API returns the caller's effective config on reuse but preserves the original
shared YAML. Use that shared YAML as a base for model/task overrides later.
Preparation resolves only split identity, normalization constants/source, and
operational data paths. Every unrelated pending decision remains pending.
Schema-2/3 saved preparation configurations remain usable: migrations through
schema 4 add fixed model/initialization/scheduler and training/runtime randomness
contracts in memory without changing stored bytes or manifest checksums. Apply
the appropriate model declaration after this base. Do not overlay unresolved dataset declarations onto its
finalized split/normalization. Changing model selection or operational paths
does not change preparation identities or rewrite shared files.

## Official source and split identities

Both official sources are loaded once with TorchVision CIFAR10/CIFAR100 and
`transform=None`, `target_transform=None`, `download=False` after any explicitly
requested acquisition. Validate all train, test, and class-metadata files against
the checksum declarations in the [pinned official CIFAR implementation](https://github.com/pytorch/vision/blob/v0.20.1/torchvision/datasets/cifar.py).
Check 50,000/10,000 counts, native uint8 HWC 32×32 RGB arrays, label bounds and
image/label association, and 10/100 class vocabularies. Training workers never
construct/download official datasets; they consume prepared source views.

Each official partition's source identity is `sha256:` plus a canonical-JSON
SHA-256 over source format version 1, dataset, partition name, length/class count,
image shape/encoding, ordered class names, decoded images digest, decoded labels
digest, and its data-file/class-metadata names with official MD5 and actual
SHA-256. Hash images in official example order as uint8 HWC C-order bytes;
hash labels as little-endian uint32 bytes. Small contiguous image blocks avoid
a duplicate complete image array. Paths, mtimes, object hashes, and provenance
are excluded. Loaders recompute identities against actual current source content.

Split generation is `numpy-pcg64-permutation-v1`: local
`numpy.random.Generator(numpy.random.PCG64(explicit_seed)).permutation(50000)`,
first 45,000 ordered indices for training, remaining 5,000 for validation.
No stratification or training-seed-dependent generation is introduced. Persist
both actual ordered arrays plus seed, algorithm, format version, and provenance.

The split identity hashes canonical JSON with exactly `format_version` (1),
`dataset`, `official_training_identity`, `train_indices`, and
`validation_indices`. Verify exact lengths, strict integer bounds, no duplicates,
disjointness, full coverage of all 50,000 indices, source/dataset association,
generation metadata, and digest. Hash the arrays in their persisted order.
Provenance/seed are separately recorded; actual content determines scientific
split identity. A conflicting seed request fails even if files otherwise exist.

## Normalization policies and identity

`explicit` requires three finite means, three positive finite stds, and a nonempty
documented source. `derive_training` requires no supplied constants/source and
uses **only the persisted 45,000 training indices**. There is no policy default
and no common CIFAR constants in the source.

For derivation, iterate images in persisted training-index order. Convert original
uint8 RGB pixels to float64 by division by 255, without crop/flip/other augmentation.
For each image, compute channel block mean and centered sum of squares over its
1,024 spatial positions, then merge with Chan's mean/M2 equations in float64.
Use population variance `M2 / (45000 * 32 * 32)` and its square root. This CPU
algorithm uses image-sized temporaries; no validation/test samples, GPU, or
extra complete dataset copy enter calculation. Record algorithm, dtype, order,
scale, correction 0, image/pixel counts, and actual software provenance.

Normalization identity hashes format version 1, dataset, official-training
identity, split identity, policy, actual mean/std, documented source, and
calculation contract. It excludes operational paths/timestamps/provenance.
The artifact stores the human source; configuration source is the content-bound
`normalization-v1:sha256:<digest>`. Loader checks constants and this exact source
against the artifact. A derivation declaration with no calculation still has
null constants; preparation never fabricates them or fills unrelated settings.
Changing a policy/source/constants request against completed preparation fails.

## Transform and randomness contracts

`cifar32-zero-pad4-flip05-float32-v1` defines the exact training transform:
constant-zero pad four pixels on each edge, independently uniform top/left crop
offsets in 0–8 inclusive, 32×32 crop, horizontal flip with probability 0.5,
CPU float32 CHW conversion using uint8/255, then per-channel normalization.
Evaluation performs only deterministic conversion and the same normalization.
No resizing, random evaluation augmentation, or other augmentations exist.
Separate wrappers share untransformed source arrays; no transform attribute is
mutated to switch between training and evaluation.

`sha256-keyed-torch-cpu-v1` derives an unsigned 64-bit local generator seed from
the first eight SHA-256 bytes interpreted little-endian. The hashed UTF-8
canonical JSON has `version`, `stream`, `dataset`, `split_identity`,
`training_seed`, zero-based `epoch`, and `example_index` (null except augmentation).
Canonical JSON uses sorted keys, compact separators, ensure_ascii=False,
allow_nan=False. Neither architecture nor alpha enters the key.

- `sample_order`: local CPU torch.randperm over persisted training positions.
- `augmentation_crop` / `augmentation_flip`: separate per-official-example local
  CPU generators; randint selects crop offsets, rand selects the flip.
- `training_workers`, `validation_workers`, `test_workers`: separate DataLoader
  base-seed generators. Their consumption never advances sample-order or
  augmentation generators or the main process's model/dropout/noise RNGs.

Order/crop draws explicitly use int64 and flip draws explicitly use float32,
so changing global default tensor dtype/device cannot change sample decisions.

Worker initialization seeds incidental Python/NumPy RNGs only in that worker;
samples never use these mutable globals. No global RNG is repeatedly reseeded
in __getitem__. Changing worker count/scheduling or restarting at the same epoch
preserves order and each example's augmentation for the same source/split/seed
and recorded RNG implementation. Paired alpha/architecture runs receive the same
data treatment. Sampler-provided `TrainSampleKey(epoch, official_index)` reliably
transmits epochs into worker copies, including persistent workers.

## Loader counts and completed-epoch continuation

Resolve the selected training seed and training.drop_last before building loaders.
Batch size remains 128. Worker count comes from execution.num_workers. Pinning
and persistent workers are explicit API bools, default false for all loaders;
persistent workers require a positive worker count. Prefetch is two batches per
worker, or None with no workers. All dataset samples stay on CPU. Evaluation is
sequential, retains the ordered validation indices/full official test order,
always has drop_last=False, and includes all 5,000/10,000 examples.

`LoaderMetadata.counts` distinguishes the full 45,000 training partition from
processed examples: drop_last=True means 351 batches and 44,928 processed
examples; false means 352 batches and all 45,000. Validation/test have 40/79
batches and full coverage. Metadata also exposes class count, content identities,
RNG version, actual torch version, and operational loader settings.

Before iterating train, call `loaders.set_epoch(epoch)` with zero-based 0–199.
Call only at an epoch boundary after finishing/discarding the previous iterator.
Save `loaders.continuation_metadata(next_epoch_index)` alongside Step 5's clean
model/optimizer/scheduler/scaler and model/noise RNG state. That record contains
its format/boundary, next epoch, original 200-epoch budget, dataset/source/split/
normalization/preprocessing identities, selected seed, randomness version,
batch/drop policy, and torch RNG implementation version. It excludes paths and
worker counts. `restore_next_epoch(record)` verifies compatibility and configures
the next epoch; index 200 means training is finished, pending evaluation remains.
There is no data RNG cursor at a completed-epoch boundary and no speculative
mid-batch resume. An unfinished epoch is replayed from the last committed boundary.

After moving shared data, preserve raw and prepared content and update both
operational data_root and split_file overrides. Do not rewrite checksummed shared
YAML just to change paths; the original preparation config remains a provenance
snapshot. Source/split/normalization/preprocessing identities are location-free.

Still pending: approved split preparation seed; normalization policy/constants
for each dataset; exact three training seeds; train drop_last; precision/backend
policy; scheduler endpoints/cadence. No dataset artifacts were generated in
Step 2. Step 3's model builders, optimizers and scheduler now exist in source,
without execution; see [models-and-optimization.md](models-and-optimization.md).
Step 4's [shared trainer](training-and-perturbation.md) now consumes these loaders
and their completed-epoch continuation interface, without changing data streams or
generating artifacts. Step 5's [continuation service](checkpoints-and-recovery.md)
validates the existing data metadata and restores its next-epoch contract without
changing prepared files or generating data. Step 7 now supplies source-only
[Slurm/signal integration](slurm-workflow.md). Data workers ignore advance warnings
without changing sample streams, prepared artifacts or the read-only data contract.
