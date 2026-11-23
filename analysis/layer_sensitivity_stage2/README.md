# Stage 2 layer sensitivity

Stage 1 relates whole-network cross-entropy change to evaluation perturbation strength using frozen summaries. Stage 2 measures the contribution of selected weight tensors and disjoint architectural stages directly from verified best-validation checkpoints. It also measures accuracy changes and prediction flips on the same examples. A layer's response depends on the checkpoint, data, perturbation scale and target size. Layer-wise sensitivity cannot establish that a layer is causally the most important for all model behavior.

## Scientific design

For a fixed clean checkpoint with eligible tensor (W_l), the offset on a selected tensor is `alpha_eval * population_std(W_l, correction=0) * Z_l`, where entries of (Z_l) are standard normal. The offsets on the other tensors are zero. Each realization holds the same temporary network fixed throughout a full pass. The repository's Gaussian offset helper produces detached float32 noise. The pass uses the saved effective precision policy, inference mode, unchanged learned normalization buffers and the original float32 unsmoothed cross-entropy. Functional substitution preserves original parameter bindings, and the original clean-state guard verifies parameters, buffers and bindings and restores each prior module mode on exit.

For group G, sensitivity is the expected perturbed mean loss minus the same checkpoint's clean mean loss, in nats per example. If a second-order expansion is valid near the checkpoint, centering cancels the first-order term in expectation. The local leading term is `(alpha_eval^2 / 2) * sum(sigma_l^2 * trace(H_ll) for l in G)`. This motivates comparing responses at small alpha and examining departures as alpha grows. Stage 3 can independently estimate the weighted Hessian-block traces and check where this approximation holds.

Tensor seeds are the first eight bytes of a SHA-256 digest interpreted as a little-endian unsigned integer. The canonical JSON key contains the Stage 2 randomness version, best generation and SHA-256, canonical evaluation alpha, realization ID and exact qualified tensor name. A private device generator draws independent tensor entries. A tensor receives the same draw in its single-tensor, stage, diagnostic and whole-network passes at a given alpha and realization. Ordering and target choice do not enter the key. Planning checks for collisions across all tensor/alpha/realization keys within each checkpoint. Alpha zero uses the ordinary clean forward without requesting a perturbation generator. Device/software differences can change generated values and inference, so reuse requires matching recorded runtime policy and hardware properties.

## Targets and architecture mapping

Every owned `Conv2d.weight` or `Linear.weight` returned by `discover_eligible_weights` can be selected. Biases, normalization parameters, layer scales and buffers remain clean. The model comes from the original CIFAR builder. The package verifies its eligible inventory against the checkpoint and checks the native blocks and stage structure before accepting the partition.

| Architecture | Disjoint stage groups |
| --- | --- |
| ResNet50 | Stem, four bottleneck stages including residual-path projections, classifier |
| MobileNetV3-Large | Stem, five contiguous inverted-residual groups split at depthwise stride-two boundaries, feature head, classifier |
| EfficientNet-B0 | Stem, seven native MBConv sequences, feature head, classifier |
| ConvNeXt-Tiny | Stem, four CNBlock sequences, three interstage downsamplers, classifier |

These mappings follow the pinned source trees. The audit checks the actual in-memory structure using the original builders. [ResNet](https://github.com/pytorch/vision/blob/v0.20.1/torchvision/models/resnet.py) · [MobileNetV3](https://github.com/pytorch/vision/blob/v0.20.1/torchvision/models/mobilenetv3.py) · [EfficientNet](https://github.com/pytorch/vision/blob/v0.20.1/torchvision/models/efficientnet.py) · [ConvNeXt](https://github.com/pytorch/vision/blob/v0.20.1/torchvision/models/convnext.py)

`--scopes stages` is the pilot default. `--scopes tensors` addresses every eligible tensor by name, and `--scopes diagnostics` enables available depthwise, pointwise, expansion, projection and squeeze-and-excitation categories. Scopes can be combined. Diagnostic categories can overlap one another and the stage partition. Additivity uses only the nonoverlapping complete stage partition. Every plan includes a full-network target. The audit stores names, shapes and counts. The execution inventory also stores population standard deviations from the actual clean weights.

## Checkpoint authority and corrected MobileNet

The audit takes explicit original full manifests and storage bases. A source can nominate exact run IDs and provide the original typed execution-only storage overrides. It reads the original full-budget journal and best binding through the read-only evaluation input reader. That reader validates the original run ownership, scientific fingerprint, role, generation, byte digest, clean checkpoint format and recorded data/model contracts. It checks the latest native optimizer state and the saved training history. Stage 2 also checks the best checkpoint's actual scientific declaration and saved optimizer metadata. The audit records the original run provenance, best checkpoint provenance and selection, declared and effective optimizer settings, data identities, and a digest of the current analysis and experiment source.

Both MobileNet datasets require `rmsprop` with declared `epsilon=0.0316` and best optimizer metadata with effective `eps=0.0316`. The known CIFAR-100 revision has its own specification, run ID suffix and scientific fingerprint. The current repository has no corresponding validated CIFAR-10 specification. A legacy CIFAR-10 run with epsilon 0.001 is excluded. A corrected CIFAR-10 manifest or checkpoint rejected by the original reader remains excluded, with the source error retained. Supported inputs use the original clean best checkpoint format 1, supported component versions, and the original readers' historical journal compatibility. A bare model state, unsupported format, filename match, reported accuracy, Stage 1 summary, or manual edit to a saved artifact does not establish checkpoint authority.

The corrected artifact locations were not supplied during implementation. The full pilot requires authoritative corrected MobileNet checkpoints for each selected training alpha and seed on both datasets, including a compatible validated reader and manifest contract for the corrected CIFAR-10 runs. Upstream changes or recovery for that provenance require separate review. Stage 2 rejects missing or ambiguous candidates. An eligible subset may run by explicit selection, and its excluded configurations remain visible in the plan and reports.

At execution the worker rereads the original binding and requires equality with the audit, including the original manifest identity and committed training transaction. Data loading validates finalized preparation and checks the decoded source against its saved identity. The exploratory validation run decodes the official training partition and uses its ordered 5,000-example validation subset. The shared preparation integrity verifier may check official test file digests. Validation execution does not decode test images or use test predictions. An explicitly planned test run decodes the official 10,000-example test set.

## Pilot and extension

The pilot requests 24 checkpoints from both datasets, all four architectures, training alphas 0, 0.10 and 0.30, and the first seed in the predeclared training seed order. There is one clean pass per checkpoint and eight realizations per stage or full-network target at each positive evaluation alpha 0.01, 0.02, 0.05, 0.10 and 0.20. Individual tensors and diagnostics are available by explicitly adding their scopes. A fully eligible stage-only pilot would have 222 targets, 8,904 passes and 44,520,000 evaluated examples. Exact executable counts depend on audit eligibility. Plan creation requires an explicit cap and execution requires one exact checkpoint ID and a separate pass cap.

The validation partition selected each best checkpoint, so the pilot is exploratory. Freeze the analysis choices after that review. Use a fresh audit with `--seeds all` and a distinct plan with `--split test` for confirmatory work. An explicit `--all-training-alphas --seeds all` audit requests all 240 configurations. Choose a pass budget before adding individual-tensor scopes.

## Local and HPC commands

Use Python 3.10–3.12 with the existing PyTorch 2.5.1 and TorchVision 0.20.1 environment. Figures use the existing reporting extra. An authorized CPU allocation is sufficient for auditing, planning, inspecting, summarizing and plotting. Real passes use the original CUDA runtime contract and one assigned GPU. Supply the effective execution device and data locations as overrides if the manifest locations differ from the allocation. On a deterministic CUDA policy, set the required `CUBLAS_WORKSPACE_CONFIG` in the launch environment before Python starts.

Prepare a separate input JSON file. Replace the illustrative paths and use the actual source declarations. The first readable manifest supplies the predetermined seed order. Optionally put its complete three-seed list in `training_seed_plan` to record exclusions even when all manifests are unreadable.

```json
{
  "protected_roots": ["/shared/experiment", "/shared/workflow", "/shared/original-reports"],
  "sources": [
    {
      "manifest": "/shared/manifests/full.json",
      "base_dir": "/shared/experiment"
    }
  ]
}
```

For multiple sources, use `"run_ids": ["<exact ID from this manifest>"]` where needed to limit declarations to the intended runs. Include actual external evaluation, workflow and derived-report roots in `protected_roots`. The audit also derives the original training, checkpoint, results, data and logs roots from the supported manifest and overrides. Output directories must be separate from the checkout, these roots and their respective inputs.

From the repository root, in a previously authorized CPU environment, set the path variables and run:

```bash
python -B -m analysis.layer_sensitivity_stage2 audit \
  --inputs "$STAGE2_INPUTS" --output "$NEW_AUDIT_DIR"

python -B -m analysis.layer_sensitivity_stage2 plan \
  --audit-dir "$NEW_AUDIT_DIR" --output "$NEW_PLAN_DIR" --max-plan-passes 9000

python -B -m analysis.layer_sensitivity_stage2 inspect --plan-dir "$NEW_PLAN_DIR"
```

These commands also accept the same arguments in PowerShell. From an authorized OSU GPU allocation, use an exact eligible checkpoint ID listed by `inspect`. The following example advances at most eight missing passes for one selected target:

```bash
python -B -m analysis.layer_sensitivity_stage2 run \
  --plan-dir "$NEW_PLAN_DIR" --results "$STAGE2_RUNS" \
  --checkpoint "$CHECKPOINT_ID" --target 'network:all' \
  --evaluation-alpha 0.05 --max-passes 8
```

Repeat `--target`, `--evaluation-alpha` or `--realization` to select particular planned passes. Qualifiers omitted from the command allow every planned value in that dimension, subject to the pass cap. A missing clean baseline runs first and counts toward the cap. Reissue the selection to resume. Jobs can work on distinct checkpoint IDs concurrently. For a single checkpoint the process holds one filesystem lease, so its target selections run serially. Use shared storage supporting advisory locks and same-directory hard links. A killed worker leaves its attempted paths and the OS releases its lease. A recorded pass failure requires `--retry-failed` to attempt it again. An interrupted pass is restarted with its original seeds. Slot results, staging evidence and previous attempts are preserved. A successful slot is validated and skipped on rerun.

```bash
python -B -m analysis.layer_sensitivity_stage2 inspect \
  --plan-dir "$NEW_PLAN_DIR" --results "$STAGE2_RUNS" --include-slots

python -B -m analysis.layer_sensitivity_stage2 summarize \
  --plan-dir "$NEW_PLAN_DIR" --results "$STAGE2_RUNS" --output "$NEW_SUMMARY_DIR"

python -B -m analysis.layer_sensitivity_stage2 plot \
  --summary-dir "$NEW_SUMMARY_DIR" --output "$NEW_FIGURES_DIR" --evaluation-alpha 0.05

python -B -m analysis.layer_sensitivity_stage2 inspect \
  --publication-dir "$NEW_FIGURES_DIR" --kind figures
```

## Measurements and outputs

Sample-weighted mean CE and accuracy use the full split. Accuracy change is `100 * (perturbed_accuracy - clean_accuracy)`, in percentage points. Prediction flips are the number of changed per-example predicted classes divided by the number of examples. A flip can leave correctness unchanged. Correct-to-incorrect and incorrect-to-correct counts reconstruct the net accuracy change. The raw result saves compact hex-encoded class IDs in evaluation order and stores the labels once in the clean record. Validation recomputes flip and transition counts against that clean record.

Expected squared perturbation energy is `alpha_eval^2 * sum(numel_l * sigma_l^2)` across the selected tensors. Report signed raw loss change, loss change per million selected weights, loss change divided by expected energy when positive, and fraction of eligible elements selected. The paired group/full ratio is unavailable when `abs(full_delta_loss) <= max(1e-8, 1e-6 * abs(clean_loss))`. This is a numerical guard, and ratios just above it can still be scientifically unstable. These quantities describe differently scaled responses and have no universal cross-architecture ranking interpretation.

Each isolated stage uses the same tensor draws as the full network for its realization. For complete pairs, compute the sum A of isolated stage loss changes, full-network change B, and residual `B - A`. The local second-order expectation is additive across disjoint groups with zero cross-covariance. Finite realization noise and higher-order interactions can both affect the residual. A residual cannot identify specific interacting pairs.

Each checkpoint summary requires all prescribed realizations for its mean. The training-seed summary uses the checkpoint means and requires all selected compatible seeds. Sample SD uses denominator n−1, with null SD for fewer than two observations. The pilot has one seed per configuration and can report perturbation SD only. It does not estimate training-seed uncertainty. Paired training profiles match seeds and the compatible scientific and runtime settings. Architecture comparisons remain descriptive because layer definitions and original optimization recipes differ.

```text
<audit>/audit.json                                  + COMPLETE.json
<plan>/plan.json                                    + COMPLETE.json
<runs>/OWNER.json
<runs>/checkpoints/<checkpoint_id>/inventory.json
<runs>/checkpoints/<checkpoint_id>/workers/*.json
<runs>/checkpoints/<checkpoint_id>/attempts/<slot_id>/*.json
<runs>/checkpoints/<checkpoint_id>/slots/<slot_id>.json
<runs>/checkpoints/<checkpoint_id>/checkpoint_complete.json + COMPLETE.json
<summary>/raw.csv
<summary>/checkpoint_summaries.csv
<summary>/cross_seed_summaries.csv
<summary>/training_pairs.csv
<summary>/residuals.csv
<summary>/residual_summaries.csv
<summary>/coverage.json
<summary>/findings.md
<summary>/summary.json                              + COMPLETE.json
<figures>/figures.json                              + COMPLETE.json
<figures>/*.png and *.pdf
```

Figures cover stage loss, accuracy degradation, raw and normalized sensitivity, training-alpha heatmaps, paired clean/noise training contrasts, the isolated sum versus whole-network comparison, the full and small-alpha responses, and prediction flips. Axes and stage order are consistent within each architecture. Hollow points show individual checkpoint means, and displayed cross-seed error bars are sample SD. Missing observations stay absent, and the figure index explains panels with insufficient evidence. For planned diagnostic targets pass `--profile-scope diagnostic`. For individual tensors pass `--profile-scope tensor` and repeat `--profile-target` with one to twelve exact qualified tensor names from the plan. This keeps labels readable and selection explicit. The additivity figure always uses the stage partition.

Publication markers bind bytes by SHA-256 and distinguish a completed write from completed scientific coverage. A valid summary may describe missing, excluded, partial or failed work. The raw inspection verifies each immutable record and the checkpoint completion marker. Audit, plan, summary and plot commands create fresh directories. A partial publication remains visible through the absence of its marker. Use a new directory for repeated reports, while resuming raw passes in the original plan-owned namespace.

## Software tests

Run the isolated synthetic suite from the repository root. Only the small synthetic CUDA check is conditional on local CUDA availability.

```bash
PYTHONPATH=src python -B -m unittest discover -s analysis/layer_sensitivity_stage2/tests -t . -v
```

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
python -B -m unittest discover -s analysis/layer_sensitivity_stage2/tests -t . -v
```
