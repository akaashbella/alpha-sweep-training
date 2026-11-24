# Stage 3 noise-weighted Hessian geometry

Stage 3 independently measures the local curvature of mean unsmoothed cross-entropy at the exact best checkpoints selected by the completed Stage 2 pilot. Stage 1 fit a whole-network quadratic to observed test loss changes. Stage 2 measured validation loss changes under isolated and full-network Gaussian perturbations. Stage 3 estimates a derivative of the clean loss and compares its prediction with both empirical studies.

Empirical sensitivity measures a finite perturbation response. The weighted Hessian trace describes a local derivative in one parameterization with one noise covariance. Classification robustness measures prediction or correctness changes. Generalization concerns performance on unseen data. Architectural importance concerns how a component contributes to model behavior. None of these measures alone establishes the others or general loss-landscape flatness.

## Estimator

Let \(L(W)\) be the complete split's mean cross-entropy, differentiated with respect to every owned eligible Conv2d and Linear weight. Let \(D\) be the diagonal operator that multiplies the coordinates of tensor \(l\) by its clean population SD \(\sigma_l\), with correction zero. The original Gaussian perturbation has the form \(\epsilon=\alpha DZ\). Its mean is zero and its covariance is \(\alpha^2D^2\).

For a valid local Taylor approximation, the first-order term vanishes in expectation. The leading loss change is \(\alpha^2\kappa/2\), where \(\kappa=\operatorname{Tr}(DHD)=\sum_l\sigma_l^2\operatorname{Tr}(H_{ll})\). The approximation can fail when perturbations cross activation regions or higher-order terms matter. A small nominal alpha alone does not guarantee locality.

For each probe use independent balanced Rademacher entries \(z_i\in\{-1,+1\}\) and set \(v_l=\sigma_l z_l\). Then \(E[v^\top Hv]=\kappa\) because \(E[zz^\top]=I\). Rademacher is the estimator distribution. The scientific perturbations remain Gaussian. Seeds are the first eight SHA-256 bytes interpreted as little-endian uint64. Canonical JSON binds the best generation and byte SHA, the Stage 3 randomness version, probe ID and qualified tensor name. A private generator supplies each tensor stream, without drawing from training, evaluation or Stage 2 RNGs. Planning checks for seed collisions and retries reuse the original seeds.

The differentiable FP32 clean forward uses the original unsmoothed FP32 classification loss. Reverse differentiation with `torch.autograd.grad(create_graph=True)` gives \(g\). Differentiating \(g^\top v\) yields the full Hessian-vector product. The computation uses the original eligible parameters as the differentiation inputs. The probe vectors and scales are detached constants. This calculation constructs neither a dense Hessian nor a matrix for \(D\), and it uses neither an optimizer nor parameter updates. The implementation rejects detached, unused, nonfinite or unavailable derivatives, including unsupported double backward. A zero-scale tensor still participates in the differentiated graph and contributes exactly zero to its own quadratic form.

The same vector is held for all batches of a complete split. If batch \(b\) has \(n_b\) examples, its mean-loss HVP gets weight \(n_b/N\). The final partial batch receives its actual weight. Loss and quadratic-form reductions and dataset accumulators use FP64 scalars. Each batch releases its graph before the next batch. The clean loss and accuracy are obtained from the same forwards. The first accepted probe supplies the recorded clean baseline, and subsequent deterministic passes must reproduce it. The original clean-state guard checks clean parameter and buffer values, module and tensor bindings, and requires-grad flags. Stage 3 extends it to check and, if needed, restore existing .grad fields. Original per-module modes are restored even on exceptions.

One full HVP yields the probe contributions \(q_l=v_l^\top(Hv)_l\). Cross-block terms appear in individual probes and cancel in expectation, so the mean estimates each tensor's weighted Hessian-block trace. Stage contributions sum the paired tensor values. Every probe checks the disjoint-stage sum against the network estimate with an absolute tolerance of \(10^{-12}\max(1,\sum_l|q_l|)\). Diagnostic groups can overlap. The sum check is algebraic and provides no independent evidence that finite Stage 2 perturbation responses are additive.

The analytical double-backward interface follows the pinned PyTorch implementation. [PyTorch 2.5.1 autograd](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/autograd/__init__.py)

## Checkpoint provenance

The audit consumes the actual Stage 2 plan publication and run namespace. It requires the exact 24 checkpoint keys, 8,904 fully published inference passes, resolved best checkpoints, original eligible inventory and stage partitions. It uses the existing Stage 2 validators for document seals, file digests, clean pairing, data coverage and immutable checkpoint completion markers. It records rejections and supplies no fallback checkpoint. The plan requires all 24 audit rows to be eligible.

Both MobileNet datasets require verified RMSprop epsilon `0.0316` in the scientific declaration and effective best checkpoint optimizer metadata. Corrected CIFAR-10 additionally requires the historical native checkout at `8010d07d9d1f0886ab1e180523f316623ef622a4` with source inventory SHA-256 `a5726bac3f449455d36e2f53840f3e0fd77768a7ab2b989d2884da20f920005c`. The local handoff corroborates those pins. The audit verifies the native manifest digest from each Stage 2 source row using its own reader. Each reader runs in a separate isolated Python process with an explicit checkout root, not on a shared Python import path. Reader pins cover the actual interpreter bytes, Git revision and native source inventory. The child invokes the original restricted checkpoint and journal reader and requires the original best binding, source, recipe, transaction and prepared-data identities to match Stage 2. This also happens before live computations. [Historical native reader](https://github.com/akaashbella/alpha-sweep-training/blob/8010d07d9d1f0886ab1e180523f316623ef622a4/src/alpha_sweep_training/evaluation_inputs.py)

The historical source commit was inspected via GitHub but is absent from the local Git object database. The local project contains an earlier corrected MobileNet handoff, not the completed Stage 2 pilot or the separate operational Stage 2 patch that produced it. Supply the actual HPC plan and completed result paths, the historical checkout, and any operational patch needed to identify their emitted schema. Stage 3 currently recognizes the Stage 2 v1 plan and result schema exposed by this checkout. Other formats and absent bindings produce explicit rejection. Neither the training run's filename nor the reported accuracy can establish corrected provenance.

## Pilot and precision

The initial plan uses the exact Stage 2 seed for each CIFAR-10/CIFAR-100, architecture and training-alpha combination. It uses the ordered complete 5,000-example validation partition with original normalization, no augmentation, and all four architectures at training alphas 0, 0.10 and 0.30. The audit validates prepared data identities and the worker verifies the actual example order and decoded selected source against the original preparation. The validation objective is exploratory because those examples selected the best checkpoints.

The initial 16 probes per checkpoint prescribe 384 complete dataset-level HVPs and 1,920,000 differentiated examples. At the default batch size 32, this means 157 batch HVPs per probe and 60,288 batch HVPs overall. No measured wall time or memory estimate is assumed. The initial pilot batch size, split and named backend are frozen. The plan output reports the actual bounds. An explicitly planned future `--split test` uses all 10,000 official test examples with identical checkpoint selection and records a different plan identity. Test outcomes play no role in checkpoint, probe count or layer selection.

Stage 3 uses Python 3.10–3.12, PyTorch 2.5.1 and TorchVision 0.20.1. The Stage 3 numerical policy disables autocast and TF32, uses highest FP32 matmul precision, disables cuDNN benchmarking and requires deterministic algorithms. The launch environment must supply `CUBLAS_WORKSPACE_CONFIG=:4096:8` or `:16:8` before Python starts. The default `cudnn` backend enables cuDNN. A separately planned `native` backend explicitly disables it. Unsupported operations or resource failures are recorded and do not switch backends. Confirm a selected backend in the pinned environment on the assigned GPU before production execution.

## Commands

Use the repository root and set the named path variables to verified existing inputs or fresh output directories as appropriate. Use separate storage outside the checkout and all scientific namespaces. Repeat `--protected-root` for Stage 1 results and any external Stage 2 derived report, evaluation or workflow roots that the Stage 2 plan does not list. The audit also protects native manifests, data and training storage roots.

```bash
python -B -m analysis.hessian_geometry_stage3 audit \
  --stage2-plan-dir "$STAGE2_PLAN" --stage2-results "$STAGE2_RUNS" \
  --historical-checkout "$MOBILE_NATIVE" --protected-root "$STAGE1_ROOT" \
  --protected-root "$EXTRA_RESEARCH_ROOT" --output "$NEW_AUDIT"

python -B -m analysis.hessian_geometry_stage3 inspect \
  --publication-dir "$NEW_AUDIT" --kind audit

python -B -m analysis.hessian_geometry_stage3 plan \
  --audit-dir "$NEW_AUDIT" --output "$NEW_PLAN" --split validation \
  --backend cudnn --batch-size 32 --max-total-probes 384

python -B -m analysis.hessian_geometry_stage3 inspect --plan-dir "$NEW_PLAN"
```

### Bounded GPU integration and resumption

Once separately authorized, use an existing OSU GPU allocation, one verified checkpoint ID from the plan, and one or two full-split probes. This can use the pilot's run namespace because the probes follow its frozen protocol and count toward the prescribed initial 16. The command launches the pinned reader in a separate process on the assigned GPU. The package doesn't submit jobs.

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -B -m analysis.hessian_geometry_stage3 run \
  --plan-dir "$NEW_PLAN" --results "$STAGE3_RUNS" \
  --checkpoint "$CHECKPOINT_ID" --probe 1 2 --max-probes 2 --device cuda:0

python -B -m analysis.hessian_geometry_stage3 inspect \
  --plan-dir "$NEW_PLAN" --results "$STAGE3_RUNS"
```

Review finite tensor and full-network quadratics, stage partition sums, exact-state flags, 5,000 examples and 157 batches at the default size. Check duration, allocated and reserved peak GPU memory, software and actual backend in the probe records. Summarize and compare below to review the actual clean-loss and clean-accuracy differences against Stage 2. The prefix table can show the first two probes while withholding the incomplete 16-probe mean. If a required double backward fails, preserve its diagnostic. A different backend or batch size uses a new plan and run namespace following review.

Reissuing an exact selection checks and skips successful records. Interrupted attempts retain their original IDs and seeds. Explicitly use `--retry-failed` for recorded failures. Different checkpoints can run concurrently. Each checkpoint has a process-scoped advisory lease, so concurrent attempts at the same checkpoint are serialized by running subsequent selections after the current worker exits. The filesystem must support advisory locks and exclusive same-directory hard-link publication. The executable work is bounded by one checkpoint, explicit probe IDs and `--max-probes`.

After a documented, prespecified precision assessment, extend 16→32 and 32→64 with a new plan directory, preserving the same run namespace and initial IDs. Use estimator precision or a sampling rule defined independently of the sign or favorability of an estimate. A probe budget of 16 may be insufficient.

```bash
python -B -m analysis.hessian_geometry_stage3 plan \
  --parent-plan "$NEW_PLAN" --output "$NEW_PLAN_32" \
  --probes 32 --max-total-probes 768 \
  --justification 'Prespecified precision extension after the 16-probe pilot'
```

The extension seals the parent and checks the entire ancestry. It changes only the probe budget and adaptation record. The old checkpoint completion markers remain evidence for their original prefixes.

### Summarization and comparison

```bash
python -B -m analysis.hessian_geometry_stage3 summarize \
  --plan-dir "$NEW_PLAN" --results "$STAGE3_RUNS" --output "$NEW_SUMMARY"

python -B -m analysis.hessian_geometry_stage3 compare \
  --plan-dir "$NEW_PLAN" --summary-dir "$NEW_SUMMARY" --output "$NEW_COMPARISON" \
  --stage1-report "$STAGE1_ORIGINAL_SUMMARIES" --stage1-collection "$STAGE1_FROZEN_COLLECTION" \
  --historical-evaluation-root "$CORRECTED_MOBILE_TEST_SWEEP"

python -B -m analysis.hessian_geometry_stage3 plot \
  --summary-dir "$NEW_SUMMARY" --comparison-dir "$NEW_COMPARISON" --output "$NEW_FIGURES"

python -B -m analysis.hessian_geometry_stage3 inspect \
  --publication-dir "$NEW_FIGURES" --kind figures
```

The Stage 1 arguments are optional. Together, the summaries and their matching frozen original collection establish empirical eligibility, fingerprint, best generation and digest, data and effective precision. Legacy MobileNet rows are excluded through the exact identity join. When a corrected CIFAR-10 report cannot be represented by the main Stage 1 reader, the optional historical evaluation root allows read-only recovery of the same empirical fits from the complete corrected native test sweep. These fits are labeled as reconstructed from native results with the Stage 1 method. The historical reader validates the native evaluation binding and sweep and checks its immutable records. It performs no forward passes for this operation. Omit these arguments to compare Stage 2 alone. A validation Hessian versus a test fit is explicitly cross-split and exploratory. A Stage 3 test plan instead supports a checkpoint- and split-matched test comparison.

The primary Stage 2 comparison requires the same ordered validation data and the same checkpoint and stage targets. For each evaluation alpha, the predicted response is \(\alpha^2\hat\kappa_G/2\) and the signed residual is observed minus predicted. The table includes the absolute residual, each experiment's replication counts and SDs/SEs, and the prediction's propagated MC SE \(\alpha^2\mathrm{SE}(\hat\kappa_G)/2\). It records observed precision, backend, device and software differences and flags unknown settings. Clean accuracy and loss are compared directly; there is no rounding to force agreement. The two uncertainty sources remain separate.

The Stage 2 local empirical fit is origin-constrained OLS with primary alphas 0.01, 0.02 and 0.05 and secondary adding 0.10. All points in a window are required. The resulting \(2c\) is compared with the independently estimated \(\hat\kappa\). Relative stage ranks include exact ties. Stage 2 finite-perturbation interactions use paired realizations. Stage 3 additive local predictions have only the numerical partition residual. These comparisons can reveal agreement or disagreement. They cannot uniquely identify interacting layer pairs, and there is no formal breakdown threshold.

## Statistics and files

Checkpoint and target means require the entire plan's probe set. For running estimates only, the prespecified complete prefixes are 1, 2, 4, 8, 16, 32 and 64, as applicable. SD is sample SD and MC SE is SD divided by \(\sqrt K\). The SD uses actual paired group sums. Signed curvature, including negative values, remains unchanged. The pilot's one training seed gives no estimate of training-level uncertainty. Seed aggregation uses checkpoint means separately from perturbation or probe replication.

Raw traces are primary. The normalized tables include trace per million eligible elements, parameter fraction, trace per summed variance energy \(\sum_l n_l\sigma_l^2\), and group fraction of the whole trace. Ratios require a denominator above an absolute \(10^{-12}\) guard. The whole-trace fraction also uses a relative \(10^{-9}\) guard against cancellation, scaled by the sum of absolute estimated tensor traces. Signed ratios and unstable denominators are visible as applicable. These normalizations are conditional on the selected target and the scientific noise model.

```text
<audit>/audit.json, COMPLETE.json
<plan>/plan.json, COMPLETE.json
<runs>/OWNER.json
<runs>/checkpoints/<id>/inventory.json
<runs>/checkpoints/<id>/attempts/<probe>/*.json
<runs>/checkpoints/<id>/probes/<probe>.json
<runs>/checkpoints/<id>/COMPLETE-016.json (or 032 and 064)
<summary>/whole_network_traces.csv, tensor_traces.csv, stage_traces.csv
<summary>/diagnostic_traces.csv, normalized_curvature.csv, probe_convergence.csv
<summary>/paired_training_comparisons.csv, training_seed_summaries.csv
<summary>/coverage.json, analysis_metadata.json, findings.md, summary.json, COMPLETE.json
<comparison>/stage1_comparisons.csv, stage2_comparisons.csv, empirical_quadratic_fits.csv
<comparison>/approximation_residuals.csv, stage2_interactions.csv, baseline_comparisons.csv
<comparison>/stage_rankings.csv, analysis_metadata.json, findings.md, comparison.json, COMPLETE.json
<figures>/figure_index.json, figures.json, COMPLETE.json, *.png, *.pdf
```

The ten figure families cover whole-network traces, signed stages, convergence, predicted versus observed, response curves, empirical proxies, stage comparisons, paired training profiles, residuals and accuracy. Figures are stratified by dataset and architecture, and the index records missing observations. Coverage records enumerate all expected probes and attempts. A completed file publication may faithfully describe missing scientific work.

## Software verification

Only synthetic examples are used during implementation testing. The optional CUDA test exercises a small convolutional model with BatchNorm and the same autograd implementation. It skips when CUDA is unavailable. To verify on a pinned CPU environment run the suite below; for the optional synthetic CUDA test use a separately authorized local GPU or GPU allocation.

```bash
PYTHONPATH=src python -B -m unittest discover -s analysis/hessian_geometry_stage3/tests -t . -v
```

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
python -B -m unittest discover -s analysis/hessian_geometry_stage3/tests -t . -v
```
