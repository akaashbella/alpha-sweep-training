# CIFAR-10 optimization audit and diagnostic source handoff

Date: October 30, 2026. Checked local HEAD:
`1513d0f7f3f0a28a52cf66662e1e4ab4a5b4c9b1`, exactly the supplied audited commit.
Initial Git status contained only the user's untracked `AGENTS.md`; it is preserved.
No causal coding defect has been reproduced. This is source work, not runtime
certification, a recipe change, or authorization to start CIFAR-100.

## Evidence available and unavailable

The user's audit/handoff and request supply these observations: MobileNet alpha-zero
selected-checkpoint clean test accuracies 57.60%, 87.76%, 24.18%; reported seed 2
selects epoch 3 and finishes near chance; 15/30 MobileNet final training accuracies
are at most 12%; all alpha >= .3 runs finish at 9.06% validation accuracy. Reported
validation loss reaches 2.535618e28 at alpha 1/seed 1/epoch 9 and 8.592712e20 at
alpha zero/seed 2/epoch 17. The supplied audit reports all 24,000 LR records matching
the schedule, no skipped updates, and matching core computation hashes in all 120
run inventories. **These record-level results were not independently reconstructed
in this session.** Seed labels 1/2 are report labels; do not assume they equal the
numeric `training_seed` fields. Discover the actual identities before selecting files.

The supplied audit also reports median eligible-weight population SD falling from
about .104984 at epoch 1 to 6.84e-12 at epoch 200 for baseline seed 2, versus about
.033303 at the control seed's final epoch. This supports degeneration, not a proof
of all-zero weights. Repeated 9.06% validation is consistent with constant-class
collapse but requires prediction histograms. Diagnostics record training and clean
validation separately, including actual forward activation/logit histograms.

The four supplied paths were checked read-only and are inaccessible here:

- `/workspace/scratch/9da864677e22/upload/cifar10-report-complete.tar(3).gz`
- `/workspace/scratch/9da864677e22/report_source/cifar10-reports.3JgYJT/`
- `/workspace/scratch/9da864677e22/deliverables/alpha-sweep-cifar10-expanded/`
- `/workspace/scratch/9da864677e22/deliverables/audit/alpha-sweep-cifar10-audit-and-codex-handoff.md`

This is a Windows repository workspace, not that ChatGPT filesystem. The earlier
provided Windows Downloads handoff was readable, but contains no raw trajectories
or model state. No full manifest, saved-run/config directories, checkpoint paths,
initial state, actual seed mapping, or presentation source is available locally.
The user reports `collection/collection.json` is truncated: do not repair it by
guessing, read partial JSON as authority, or require it for descriptive CSV analysis.
Use the existing `collection/` CSVs and `summaries/` when accessible. The new CSV
path never opens collection.json and labels completion as unverified export evidence.

## Findings

Paths/lines below refer to the pinned original source, which this change preserves.

| Finding | Classification and evidence | Consequence |
| --- | --- | --- |
| Alpha-zero MobileNet does not sustain learning | Confirmed in supplied record audit; raw records unavailable here | Prioritize the reported seed-2 trajectory, using seed 1 as a learning control. Noise during independent test evaluation cannot explain a clean-training collapse. |
| Extreme but finite losses can retain training-completed status | Source-confirmed observability gap: [numerical.py:23](../src/alpha_sweep_training/numerical.py#L23), [training.py:307](../src/alpha_sweep_training/training.py#L307) | Finiteness/completion and optimization health are different. Descriptive flags must not stop, discard or relabel the run. |
| Per-architecture optimizers are intentional | [spec.py:48](../src/alpha_sweep_training/spec.py#L48), [optimizers.py:76](../src/alpha_sweep_training/optimizers.py#L76) | Faithful implementation of fixed recipes; not evidence of an optimizer-argument defect. Cross-architecture effects include the recipes. |
| RMSprop epsilon/LR may be unstable here | Untested recipe hypothesis, not a demonstrated coding error | Compare epsilon alone, LR alone, then precision alone against a repeatable baseline; observe actual update ratios and optimizer buffers. |
| Alpha zero bypasses noise | [perturbation.py:41](../src/alpha_sweep_training/perturbation.py#L41) | No scale calculation, draw or generator access on that branch. Runtime equivalence is prepared as a regression, not claimed proven. |
| Positive-noise functional tensors retain gradient ownership | [perturbation.py:59](../src/alpha_sweep_training/perturbation.py#L59), [weights.py](../src/alpha_sweep_training/weights.py) | Scale/noise are detached; the clean-plus-offset addition is tracked. No obvious disconnected-parameter, accumulated-noise or correction=1 defect was found statically. |
| BN buffers update natively during training; validation uses frozen running state | [perturbation.py:70](../src/alpha_sweep_training/perturbation.py#L70), [evaluation.py](../src/alpha_sweep_training/evaluation.py) | Matches the specified protocol. Nonzero-noise distribution mismatch is plausible; it alone cannot explain alpha-zero failures. Native normalization suitability remains untested. |
| Bfloat16 has FP32 master weights/loss and no scaler | [runtime.py](../src/alpha_sweep_training/runtime.py), [evaluation.py:28](../src/alpha_sweep_training/evaluation.py#L28) | No missing-GradScaler defect demonstrated. Float32 is a separate diagnostic arm, not a silent production fallback. |
| LR counters/restore ordering and metric arithmetic have explicit contracts | [scheduler.py:50](../src/alpha_sweep_training/scheduler.py#L50), [continuation.py:94](../src/alpha_sweep_training/continuation.py#L94), [metrics.py:74](../src/alpha_sweep_training/metrics.py#L74) | Source aligns with original horizon, optimizer-before-scheduler restore, and sample-weighted loss. The supplied all-epoch LR check is reported evidence; reference checks remain unexecuted here. |
| Fixed evaluation network and owned-state verification are implemented | [evaluation.py:91](../src/alpha_sweep_training/evaluation.py#L91), [evaluation_state.py](../src/alpha_sweep_training/evaluation_state.py) | Draws occur before the complete pass. Exception/preservation checks are now prepared; source alone is not certification. |
| No computation-level test suite existed at the pinned baseline | Baseline tests covered planning/queued waves | New opt-in CPU/CUDA references are source-only until an execution stage is authorized. |
| Presentation claims all models used SGD | Reported methods error; LaTeX source inaccessible | Correction text is supplied below; the remote presentation has not been edited. |

The [pinned TorchVision classification trainer](https://github.com/pytorch/vision/blob/v0.20.1/references/classification/train.py#L254)
passes RMSprop epsilon .0316 and smoothing alpha .9. The
[pinned native RMSprop implementation](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/optim/rmsprop.py#L289)
uses sqrt(square-average) plus epsilon before the momentum update. Therefore epsilon
can materially change updates when the adaptive average is small. This does not
establish .0316 as correct for CIFAR, identify the original trigger, or imply a
uniform 31.6-fold change in real updates. The ImageNet recipe is not copied.

## Source prepared for review

| File | Implemented role |
| --- | --- |
| `src/alpha_sweep_training/diagnostic_audit.py` | Read-only historical journal/checkpoint/config/component audit; actual source/software differences, native optimizer state summaries, data/runtime/best bindings, exact export list. Epoch-zero state is explicitly unavailable unless supplied. |
| `src/alpha_sweep_training/diagnostic_protocol.py` | Separate protocol identity, explicit one-factor decisions, no defaults for diagnostic thresholds/budgets, protected/exclusive output roots. |
| `src/alpha_sweep_training/diagnostic_training.py` | Alpha-zero MobileNet baseline/epsilon/lower-LR/float32 arms using original builders and private diagnostic optimizer; original 200-epoch schedule horizon; full original validation recheck on a separate best model. No production writer or test pass. |
| `src/alpha_sweep_training/diagnostic_observer.py` | Optional prediction/logit/loss/activation, gradient/parameter/update, RMSprop and BN observations. Owned read-only copies, no extra model forward/backward, no RNG draw, no optimizer-state creation. |
| `src/alpha_sweep_training/diagnostic_failures.py` | Typed stage/cause records: origin, epoch/batch indices and numerical stage/reason/tensor details; secondary observation/output failures remain separate. No compute imports at CLI startup. |
| `src/alpha_sweep_training/optimization_health.py` | Separate descriptive health policy/report for complete validated exported collections; no score filtering or alpha selection. |
| `src/alpha_sweep_training/diagnostic_csv.py` | CSV/summary-only fallback with full 120-identity joins, internal consistency, earliest-best checks, streaming layer medians and file hashes. Preserves reported lifecycle separately from unknown checkpoint authority. |
| `src/alpha_sweep_training/diagnostics.py`, `pyproject.toml` | Explicit `--execute` CLI for later authorized stages; source-phase implementation invokes none of it. |
| `tests/test_computation_contracts.py`, `tests/test_diagnostic_contracts.py` | Deferred arithmetic/ownership, state, counter, observation, policy, isolation and truncated-collection checks. |
| `tests/test_diagnostic_integration.py`, `tests/run_diagnostic_cpu_checks.py` | CPU-only mocked runner/recheck wiring and explicit environment-gated suite; no CUDA test collection. Covers effective options, actual step LR sequence, arm initialization, validation delegation and failure propagation. |

The descriptive weight-SD ratio is now named
`median_weight_sd_last_to_first_recorded_epoch_ratio`, with explicit 1-based
numerator and denominator epoch labels. Both endpoints are post-training epoch
statistics, not scratch initialization. Missing/zero denominator evidence stays null.
Execution examples start with four traced training batches and one selected
activation module, then a separate tracing-disabled baseline reproduction.

Production spec/config/optimizer/trainer/evaluator/identity/checkpoint/Slurm modules
are unchanged. The diagnostic raw RMSprop intentionally has no production-builder
ownership descriptor, so it cannot be passed to the strict main-v1 trainer. Epsilon
and LR stay in the separate diagnostic protocol, not a falsified RunConfig recipe.
The float32 arm constructs an independently identified precision declaration without
rewriting the original config. Effective instantiated options, hardware, backend,
software and prepared-data identities are checked before controlled comparisons.
The selected config must also match the actual archived resolved config after
path resolution, including its execution/data bindings; a current config with
the same scientific fingerprint cannot silently replace that evidence.
Core computation digests cannot be waived. Every other recorded-file difference
requires exact saved/actual hashes and a reason in an explicit diagnostic source
review, including the packaging entry point added here. This is not production
continuation permission; original full-source compatibility checks are unchanged.

## Ordered next evidence and decision

1. Obtain the original manifest, report CSVs/summary evidence, and actual seed-2
   and seed-1 alpha-zero saved configuration/journal/checkpoint bindings. Verify
   source bytes, native optimizer defaults/state, schedule position and initial
   state availability. Do not infer these from current declarations or Git alone.
2. Recheck each selected best on the original 5,000-image validation split under
   matching core bytes/software/GPU/precision/backend/data. Compare exact correct
   counts and examples; record any loss difference. No official test tuning.
3. Reconstruct the first divergent logged epoch from training and validation
   separately. Existing epoch aggregates cannot identify the first bad batch or
   layer. If intermediate optimizer/checkpoint histories were pruned, record that
   loss of evidence; do not fabricate trajectories from the final checkpoint.
4. Establish repeated baseline diagnostics, then epsilon-only .001 vs .0316,
   explicitly chosen lower LR at fixed epsilon, and float32 at the original recipe.
   Keep initialization/data streams/budget/tracing/hardware fixed. Prefer an actual
   epoch-zero snapshot; otherwise say seeded reconstruction, not original-state proof.
5. Inspect the actual first abnormal update, logits, gradients and BN state. No BN
   recalibration is implemented or justified from the missing trajectories. A later
   recalibration policy would require a clone and training-only data, never changing
   original buffers or the primary frozen-buffer robustness protocol.

No blanket rerun is justified yet. If a recipe change is supported, stabilize the
clean baseline across all three original seed identities/full 200 epochs, then
create a reviewed new scientific experiment identity and rerun all ten alphas ×
three seeds for the affected architecture (30 MobileNet runs if only its recipe
changes). Do not replace only failing seeds. EfficientNet sharing RMSprop does not
automatically make its existing results invalid. A demonstrated shared coding
defect determines a potentially broader scope; an evaluation-only defect may need
only checkpoint re-evaluation. Current evidence authorizes none of those reruns.

## Presentation/methods correction to apply when its source is available

Replace the all-SGD claim with:

> We study training-noise effects within four fixed architecture-specific training
> recipes. ResNet-50 uses SGD (LR .1, momentum .9, weight decay .0005).
> MobileNetV3-Large and EfficientNet-B0 use RMSprop (LR .01, momentum .9,
> smoothing alpha .9, epsilon .001, weight decay .00001). ConvNeXt-Tiny uses AdamW
> (LR .001, betas .9/.999, weight decay .05). Within an architecture, the recipe is
> fixed across all training alphas and seeds. Cross-architecture differences reflect
> both architecture and training recipe; pure architecture or optimizer-independent
> effects are not identified. MobileNet's unstable clean baseline is preliminary
> evidence and remains prominently visible.

This corrects the description without changing the experiment. Different fixed
optimizers are acceptable within-model controls. No forced common optimizer,
clipping, EMA, loss cap, new normalization policy or changed checkpoint selection
has been added. The inaccessible presentation still requires this edit.

## Commands and limits

Executed: Git status/log/HEAD/diff inspection, source/spec/document reading and
search, four `Test-Path` availability checks, and read-only upstream pinned-source
lookups. Source edits create the implementation and this handoff only. The later
review explicitly authorized focused CPU checks in an existing compatible
environment. Python launcher/PATH and installed-distribution inspection found
Python 3.13.12, Torch 2.9.1 and TorchVision 0.24.1, incompatible with the project.
The explicit CPU-only entry point was attempted and stopped at its metadata gate
before tests/project imports. No packages were installed; CUDA tests were excluded.

Not executed: project imports, tests, compilation/linting, model construction,
checkpoint loading, health reporting, validation, diagnostic training, benchmarks,
dataset preparation/downloads, installs, Slurm commands/jobs or artifact publication.
No research outputs/checkpoints/seeds/fingerprints were modified. Runtime correctness
and the causal trigger remain unverified. The
[execution/export handoff](diagnostic-execution.md) contains concrete future commands,
artifact requirements, explicit runtime gates and the source-frozen transfer constraint.
