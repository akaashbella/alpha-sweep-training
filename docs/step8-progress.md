# Step 8 source progress

Baseline: clean working tree at reviewed local/remote HEAD
`45ce5507ca3d702c265f84b503744e7d48d5bbfc`. No reset or user-edit replacement.

Implemented modules: `reporting_contract.py` (finite sealed snapshots, conventions,
membership/raw/epoch validation and own provenance); `results.py` (historical,
read-only 240/6720 authority joins); `aggregation.py` (complete prescribed summaries,
two-stage robustness, validation-only co-winners and paired descriptions);
`report_publication.py` (exclusive guarded derived namespaces, CSV/JSON exports,
checksums/last marker and inspection); `plotting.py` (lazy headless six figure
families, individual seeds/realizations, per-seed heatmaps and coverage);
`reporting.py` (collect/summarize/plot/inspect CLI). The plotting dependency is an
optional extra; existing research package pins remain unchanged.

Statistics are fraction accuracy, percentage-point differences, sample SD ddof=1
(null for n<2), no pre-comparison rounding, complete prescribed counts only and
realizations within checkpoint before training-seed aggregation. Global validation
selection requires all ten complete three-seed completed-training candidates and
retains every exact mean tie in grid order. Test data never select a checkpoint,
training alpha or tie-break.

Static review cases and outcomes:

| Case | Source behavior |
| --- | --- |
| Complete 240 tasks/6720 slots | Fixed manifest/slot order; all identities/refs retained, complete three-seed and 3-realization statistics, sweep proof separate |
| Entirely unexecuted grid | 240 missing training and 6720 unbound expected slot rows; actual checkpoints/private seeds/raw references remain null; header-only epoch/layer exports |
| Missing/failed training seed | Stored status/condition/evidence retained; official validation group null with exclusions; no zero imputation |
| Full budget awaiting training publication | Verified best/slots reportable; validation-selection seed ineligible until training completion validates |
| Partial robustness with clean success | Clean/raw successful slots reportable; incomplete positive checkpoint/group means null; no robustness-completion inference |
| Three realizations per training seed | Checkpoint mean/SD first, cross-seed mean/SD second; distinct within-checkpoint and cross-seed uncertainty |
| Exact validation ties/incomplete alpha grid | Unrounded correct-count means; all co-winners in grid order; provisional ranking only when grid incomplete |
| Deferred sweep finalization | Preserves deferred/reason/error/pending coverage; valid raw slots remain usable, completed marker not invented |
| Legacy clean evidence | Separate original evidence; never auto-imported or duplicated as independent replicate 0 |
| Incompatible protocol/checkpoint/provenance | Existing validators reject; historical source/provenance retained instead of reporter source equality; official pairing/group compatibility explicit |
| Concurrent authority change | Task/checkpoint stamps invalidate accepted bracket and mark unstable; no mixed generation or reader writer locks |
| Output/figure failure | New directory ownership and overlap/link rejection; preserve partial files; final checksummed marker only after all outputs; failure/unconfirmed marker never science authority |

Source review corrected distinctions between training seed replication and data
compatibility, removed per-layer full-provenance duplication, retained successful
raw observations when official compatibility is unavailable, captured sweep
documents, required receipt/epoch/best linkage and retained explicit output-failure
states. Output overlap decisions normalize paths after rejecting links, preventing
traversal aliases from bypassing protected-store checks. Evaluation inspection is isolated from training validation eligibility so
test availability cannot affect alpha selection; auxiliary unexpected-artifact
inspection remains diagnostic rather than discarding otherwise verified results.
The requirement-by-requirement source review and tracked diffs are complete.
No further known Step 8 source gap was identified; actual reader compatibility,
resource usage, filesystem durability and renderer behavior require later HPC
verification. Source inspection does not establish runtime correctness.

Final Git inventory: 13 modified tracked files and eight untracked files (six
reporting modules and two guides/progress documents), nothing staged. No commit or
push performed. Scientific configuration/spec constants and identity code,
training/optimizer/scheduler/randomness, dataset/checkpoint/evaluation authority
sources and Step 7 workflow/Slurm source remain unchanged. The scientific document
only finalizes the explicitly requested reporting conventions.

Remaining inputs: training seeds, split preparation seed/identity and normalization,
drop_last, precision/backend and scheduler endpoints/cadence; actual shared storage,
fixed compatible environment/CUDA/Matplotlib setup, account/QOS evidence and Slurm/
filesystem capabilities. Reporting conventions above are explicitly approved Step 8
choices, not new training scientific defaults. Step 7 authority/recovery and all
training/evaluation fingerprints, bindings and seeds remain unchanged.

The [operating guide](results-and-figures.md) documents exact schemas/APIs, formulas,
coverage/ownership and future full-experiment commands. Stop before Step 9.
No tests, project execution/imports, executable verification, compilation/linting,
installations/downloads/preparation, actual aggregates/figures/reports/results,
experiment/workflow artifacts, Slurm commands or HPC jobs were performed. Only
source/configuration/documentation edits, source/Git inspection and official
plotting-documentation consultation occurred. Runtime correctness remains unverified.
