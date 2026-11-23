"""Synthetic reporting-v1 documents exclusively for temporary unit-test storage."""

import hashlib
from pathlib import Path

from alpha_sweep_training.reporting_contract import CONVENTIONS, sealed

from ..contracts import ALPHAS, ARCHITECTURES, DATASETS, EVALUATION_ALPHAS, canonical_alpha
from ..loading import REQUIRED_FILES, digest
from ..output import write_json
from ..statistics import complete_summary

SEEDS = (11, 22, 33)
SIGNATURE = "sha256:" + "a"*64


def stat(individuals: list[dict], expected: int) -> dict:
    values = [r["value"] for r in individuals]
    s = complete_summary(values, expected)
    return dict(expected_count=expected, eligible_count=s["eligible_count"], complete=s["complete"],
                individuals=individuals, exclusions=[r for r in individuals if r["value"] is None],
                official=None if not s["complete"] else dict(mean=s["mean"], sample_sd=s["sample_sd"], count=expected))


def fixture(*, missing: set[tuple[int, float, int]] | None = None) -> dict:
    missing = missing or set()
    training_groups, checkpoint_groups, robust_groups, clean_pairs = [], [], [], []
    run_index = 0
    for dataset in DATASETS:
        for architecture in ARCHITECTURES:
            for train in ALPHAS:
                header = dict(dataset=dataset, architecture=architecture, training_alpha=train, canonical_training_alpha=canonical_alpha(train))
                runs = [f"{dataset}__{architecture}__alpha-{canonical_alpha(train)}__seed-{s}" for s in SEEDS]
                clean_by, by_eval = {}, {e: [] for e in EVALUATION_ALPHAS}
                for run, seed in zip(runs, SEEDS):
                    clean_loss = 1.0 + train + seed/100
                    clean_acc = .8 + seed/1000
                    coefficient = 3/(1+train) + seed/100
                    checkpoint = dict(generation=f"synthetic-generation-{run_index}", sha256=hashlib.sha256(run.encode()).hexdigest())
                    clean_ok = (run_index, 0., 0) not in missing
                    for offset, alpha in enumerate(EVALUATION_ALPHAS):
                        reps = (0,) if alpha == 0 else (1, 2, 3)
                        metrics = {}
                        for metric in ("loss", "accuracy"):
                            inds = []
                            for rep in reps:
                                ok = (run_index, alpha, rep) not in missing
                                # Symmetric tiny realization variation has zero mean.
                                variation = 0. if alpha == 0 else (rep-2)*alpha*.001
                                value = clean_loss+coefficient*alpha**2+variation if metric == "loss" else clean_acc-.02*coefficient*alpha**2+variation
                                logical = run_index*28 + (0 if alpha == 0 else 1+3*(offset-1)+rep-1)
                                inds.append(dict(training_run_id=run, training_seed=seed, logical_index=logical,
                                    raw_reference=dict(relative_path=f"synthetic/slot-{logical}.json", sha256="f"*64),
                                    observed_value=value if ok else None, value=value if ok else None, reason=None if ok else "synthetic missing slot"))
                            metrics[metric] = stat(inds, len(reps))
                        changes = None if alpha == 0 else stat([dict(training_run_id=run, training_seed=seed,
                            canonical_evaluation_alpha=canonical_alpha(alpha), evaluation_replicate=rep,
                            value=100*(metrics["accuracy"]["individuals"][rep-1]["value"]-clean_acc) if clean_ok and metrics["accuracy"]["individuals"][rep-1]["value"] is not None else None,
                            reason=None if clean_ok and metrics["accuracy"]["individuals"][rep-1]["value"] is not None else "synthetic missing target/baseline") for rep in reps], 3)
                        row = dict(**header, training_run_id=run, training_seed=seed, evaluation_alpha=alpha,
                            canonical_evaluation_alpha=canonical_alpha(alpha), **metrics, noise_change_pp=changes,
                            compatibility_signature=SIGNATURE if metrics["accuracy"]["complete"] else None, best_checkpoint_identity=checkpoint)
                        checkpoint_groups.append(row)
                        by_eval[alpha].append(row)
                        if alpha == 0:
                            clean_by[run] = row
                    clean_pairs.append(dict(**header, training_run_id=run, training_seed=seed,
                        baseline_run_id=f"{dataset}__{architecture}__alpha-0__seed-{seed}", value=0. if clean_ok else None,
                        reason=None if clean_ok else "synthetic missing clean"))
                    run_index += 1
                def across(metric: str, rows: list[dict]) -> dict:
                    return stat([dict(training_run_id=r["training_run_id"], training_seed=r["training_seed"],
                        value=None if r[metric]["official"] is None else r[metric]["official"]["mean"],
                        reason=None if r[metric]["official"] is not None else "synthetic incomplete checkpoint") for r in rows], 3)
                training_groups.append(dict(**header, training_run_ids=runs,
                    validation_accuracy=stat([dict(training_run_id=r, training_seed=s, value=.8, reason=None) for r, s in zip(runs, SEEDS)], 3),
                    clean_accuracy=across("accuracy", list(clean_by.values())), clean_loss=across("loss", list(clean_by.values()))))
                for alpha, rows in by_eval.items():
                    robust_groups.append(dict(**header, evaluation_alpha=alpha, canonical_evaluation_alpha=canonical_alpha(alpha),
                        accuracy=across("accuracy", rows), loss=across("loss", rows), noise_change_pp=None if alpha == 0 else across("noise_change_pp", rows)))
    return sealed(dict(reporting_version=1, kind="result_summaries", conventions=CONVENTIONS,
        grid_id="synthetic-software-fixture-only", manifest_document_sha256="b"*64, collection_sha256="c"*64,
        training_groups=training_groups, checkpoint_robustness=checkpoint_groups, robustness_groups=robust_groups,
        clean_seed_pairs=clean_pairs, coverage=dict(expected_training=240, expected_evaluation_slots=6720,
            successful_slots=6720-len(missing), source_diagnostics=["SYNTHETIC SOFTWARE FIXTURE; NOT SCIENTIFIC DATA"])))


def reseal(document: dict) -> dict:
    return sealed({k: v for k, v in document.items() if k != "document_sha256"})


def publication(root: Path, document: dict) -> None:
    root.mkdir()
    write_json(root / "summaries.json", document)
    write_json(root / "coverage.json", document["coverage"])
    for name in REQUIRED_FILES - {"summaries.json", "coverage.json"}:
        (root / name).write_text("synthetic software fixture only\n", encoding="utf-8")
    owner = sealed(dict(reporting_version=1, namespace="derived-reporting-v1", kind="summaries",
        scientific_completion_authority=False, grid_id=document["grid_id"], manifest_document_sha256=document["manifest_document_sha256"],
        collection_sha256=document["collection_sha256"], publisher_provenance=dict(synthetic_fixture=True)))
    write_json(root / "OWNER.json", owner)
    refs = [dict(relative_path=name, sha256=digest(root / name), role="derived_summaries" if name == "summaries.json" else "coverage" if name == "coverage.json" else "fixture") for name in sorted(REQUIRED_FILES)]
    write_json(root / "COMPLETE.json", sealed(dict(reporting_version=1, status="derived_publication_complete",
        owner_sha256=owner["document_sha256"], scientific_completion_authority=False, files=refs)))
