"""Checkpoint-first loss responses, local fits, descriptive pairing and coverage."""

from collections import defaultdict

from .contracts import ALPHAS, ARCHITECTURES, DATASETS, EVALUATION_ALPHAS, IDENTITY, WINDOWS
from .statistics import complete_summary, pearson, quadratic_fit


def _mean(stat: dict) -> float | None:
    return None if stat["official"] is None else stat["official"]["mean"]


def _reason(stat: dict) -> str:
    return "; ".join(str(i["reason"]) for i in stat["exclusions"]) or "incomplete prescribed observations"


def analyze(document: dict, seeds: tuple[int, ...]) -> dict:
    """Accept a validated frozen summary; preserve every expected checkpoint/alpha."""
    source = {(r["training_run_id"], r["evaluation_alpha"]): r for r in document["checkpoint_robustness"]}
    groups = {(g["dataset"], g["architecture"], g["training_alpha"], g["evaluation_alpha"]): g for g in document["robustness_groups"]}
    training = {(g["dataset"], g["architecture"], g["training_alpha"]): g for g in document["training_groups"]}
    sensitivities, fits, predictions, slots = [], [], [], []
    checkpoint_rows, fit_rows = {}, {}
    for dataset in DATASETS:
        for architecture in ARCHITECTURES:
            for training_alpha in ALPHAS:
                tg = training[(dataset, architecture, training_alpha)]
                for run, seed in zip(tg["training_run_ids"], seeds):
                    clean = source[(run, 0)]
                    identity = {k: clean[k] for k in IDENTITY}
                    # A missing clean slot can have null binding while a positive
                    # slot has one. Preserve that actual checkpoint evidence.
                    identity["best_checkpoint_identity"] = next((source[(run, e)]["best_checkpoint_identity"] for e in EVALUATION_ALPHAS
                        if source[(run, e)]["best_checkpoint_identity"] is not None), None)
                    own = []
                    for alpha in EVALUATION_ALPHAS:
                        row = source[(run, alpha)]
                        reasons = []
                        if not clean["loss"]["complete"]:
                            reasons.append("Clean baseline: " + _reason(clean["loss"]))
                        if not row["loss"]["complete"]:
                            reasons.append("Evaluation mean: " + _reason(row["loss"]))
                        if row["loss"]["complete"] and clean["loss"]["complete"] and row["compatibility_signature"] != clean["compatibility_signature"]:
                            reasons.append("Clean/perturbed data/protocol/runtime/software mismatch")
                        if alpha > 0 and not row["noise_change_pp"]["complete"]:
                            reasons.append("Paired comparison: " + _reason(row["noise_change_pp"]))
                        eligible = not reasons
                        group = groups[(dataset, architecture, training_alpha, alpha)]
                        baseline_group = groups[(dataset, architecture, training_alpha, 0)]
                        cross_seed = all(stat["complete"] for stat in (group["loss"], group["accuracy"], baseline_group["loss"], baseline_group["accuracy"]))
                        cross_seed = cross_seed and (alpha == 0 or group["noise_change_pp"]["complete"])
                        clean_loss, clean_accuracy = _mean(clean["loss"]), _mean(clean["accuracy"])
                        loss, accuracy = _mean(row["loss"]), _mean(row["accuracy"])
                        result = dict(**identity, evaluation_alpha=alpha, canonical_evaluation_alpha=row["canonical_evaluation_alpha"],
                            clean_loss=clean_loss, perturbed_loss=loss, clean_accuracy=clean_accuracy, perturbed_accuracy=accuracy,
                            clean_accuracy_percent=None if clean_accuracy is None else 100*clean_accuracy,
                            perturbed_accuracy_percent=None if accuracy is None else 100*accuracy,
                            delta_loss=loss-clean_loss if eligible else None,
                            delta_accuracy_pp=100*(accuracy-clean_accuracy) if eligible else None,
                            accuracy_degradation_pp=100*(clean_accuracy-accuracy) if eligible else None,
                            loss_within_checkpoint_sample_sd=None if row["loss"]["official"] is None else row["loss"]["official"]["sample_sd"],
                            accuracy_within_checkpoint_sample_sd=None if row["accuracy"]["official"] is None else row["accuracy"]["official"]["sample_sd"],
                            expected_realizations=row["loss"]["expected_count"], eligible_realizations=row["loss"]["eligible_count"],
                            eligible=eligible, exclusion_reason="; ".join(reasons) or None,
                            cross_seed_eligible=cross_seed, compatibility_signature=row["compatibility_signature"])
                        own.append(result)
                        sensitivities.append(result)
                        for loss_i, acc_i in zip(row["loss"]["individuals"], row["accuracy"]["individuals"]):
                            slots.append(dict(**identity, evaluation_alpha=alpha,
                                evaluation_replicate=0 if alpha == 0 else loss_i["logical_index"]-row["loss"]["individuals"][0]["logical_index"]+1,
                                logical_index=loss_i["logical_index"], raw_reference=loss_i["raw_reference"],
                                accepted_loss=loss_i["value"], observed_loss=loss_i.get("observed_value"),
                                accepted_accuracy=acc_i["value"], observed_accuracy=acc_i.get("observed_value"),
                                loss_exclusion=loss_i["reason"], accuracy_exclusion=acc_i["reason"]))
                    checkpoint_rows[run] = own
                    for window in WINDOWS:
                        fit = quadratic_fit({r["evaluation_alpha"]: r["delta_loss"] for r in own}, window)
                        fit.update(identity)
                        fit["cross_seed_eligible"] = all(r["cross_seed_eligible"] for r in own if r["evaluation_alpha"] in WINDOWS[window])
                        fit["clean_accuracy"] = _mean(clean["accuracy"])
                        fit["clean_loss"] = _mean(clean["loss"])
                        fits.append(fit)
                        fit_rows[(run, window)] = fit
                        for row in own:
                            a = row["evaluation_alpha"]
                            predicted = fit["coefficient_c"]*a*a if fit["eligible"] else None
                            observed = row["delta_loss"]
                            predictions.append(dict(**identity, fit_window=window, evaluation_alpha=a,
                                point_role="clean_anchor" if a == 0 else "fitting" if a in WINDOWS[window] else "extrapolation",
                                observed_delta_loss=observed, predicted_delta_loss=predicted,
                                residual=observed-predicted if observed is not None and predicted is not None else None,
                                absolute_residual=abs(observed-predicted) if observed is not None and predicted is not None else None,
                                exclusion_reason=fit["exclusion_reason"] or row["exclusion_reason"]))
    summaries = []
    for key, tg in training.items():
        header = {k: tg[k] for k in ("dataset", "architecture", "training_alpha", "canonical_training_alpha")}
        for alpha in EVALUATION_ALPHAS:
            rows = [next(r for r in checkpoint_rows[run] if r["evaluation_alpha"] == alpha) for run in tg["training_run_ids"]]
            for metric in ("delta_loss", "delta_accuracy_pp", "clean_loss", "clean_accuracy", "perturbed_loss", "perturbed_accuracy"):
                values = [r[metric] if r["eligible"] else None for r in rows]
                summaries.append(dict(**header, summary_kind="loss_response", evaluation_alpha=alpha, fit_window=None,
                    metric=metric, **complete_summary(values, compatible=all(r["cross_seed_eligible"] for r in rows)),
                    individuals=[dict(training_run_id=r["training_run_id"], training_seed=r["training_seed"], value=v,
                                      exclusion_reason=r["exclusion_reason"]) for r, v in zip(rows, values)]))
        for window in WINDOWS:
            rows = [fit_rows[(run, window)] for run in tg["training_run_ids"]]
            for metric in ("coefficient_c", "empirical_curvature_proxy_2c"):
                summaries.append(dict(**header, summary_kind="quadratic_fit", evaluation_alpha=None, fit_window=window,
                    metric=metric, **complete_summary([r[metric] for r in rows], compatible=all(r["cross_seed_eligible"] for r in rows)),
                    individuals=[dict(training_run_id=r["training_run_id"], training_seed=r["training_seed"], value=r[metric],
                                      exclusion_reason=r["exclusion_reason"]) for r in rows]))
    paired, paired_summaries = _training_pairs(document, sensitivities, fits)
    relationships = _relationships(sensitivities, fits)
    return dict(loss_sensitivity=sensitivities, quadratic_fits=fits, predictions=predictions,
                summaries=summaries, training_pairs=paired, paired_summaries=paired_summaries,
                relationships=relationships, coverage=dict(
                    expected_checkpoints=240, expected_checkpoint_alpha_records=2400, expected_realization_slots=6720,
                    expected_fits=480, eligible_loss_responses=sum(r["eligible"] for r in sensitivities),
                    eligible_fits=sum(r["eligible"] for r in fits),
                    original_reporting_coverage=document["coverage"],
                    checkpoint_alpha_observations=sensitivities, realization_slots=slots,
                    fit_exclusions=[r for r in fits if not r["eligible"]],
                    cross_seed_summary_exclusions=[r for r in summaries if not r["complete"]]))


def _training_pairs(document: dict, sensitivity: list[dict], fits: list[dict]) -> tuple[list[dict], list[dict]]:
    # Existing clean seed-pair eligibility also enforces equal training settings
    # except alpha, including versioned optimizer recipe changes.
    approved = {r["training_run_id"]: r for r in document["clean_seed_pairs"]}
    pairs = []
    for rows, metric, suffix in ((sensitivity, "delta_loss", "evaluation_alpha"),
                                  (fits, "empirical_curvature_proxy_2c", "fit_window")):
        baseline = {(r["dataset"], r["architecture"], r["training_seed"], r[suffix]): r for r in rows if r["training_alpha"] == 0}
        for row in rows:
            if row["training_alpha"] == 0:
                continue
            base = baseline[(row["dataset"], row["architecture"], row["training_seed"], row[suffix])]
            gate = approved.get(row["training_run_id"])
            reason = None
            if gate is None or gate["value"] is None:
                reason = "Existing paired-training eligibility unavailable: " + str(None if gate is None else gate["reason"])
            elif not row["eligible"] or not base["eligible"]:
                reason = "Target/baseline measurement unavailable"
            elif suffix == "evaluation_alpha" and row["compatibility_signature"] != base["compatibility_signature"]:
                reason = "Incompatible target/baseline evaluation settings"
            pairs.append(dict(**{k: row[k] for k in IDENTITY}, comparison="perturbation_training_minus_clean_training",
                metric=metric, evaluation_alpha=row.get("evaluation_alpha"), fit_window=row.get("fit_window"),
                baseline_run_id=base["training_run_id"], target_value=row[metric], baseline_value=base[metric],
                difference=row[metric]-base[metric] if reason is None else None,
                target_clean_accuracy=row["clean_accuracy"], baseline_clean_accuracy=base["clean_accuracy"],
                cross_seed_eligible=row["cross_seed_eligible"] and base["cross_seed_eligible"],
                eligible=reason is None, exclusion_reason=reason))
    bins = defaultdict(list)
    for row in pairs:
        bins[(row["dataset"], row["architecture"], row["training_alpha"], row["metric"], row["evaluation_alpha"], row["fit_window"])].append(row)
    summaries = []
    for key, rows in bins.items():
        summaries.append(dict(zip(("dataset", "architecture", "training_alpha", "metric", "evaluation_alpha", "fit_window"), key),
            **complete_summary([r["difference"] for r in rows], compatible=all(r["cross_seed_eligible"] for r in rows)),
            individuals=[dict(training_seed=r["training_seed"], difference=r["difference"], exclusion_reason=r["exclusion_reason"]) for r in rows]))
    return pairs, summaries


def _relationships(sensitivity: list[dict], fits: list[dict]) -> list[dict]:
    fit_by = {(f["training_run_id"], f["fit_window"]): f for f in fits}
    output = []
    for dataset in DATASETS:
        for architecture in ARCHITECTURES:
            for alpha in EVALUATION_ALPHAS[1:]:
                # Never pool evaluation alphas. Also provide within-training-alpha
                # descriptions so training-alpha confounding remains inspectable.
                for train in (None, *ALPHAS):
                    rows = [r for r in sensitivity if r["dataset"] == dataset and r["architecture"] == architecture and
                            r["evaluation_alpha"] == alpha and (train is None or r["training_alpha"] == train)]
                    signatures = sorted({r["compatibility_signature"] for r in rows if r["eligible"]}) or [None]
                    for signature in signatures:
                        cohort = [r for r in rows if r["eligible"] and r["compatibility_signature"] == signature]
                        for metric in ("delta_loss", "primary_proxy_2c", "secondary_proxy_2c"):
                            pairs, identities = [], []
                            for row in cohort:
                                value = row["delta_loss"] if metric == "delta_loss" else fit_by[(row["training_run_id"], metric.split("_")[0])]["empirical_curvature_proxy_2c"]
                                if value is not None:
                                    pairs.append((value, row["accuracy_degradation_pp"]))
                                    identities.append(row["training_run_id"])
                            output.append(dict(dataset=dataset, architecture=architecture, evaluation_alpha=alpha, training_alpha=train,
                                compatibility_signature=signature, compatible_cohort_count=len(cohort),
                                x_metric=metric, y_metric="accuracy_degradation_pp", expected_count=30 if train is None else 3,
                                **pearson(pairs), training_run_ids=identities,
                                interpretation="descriptive compatible available pairs; no p-value/CI; training alpha varies" if train is None else "within training alpha and compatible settings; only three training seeds"))
    return output
