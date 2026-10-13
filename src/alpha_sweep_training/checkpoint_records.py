"""Committed record reconstruction and strict full-epoch/selection validation."""

from dataclasses import asdict, fields
import math

from .checkpoint_io import exact, fail, same
from .checkpoints import compatible_provenance
from .identity import canonical_alpha, identity_metadata
from .metrics import AmpSkip, BestSelection, ClassificationMetrics, Counters, EpochRecord, LayerStatistic, RecordContext
from .scheduler import saved_schedule_rates
from .spec import EPOCHS, TRAINING_ENGINE_VERSION, TRAINING_RECORD_VERSION, VALIDATION_SIZE


def validate_context(context: dict, config, provenance) -> RecordContext:
    exact(context, set(identity_metadata(config)) | {"result_format_version", "engine_version", "dataset", "architecture",
        "alpha", "canonical_alpha", "training_seed", "split_identity", "preprocessing_identity", "normalization_identity",
        "attempt_id", "provenance"}, "record context")
    for key, value in identity_metadata(config).items():
        same(context[key], value, f"record {key}")
    science = config.scientific
    for key in ("dataset", "architecture", "alpha", "training_seed"):
        same(context[key], science[key], f"record {key}")
    same(context["canonical_alpha"], canonical_alpha(science["alpha"]), "record canonical alpha")
    if type(context["attempt_id"]) is not str or not context["attempt_id"]:
        fail("Record requires actual attempt identity")
    same(context["split_identity"], science["data"]["split"]["identity"], "record split")
    same(f"normalization-v1:{context['normalization_identity']}", science["data"]["normalization"]["source"], "record normalization")
    same(context["result_format_version"], TRAINING_RECORD_VERSION, "record format")
    same(context["engine_version"], TRAINING_ENGINE_VERSION, "record engine")
    compatible_provenance(context["provenance"], provenance)
    return RecordContext(context)


def classification(value: dict, *, clean: bool = False) -> ClassificationMetrics:
    keys = {"loss", "correct", "examples", "batches", "accuracy", "accuracy_unit", "loss_unit"}
    exact(value, keys | ({"clean_weights"} if clean else set()), "classification metrics")
    result = ClassificationMetrics(**{key: value[key] for key in ("loss", "correct", "examples", "batches")})
    expected = result.to_dict()
    if clean:
        expected["clean_weights"] = True
    same(value, expected, "metric values/units")
    return result


def parse_epoch(document: dict, config, provenance) -> tuple[RecordContext, EpochRecord, tuple[LayerStatistic, ...]]:
    exact(document, {"context", "record", "statistics"}, "epoch document")
    context = validate_context(document["context"], config, provenance)
    value = exact(document["record"], {field.name for field in fields(EpochRecord)}, "epoch record")
    kwargs = dict(value)
    kwargs["training"] = classification(value["training"])
    kwargs["validation"] = classification(value["validation"], clean=True)
    kwargs["totals"] = Counters(**exact(value["totals"], set(asdict(Counters())), "epoch counters"))
    kwargs["best"] = BestSelection(**exact(value["best"], {"epoch_index", "correct", "examples"}, "epoch best"))
    skips = []
    for entry in value["amp_skips"]:
        exact(entry, {"batch_index", "gradient_names", "scale_before", "scale_after"}, "AMP skip")
        skips.append(AmpSkip(entry["batch_index"], tuple(entry["gradient_names"]), entry["scale_before"], entry["scale_after"]))
    kwargs["amp_skips"] = tuple(skips)
    record = EpochRecord(**kwargs)
    statistics = tuple(LayerStatistic(**{**exact(entry, {field.name for field in fields(LayerStatistic)}, "layer statistic"),
                                        "shape": tuple(entry["shape"])}) for entry in document["statistics"])
    same(record.to_dict(), value, "epoch roundtrip")
    return context, record, statistics


def validate_epoch(record: EpochRecord, statistics: tuple[LayerStatistic, ...], *, previous: Counters,
                   previous_best: BestSelection | None, config, components: dict) -> None:
    counts = components["data"]["counts"]
    completed = previous.completed_epochs + 1
    same(record.epoch_index, previous.completed_epochs, "record epoch index")
    same(record.completed_epochs, completed, "record completed count")
    same(record.status, "trained_pending_test" if completed == EPOCHS else "training", "epoch status")
    same(record.training_kind, "clean_training" if config.scientific["alpha"] == 0 else "perturbed_training", "training metric kind")
    record.totals.validate(complete_boundary=True)
    for name in ("attempted_batches", "successful_updates", "skipped_updates"):
        value = getattr(record, name)
        if type(value) is not int or value < 0:
            fail("Invalid epoch update counter")
        same(getattr(record.totals, name), getattr(previous, name) + value, f"epoch cumulative {name}")
    same(record.attempted_batches, counts["training_batches"], "epoch attempts")
    same(record.successful_updates + record.skipped_updates, record.attempted_batches, "epoch update accounting")
    same(record.totals.completed_epochs, completed, "epoch total count")
    same(record.totals.processed_examples, previous.processed_examples + counts["training_processed_per_epoch"], "epoch samples")
    same(record.training.examples, counts["training_processed_per_epoch"], "training coverage")
    same(record.training.batches, counts["training_batches"], "training batches")
    same(record.validation.examples, VALIDATION_SIZE, "validation coverage")
    same(record.validation.batches, counts["validation_batches"], "validation batches")
    selected = previous_best is None or record.validation.correct > previous_best.correct
    same(record.selected_best, selected, "strict best decision")
    best = BestSelection(record.epoch_index, record.validation.correct) if selected else previous_best
    same(asdict(record.best), asdict(best), "best selection across epochs")
    record.best.validate(completed)
    skip_indices = [skip.batch_index for skip in record.amp_skips]
    if (len(skip_indices) != record.skipped_updates or len(set(skip_indices)) != len(skip_indices)
            or any(type(index) is not int or not 0 <= index < record.attempted_batches for index in skip_indices)):
        fail("AMP skip locations/counts disagree")
    if skip_indices and config.scientific["precision"]["mode"] != "amp_float16":
        fail("Managed AMP skips require float16 scaling")
    names = set(components["optimizer"]["parameter_names"])
    for skip in record.amp_skips:
        if not skip.gradient_names or not set(skip.gradient_names) <= names or any(
                type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0
                for scale in (skip.scale_before, skip.scale_after)):
            fail("Invalid AMP skip diagnostics")
        if skip.scale_after > skip.scale_before:
            fail("Scaler-managed overflow cannot increase its scale")
    interval = config.scientific["training"]["scheduler_interval"]
    same(record.scheduler_interval, interval, "record scheduler cadence")
    before_position = previous.completed_epochs if interval == "epoch" else previous.successful_updates
    after_position = completed if interval == "epoch" else record.totals.successful_updates
    same(record.scheduler_position_before, before_position, "scheduler position before epoch")
    same(record.scheduler_position_after, after_position, "scheduler position after epoch")
    descriptor = {key: value for key, value in components["scheduler"].items()
                  if key not in ("total_units", "warmup_units", "current_learning_rates", "position_semantics")}
    for field, position, epochs, updates in (
        ("lr_first_attempt", before_position, previous.completed_epochs, previous.successful_updates),
        ("lr_at_boundary", after_position, completed, record.totals.successful_updates),
        ("lr_last_attempt", before_position if interval == "epoch" else after_position - (0 if record.attempted_batches - 1 in skip_indices else 1),
         previous.completed_epochs, record.totals.successful_updates - (0 if record.attempted_batches - 1 in skip_indices else 1)),
    ):
        rates = saved_schedule_rates(config, counts["training_batches"], {**descriptor, "completed_units": position}, epochs, updates)
        if not math.isclose(getattr(record, field), rates[0], rel_tol=1e-12, abs_tol=0.0):
            fail(f"Epoch {field} disagrees with its scheduler position")
    if type(record.duration_seconds) not in (int, float) or not math.isfinite(record.duration_seconds) or record.duration_seconds < 0:
        fail("Invalid epoch duration")
    eligible = components["model"]["eligible_weights"]
    if len(statistics) != len(eligible) or len({entry.name for entry in statistics}) != len(statistics):
        fail("Incomplete/duplicate clean layer statistics")
    for statistic, expected in zip(statistics, eligible):
        for key in ("name", "module_type", "shape", "numel"):
            same(getattr(statistic, key), expected[key], f"clean layer {key}")
        if type(statistic.population_std) not in (int, float) or not math.isfinite(statistic.population_std) or statistic.population_std < 0:
            fail("Nonfinite/negative clean layer population standard deviation")
