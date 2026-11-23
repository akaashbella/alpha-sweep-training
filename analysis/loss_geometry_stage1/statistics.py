"""Deterministic descriptive statistics; no Hessian estimation or significance tests."""

import math
import statistics as stdlib_stats

from .contracts import WINDOWS, require


def quadratic_fit(points: dict[float, float | None], window: str) -> dict:
    alphas = WINDOWS[window]
    valid = [(a, points.get(a)) for a in alphas if points.get(a) is not None]
    result = dict(fit_window=window, fitting_alphas=list(alphas), fitting_alpha_min=min(alphas),
                  fitting_alpha_max=max(alphas), expected_points=len(alphas), valid_points=len(valid),
                  eligible=len(valid) == len(alphas), exclusion_reason=None, coefficient_c=None,
                  empirical_curvature_proxy_2c=None, rss=None, rmse=None, r_squared=None,
                  r_squared_status="fit_unavailable", fitted_points=[])
    if len(valid) != len(alphas):
        result["exclusion_reason"] = "Missing/ineligible fitting alphas: " + ", ".join(str(a) for a in alphas if points.get(a) is None)
        return result
    require(all(type(y) in (int, float) and math.isfinite(y) for _, y in valid), "Nonfinite fit observations")
    # x = alpha^2, OLS through origin: c = sum(x*y)/sum(x*x).
    c = math.fsum(a*a*y for a, y in valid) / math.fsum(a**4 for a, _ in valid)
    fitted = [dict(evaluation_alpha=a, observed_delta_loss=y, predicted_delta_loss=c*a*a,
                   residual=y-c*a*a) for a, y in valid]
    rss = math.fsum(p["residual"]**2 for p in fitted)
    mean = math.fsum(y / len(valid) for _, y in valid)
    tss = math.fsum((y-mean)**2 for _, y in valid)
    r2 = None if tss == 0 else 1-rss/tss
    require(all(math.isfinite(v) for v in (c, 2*c, rss, math.sqrt(rss/len(valid)))) and
            (r2 is None or math.isfinite(r2)), "Nonfinite quadratic fit result")
    result.update(coefficient_c=c, empirical_curvature_proxy_2c=2*c, rss=rss,
                  rmse=math.sqrt(rss/len(valid)), r_squared=r2,
                  r_squared_status="undefined_constant_observed_delta_loss" if tss == 0 else "defined_centered",
                  fitted_points=fitted)
    return result


def complete_summary(values: list[float | None], expected: int = 3, compatible: bool = True) -> dict:
    eligible = [v for v in values if v is not None]
    require(all(math.isfinite(v) for v in eligible), "Nonfinite cross-seed observation")
    complete = len(eligible) == expected and len(values) == expected and compatible
    return dict(expected_count=expected, eligible_count=len(eligible), compatible=compatible, complete=complete,
                mean=math.fsum(v/expected for v in eligible) if complete else None,
                sample_sd=stdlib_stats.stdev(eligible) if complete and expected >= 2 else None)


def pearson(pairs: list[tuple[float, float]]) -> dict:
    """Centered Pearson r, conditional on the explicitly reported available pairs."""
    n = len(pairs)
    if n < 3:
        return dict(pearson_r=None, count=n, status="fewer_than_three_pairs")
    xs, ys = zip(*pairs)
    mx, my = math.fsum(x/n for x in xs), math.fsum(y/n for y in ys)
    xx, yy = math.fsum((x-mx)**2 for x in xs), math.fsum((y-my)**2 for y in ys)
    if xx == 0 or yy == 0:
        return dict(pearson_r=None, count=n, status="constant_variable")
    r = math.fsum((x-mx)*(y-my) for x, y in pairs) / (math.sqrt(xx)*math.sqrt(yy))
    require(math.isfinite(r), "Nonfinite correlation")
    return dict(pearson_r=r, count=n, status="available_pairs_descriptive_only")
