"""NumPy-only fixed-effects projection and exploratory calendar-score HAC.

The covariance calculation sums scores across ALL municipalities in each calendar
year before applying Bartlett HAC. It thus retains contemporaneous cross-sectional
dependence and temporal dependence through the selected lag. No finite-sample
coverage claim, p value, degrees-of-freedom correction, or causal interpretation is
made. All public model results are JSON serializable, with no NaN/Infinity values.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


PROJECTION_TOL = 1e-11
MAX_PROJECTION_ITER = 20000
ABSORPTION_TOL = 1e-10
RANK_RELATIVE_TOL = 1e-10
NORMAL_975 = 1.959963984540054


def _labels(values: Any, n: int, label: str) -> np.ndarray:
    a = np.asarray(values)
    if a.ndim != 1 or len(a) != n:
        raise ValueError(f"{label} must be a one-dimensional array of length {n}")
    if any(v is None for v in a.tolist()):
        raise ValueError(f"{label} contains a missing identifier")
    s = a.astype(str)
    if any(v.strip() in {"", "nan", "None"} for v in s.tolist()):
        raise ValueError(f"{label} contains an empty/missing identifier")
    return s


def _groups(unit: Any, state: Any, year: Any, n: int) -> dict[str, Any]:
    u = _labels(unit, n, "unit")
    s = _labels(state, n, "state")
    yr0 = np.asarray(year)
    if yr0.ndim != 1 or len(yr0) != n:
        raise ValueError(f"year must be a one-dimensional array of length {n}")
    try:
        yf = yr0.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("year must contain integer calendar years") from exc
    if not np.all(np.isfinite(yf)) or not np.all(yf == np.floor(yf)):
        raise ValueError("year must contain finite integer calendar years")
    yr = yf.astype(np.int64)
    ul, ui = np.unique(u, return_inverse=True)
    sl, si = np.unique(s, return_inverse=True)
    yl, yi = np.unique(yr, return_inverse=True)
    # Integer codes avoid accidental collisions in string concatenation.
    _, syi = np.unique(si * len(yl) + yi, return_inverse=True)
    uc = np.bincount(ui).astype(float)
    syc = np.bincount(syi).astype(float)
    tm = np.bincount(ui, weights=yf) / uc
    tc = yf - tm[ui]
    tss = np.bincount(ui, weights=tc * tc)
    parity = np.where(yr % 2 == 0, 1.0, -1.0)
    pc = parity - (np.bincount(ui, weights=parity) / uc)[ui]
    pss = np.bincount(ui, weights=pc * pc)
    pt = np.bincount(ui, weights=pc * tc)
    pc_coef = np.divide(pt, pss, out=np.zeros_like(pt), where=pss > 0)
    t_after_p = tc - pc * pc_coef[ui]
    tp_ss = np.bincount(ui, weights=t_after_p * t_after_p)
    # If a local trend is exactly spanned by parity (e.g. two observed years),
    # do not magnify floating-point residue into a nuisance direction.
    valid_tp = tp_ss > 1e-24 * np.maximum(tss, 1.0)
    t_after_p[~valid_tp[ui]] = 0.0
    tp_ss[~valid_tp] = 0.0
    for j, name in enumerate(ul):
        if len(np.unique(s[ui == j])) != 1:
            raise ValueError(f"unit {name} maps to more than one state")
    return {"unit": u, "state": s, "year": yr, "ul": ul, "ui": ui,
            "sl": sl, "si": si, "yl": yl, "yi": yi, "syi": syi,
            "uc": uc, "syc": syc, "tc": tc, "tss": tss,
            "pc": pc, "pss": pss, "t_after_p": t_after_p, "tp_ss": tp_ss}


def _demean(a: np.ndarray, ids: np.ndarray, counts: np.ndarray) -> None:
    for j in range(a.shape[1]):
        means = np.bincount(ids, weights=a[:, j], minlength=len(counts)) / counts
        a[:, j] -= means[ids]


def _remove_unit_block(a: np.ndarray, g: dict[str, Any], trends: bool, parity: bool) -> None:
    _demean(a, g["ui"], g["uc"])
    bases = []
    if parity:
        bases.append((g["pc"], g["pss"]))
    if trends:
        bases.append((g["t_after_p"], g["tp_ss"]) if parity else (g["tc"], g["tss"]))
    for basis, ss in bases:
        nz = ss > 0
        for j in range(a.shape[1]):
            beta = np.zeros(len(g["uc"]), dtype=float)
            sums = np.bincount(g["ui"], weights=basis * a[:, j],
                               minlength=len(g["uc"]))
            beta[nz] = sums[nz] / ss[nz]
            a[:, j] -= basis * beta[g["ui"]]


def _orthogonality(a: np.ndarray, g: dict[str, Any], trends: bool, parity: bool) -> dict[str, float]:
    max_unit = max_sy = max_trend = max_parity = 0.0
    for j in range(a.shape[1]):
        max_unit = max(max_unit, float(np.max(np.abs(
            np.bincount(g["ui"], weights=a[:, j]) / g["uc"]))))
        max_sy = max(max_sy, float(np.max(np.abs(
            np.bincount(g["syi"], weights=a[:, j]) / g["syc"]))))
        if trends:
            nz = g["tss"] > 0
            v = np.bincount(g["ui"], weights=g["tc"] * a[:, j])
            if np.any(nz):
                max_trend = max(max_trend, float(np.max(
                    np.abs(v[nz]) / np.sqrt(g["tss"][nz] * g["uc"][nz]))))
        if parity:
            nz = g["pss"] > 0
            v = np.bincount(g["ui"], weights=g["pc"] * a[:, j])
            if np.any(nz):
                max_parity = max(max_parity, float(np.max(
                    np.abs(v[nz]) / np.sqrt(g["pss"][nz] * g["uc"][nz]))))
    return {"unit_mean_max_abs_scaled": max_unit,
            "state_year_mean_max_abs_scaled": max_sy,
            "unit_trend_projection_max_abs_scaled": max_trend,
            "unit_parity_projection_max_abs_scaled": max_parity}


def residualize(A: Any, unit: Any, state: Any, year: Any,
                unit_trends: bool = False, unit_parity: bool = False, tol: float = PROJECTION_TOL,
                max_iter: int = MAX_PROJECTION_ITER) -> tuple[np.ndarray, dict[str, Any]]:
    """Project onto the complement of unit and state-by-calendar-year effects.

    Unit trends, when requested, form one intercept-plus-linear-trend projection
    block. Alternating orthogonal projections accommodate an unbalanced panel.
    Each input column is internally scaled by its RMS to make convergence checks
    independent of measurement units. Missing values must be removed by caller.
    The returned array preserves the input dimensionality (one or two dimensions).
    """
    a = np.asarray(A, dtype=float)
    one = a.ndim == 1
    if one:
        a = a[:, None]
    if a.ndim != 2 or not a.shape[0] or not a.shape[1]:
        raise ValueError("A must be a nonempty one- or two-dimensional numeric array")
    if not np.all(np.isfinite(a)):
        raise ValueError("A contains missing or nonfinite values; caller must apply an explicit mask")
    if tol <= 0 or max_iter < 1:
        raise ValueError("tol must be positive and max_iter at least one")
    g = _groups(unit, state, year, len(a))
    scale = np.sqrt(np.mean(a * a, axis=0))
    scale[scale == 0] = 1.0
    if not np.all(np.isfinite(scale)):
        raise ValueError("A magnitude exceeds the supported floating-point range")
    r = a.copy() / scale
    converged = False
    delta = float("inf")
    for iteration in range(1, max_iter + 1):
        old = r.copy()
        _remove_unit_block(r, g, unit_trends, unit_parity)
        _demean(r, g["syi"], g["syc"])
        delta = float(np.max(np.abs(r - old)))
        if delta <= tol:
            checks = _orthogonality(r, g, unit_trends, unit_parity)
            if max(checks.values()) <= tol * 10:
                converged = True
                break
    checks = _orthogonality(r, g, unit_trends, unit_parity)
    diagnostics: dict[str, Any] = {
        "converged": converged, "iterations": iteration,
        "last_max_absolute_change_scaled": delta,
        "tolerance": tol, "maximum_iterations": max_iter,
        "n_observations": len(r), "n_units": len(g["ul"]),
        "n_states": len(g["sl"]), "n_calendar_years": len(g["yl"]),
        "n_state_year_cells": len(g["syc"]), "unit_linear_trends": bool(unit_trends),
        "unit_specific_fixed_parity": bool(unit_parity),
        "column_input_rms": scale.tolist(), **checks,
    }
    out = r * scale
    return (out[:, 0] if one else out), diagnostics


def _share_summary(energy: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    names, ids = np.unique(labels.astype(str), return_inverse=True)
    values = np.bincount(ids, weights=energy)
    total = float(np.sum(values))
    if total <= 0:
        return {"groups": len(names), "effective_groups": 0.0,
                "maximum_share": None, "top_ten": []}
    shares = values / total
    order = np.argsort(-shares, kind="stable")
    return {"groups": len(names),
            "effective_groups": float(1.0 / np.sum(shares * shares)),
            "maximum_share": float(np.max(shares)),
            "top_ten": [{"group": str(names[i]), "share": float(shares[i])}
                        for i in order[:10]]}


def prepare_design(X: Any, unit: Any, state: Any, year: Any, names: list[str],
                   unit_trends: bool = False, unit_parity: bool = False,
                   interaction_name: str | None = "D0_x_history") -> tuple[np.ndarray, dict[str, Any]]:
    """Validate and project X without inspecting outcomes or dropping columns."""
    x = np.asarray(X, dtype=float)
    if x.ndim != 2 or not x.shape[0] or not x.shape[1]:
        raise ValueError("X must be a nonempty two-dimensional numeric array")
    if len(names) != x.shape[1] or len(set(names)) != len(names):
        raise ValueError("names must be unique and match X columns")
    if not np.all(np.isfinite(x)):
        raise ValueError("X contains missing or nonfinite values")
    xw, projection = residualize(x, unit, state, year, unit_trends=unit_trends, unit_parity=unit_parity)
    raw_rms = np.sqrt(np.mean(x * x, axis=0))
    within_rms = np.sqrt(np.mean(xw * xw, axis=0))
    ratios = np.divide(within_rms, raw_rms, out=np.zeros_like(within_rms), where=raw_rms > 0)
    absorbed = [names[j] for j in range(len(names))
                if raw_rms[j] == 0 or ratios[j] <= ABSORPTION_TOL]
    safe = within_rms.copy()
    safe[safe == 0] = 1.0
    z = xw / safe
    singular = np.linalg.svd(z, full_matrices=False, compute_uv=False)
    largest = float(singular[0]) if len(singular) else 0.0
    rank = int(np.sum(singular > RANK_RELATIVE_TOL * largest)) if largest else 0
    condition = (float(singular[0] / singular[-1])
                 if len(singular) == len(names) and singular[-1] > 0 else None)
    reasons = []
    if not projection["converged"]:
        reasons.append("fixed_effect_projection_did_not_converge")
    if absorbed:
        reasons.append("one_or_more_regressors_are_absorbed_or_numerically_negligible_after_fixed_effects")
    if rank < len(names):
        reasons.append("scaled_within_design_is_rank_deficient_at_declared_tolerance")
    status = "ESTIMABLE" if not reasons else "NONESTIMABLE"
    result: dict[str, Any] = {
        "status": status, "reasons": reasons, "columns": list(names),
        "n_columns": len(names), "rank": rank,
        "rank_relative_singular_value_tolerance": RANK_RELATIVE_TOL,
        "absorption_within_to_input_rms_tolerance": ABSORPTION_TOL,
        "scaled_singular_values": singular.tolist(),
        "scaled_condition_number": condition, "absorbed_columns": absorbed,
        "input_rms": dict(zip(names, raw_rms.tolist())),
        "within_rms": dict(zip(names, within_rms.tolist())),
        "within_to_input_rms_ratio": dict(zip(names, ratios.tolist())),
        "projection": projection,
    }
    if interaction_name is not None and interaction_name in names and status == "ESTIMABLE":
        j = names.index(interaction_name)
        other = np.delete(z, j, axis=1)
        if other.shape[1]:
            partial = z[:, j] - other @ np.linalg.lstsq(
                other, z[:, j], rcond=RANK_RELATIVE_TOL)[0]
        else:
            partial = z[:, j].copy()
        energy = partial * partial
        g = _groups(unit, state, year, len(x))
        result["interaction_identification"] = {
            "name": interaction_name,
            "partial_to_within_sum_squares_ratio": float(np.sum(energy) / np.sum(z[:, j] ** 2)),
            "partial_sum_squares_original_units": float(np.sum(energy) * within_rms[j] ** 2),
            "unit": _share_summary(energy, g["unit"]),
            "state": _share_summary(energy, g["state"]),
            "year": _share_summary(energy, g["year"]),
            "interpretation": "Shares of residualized interaction information, not independent sample counts.",
        }
    return xw, result


def calendar_score_hac(X_within: Any, residual: Any, year: Any, lag: int) -> dict[str, Any]:
    """Return calendar-score HAC covariance for a full-rank within design.

    Uses actual calendar-year distances; absent years contribute zero scores.
    Bread is (X'X)^-1 and meat uses summed scores, so there is NO N or T factor
    to add or remove. Public mainly for independent testing/reuse.
    """
    x = np.asarray(X_within, dtype=float)
    e = np.asarray(residual, dtype=float)
    yrs0 = np.asarray(year, dtype=float)
    if x.ndim != 2 or e.ndim != 1 or len(e) != len(x) or len(yrs0) != len(x):
        raise ValueError("HAC inputs have incompatible shapes")
    if len(x) == 0 or x.shape[1] == 0 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(e)):
        raise ValueError("HAC inputs must be nonempty and finite")
    if not np.all(np.isfinite(yrs0)) or not np.all(yrs0 == np.floor(yrs0)):
        raise ValueError("HAC years must be finite integers")
    if not isinstance(lag, (int, np.integer)) or lag < 0:
        raise ValueError("HAC lag must be a nonnegative integer")
    yrs = yrs0.astype(np.int64)
    unique_years = np.unique(yrs)
    # Guard against accidental malformed calendar identifiers consuming memory.
    first_year, last_year = int(np.min(yrs)), int(np.max(yrs))
    if last_year - first_year + 1 > 10000:
        raise ValueError("Calendar span exceeds 10000 years")
    years_full = np.arange(first_year, last_year + 1)
    scale = np.sqrt(np.mean(x * x, axis=0))
    if np.any(scale <= 0):
        raise ValueError("HAC design has a zero column")
    z = x / scale
    _, sv, vt = np.linalg.svd(z, full_matrices=False)
    if len(sv) < x.shape[1] or sv[-1] <= RANK_RELATIVE_TOL * sv[0]:
        raise ValueError("HAC design is rank deficient at the declared tolerance")
    bread = (vt.T / (sv * sv)) @ vt
    scores = np.zeros((len(years_full), x.shape[1]), dtype=float)
    for j in range(x.shape[1]):
        scores[:, j] = np.bincount(yrs - years_full[0], weights=z[:, j] * e,
                                   minlength=len(years_full))
    meat = scores.T @ scores
    for ell in range(1, min(lag, len(scores) - 1) + 1):
        weight = 1.0 - ell / (lag + 1.0)
        cross = scores[ell:].T @ scores[:-ell]
        meat += weight * (cross + cross.T)
    cov_scaled = bread @ meat @ bread
    cov = cov_scaled / scale[:, None] / scale[None, :]
    cov = (cov + cov.T) / 2.0
    eig = np.linalg.eigvalsh(cov)
    # Roundoff is tolerated only relative to the covariance's own magnitude.
    magnitude = float(np.max(np.abs(cov)))
    psd_tolerance = 1e-10 * max(magnitude, np.finfo(float).tiny)
    if float(eig[0]) < -psd_tolerance:
        raise ArithmeticError("Bartlett HAC covariance is materially non-positive-semidefinite")
    diagonal = np.diag(cov)
    if np.any(diagonal < -psd_tolerance):
        raise ArithmeticError("Bartlett HAC has a materially negative marginal variance")
    se = np.sqrt(np.maximum(diagonal, 0))
    return {
        "method": "calendar_year_summed_score_Bartlett_HAC",
        "lag_years": int(lag), "covariance": cov.tolist(), "standard_errors": se.tolist(),
        "n_observed_score_years": len(unique_years),
        "calendar_span_years": len(years_full), "calendar_years": years_full.tolist(),
        "absent_years_zero_scores": [int(y) for y in years_full if y not in set(unique_years.tolist())],
        "finite_sample_correction": "none", "normalization": "summed scores; (X'X)^-1 bread on both sides",
        "minimum_covariance_eigenvalue": float(eig[0]),
        "cross_section_dependence": "unrestricted within each year; all cross-unit score products retained",
        "temporal_dependence": "Bartlett weights through selected calendar lag only",
        "limitations": "Exploratory asymptotic approximation with few calendar years; omitted long-lag dependence and nonstationarity can invalidate coverage.",
    }


def fit_model(X: Any, y: Any, unit: Any, state: Any, year: Any, names: list[str],
              unit_trends: bool = False, unit_parity: bool = False,
              interaction_name: str | None = "D0_x_history") -> dict[str, Any]:
    """Fit fixed-effects OLS and two predeclared exploratory HAC references.

    Missing observations are never silently dropped. Design is assessed before
    any outcome projection. A NONESTIMABLE result contains no estimated effects.
    """
    x = np.asarray(X, dtype=float)
    outcome = np.asarray(y, dtype=float)
    if outcome.ndim != 1 or x.ndim != 2 or len(outcome) != len(x):
        raise ValueError("y must be one dimensional and match X rows")
    if not np.all(np.isfinite(outcome)):
        raise ValueError("y contains missing or nonfinite values; caller must apply an explicit mask")
    xw, design = prepare_design(x, unit, state, year, names,
                                unit_trends=unit_trends, unit_parity=unit_parity, interaction_name=interaction_name)
    result: dict[str, Any] = {"status": design["status"], "design": design,
        "coefficient_order": list(names), "unit_fixed_effects": True,
        "state_by_year_fixed_effects": True, "unit_linear_trends": bool(unit_trends),
        "unit_specific_fixed_parity": bool(unit_parity),
        "outcome_zero_rows_retained": int(np.sum(outcome == 0)),
        "interpretation": "Conditional association in the specified reported-unit panel, not an identified causal recovery effect.",
        "p_values_computed": False, "significance_stars_computed": False}
    if design["status"] != "ESTIMABLE":
        return result
    yw, y_projection = residualize(outcome, unit, state, year, unit_trends=unit_trends, unit_parity=unit_parity)
    result["outcome_projection"] = y_projection
    if not y_projection["converged"]:
        result["status"] = "NONESTIMABLE"
        result["failure_reason"] = "outcome_fixed_effect_projection_did_not_converge"
        return result
    scale = np.sqrt(np.mean(xw * xw, axis=0))
    z = xw / scale
    b_scaled, _, rank, singular = np.linalg.lstsq(z, yw, rcond=RANK_RELATIVE_TOL)
    if rank != len(names):
        result["status"] = "NONESTIMABLE"
        result["failure_reason"] = "fit_rank_disagrees_with_outcome_free_design_rank"
        return result
    b = b_scaled / scale
    e = yw - z @ b_scaled
    rss = float(e @ e)
    tss = float(yw @ yw)
    normal = z.T @ e
    relative_normal = float(np.max(np.abs(normal)) / max(
        float(np.linalg.norm(z, axis=0).max() * np.linalg.norm(yw)), np.finfo(float).tiny))
    result["status"] = "FIT_COMPLETE_EXPLORATORY"
    result["coefficients"] = dict(zip(names, b.tolist()))
    result["diagnostics"] = {
        "n_observations": len(x), "residual_sum_squares": rss,
        "within_outcome_sum_squares": tss,
        "within_r_squared": float(1 - rss / tss) if tss > 0 else None,
        "normal_equation_scaled_max_abs": float(np.max(np.abs(normal))),
        "normal_equation_relative_max_abs": relative_normal,
        "least_squares_singular_values": singular.tolist(),
        "outcome_min": float(np.min(outcome)), "outcome_max": float(np.max(outcome)),
    }
    inference = {}
    for lag in (2, 4):
        h = calendar_score_hac(xw, e, year, lag)
        ses = np.asarray(h["standard_errors"])
        h["reference_level"] = 0.95
        h["critical_value"] = NORMAL_975
        h["reference_distribution"] = "standard normal; approximate, not a finite-sample coverage guarantee"
        h["intervals"] = {name: {"estimate": float(b[j]), "standard_error": float(ses[j]),
                                  "lower": float(b[j] - NORMAL_975 * ses[j]),
                                  "upper": float(b[j] + NORMAL_975 * ses[j])}
                          for j, name in enumerate(names)}
        inference[f"HAC_L{lag}"] = h
    result["uncertainty"] = inference
    return result
