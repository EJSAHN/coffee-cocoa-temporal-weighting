"""Inspect a linear contrast without estimating a statistical model.

The contrast is sum(weight * outcome) over unit/year records.  All calculations
use fractions.  Decimal strings denote exact decimals, not binary floats.
Diagnostics describe the supplied weights; they do not establish causal
identification, covariance validity, or sampling independence.
"""

from collections import defaultdict
from fractions import Fraction
import math


class DiagnosticInputError(ValueError):
    """Invalid or internally inconsistent observation mappings."""


_MISSING = {"", "NA", "N/A", "NULL", "NONE", "NAN"}


def _missing(value):
    return value is None or (isinstance(value, str) and value.strip().upper() in _MISSING)


def _fraction(value, label):
    if isinstance(value, bool) or _missing(value):
        raise DiagnosticInputError(f"{label}: a finite number is required")
    try:
        result = Fraction(str(value).strip())
    except (ValueError, ZeroDivisionError, TypeError) as exc:
        raise DiagnosticInputError(f"{label}: invalid finite number {value!r}") from exc
    return result


def _integer(value, label):
    number = _fraction(value, label)
    if number.denominator != 1:
        raise DiagnosticInputError(f"{label}: an integer year is required")
    return number.numerator


def _identity(value, label):
    if value is None or isinstance(value, bool) or not str(value).strip():
        raise DiagnosticInputError(f"{label}: a nonempty identifier is required")
    return str(value).strip()


def _number(prefix, value):
    """Exact representation is authoritative; a nonfinite float is never emitted."""
    if value is None:
        return {prefix + "_exact": None, prefix + "_value": None}
    try:
        approximate = float(value)
    except OverflowError:
        approximate = None
    if approximate is not None and not math.isfinite(approximate):
        approximate = None
    return {prefix + "_exact": str(value), prefix + "_value": approximate}


def diagnose(records, origin=2006):
    """Return exact, JSON-serializable diagnostics for tidy weighted records.

    Required keys are ``unit``, ``year`` and ``weight``.  Optional keys are
    ``contrast`` (default ``contrast``), ``group`` and ``outcome``.  Years must
    be integers; weights and outcomes accept integer, decimal or fraction
    strings.  The four time bases are 1, u, u**2 and (-1)**u, where u=year-origin.

    Duplicate unit/year records in a contrast are consolidated by adding
    weights.  Their nonmissing outcomes must agree exactly.  Missing outcomes
    are never filled, including at zero weights or within duplicated records:
    reconstruction is available only when all original rows have outcomes.
    Group is a unit attribute and must be consistent across every input row.

    Response bounds are deterministic sensitivity bounds.  For a bound B and
    coefficient amplitude M, the absolute response is at most M*B under the
    explicitly reported coefficient scope.  They are not confidence intervals.
    """
    origin = _integer(origin, "origin")
    by_key = {}
    group_by_unit = {}
    original_counts = defaultdict(int)
    missing_counts = defaultdict(int)
    missing_nonzero_counts = defaultdict(int)
    outcome_columns = defaultdict(bool)
    for row_number, row in enumerate(records, 1):
        if not hasattr(row, "get"):
            raise DiagnosticInputError(f"row {row_number}: expected a mapping")
        label = f"row {row_number}"
        unit = _identity(row.get("unit"), label + " unit")
        contrast = _identity(row.get("contrast", "contrast"), label + " contrast")
        year = _integer(row.get("year"), label + " year")
        weight = _fraction(row.get("weight"), label + " weight")
        group_raw = row.get("group")
        group = None if group_raw is None or str(group_raw).strip() == "" else _identity(group_raw, label + " group")
        if unit in group_by_unit and group_by_unit[unit] != group:
            raise DiagnosticInputError(f"{label}: inconsistent group identity for unit {unit!r}")
        group_by_unit[unit] = group
        outcome_columns[contrast] = outcome_columns[contrast] or "outcome" in row
        missing = _missing(row.get("outcome"))
        outcome = None if missing else _fraction(row["outcome"], label + " outcome")
        original_counts[contrast] += 1
        missing_counts[contrast] += int(missing)
        missing_nonzero_counts[contrast] += int(missing and weight != 0)
        key = (contrast, unit, year)
        if key not in by_key:
            by_key[key] = {"weight": Fraction(0), "outcome": outcome,
                           "input_rows": 0, "missing_outcome_rows": 0}
        cell = by_key[key]
        if outcome is not None and cell["outcome"] is not None and outcome != cell["outcome"]:
            raise DiagnosticInputError(f"{label}: conflicting outcomes for {key!r}")
        if outcome is not None:
            cell["outcome"] = outcome
        cell["weight"] += weight
        cell["input_rows"] += 1
        cell["missing_outcome_rows"] += int(missing)
    if not by_key:
        raise DiagnosticInputError("No input records")

    output = {
        "schema_version": "1.0.0",
        "origin": origin,
        "settings": {
            "arithmetic": "exact rational; *_exact fields are authoritative",
            "duplicate_policy": "add weights; retain zero sums; reject conflicting outcomes",
            "outcome_policy": "no filling; all original outcomes required for reconstruction",
            "basis": {"constant": "1", "linear": "year - origin",
                      "quadratic": "(year - origin)^2", "parity": "(-1)^(year - origin)"},
            "response_bounds": "absolute response <= coefficient_amplitude * reported bound",
            "interpretation": "linear-contrast algebra and deterministic sensitivity; not causal or sampling inference",
        },
        "contrast_summary": [], "unit_moments": [], "year_weights": [],
        "group_year_weights": [], "response_bounds": [], "consolidated_weights": [],
    }
    contrasts = sorted(original_counts)
    for contrast in contrasts:
        selected = [(unit, year, cell) for (name, unit, year), cell in sorted(by_key.items()) if name == contrast]
        units = sorted({unit for unit, _, _ in selected})
        years = sorted({year for _, year, _ in selected})
        group_complete = all(group_by_unit[unit] is not None for unit in units)
        moments = {unit: {basis: Fraction(0) for basis in ("constant", "linear", "quadratic", "parity")} for unit in units}
        unit_l1 = defaultdict(Fraction)
        unit_nonzero = defaultdict(int)
        unit_rows = defaultdict(int)
        year_weight = defaultdict(Fraction)
        group_year_weight = defaultdict(Fraction)
        reconstruction = Fraction(0)
        for unit, year, cell in selected:
            weight = cell["weight"]
            centered = year - origin
            bases = {"constant": 1, "linear": centered, "quadratic": centered ** 2,
                     "parity": 1 if centered % 2 == 0 else -1}
            for basis, value in bases.items():
                moments[unit][basis] += weight * value
            unit_l1[unit] += abs(weight)
            unit_nonzero[unit] += int(weight != 0)
            unit_rows[unit] += 1
            year_weight[year] += weight
            if group_complete:
                group_year_weight[(group_by_unit[unit], year)] += weight
            complete_cell = cell["missing_outcome_rows"] == 0
            contribution = weight * cell["outcome"] if complete_cell else None
            if contribution is not None:
                reconstruction += contribution
            output["consolidated_weights"].append({
                "contrast": contrast, "unit": unit, "group": group_by_unit[unit], "year": year,
                **_number("weight", weight), "input_rows": cell["input_rows"],
                "missing_outcome_rows": cell["missing_outcome_rows"],
                "outcome_status": "complete" if complete_cell else "missing_in_original_rows",
                **_number("outcome", cell["outcome"] if complete_cell else None),
                **_number("contribution", contribution),
            })
        for unit in units:
            unit_result = {"contrast": contrast, "unit": unit, "group": group_by_unit[unit],
                           "consolidated_rows": unit_rows[unit], "nonzero_rows": unit_nonzero[unit],
                           "all_weights_cancelled": unit_nonzero[unit] == 0,
                           **_number("weight_l1", unit_l1[unit])}
            for basis, value in moments[unit].items():
                unit_result.update(_number(basis, value))
                unit_result[basis + "_cancelled"] = value == 0
            output["unit_moments"].append(unit_result)
        for year, weight in sorted(year_weight.items()):
            output["year_weights"].append({"contrast": contrast, "year": year, **_number("weight", weight)})
        for (group, year), weight in sorted(group_year_weight.items()):
            output["group_year_weights"].append({"contrast": contrast, "group": group, "year": year,
                                                  **_number("weight", weight)})
        def add_bound(basis, scope, bound):
            output["response_bounds"].append({"contrast": contrast, "basis": basis,
                "coefficient_scope": scope, **_number("response_per_unit_amplitude", bound)})
        for basis in ("constant", "linear", "quadratic", "parity"):
            add_bound(basis, "independently_bounded_unit_coefficients", sum((abs(moments[unit][basis]) for unit in units), Fraction(0)))
            add_bound(basis, "single_common_coefficient", abs(sum((moments[unit][basis] for unit in units), Fraction(0))))
        add_bound("arbitrary_unit_year_values", "one_bounded_value_per_unit_year", sum(unit_l1.values(), Fraction(0)))
        add_bound("arbitrary_common_year_values", "one_bounded_value_per_year", sum((abs(w) for w in year_weight.values()), Fraction(0)))
        if group_complete:
            add_bound("arbitrary_group_year_values", "one_bounded_value_per_group_year", sum((abs(w) for w in group_year_weight.values()), Fraction(0)))
        if not outcome_columns[contrast]:
            status = "not_requested"
        elif missing_counts[contrast] == 0:
            status = "complete"
        elif missing_nonzero_counts[contrast] == 0:
            status = "incomplete_missing_outcomes_on_zero_weight_rows"
        else:
            status = "incomplete_missing_outcomes"
        summary = {
            "contrast": contrast, "input_rows": original_counts[contrast],
            "consolidated_rows": len(selected), "duplicate_rows_consolidated": original_counts[contrast] - len(selected),
            "units": len(units), "years": len(years), "first_year": min(years), "last_year": max(years),
            "nonzero_rows": sum(unit_nonzero.values()), "zero_rows_retained": len(selected) - sum(unit_nonzero.values()),
            "fully_cancelled_units": [unit for unit in units if unit_nonzero[unit] == 0],
            "group_year_status": "complete" if group_complete else "unavailable_incomplete_group_mapping",
            "outcome_status": status, "missing_outcome_rows": missing_counts[contrast],
            "missing_outcome_rows_with_nonzero_input_weight": missing_nonzero_counts[contrast],
            **_number("reconstructed_contrast", reconstruction if status == "complete" else None),
            **_number("weight_sum", sum(year_weight.values(), Fraction(0))),
            **_number("weight_l1", sum(unit_l1.values(), Fraction(0))),
        }
        for basis in ("constant", "linear", "quadratic", "parity"):
            summary[basis + "_cancelled_for_every_unit"] = all(moments[unit][basis] == 0 for unit in units)
        output["contrast_summary"].append(summary)
    return output
