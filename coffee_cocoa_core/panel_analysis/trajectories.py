"""Reported production/area trajectories and exact descriptive allocations.

No regression, outcome-based exclusions, yield imputations, or causal allocation.
Annual tables include 2002--2025. Transitions include 2006->2007--2024->2025.
The full fixed cohort remains the denominator for coverage counts; numeric
aggregates state their observed support explicitly. Only Python's stdlib is used.
"""
from __future__ import annotations

import math
from collections import defaultdict


ANNUAL_YEARS = tuple(range(2002, 2026))
TRANSITION_END_YEARS = tuple(range(2007, 2026))
SELECTED_POSTHOC_CASES = frozenset({('cacao', '1500859'), ('cacao', '1505437')})


def _number(value, field):
    if value is None or value == '':
        return None
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError('Invalid nonnegative reported field: ' + field)
    return result


def _mean(values):
    values = [v for v in values if v is not None]
    return math.fsum(values) / len(values) if values else None


def _sum_or_missing(values):
    values = [v for v in values if v is not None]
    return math.fsum(values) if values else None


def _pilot(value):
    if isinstance(value, str):
        return value.strip().lower() in {'true', '1', 'yes'}
    return bool(value)


def _annual_row(row):
    q, i, h = (_number(row.get(k), k) for k in ('Q_tonnes', 'Iha', 'Hha'))
    a0 = _number(row.get('A0'), 'A0')
    if a0 is None or a0 <= 0:
        raise ValueError('Frozen baseline area must be strictly positive')
    year = int(row['year'])
    if float(row['year']) != year or year not in ANNUAL_YEARS:
        raise ValueError('Trajectory year outside the frozen 2002--2025 range')
    territory, crop, state = (str(row[k]).strip() for k in ('territory', 'crop', 'state'))
    if not all((territory, crop, state)):
        raise ValueError('Missing trajectory identity')
    return {
        'crop': crop, 'territory': territory, 'state': state, 'year': year,
        'pilot': _pilot(row.get('pilot', False)), 'A0': a0,
        'Q_tonnes': q, 'Iha': i, 'Hha': h,
        'production_kg_fixed_ha': None if q is None else 1000.0*q/a0,
        'intended_area_ratio_A0': None if i is None else i/a0,
        'harvested_area_ratio_A0': None if h is None else h/a0,
        'reported_unharvested_area_ratio_A0': None if i is None or h is None else (i-h)/a0,
        'yield_kg_harvested_ha': 1000.0*q/h if q is not None and h is not None and h > 0 else None,
        'harvested_fraction_of_intended': h/i if i is not None and i > 0 and h is not None else None,
        'common_qih_observed': q is not None and i is not None and h is not None,
        'zero_harvested_area': h == 0 if h is not None else False,
        'positive_production_zero_harvested_area': q is not None and q > 0 and h == 0,
        'harvested_exceeds_intended': h is not None and i is not None and h > i,
    }


def _annual_summary(crop, state, year, cohort_count, rows):
    common = [r for r in rows if r['common_qih_observed']]
    qsum = _sum_or_missing([r['Q_tonnes'] for r in common])
    isum = _sum_or_missing([r['Iha'] for r in common])
    hsum = _sum_or_missing([r['Hha'] for r in common])
    a0sum = _sum_or_missing([r['A0'] for r in common])
    qrows = [r for r in rows if r['Q_tonnes'] is not None]
    return {
        'crop': crop, 'geography_scope': 'crop' if state == 'ALL' else 'state',
        'state': state, 'year': year, 'fixed_cohort_units': cohort_count,
        'present_source_rows': len(rows), 'absent_source_rows': cohort_count-len(rows),
        'production_observed_units': len(qrows),
        'production_missing_or_absent_units': cohort_count-len(qrows),
        'intended_area_observed_units': sum(r['Iha'] is not None for r in rows),
        'harvested_area_observed_units': sum(r['Hha'] is not None for r in rows),
        'common_qih_observed_units': len(common),
        'common_qih_missing_or_absent_units': cohort_count-len(common),
        'production_observed_equal_unit_mean_kg_fixed_ha': _mean([r['production_kg_fixed_ha'] for r in qrows]),
        'common_qih_sum_Q_tonnes': qsum, 'common_qih_sum_Iha': isum,
        'common_qih_sum_Hha': hsum, 'common_qih_sum_A0': a0sum,
        'common_qih_equal_unit_mean_kg_fixed_ha': _mean([r['production_kg_fixed_ha'] for r in common]),
        'common_qih_baseline_area_weighted_kg_fixed_ha': None if a0sum is None else 1000.0*qsum/a0sum,
        'common_qih_equal_unit_mean_intended_ratio_A0': _mean([r['intended_area_ratio_A0'] for r in common]),
        'common_qih_equal_unit_mean_harvested_ratio_A0': _mean([r['harvested_area_ratio_A0'] for r in common]),
        'common_qih_baseline_area_weighted_intended_ratio_A0': None if a0sum is None else isum/a0sum,
        'common_qih_baseline_area_weighted_harvested_ratio_A0': None if a0sum is None else hsum/a0sum,
        'common_qih_aggregate_reported_kg_per_harvested_ha': 1000.0*qsum/hsum if hsum is not None and hsum > 0 else None,
        'common_qih_zero_harvested_units': sum(r['zero_harvested_area'] for r in common),
        'positive_production_zero_harvested_units': sum(r['positive_production_zero_harvested_area'] for r in rows),
        'harvested_exceeds_intended_units': sum(r['harvested_exceeds_intended'] for r in rows),
        'aggregation_note': 'Observed supports explicit; fixed-cohort subset, not Brazil total or regression estimand',
    }


def _transition(crop, territory, state, a0, year, before, after):
    def value(row, key):
        return None if row is None else row[key]
    q0, q1 = value(before, 'Q_tonnes'), value(after, 'Q_tonnes')
    h0, h1 = value(before, 'Hha'), value(after, 'Hha')
    i0, i1 = value(before, 'Iha'), value(after, 'Iha')
    delta = None if q0 is None or q1 is None else 1000.0*(q1-q0)/a0
    eligible = delta is not None and h0 is not None and h1 is not None and h0 > 0 and h1 > 0
    area = productivity = unallocated = residual = None
    if eligible:
        hratio0, hratio1 = h0/a0, h1/a0
        y0, y1 = 1000.0*q0/h0, 1000.0*q1/h1
        area = (hratio1-hratio0)*(y1+y0)/2.0
        productivity = (y1-y0)*(hratio1+hratio0)/2.0
        unallocated = 0.0
        residual = delta-area-productivity
        status = 'ALLOCATED_POSITIVE_HARVESTED_BOTH_ENDPOINTS'
    elif delta is not None:
        unallocated = delta
        status = 'UNALLOCATED_MISSING_OR_ZERO_HARVESTED_AREA'
    else:
        status = 'MISSING_PRODUCTION_CHANGE'
    return {
        'crop': crop, 'territory': territory, 'state': state,
        'start_year': year-1, 'end_year': year, 'A0': a0,
        'source_row_start_present': before is not None,
        'source_row_end_present': after is not None,
        'Q_start_tonnes': q0, 'Q_end_tonnes': q1,
        'I_start_ha': i0, 'I_end_ha': i1, 'H_start_ha': h0, 'H_end_ha': h1,
        'yield_start_kg_harvested_ha': 1000.0*q0/h0 if q0 is not None and h0 is not None and h0 > 0 else None,
        'yield_end_kg_harvested_ha': 1000.0*q1/h1 if q1 is not None and h1 is not None and h1 > 0 else None,
        'observed_delta_kg_fixed_ha': delta,
        'intended_area_ratio_change_A0': None if i0 is None or i1 is None else (i1-i0)/a0,
        'harvested_area_ratio_change_A0': None if h0 is None or h1 is None else (h1-h0)/a0,
        'allocation_status': status, 'allocation_eligible': eligible,
        'area_midpoint_allocation_kg_fixed_ha': area,
        'productivity_midpoint_allocation_kg_fixed_ha': productivity,
        'unallocated_observed_change_kg_fixed_ha': unallocated,
        'allocation_identity_residual_kg_fixed_ha': residual,
    }


def _transition_summary(crop, state, year, rows):
    observed = [r for r in rows if r['observed_delta_kg_fixed_ha'] is not None]
    allocated = [r for r in observed if r['allocation_eligible']]
    unallocated = [r for r in observed if not r['allocation_eligible']]
    # A zero component sum for an empty allocation category means no allocated
    # contribution, NOT an imputed outcome. Entirely missing outcomes stay None.
    delta_sum = _sum_or_missing([r['observed_delta_kg_fixed_ha'] for r in observed])
    area_sum = math.fsum(r['area_midpoint_allocation_kg_fixed_ha'] for r in allocated) if observed else None
    prod_sum = math.fsum(r['productivity_midpoint_allocation_kg_fixed_ha'] for r in allocated) if observed else None
    unalloc_sum = math.fsum(r['unallocated_observed_change_kg_fixed_ha'] for r in unallocated) if observed else None
    abs_unalloc = math.fsum(abs(r['unallocated_observed_change_kg_fixed_ha']) for r in unallocated) if observed else None
    denominator = len(observed)
    return {
        'crop': crop, 'geography_scope': 'crop' if state == 'ALL' else 'state',
        'state': state, 'start_year': year-1, 'end_year': year,
        'fixed_cohort_units': len(rows), 'production_change_observed_units': denominator,
        'production_change_missing_units': len(rows)-denominator,
        'allocation_eligible_units': len(allocated), 'unallocated_observed_units': len(unallocated),
        'sum_observed_delta_kg_fixed_ha': delta_sum,
        'sum_area_midpoint_allocation_kg_fixed_ha': area_sum,
        'sum_productivity_midpoint_allocation_kg_fixed_ha': prod_sum,
        'sum_unallocated_observed_change_kg_fixed_ha': unalloc_sum,
        'sum_absolute_unallocated_observed_change_kg_fixed_ha': abs_unalloc,
        'mean_observed_delta_kg_fixed_ha': delta_sum/denominator if denominator else None,
        'mean_area_allocation_same_observed_denominator': area_sum/denominator if denominator else None,
        'mean_productivity_allocation_same_observed_denominator': prod_sum/denominator if denominator else None,
        'mean_unallocated_change_same_observed_denominator': unalloc_sum/denominator if denominator else None,
        'allocation_identity_residual_kg_fixed_ha': delta_sum-area_sum-prod_sum-unalloc_sum if denominator else None,
        'allocation_note': 'Exact signed arithmetic allocation; no causal mechanism or biological attribution',
    }


def _level_components(units, index, transition_index):
    """Accumulate valid midpoint allocations from the fixed 2006 reference.

    The residual includes undefined-yield changes and net movement across
    missing-production gaps. Its meaning is arithmetic, not measurement error.
    Missing outcomes are never filled. A missing 2006 outcome makes this entire
    decomposition unavailable for the unit; it does not remove any model row.
    """
    result = []
    for (crop, territory), (state, a0) in sorted(units.items()):
        baseline_row = index.get((crop, territory, 2006))
        ybase = None if baseline_row is None else baseline_row['production_kg_fixed_ha']
        area_parts, productivity_parts = [], []
        n_allocated = n_undefined_yield = n_missing_q = 0
        for year in TRANSITION_END_YEARS:
            transition = transition_index[(crop, territory, year)]
            if transition['allocation_eligible']:
                area_parts.append(transition['area_midpoint_allocation_kg_fixed_ha'])
                productivity_parts.append(transition['productivity_midpoint_allocation_kg_fixed_ha'])
                n_allocated += 1
            elif transition['observed_delta_kg_fixed_ha'] is None:
                n_missing_q += 1
            else:
                n_undefined_yield += 1
            source = index.get((crop, territory, year))
            y = None if source is None else source['production_kg_fixed_ha']
            ac = math.fsum(area_parts) if ybase is not None else None
            pc = math.fsum(productivity_parts) if ybase is not None else None
            uc = y-ybase-ac-pc if y is not None and ybase is not None else None
            identity_residual = y-math.fsum([ybase, ac, pc, uc]) if uc is not None else None
            result.append({
                'crop': crop, 'territory': territory, 'state': state, 'year': year,
                'A0': a0, 'reference_year': 2006,
                'baseline_2006_production_kg_fixed_ha': ybase,
                'observed_production_kg_fixed_ha': y,
                'reference_production_observed': ybase is not None,
                'level_decomposition_available': uc is not None,
                'cumulative_area_allocation_kg_fixed_ha': ac,
                'cumulative_productivity_allocation_kg_fixed_ha': pc,
                'cumulative_unallocated_change_kg_fixed_ha': uc,
                'cumulative_unallocated_absolute_kg_fixed_ha': None if uc is None else abs(uc),
                'allocated_transitions_since_reference': n_allocated,
                'undefined_yield_observed_transitions_since_reference': n_undefined_yield,
                'missing_production_transitions_since_reference': n_missing_q,
                'level_identity_residual_kg_fixed_ha': identity_residual,
                'allocation_note': 'Pathwise midpoint arithmetic; residual retains undefined-yield changes and missing-production gaps',
            })
    return result


def analyse_trajectories(panel):
    """Return scalar-only output rows and independent arithmetic diagnostics.

    ``panel`` contains one row per original crop/territory/year, including
    pre-followup baseline years. All inputs are copied; source values never change.
    Missing expected years are exposed by coverage and missing transition counts.
    """
    rows = [_annual_row(r) for r in panel]
    if not rows:
        raise ValueError('The frozen trajectory panel is empty')
    rows.sort(key=lambda r: (r['crop'], r['territory'], r['year']))
    index, units = {}, {}
    groups = defaultdict(set)
    annual_groups = defaultdict(list)
    for row in rows:
        crop, territory, state, year = (row[k] for k in ('crop', 'territory', 'state', 'year'))
        key, unit = (crop, territory, year), (crop, territory)
        if key in index:
            raise ValueError('Duplicate trajectory crop/territory/year: ' + repr(key))
        signature = (state, row['A0'])
        if unit in units and units[unit] != signature:
            raise ValueError('Frozen unit state/baseline area changed: ' + repr(unit))
        units[unit], index[key] = signature, row
        for geography in ('ALL', state):
            groups[(crop, geography)].add(unit)
            annual_groups[(crop, geography, year)].append(row)
    annual_summary = [_annual_summary(crop, state, year, len(members), annual_groups[(crop, state, year)])
                      for (crop, state), members in sorted(groups.items()) for year in ANNUAL_YEARS]
    transitions, transition_groups = [], defaultdict(list)
    for (crop, territory), (state, a0) in sorted(units.items()):
        for year in TRANSITION_END_YEARS:
            row = _transition(crop, territory, state, a0, year,
                              index.get((crop, territory, year-1)), index.get((crop, territory, year)))
            transitions.append(row)
            for geography in ('ALL', state):
                transition_groups[(crop, geography, year)].append(row)
    transition_summary = [_transition_summary(crop, state, year, members)
                          for (crop, state, year), members in sorted(transition_groups.items())]
    level_components = _level_components(units, index,
                         {(r['crop'], r['territory'], r['end_year']): r for r in transitions})
    selected = [dict(r, selection='POSTHOC_PANEL_EXAMPLES_NOT_EXCLUSIONS')
                for r in rows if (r['crop'], r['territory']) in SELECTED_POSTHOC_CASES]
    residuals = [r['allocation_identity_residual_kg_fixed_ha'] for r in transitions if r['allocation_eligible']]
    summary_residuals = [r['allocation_identity_residual_kg_fixed_ha'] for r in transition_summary
                         if r['allocation_identity_residual_kg_fixed_ha'] is not None]
    max_scaled = max((abs(r['allocation_identity_residual_kg_fixed_ha']) /
                      max(1.0, abs(r['observed_delta_kg_fixed_ha']),
                          abs(r['area_midpoint_allocation_kg_fixed_ha']),
                          abs(r['productivity_midpoint_allocation_kg_fixed_ha']))
                      for r in transitions if r['allocation_eligible']), default=0.0)
    if max_scaled > 1e-10:
        raise ValueError('Midpoint transition identity did not reconcile')
    level_residuals = [r['level_identity_residual_kg_fixed_ha'] for r in level_components
                       if r['level_decomposition_available']]
    level_scaled = max((abs(r['level_identity_residual_kg_fixed_ha']) / max(
        1.0, abs(r['observed_production_kg_fixed_ha']),
        abs(r['baseline_2006_production_kg_fixed_ha']),
        abs(r['cumulative_area_allocation_kg_fixed_ha']),
        abs(r['cumulative_productivity_allocation_kg_fixed_ha']),
        abs(r['cumulative_unallocated_change_kg_fixed_ha']))
        for r in level_components if r['level_decomposition_available']), default=0.0)
    if level_scaled > 1e-10:
        raise ValueError('Cumulative level component identity did not reconcile')
    return {
        'annual_unit_rows': rows, 'annual_summary_rows': annual_summary,
        'transition_rows': transitions, 'transition_summary_rows': transition_summary,
        'selected_case_rows': selected,
        'level_component_rows': level_components,
        'validation': {
            'status': 'PASS', 'input_rows': len(rows), 'fixed_crop_units': len(units),
            'transition_rows': len(transitions), 'transition_allocation_identities_checked': len(residuals),
            'transition_summary_identities_checked': len(summary_residuals),
            'maximum_absolute_unit_identity_residual': max(map(abs, residuals), default=0.0),
            'maximum_absolute_summary_identity_residual': max(map(abs, summary_residuals), default=0.0),
            'maximum_scaled_unit_identity_residual': max_scaled,
            'level_reference_year': 2006,
            'level_component_rows': len(level_components),
            'level_identities_checked': len(level_residuals),
            'maximum_absolute_level_identity_residual': max(map(abs, level_residuals), default=0.0),
            'maximum_scaled_level_identity_residual': level_scaled,
            'units_missing_2006_production': sum(
                index.get((crop, territory, 2006)) is None or
                index[(crop, territory, 2006)]['production_kg_fixed_ha'] is None
                for crop, territory in units),
            'zero_or_missing_yields_imputed': False,
            'selected_case_rows': len(selected), 'selected_cases_are_posthoc': True,
            'causal_or_biological_mechanism_claim': False,
        },
    }
