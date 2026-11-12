"""Frozen temporal endpoint support and temporal-specification designs.

These functions prepare inputs only.  They do not fit models or select a model
using outcomes.  All fits for a crop share the same eligible transition endpoints;
first differences additionally use the immediately preceding outcome records.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from .features import BASE_NAMES, OUTCOMES

CROPS = ('coffee_total', 'cacao')
FIRST_ENDPOINT = 2008
LAST_ENDPOINT = 2023
FUTURE_NAMES = ['D_lead1', 'D_lead2', 'T_lead1', 'T_lead2']
CORE_VARIANTS = ('levels', 'levels_unit_trend', 'first_difference')


def specifications():
    """Return 18 core specifications and two production-only lead diagnostics."""
    result = []
    for crop in CROPS:
        for outcome in OUTCOMES:
            variants = list(CORE_VARIANTS)
            if outcome == 'production_fixed_ha':
                variants.append('levels_unit_trend_future')
            for variant in variants:
                result.append({
                    'model_id': '__'.join((crop, outcome, variant)),
                    'crop': crop, 'outcome': outcome, 'variant': variant,
                    'unit_trends': variant in ('levels_unit_trend', 'levels_unit_trend_future'),
                    'unit_parity': crop == 'coffee_total',
                    'role': 'temporal_lead_diagnostic' if variant.endswith('_future') else 'temporal_specification_comparison',
                })
    return result


def _finite(value):
    try:
        return value is not None and not isinstance(value, (bool, np.bool_)) and math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _label(value, name):
    if value is None or isinstance(value, (bool, np.bool_)):
        raise ValueError('Missing/invalid ' + name)
    label = str(value).strip()
    if not label or label.lower() in ('none', 'nan', 'null'):
        raise ValueError('Missing/invalid ' + name)
    return label


def _key(row):
    crop = _label(row.get('crop'), 'crop')
    if crop not in CROPS:
        raise ValueError('Unsupported crop: ' + crop)
    territory = _label(row.get('territory'), 'territory')
    value = row.get('year')
    if not _finite(value) or float(value) != int(float(value)):
        raise ValueError('Invalid calendar year')
    return crop, territory, int(float(value))


def _level_x(row):
    values = [row.get(name) for name in BASE_NAMES[:6]]
    if not all(_finite(v) for v in values):
        return None
    values = [float(v) for v in values]
    history = (values[1] / 2.0) + (values[2] / 2.0)
    product = values[0] * history
    if not math.isfinite(history) or not math.isfinite(product):
        return None
    return values + [product]


def _canonical_panel(panel):
    by_key = {}
    unit_properties = {}
    municipality_states = {}
    for original in panel:
        key = _key(original)
        if key in by_key:
            raise ValueError('Duplicate panel identity: ' + repr(key))
        state = _label(original.get('state'), 'state')
        if not _finite(original.get('A0')) or float(original['A0']) < 100.0:
            raise ValueError('Invalid frozen baseline area: ' + repr(key))
        area = float(original['A0'])
        unit = key[:2]
        properties = (state, area)
        if unit in unit_properties and unit_properties[unit] != properties:
            raise ValueError('State or frozen A0 changes within crop-unit: ' + repr(unit))
        unit_properties[unit] = properties
        if key[1] in municipality_states and municipality_states[key[1]] != state:
            raise ValueError('Municipality maps to multiple states across crops: ' + key[1])
        municipality_states[key[1]] = state
        by_key[key] = dict(original, crop=key[0], territory=key[1], year=key[2], state=state, A0=area)
    return by_key


def _registry(reference_rows, panel):
    result = {}
    for original in reference_rows:
        key = _key(original)
        if key in result:
            raise ValueError('Duplicate common-support common-reference identity: ' + repr(key))
        if key not in panel:
            raise ValueError('common-support common-reference identity is absent from panel: ' + repr(key))
        state = _label(original.get('state'), 'reference state')
        if state != panel[key]['state']:
            raise ValueError('Reference state differs from panel: ' + repr(key))
        # Published registry climate columns are checked when present.  They are
        # identity/provenance checks, not alternative exposure definitions.
        source = panel[key]
        d0 = float(source['D0']) if _finite(source.get('D0')) else None
        h = (float(source['D1'])/2.0 + float(source['D2'])/2.0
             if all(_finite(source.get(n)) for n in ('D1', 'D2')) else None)
        for field, actual in (('D0', d0), ('history', h)):
            if field in original:
                if actual is None or not _finite(original[field]) or not math.isclose(float(original[field]), actual, rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError('Reference ' + field + ' differs from panel: ' + repr(key))
        result[key] = dict(original, crop=key[0], territory=key[1], year=key[2], state=state)
    return result


def _counts(rows):
    years = sorted({r['year'] for r in rows})
    states = sorted({r['state'] for r in rows})
    return {
        'rows': len(rows), 'units': len({r['territory'] for r in rows}),
        'states': len(states), 'years': len(years), 'endpoint_years': years,
        'underlying_difference_outcome_years': sorted(set(years) | {y-1 for y in years}),
        'rows_by_year': {str(y): n for y, n in sorted(Counter(r['year'] for r in rows).items())},
        'rows_by_state': {str(s): n for s, n in sorted(Counter(r['state'] for r in rows).items())},
        'rows_by_state_year': [dict(state=s, year=y, rows=n) for (s, y), n in sorted(Counter((r['state'], r['year']) for r in rows).items())],
    }


def prepare(panel, reference_rows):
    """Build the finite model menu on one support per crop, without fitting.

    All candidate panel endpoints 2008--2023 are retained or assigned explicit
    reasons.  Missing outcome values, including predecessor values, are never
    imputed; genuine numerical zeros remain eligible.  Future climate joins use
    exact crop/municipality/year identities and never positional row shifts.
    """
    panel_by_key = _canonical_panel(panel)
    reference = _registry(reference_rows, panel_by_key)
    specs = specifications()
    selected = {crop: [] for crop in CROPS}
    support_rows, exclusion_rows = [], []
    for key in sorted(panel_by_key):
        crop, territory, year = key
        if not FIRST_ENDPOINT <= year <= LAST_ENDPOINT:
            continue
        row = panel_by_key[key]
        prior_key = (crop, territory, year-1)
        prior = panel_by_key.get(prior_key)
        reasons = []
        if key not in reference:
            reasons.append('endpoint_not_in_common_support_common_reference')
        if prior_key not in reference:
            reasons.append('predecessor_not_in_common_support_common_reference')
        if prior is None:
            reasons.append('consecutive_predecessor_panel_row_missing')
        for label, source in (('endpoint', row), ('predecessor', prior)):
            if source is not None and not all(_finite(source.get(o)) for o in OUTCOMES):
                reasons.append(label + '_three_outcome_support_missing')
        current_x = _level_x(row)
        prior_x = None if prior is None else _level_x(prior)
        if current_x is None:
            reasons.append('endpoint_original_climate_design_missing')
        if prior is not None and prior_x is None:
            reasons.append('predecessor_original_climate_design_missing')
        future = [panel_by_key.get((crop, territory, year+j)) for j in (1, 2)]
        for j, source in enumerate(future, 1):
            if source is None or not all(_finite(source.get(name)) for name in ('D0', 'T0')):
                reasons.append('future_climate_' + str(j) + 'y_missing')
        # Subtraction can overflow despite finite source values.  Treat it as an
        # explicit unsupported numeric range, identically across all models.
        if not reasons:
            difference_x = [a-b for a, b in zip(current_x, prior_x)]
            differences_y = {o: float(row[o])-float(prior[o]) for o in OUTCOMES}
            if not all(math.isfinite(v) for v in difference_x + list(differences_y.values())):
                reasons.append('nonfinite_difference_numeric_range')
        record = {'crop': crop, 'territory': territory, 'state': row['state'], 'year': year,
                  'predecessor_year': year-1, 'lead1_year': year+1, 'lead2_year': year+2}
        if reasons:
            exclusion_rows.append(dict(record, reasons=';'.join(reasons)))
            continue
        history = current_x[1]/2.0 + current_x[2]/2.0
        support_rows.append(dict(record, A0=row['A0'], D0=current_x[0], history=history,
                                 endpoint_and_predecessor_common_reference=True,
                                 outcome_support='all_three_at_both_endpoints',
                                 future_climate_required_for_all_models=True))
        selected[crop].append({
            'row': dict(row, history=history, predecessor_year=year-1,
                        exposure_calendar_year=year),
            'level_x': current_x, 'difference_x': difference_x,
            'difference_y': differences_y,
            'future_x': [float(future[0]['D0']), float(future[1]['D0']),
                         float(future[0]['T0']), float(future[1]['T0'])],
        })
    data_by_model, reference_by_crop = {}, {}
    for crop in CROPS:
        records = selected[crop]
        x = np.asarray([r['level_x'] for r in records], dtype=float).reshape((-1, len(BASE_NAMES)))
        h = x[:, 1]/2.0 + x[:, 2]/2.0
        reference_by_crop[crop] = {
            'X': x, 'names': list(BASE_NAMES), 'rows': [dict(r['row']) for r in records],
            'history_mean': float(np.mean(h)) if len(h) else None,
            'history_q25': float(np.quantile(h, 0.25)) if len(h) else None,
            'history_q75': float(np.quantile(h, 0.75)) if len(h) else None,
            'history_quantile_method': 'NumPy linear interpolation',
            'interpretation': 'Undifferenced original endpoint climate for every slope; fixed within crop across all specifications.',
        }
    for spec in specs:
        records = selected[spec['crop']]
        difference = spec['variant'] == 'first_difference'
        future = spec['variant'] == 'levels_unit_trend_future'
        names = list(BASE_NAMES) + (list(FUTURE_NAMES) if future else [])
        xrows = [list(r['difference_x'] if difference else r['level_x']) + (r['future_x'] if future else []) for r in records]
        y = [r['difference_y'][spec['outcome']] if difference else float(r['row'][spec['outcome']]) for r in records]
        rows = [dict(r['row'], response_transformation='first_difference' if difference else 'levels') for r in records]
        data_by_model[spec['model_id']] = {
            'X': np.asarray(xrows, dtype=float).reshape((-1, len(names))),
            'y': np.asarray(y, dtype=float), 'names': names, 'rows': rows,
            'unit': np.asarray([r['territory'] for r in rows], dtype=str),
            'state': np.asarray([r['state'] for r in rows], dtype=str),
            'year': np.asarray([r['year'] for r in rows], dtype=int),
        }
    summary = {
        'status': 'SUPPORT_PREPARED_WITHOUT_FITTING', 'model_specifications': len(specs),
        'requested_endpoint_years': [FIRST_ENDPOINT, LAST_ENDPOINT],
        'possible_underlying_difference_outcome_years': [FIRST_ENDPOINT-1, LAST_ENDPOINT],
        'candidate_endpoints': len(support_rows)+len(exclusion_rows),
        'eligible_endpoints': len(support_rows), 'excluded_endpoints': len(exclusion_rows),
        'by_crop': {crop: _counts([r for r in support_rows if r['crop'] == crop]) for crop in CROPS},
        'exclusion_reason_counts': dict(sorted(Counter(reason for r in exclusion_rows for reason in r['reasons'].split(';')).items())),
        'support_definition': 'Same eligible transition endpoints within crop; FD also uses predecessor outcomes.',
        'first_difference_unit_intercept': 'Represents a unit linear drift in the underlying level model; no trend is added to FD.',
        'first_difference_coffee_parity': 'Parity span preserved because parity[t]-parity[t-1]=2*parity[t].',
        'climate_lead_diagnostic': 'Production only; future association is not a randomized placebo or an automatic causal validity test.',
    }
    return dict(specs=specs, data_by_model=data_by_model, reference_by_crop=reference_by_crop,
                support_rows=support_rows, exclusion_rows=exclusion_rows, summary=summary)
