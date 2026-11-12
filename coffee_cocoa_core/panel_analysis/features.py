"""Predeclared calendar exposures and outcome-specific observed supports.

No event selection, second-event baseline, outcome-dependent climate windows,
or imputations are used. Only standard library and NumPy are required.
"""
from __future__ import annotations
import math
from collections import Counter
import numpy as np

BASE_NAMES = ['D0', 'D1', 'D2', 'T0', 'T1', 'T2', 'D0_x_history']
OUTCOMES = ('production_fixed_ha', 'intended_area_fixed_ha', 'unharvested_area_fixed_ha')


def finite(x):
    return x is not None and math.isfinite(float(x))


def make_panel(annual_rows, climate_annual):
    """Return independent new dicts; original records are never modified."""
    seen = set()
    panel = []
    for original in annual_rows:
        row = dict(original)
        key = (str(row['territory']), row['crop'], int(row['year']))
        if key in seen:
            raise ValueError('Duplicate annual target key: ' + repr(key))
        seen.add(key)
        row['year'] = key[2]
        row['territory'] = key[0]
        area = float(row['A0'])
        if not math.isfinite(area) or area < 100:
            raise ValueError('Invalid frozen denominator')
        for field in ('Q_tonnes', 'Iha', 'Hha'):
            value = row.get(field)
            if value is not None and (not finite(value) or float(value) < 0):
                raise ValueError('Invalid original value: ' + field)
        q, i, h = row.get('Q_tonnes'), row.get('Iha'), row.get('Hha')
        row['production_fixed_ha'] = None if q is None else float(q)*1000.0/area
        row['intended_area_fixed_ha'] = None if i is None else float(i)/area
        # A negative reported I-H is retained and flagged, not silently censored.
        row['unharvested_area_fixed_ha'] = None if i is None or h is None else (float(i)-float(h))/area
        row['harvested_exceeds_intended'] = i is not None and h is not None and float(h)>float(i)
        row['harvest_fraction'] = float(h)/float(i) if i is not None and h is not None and float(i)>0 else None
        row['yield_kg_harvested_ha'] = float(q)*1000.0/float(h) if q is not None and h is not None and float(h)>0 else None
        for lag in range(4):
            climate = climate_annual.get((key[0], key[2]-lag))
            if climate is None:
                row['D'+str(lag)] = None
                row['T'+str(lag)] = None
            else:
                row['D'+str(lag)] = climate.get('D_mean')
                row['T'+str(lag)] = climate.get('Tz_mean')
        panel.append(row)
    return sorted(panel, key=lambda r: (r['crop'], r['territory'], r['year']))


def specifications():
    """Finite menu fixed before this runner fits any new observed outcomes."""
    specs = []
    for crop in ('coffee_total', 'cacao'):
        for outcome in OUTCOMES:
            specs.append(dict(crop=crop, outcome=outcome, variant='primary', role='primary' if outcome==OUTCOMES[0] else 'secondary'))
            specs.append(dict(crop=crop, outcome=outcome, variant='common_component_rows', role='common_support_secondary'))
        for variant in ('nonlinear_main_effects', 'unit_linear_trend', 'exclude_prior_pilot',
                        'omit_2025', 'discordant_cells_as_missing', 'preceding_calendar_window'):
            specs.append(dict(crop=crop, outcome=OUTCOMES[0], variant=variant, role='sensitivity'))
        if crop == 'coffee_total':
            specs.append(dict(crop=crop, outcome=OUTCOMES[0], variant='without_local_biennial_term', role='sensitivity'))
    for s in specs:
        s['model_id'] = '__'.join((s['crop'], s['outcome'], s['variant']))
        s['unit_trends'] = s['variant']=='unit_linear_trend'
        s['unit_parity'] = s['crop']=='coffee_total' and s['variant']!='without_local_biennial_term'
    return specs


def select_data(panel, spec):
    names = BASE_NAMES.copy()
    if spec['variant']=='nonlinear_main_effects':
        names += ['D0_squared', 'history_squared', 'T0_squared']
    retained = []
    excluded = []
    xrows = []
    yrows = []
    lagshift = 1 if spec['variant']=='preceding_calendar_window' else 0
    for row in panel:
        if row['crop'] != spec['crop'] or not 2007 <= row['year'] <= 2025:
            continue
        reasons = []
        if spec['variant']=='exclude_prior_pilot' and row['pilot']:
            reasons.append('prior_pilot_unit')
        if spec['variant']=='omit_2025' and row['year']==2025:
            reasons.append('prespecified_2025_omission')
        if spec['variant']=='discordant_cells_as_missing' and row['source_discordant']:
            reasons.append('alternative_source_discordance_missing')
        if spec['variant']=='common_component_rows' and not all(finite(row.get(o)) for o in OUTCOMES):
            reasons.append('not_all_three_components_observed')
        if not finite(row.get(spec['outcome'])):
            reasons.append('outcome_missing')
        d = [row.get('D'+str(k+lagshift)) for k in range(3)]
        t = [row.get('T'+str(k+lagshift)) for k in range(3)]
        if not all(finite(v) for v in d+t):
            reasons.append('climate_or_history_missing')
        record = {'model_id':spec['model_id'], 'territory':row['territory'], 'crop':row['crop'],
                  'state':row['state'], 'year':row['year']}
        if reasons:
            excluded.append(dict(record, reasons=';'.join(reasons)))
            continue
        d = [float(v) for v in d]; t = [float(v) for v in t]
        history = (d[1]+d[2])/2
        xx = d+t+[d[0]*history]
        if spec['variant']=='nonlinear_main_effects':
            xx += [d[0]**2, history**2, t[0]**2]
        xrows.append(xx); yrows.append(float(row[spec['outcome']]))
        retained.append(dict(row, history=history, exposure_calendar_year=row['year']-lagshift))
    x = np.asarray(xrows, dtype=float).reshape((-1,len(names)))
    y = np.asarray(yrows, dtype=float)
    return dict(X=x,y=y,names=names,rows=retained,excluded=excluded,
                unit=np.asarray([r['territory'] for r in retained]),
                state=np.asarray([r['state'] for r in retained]),
                year=np.asarray([r['year'] for r in retained],dtype=int))


def support_counts(data, spec):
    counts = Counter((r['state'],r['year']) for r in data['rows'])
    return [dict(model_id=spec['model_id'], crop=spec['crop'], outcome=spec['outcome'],
                 variant=spec['variant'], state=s,year=y,observed_rows=n)
            for (s,y),n in sorted(counts.items())]
