"""Exact temporal responses of preserved descriptive contrast weights.

This module never reads production outcomes. It evaluates known basis sequences
against Fraction weights and independently reconstructs the same quantities from
frozen event windows. A moment is an algebraic sensitivity, not an estimated bias,
causal effect, uncertainty interval, or independent information count.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction
import re

YEARS = tuple(range(2002, 2026))
KINDS = ('original_confirmation', 'original_onset', 'matched_onset', 'matched_anchor_change')
CROPS = ('coffee_total', 'cacao')
POPULATIONS = ('full_target', 'exclude_43_pilot_units')
MOMENTS = ('m0', 'm1', 'm2', 'mp')


def _integer(value, name):
    if isinstance(value, bool):
        raise ValueError(name+' must be an integer')
    try:
        parsed = int(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(name+' must be an integer') from exc
    if str(parsed) != str(value).strip():
        raise ValueError(name+' must be an exact integer or its decimal string')
    return parsed


def _label(value, name):
    if value is None or isinstance(value, bool):
        raise ValueError('Missing '+name)
    text = str(value).strip()
    if not text or text.lower() in ('none', 'nan', 'null'):
        raise ValueError('Missing '+name)
    return text


def _fraction(value, name):
    if not isinstance(value, Fraction):
        raise ValueError(name+' must retain its exact Fraction representation')
    return value


def _fields(name, value):
    value = Fraction(value)
    return {name: float(value), name+'_exact':str(value),
            name+'_numerator':str(value.numerator), name+'_denominator':str(value.denominator)}


def _basis(year):
    tau = year-2006
    return (1, tau, tau*tau, 1 if year%2==0 else -1)


def _window(event, prefix, expected_first, expected_last):
    first = _integer(event[prefix+'first_year'], prefix+'first_year')
    last = _integer(event[prefix+'last_year'], prefix+'last_year')
    if (first, last) != (expected_first, expected_last):
        raise ValueError('Preserved '+prefix+' window differs from the frozen definition')
    return tuple(range(first, last+1))


def _event_definition(event, kind):
    start = _label(event.get('current_start_month'), 'current_start_month')
    if not re.fullmatch(r'\d{4}-(?:0[1-9]|1[0-2])', start):
        raise ValueError('Malformed frozen onset month')
    onset = int(start[:4])
    baseline = _window(event, 'benchmark_', onset-4, onset-1)
    if kind.startswith('original_'):
        anchor = _integer(event['anchor_year'], 'anchor_year')
        if kind=='original_onset' and anchor!=onset:
            raise ValueError('Original onset anchor differs from onset year')
        if anchor<onset:
            raise ValueError('Confirmation anchor precedes onset')
        post = _window(event, 'post_', anchor+1, anchor+2)
        required = set(baseline+post)
        terms = [(Fraction(-1, 4), baseline), (Fraction(1, 2), post)]
        confirmation = anchor
    else:
        anchor = _integer(event['onset_anchor_year'], 'onset_anchor_year')
        confirmation = _integer(event['confirmation_anchor_year'], 'confirmation_anchor_year')
        if anchor!=onset or confirmation<onset:
            raise ValueError('Matched event anchors differ from the frozen event chronology')
        onset_post = _window(event, 'onset_post_', onset+1, onset+2)
        confirmation_post = _window(event, 'confirmation_post_', confirmation+1, confirmation+2)
        required = set(baseline+onset_post+confirmation_post)
        terms = ([(Fraction(-1, 4), baseline), (Fraction(1, 2), onset_post)]
                 if kind=='matched_onset' else
                 [(Fraction(-1, 2), onset_post), (Fraction(1, 2), confirmation_post)])
    if not required.issubset(YEARS):
        raise ValueError('Required outcome year lies outside the fixed 2002--2025 grid')
    return terms, required, onset, confirmation


def _analytic_event_moments(kind, onset, confirmation):
    s, c = onset-2006, confirmation-2006
    if kind in ('original_onset', 'matched_onset'):
        return (0, 4, 8*s-5, 0)
    if kind=='original_confirmation':
        return (0, c-s+4, c*c-s*s+3*c+5*s-5, 0)
    return (0, c-s, c*c-s*s+3*(c-s), 0)


def _metadata(design):
    meta = {name:_label(design.get(name), name) for name in ('design_id','kind','crop','population')}
    if meta['kind'] not in KINDS or meta['crop'] not in CROPS or meta['population'] not in POPULATIONS:
        raise ValueError('Unexpected preserved design identity')
    if tuple(_integer(y,'year') for y in design['years'])!=YEARS:
        raise ValueError('Annual weight grid differs from the fixed 2002--2025 grid')
    return meta


def _one_design(design):
    meta = _metadata(design)
    units = {}
    for original in design['units']:
        territory = _label(original.get('territory'), 'territory')
        if territory in units:
            raise ValueError('Duplicate municipality in preserved design')
        unit = dict(territory=territory, state=_label(original.get('state'), 'state'))
        for field in ('n_short','n_long'):
            unit[field] = _integer(original[field], field)
            if unit[field]<=0:
                raise ValueError('Each preserved municipality must have both contrast groups')
        units[territory] = unit
    n = len(units)
    if not n:
        raise ValueError('Preserved contrast has no municipalities')
    if 'territories' in design and set(map(str,design['territories']))!=set(units):
        raise ValueError('Territory inventory differs from unit metadata')
    given = {}
    for key, original in design['weights_exact'].items():
        if not isinstance(key, tuple) or len(key)!=2:
            raise ValueError('Malformed exact annual weight identity')
        territory, year = str(key[0]), _integer(key[1], 'weight year')
        if territory not in units or year not in YEARS:
            raise ValueError('Annual weight refers to an unknown unit or year')
        if (territory,year) in given:
            raise ValueError('Repeated canonical annual weight identity')
        given[territory,year] = _fraction(original, 'annual weight')
    reconstructed = defaultdict(Fraction)
    direct = {u:[Fraction(0) for _ in MOMENTS] for u in units}
    analytic = {u:[Fraction(0) for _ in MOMENTS] for u in units}
    onset_groups = defaultdict(list)
    duration_groups = defaultdict(list)
    required = set()
    counts, seen_events = Counter(), set()
    for event in design['events']:
        territory = _label(event.get('territory'), 'event territory')
        if territory not in units:
            raise ValueError('Event municipality absent from preserved design')
        if _label(event.get('state'),'event state')!=units[territory]['state']:
            raise ValueError('Event and unit states differ')
        contrast = _label(event.get('contrast'), 'contrast')
        if contrast not in ('short','long'):
            raise ValueError('Unexpected preserved contrast group')
        event_key = (territory, _label(event.get('pair_uid'), 'pair_uid'))
        if event_key in seen_events:
            raise ValueError('Duplicate preserved event pair')
        seen_events.add(event_key); counts[territory,contrast] += 1
        factor = Fraction(1 if contrast=='short' else -1, n*units[territory]['n_'+contrast])
        terms, needed, onset, confirmation = _event_definition(event,meta['kind'])
        required.update((territory,year) for year in needed)
        onset_groups[territory,contrast].append(onset)
        duration_groups[territory,contrast].append(confirmation-onset)
        for within_factor, years in terms:
            for year in years:
                reconstructed[territory,year] += factor*within_factor
            # Evaluate window averages directly rather than reusing the merged W.
            for j in range(4):
                direct[territory][j] += factor*within_factor*sum(_basis(year)[j] for year in years)
        for j,value in enumerate(_analytic_event_moments(meta['kind'],onset,confirmation)):
            analytic[territory][j] += factor*value
    for territory, unit in units.items():
        if any(counts[territory,c]!=unit['n_'+c] for c in ('short','long')):
            raise ValueError('Preserved nested-average denominators differ from event counts')
    source_required = {(str(u),_integer(y,'required outcome year')) for u,y in design['required_outcomes']}
    if source_required!=required:
        raise ValueError('Required outcome identities differ, including cancelled baseline years')
    if set(given)!=set(reconstructed):
        raise ValueError('Exact annual weight support differs, including touched zero-weight years')
    if any(given[key]!=reconstructed[key] for key in given):
        raise ValueError('Exact annual weights differ from independent event-window reconstruction')
    unit_rows, moment_map = [], {}
    for territory, unit in sorted(units.items()):
        moments = tuple(sum((given.get((territory,year),Fraction(0))*_basis(year)[j]
                             for year in YEARS), Fraction(0)) for j in range(4))
        if moments!=tuple(direct[territory]) or moments!=tuple(analytic[territory]):
            raise ValueError('Weight, direct-window and analytic temporal moments disagree')
        if moments[0]!=0 or moments[3]!=0:
            raise ValueError('B4/post2 contrast failed exact constant/parity annihilation')
        if meta['kind'] in ('original_onset','matched_onset'):
            if moments[1]!=0:
                raise ValueError('Onset contrast failed exact linear-trend annihilation')
            sbar = {c:sum(onset_groups[territory,c],Fraction(0))/unit['n_'+c] for c in ('short','long')}
            if moments[2]!=Fraction(8,n)*(sbar['short']-sbar['long']):
                raise ValueError('Onset quadratic moment differs from closed-form identity')
        if meta['kind']=='original_confirmation':
            dbar = {c:sum(duration_groups[territory,c],Fraction(0))/unit['n_'+c] for c in ('short','long')}
            if moments[1]!=(dbar['short']-dbar['long'])/n:
                raise ValueError('Confirmation linear moment differs from closed-form identity')
        row = dict(meta,**unit)
        for name,value in zip(MOMENTS,moments):
            row.update(_fields(name,value))
        row.update(_fields('annual_weight_l1',sum((abs(given.get((territory,year),Fraction(0))) for year in YEARS),Fraction(0))))
        unit_rows.append(row); moment_map[territory] = moments
    year_weights = {year:sum((given.get((u,year),Fraction(0)) for u in units),Fraction(0)) for year in YEARS}
    states = sorted({unit['state'] for unit in units.values()})
    state_year_weights = {(state,year):sum((given.get((u,year),Fraction(0)) for u,unit in units.items() if unit['state']==state),Fraction(0))
                          for state in states for year in YEARS}
    if sum(year_weights.values(),Fraction(0))!=0 or any(sum((state_year_weights[state,y] for y in YEARS),Fraction(0))!=0 for state in states):
        raise ValueError('Temporal pulse profile failed exact zero-sum identity')
    design_row = dict(meta,n_units=n,n_short_events=sum(unit['n_short'] for unit in units.values()),
                      n_long_events=sum(unit['n_long'] for unit in units.values()),
                      moment_origin_year=2006,weight_support_rows=len(given),required_outcome_rows=len(required))
    for j,name in enumerate(MOMENTS):
        values = [moments[j] for moments in moment_map.values()]
        total = sum(values,Fraction(0))
        if total!=sum((year_weights[y]*_basis(y)[j] for y in YEARS),Fraction(0)):
            raise ValueError('Calendar profile and summed unit moments disagree')
        for suffix,value in [('sum',total),('l1',sum(map(abs,values),Fraction(0))),('max_abs',max(map(abs,values)))]:
            design_row.update(_fields(name+'_'+suffix,value))
        design_row[name+'_nonzero_units'] = sum(v!=0 for v in values)
    for name,values in [('annual_weight_l1',given.values()),('common_year_weight_l1',year_weights.values()),
                        ('state_year_weight_l1',state_year_weights.values())]:
        design_row.update(_fields(name,sum((abs(v) for v in values),Fraction(0))))
    design_row.update(status='PASS',basis_response_is_observed_effect=False,
        l1_interpretation='Sharp absolute response bound for coefficient/shock amplitude one under its stated unit, common-year or state-year scope; not a confidence interval or estimated real-data bias')
    year_rows = [dict(meta,year=year,tau=year-2006,**_fields('weight',value)) for year,value in year_weights.items()]
    state_year_rows = [dict(meta,state=state,year=year,tau=year-2006,**_fields('weight',value))
                       for (state,year),value in state_year_weights.items()]
    return dict(unit_rows=unit_rows,design_row=design_row,year_rows=year_rows,state_year_rows=state_year_rows,
                given=given,moments=moment_map,n_events=len(seen_events))


def analyse(designs,log=print):
    """Validate and profile sixteen outcome-free descriptive contrasts.

    Parameters
    ----------
    designs : sequence of mappings
        Exactly one design for every combination of ``CROPS``, ``POPULATIONS``
        and ``KINDS``. Each design supplies an opaque unique ``design_id``,
        ``crop``, ``population``, ``kind``, the annual ``years`` grid, ``units``,
        ``events``, ``weights_exact`` and ``required_outcomes``. Unit records
        contain territory, state and positive n_short/n_long event counts.
        Annual weights map (territory, year) tuples to ``fractions.Fraction``;
        touched zero-weight entries must be retained. ``required_outcomes``
        retains baseline years even when their contributions cancel.
    log : callable
        Receives a progress string for each design; defaults to ``print``.

    Returns
    -------
    dict
        JSON-serializable unit_rows, design_rows, year_rows, state_year_rows
        and validation. Every moment retains float, exact fraction string,
        numerator and denominator representations. The basis is constant,
        year minus 2006, its square, and fixed alternating year parity.

    Notes
    -----
    Input records are not modified, and no outcome values are read. This
    function preserves the fixed 2002--2025, B4/post2 design contract; it is
    not a general estimator for arbitrary event windows. Invalid identities,
    chronology, denominators, support or exact coefficients raise ValueError.
    Missing required mapping fields raise KeyError. No coefficients are
    repaired, reconstructed as floating point, or silently dropped.
    """
    if len(designs)!=16:
        raise ValueError('The preserved temporal audit requires all sixteen designs')
    results = {}; ids = set()
    for design in designs:
        meta = _metadata(design)
        identity = (meta['crop'],meta['population'],meta['kind'])
        if identity in results or meta['design_id'] in ids:
            raise ValueError('Duplicate preserved design identity')
        ids.add(meta['design_id'])
        log('Exact temporal weight profile '+meta['design_id'])
        results[identity] = _one_design(design)
    expected = {(c,p,k) for c in CROPS for p in POPULATIONS for k in KINDS}
    if set(results)!=expected:
        raise ValueError('Preserved sixteen-design inventory differs')
    cross_checks = 0
    for crop in CROPS:
        for population in POPULATIONS:
            confirmation = results[crop,population,'original_confirmation']
            onset = results[crop,population,'matched_onset']
            change = results[crop,population,'matched_anchor_change']
            keys = set(confirmation['given'])|set(onset['given'])|set(change['given'])
            for key in keys:
                if confirmation['given'].get(key,Fraction(0))!=onset['given'].get(key,Fraction(0))+change['given'].get(key,Fraction(0)):
                    raise ValueError('Confirmation != matched onset + anchor change in exact annual weights')
            units = set(confirmation['moments'])|set(onset['moments'])|set(change['moments'])
            if any(set(r['moments'])!=units for r in (confirmation,onset,change)):
                raise ValueError('Matched anchor comparisons have different municipality support')
            for unit in units:
                if any(confirmation['moments'][unit][j]!=onset['moments'][unit][j]+change['moments'][unit][j] for j in range(4)):
                    raise ValueError('Matched anchor temporal moment identity failed')
            cross_checks += 1
    unit_rows=[];design_rows=[];year_rows=[];state_year_rows=[]
    for key in sorted(results):
        result=results[key]
        unit_rows.extend(result['unit_rows']);design_rows.append(result['design_row'])
        year_rows.extend(result['year_rows']);state_year_rows.extend(result['state_year_rows'])
    validation = dict(status='PASS',designs_checked=len(results),unit_design_rows_checked=len(unit_rows),
        event_design_rows_checked=sum(r['n_events'] for r in results.values()),
        exact_weight_event_window_reconciliations=len(results),closed_form_moment_reconciliations=len(unit_rows),
        matched_anchor_identity_groups=cross_checks,year_rows=len(year_rows),state_year_rows=len(state_year_rows),
        constant_and_fixed_parity_annihilated_for_every_unit=True,
        onset_linear_trend_annihilated_for_every_unit=True,confirmation_duration_difference_formula_verified=True,
        onset_quadratic_eight_year_formula_verified=True,zero_sum_year_and_state_year_profiles=True,
        observed_outcomes_read=False,new_actual_effects_computed=False,confidence_intervals_computed=False,
        moments_interpreted_as_actual_bias=False,exact_fraction_arithmetic=True)
    return dict(unit_rows=unit_rows,design_rows=design_rows,year_rows=year_rows,
                state_year_rows=state_year_rows,validation=validation)
