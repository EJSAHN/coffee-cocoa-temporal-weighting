"""Frozen legacy m6 event contrast applied only to generated climate/outcomes.

Calendar standardization, hysteresis event detection, pair eligibility and
nested annual weights preserve the validated complete-panel definitions.
No outcome values enter the weights.
"""
from __future__ import annotations

import collections as C
from fractions import Fraction
import statistics
import numpy as np

MONTHS = list(range(1980 * 12, 2026 * 12))
ROLES = ('short_post', 'short_benchmark', 'long_post', 'long_benchmark')

class AuditError(ValueError):
    pass

def require(ok, message):
    if not ok:
        raise ValueError(message)

def truth(value):
    if value is True or value == 'True':
        return True
    if value is False or value == 'False':
        return False
    raise ValueError('Expected frozen explicit boolean: ' + str(value))

def unit(row):
    return row['territory'], row['crop']

SOURCE_PROVENANCE = {
    'calendar_standardization': {'sha256': 'b6935ff0faad33179cf923b806a88c8aef00af58b691c897c676b641b63fb5d0'},
    'event_support': {'sha256': '9479d6e2ab8f1c55e15f1775893072218e1263a47dd8013aa1fb3c4bb2f2dc02'},
    'nested_annual_weights': {'sha256': '5ff35ea7670627f70c75146d122c332931369ad6bf052514b43a3f31f7ac7c00'},
}


def standardize(values, scale):
    sums = []
    for i in range(len(values)):
        segment = values[max(0, i - scale + 1):i + 1]
        sums.append(sum(segment) if len(segment) == scale and all(v is not None for v in segment) else None)
    ref = {}
    for calendar_month in range(12):
        sample = [v for i, v in enumerate(sums) if 1981 <= MONTHS[i] // 12 <= 2010
                  and MONTHS[i] % 12 == calendar_month and v is not None]
        if len(sample) != 30:
            raise AuditError('Incomplete 1981-2010 monthly climate reference period.')
        sd = statistics.stdev(sample)
        if sd <= 1e-12:
            raise AuditError('Zero-variance monthly climate reference period.')
        ref[calendar_month] = (statistics.mean(sample), sd)
    z = [None if v is None else (v - ref[MONTHS[i] % 12][0]) / ref[MONTHS[i] % 12][1]
         for i, v in enumerate(sums)]
    return z


def detect(z, offset=12, exit_threshold=-0.5, exit_months=3):
    """Hysteresis event detection; missing values censor episodes."""
    events, active = [], None
    severe = wet = 0
    recovery_seen = False
    def finish(end, censored, confirmed=None, reason='threshold_recovery'):
        vals = z[active['start_index']:end+1]
        require(vals and all(v is not None for v in vals), 'Invalid event span')
        events.append(dict(active, end_index=end, recovery_confirmed_index=confirmed,
            right_censored=censored, end_reason=reason, duration_months=len(vals),
            severity_z_months=sum(max(0,-v) for v in vals), minimum_z=min(vals),
            severe_months=sum(v <= -1.5 for v in vals)))
    for i in range(offset, len(z)):
        v = z[i]
        if v is None:
            if active is not None:
                finish(i-1, True, reason='missing_climate')
            active = None
            severe = wet = 0
            recovery_seen = False
            continue
        severe = severe+1 if v <= -1.5 else 0
        wet = wet+1 if v > exit_threshold else 0
        if active is None and severe >= 2:
            active = dict(start_index=i-1, left_censored=not recovery_seen)
        if wet >= exit_months:
            if active is not None:
                finish(i-exit_months, False, i)
                active = None
            recovery_seen = True
    if active is not None:
        finish(len(z)-1, True, reason='end_of_record')
    return events


def annual_contributions(pairs,anchor):
    """Within-one-unit rational coefficients, aggregated before simulation.

    Pairs must already be eligible. Short mean(delta2) minus long mean(delta2).
    Returned role terms are signed and omit the final equal-municipality 1/N.
    Exactly-zero touched years are retained. No observed outcomes are accepted.
    """
    groups={g:[p for p in pairs if p['contrast_group']==g] for g in ('short','long')}
    require(all(groups.values()),'Both fixed gap groups are required')
    require(len({unit(p) for p in pairs})==1,'Annual unit design mixes crop/municipality identities')
    require(len({p['pair_uid'] for p in pairs})==len(pairs),'Duplicate eligible pair')
    weights=C.defaultdict(lambda:C.defaultdict(Fraction));counts=C.Counter()
    for g,pp in groups.items():
        sign=1 if g=='short' else -1
        for p in pp:
            sy=int(p['current_start_month'][:4])
            a,b=int(p['prebenchmark_first_year']),int(p['prebenchmark_last_year'])
            require((a,b)==(sy-4,sy-1),'Frozen B4 dates changed')
            ay=int(p['confirmation_year']) if anchor=='confirmation' else sy
            require(2007<=sy<=2025 and ay+2<=2025,'Ineligible onset or post period reached linear design')
            require(truth(p['prebenchmark_available']),'Unavailable benchmark reached linear design')
            if anchor=='confirmation':require(truth(p['current_episode_resolved']),'Unresolved confirmation reached linear design')
            for y in (ay+1,ay+2):
                weights[y][g+'_post']+=Fraction(sign,2*len(pp));counts[y]+=1
            for y in range(a,b+1):
                weights[y][g+'_benchmark']+=Fraction(-sign,4*len(pp));counts[y]+=1
    require(sum((sum(v.values(),Fraction()) for v in weights.values()),Fraction())==0,
            'Within-unit coefficient sum is not zero')
    # Every gap-group mean gives total post weight +/-1 and benchmark -/+1.
    for role,expected in [('short_post',1),('short_benchmark',-1),('long_post',-1),('long_benchmark',1)]:
        require(sum((v[role] for v in weights.values()),Fraction())==expected,'Role normalization failed')
    return {y:{role:v[role] for role in ROLES} for y,v in sorted(weights.items())},counts


def _month(index):
    month = MONTHS[index]
    return f'{month // 12:04d}-{month % 12 + 1:02d}'


def _pair_rows(sequence, territory, observed_years):
    """Minimal faithful register_pair adapter; every supplied annual cell observed.

    Eligibility does not require a four-year post period or exclude
    subsequent droughts.
    An unresolved current episode is allowed only for the onset companion.
    """
    rows = []
    for index in range(1, len(sequence)):
        previous, current = sequence[index - 1], sequence[index]
        start_year = MONTHS[current['start_index']] // 12
        gap = current['start_index'] - previous['end_index'] - 1
        group = ('short' if 3 <= gap <= 24 else
                 'long' if 48 <= gap <= 72 else 'outside_contrast')
        history_known = (not previous['left_censored'] and
                         not previous['right_censored'] and
                         not current['left_censored'])
        core = 2007 <= start_year <= 2025 and group != 'outside_contrast' and history_known
        resolved = (not current['right_censored'] and
                    current['recovery_confirmed_index'] is not None)
        confirmation_year = (MONTHS[current['recovery_confirmed_index']] // 12
                             if resolved else None)
        pre_available = all(year in observed_years for year in range(start_year - 4, start_year))
        confirmation_calendar = confirmation_year is not None and confirmation_year + 2 <= 2025
        confirmation_observed = confirmation_calendar and all(
            year in observed_years for year in (confirmation_year + 1, confirmation_year + 2))
        onset_calendar = start_year + 2 <= 2025
        onset_observed = onset_calendar and all(
            year in observed_years for year in (start_year + 1, start_year + 2))
        primary_eligible = bool(core and pre_available and confirmation_observed and resolved)
        onset_eligible = bool(core and pre_available and onset_observed)
        reasons = []
        if not 2007 <= start_year <= 2025:
            reasons.append('ONSET_OUTSIDE_ANALYSIS_YEARS')
        if group == 'outside_contrast':
            reasons.append('GAP_OUTSIDE_FROZEN_CONTRAST')
        if not history_known:
            reasons.append('CENSORED_EPISODE_HISTORY')
        if not resolved:
            reasons.append('CURRENT_EPISODE_NOT_CONFIRMED_ENDED')
        if not pre_available:
            reasons.append('PREBENCHMARK_INCOMPLETE')
        if resolved and not confirmation_calendar:
            reasons.append('ADMINISTRATIVE_2Y_FOLLOWUP_UNAVAILABLE')
        rows.append(dict(
            territory=territory, crop='synthetic_crop', state='SYNTHETIC',
            definition='m6_original',
            pair_uid='m6_original|' + territory + '_' + _month(current['start_index']),
            previous_start_month=_month(previous['start_index']),
            previous_end_month=_month(previous['end_index']),
            current_start_month=_month(current['start_index']),
            current_end_month=_month(current['end_index']),
            gap_months=gap, contrast_group=group,
            onset_in_2007_2025=2007 <= start_year <= 2025,
            history_known_for_comparison=bool(history_known),
            base_pair_eligible=bool(core), current_episode_resolved=bool(resolved),
            confirmation_month=(_month(current['recovery_confirmed_index']) if resolved else ''),
            confirmation_year=confirmation_year,
            prebenchmark_first_year=start_year - 4,
            prebenchmark_last_year=start_year - 1,
            prebenchmark_available=bool(pre_available),
            primary_2y_computable=primary_eligible,
            onset_2y_computable=onset_eligible,
            primary_2y_unavailable_reasons=';'.join(reasons),
            previous_left_censored=bool(previous['left_censored']),
            previous_right_censored=bool(previous['right_censored']),
            current_left_censored=bool(current['left_censored']),
            current_right_censored=bool(current['right_censored']),
            subsequent_onset_month=(_month(sequence[index + 1]['start_index'])
                                    if index + 1 < len(sequence) else ''),
        ))
    return rows


def build_weights(z6, unit_ids, years=np.arange(2002, 2026)):
    """Build exact legacy event contrasts without seeing generated outcomes.

    Parameters
    ----------
    z6 : (N, 552) array
        Calendar-month standardized six-month balance, January 1980 through
        December 2025. NaN/None censor episodes; infinity is rejected.
    unit_ids : N distinct identifiers
        Converted to strings. No outcome-dependent inclusion is permitted.
    years : exactly 2002,...,2025
        Annual output grid. Result vectors are flattened unit-major N*24.

    Each anchor independently retains municipalities with both eligible gap
    groups. Nested equal-event/equal-municipality weights use exact Fractions
    until their final conversion to floats. Empty support returns all zeros
    with an explicit NONESTIMABLE_NO_WITHIN_UNIT_OVERLAP status.
    """
    units = [str(value) for value in unit_ids]
    require(len(set(units)) == len(units), 'Duplicate synthetic unit identifier')
    years = np.asarray(years)
    require(years.ndim == 1 and np.array_equal(years, np.arange(2002, 2026)),
            'Legacy annual outcome grid must be exactly 2002-2025')
    values = np.asarray(z6, dtype=float)
    require(values.ndim == 2 and values.shape == (len(units), 552),
            'Legacy z6 array must have shape (number of units, 552)')
    require(not np.isinf(values).any(), 'Infinite synthetic climate value')
    observed_years = set(int(year) for year in years)
    pairs = []
    event_counts = []
    for territory, unit_values in zip(units, values):
        z = [None if np.isnan(value) else float(value) for value in unit_values]
        sequence = detect(z, offset=12)
        event_counts.append(dict(territory=territory, events=len(sequence),
                                 left_censored_events=sum(bool(e['left_censored']) for e in sequence),
                                 right_censored_events=sum(bool(e['right_censored']) for e in sequence)))
        pairs.extend(_pair_rows(sequence, territory, observed_years))
    index_by_unit = {territory: index for index, territory in enumerate(units)}
    index_by_year = {int(year): index for index, year in enumerate(years)}
    all_weights, anchor_diagnostics = {}, {}
    for anchor in ('confirmation', 'onset'):
        eligibility = 'primary_2y_computable' if anchor == 'confirmation' else 'onset_2y_computable'
        eligible = [pair for pair in pairs if pair[eligibility]]
        grouped = C.defaultdict(list)
        for pair in eligible:
            grouped[pair['territory']].append(pair)
        overlap = {territory: unit_pairs for territory, unit_pairs in grouped.items()
                   if {pair['contrast_group'] for pair in unit_pairs} == {'short', 'long'}}
        n_units = len(overlap)
        vector = np.zeros(len(units) * len(years), dtype=float)
        rational_sum = Fraction()
        zero_units = 0
        for territory, unit_pairs in sorted(overlap.items()):
            roles, _ = annual_contributions(unit_pairs, anchor)
            unit_nonzero = 0
            for year, parts in roles.items():
                exact = sum(parts.values(), Fraction()) / n_units
                vector[index_by_unit[territory] * len(years) + index_by_year[year]] = float(exact)
                rational_sum += exact
                unit_nonzero += exact != 0
            zero_units += unit_nonzero == 0
        require(rational_sum == 0, 'Exact legacy contrast weights do not sum to zero')
        name = anchor + '2y'
        all_weights[name] = vector
        used_pairs = [pair for unit_pairs in overlap.values() for pair in unit_pairs]
        anchor_diagnostics[name] = dict(
            status=('COMPLETE_LEGACY_EVENT_WEIGHTS' if n_units else
                    'NONESTIMABLE_NO_WITHIN_UNIT_OVERLAP'),
            estimable=bool(n_units), eligible_units=n_units,
            total_units=len(units), eligible_pairs_before_overlap=len(eligible),
            pairs_in_contrast=len(used_pairs),
            short_pairs=sum(pair['contrast_group'] == 'short' for pair in used_pairs),
            long_pairs=sum(pair['contrast_group'] == 'long' for pair in used_pairs),
            eligible_unit_ids=sorted(overlap),
            nonzero_weight_rows=int(np.count_nonzero(vector)),
            exact_zero_vector_units=zero_units, exact_weight_sum_zero=True,
            floating_weight_sum=float(vector.sum()),
        )
    diagnostics = dict(
        definition='m6_original', climate_months=552, climate_first_month='1980-01',
        climate_last_month='2025-12', detection_first_month='1981-01',
        annual_years=[int(year) for year in years], units=len(units),
        total_events=sum(row['events'] for row in event_counts),
        total_adjacent_pairs=len(pairs), per_unit_events=event_counts,
        anchors=anchor_diagnostics, outcome_values_used=False,
        annual_values_assumed_observed=True,
        source_provenance=SOURCE_PROVENANCE,
        interpretation='Descriptive event functional; not the panel interaction coefficient.',
    )
    return dict(weights=all_weights, diagnostics=diagnostics, pairs=pairs)
