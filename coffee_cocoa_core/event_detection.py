"""Reconstruct the original event designs from monthly climate and annual availability.

Monthly standardization and hysteresis detection reuse the exact validated
functions used in the synthetic benchmark. Pair eligibility here additionally
uses the actual annual missingness, which the fully observed benchmark does
not have. No production magnitudes or effect estimates enter selection.
"""
from __future__ import annotations
from collections import defaultdict, Counter
from fractions import Fraction
import csv
import gzip
import itertools
import json
import math
from pathlib import Path
import statistics

from .benchmark.legacy_events import standardize, detect, MONTHS
from .data_import import MONTH_COLUMNS, MONTHS as MONTH_LABELS, finite_number
from .event_designs import event_weights, _original_id
from .io import sha256, write_json, write_table


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _month(index):
    value = MONTHS[index]
    return f'{value // 12:04d}-{value % 12 + 1:02d}'


def _truth(value):
    if value is True or value == 'True':
        return True
    if value is False or value == 'False':
        return False
    raise ValueError('Invalid explicit preserved boolean')


def panel_availability(panel):
    """Preserve crop-unit identities and distinguish numeric zero from missing."""
    cohorts, available = {}, {}
    for row in panel:
        unit = str(row['territory']), row['crop']
        year = int(row['year'])
        identity = dict(territory=unit[0], crop=unit[1], state=row['state'], pilot=_truth(row['pilot']))
        _require(unit not in cohorts or cohorts[unit] == identity, 'Unit state/pilot identity changed')
        cohorts[unit] = identity
        key = unit, year
        _require(key not in available, 'Duplicate annual availability row')
        value = row.get('production_fixed_ha')
        available[key] = value is not None and math.isfinite(float(value))
    _require(set(available) == {(unit, year) for unit in cohorts for year in range(2002, 2026)},
             'Event reconstruction requires the complete preserved annual grid')
    return cohorts, available


def _unit_events(identity, sequence, temperature_z):
    events = []
    for event in sequence:
        start, end = event['start_index'], event['end_index']
        span = temperature_z[start:end + 1]
        events.append(dict(
            **identity, event_id=f"{identity['territory']}_{identity['crop']}_6m_{_month(start)}",
            start_month=_month(start), end_month=_month(end),
            recovery_confirmed_month=(_month(event['recovery_confirmed_index'])
                                      if event['recovery_confirmed_index'] is not None else ''),
            duration_months=event['duration_months'], severity_z_months=event['severity_z_months'],
            minimum_z=event['minimum_z'], severe_months=event['severe_months'],
            left_censored=event['left_censored'], right_censored=event['right_censored'],
            end_reason=event['end_reason'], mean_tmax_z_during_episode=statistics.mean(span)))
    return events


def _register_pairs(identity, events, availability):
    """Stage 3G register_pair rules, restricted to the original m6 two-year design."""
    unit = identity['territory'], identity['crop']
    pairs = []
    for index in range(1, len(events)):
        previous, current = events[index - 1], events[index]
        month_index = lambda value: int(value[:4]) * 12 + int(value[5:7]) - 1
        gap = month_index(current['start_month']) - month_index(previous['end_month']) - 1
        contrast = ('short' if 3 <= gap <= 24 else 'long' if 48 <= gap <= 72 else 'outside_contrast')
        sy = int(current['start_month'][:4])
        known = not previous['left_censored'] and not previous['right_censored'] and not current['left_censored']
        core = 2007 <= sy <= 2025 and contrast != 'outside_contrast' and known
        confirmed = not current['right_censored'] and bool(current['recovery_confirmed_month'])
        cy = int(current['recovery_confirmed_month'][:4]) if confirmed else None
        preyears = list(range(sy - 4, sy))
        pre_available = all(availability.get((unit, year), False) for year in preyears)
        premin, premax = f'{preyears[0]}-01', f'{preyears[-1]}-12'
        contains = any(event['start_month'] <= premax and event['end_month'] >= premin for event in events)
        nxt = events[index + 1]['start_month'] if index + 1 < len(events) else ''
        pair = dict(**identity, pair_uid='m6_original|' + current['event_id'], contrast=contrast,
                    current_start_month=current['start_month'], current_end_month=current['end_month'],
                    previous_start_month=previous['start_month'], previous_end_month=previous['end_month'],
                    gap_months=gap, history_known_for_comparison=known, base_pair_eligible=core,
                    current_episode_resolved=confirmed, confirmation_year=cy, onset_year=sy,
                    benchmark_first_year=sy - 4, benchmark_last_year=sy - 1,
                    prebenchmark_available=pre_available, prebenchmark_contains_prior_drought=contains,
                    subsequent_onset_month=nxt, previous_duration_months=previous['duration_months'],
                    current_duration_months=current['duration_months'],
                    previous_severity_z_months=previous['severity_z_months'],
                    current_severity_z_months=current['severity_z_months'],
                    current_mean_tmax_z=current['mean_tmax_z_during_episode'])
        for anchor, year in (('confirmation', cy), ('onset', sy)):
            calendar = year is not None and year + 2 <= 2025
            observed = calendar and all(availability.get((unit, y), False) for y in range(year + 1, year + 3))
            pair[anchor + '_2y_computable'] = bool(core and pre_available and observed and (anchor == 'onset' or confirmed))
            pair[anchor + '_further_onset'] = bool(nxt and nxt <= f'{year + 2}-12') if calendar else None
        pairs.append(pair)
    return pairs


def reconstruct_from_series(panel, series):
    """Outcome-blind reconstruction from iterable (territory,z6,Tz) series.

    This separate pure interface supports fictional detector/eligibility tests.
    Production run obtains z6 by re-standardizing raw monthly water balance.
    """
    cohorts, available = panel_availability(panel)
    towns = defaultdict(list)
    for unit, identity in cohorts.items():
        towns[unit[0]].append(identity)
    seen, ledgers, pairs = set(), [], []
    for territory, z6, tz in series:
        _require(territory in towns and territory not in seen, 'Unexpected or duplicated monthly municipality')
        _require(len(z6) == len(tz) == 552, 'Monthly event history must contain exactly 552 months')
        _require(all(v is None or math.isfinite(float(v)) for v in z6), 'Nonfinite standardized balance')
        _require(all(v is not None and math.isfinite(float(v)) for v in tz), 'Missing/nonfinite temperature history')
        seen.add(territory)
        sequence = detect(z6, offset=12)
        for identity in towns[territory]:
            events = _unit_events(identity, sequence, tz)
            ledgers.extend(events)
            pairs.extend(_register_pairs(identity, events, available))
    _require(seen == set(towns), 'Monthly source omits target municipalities')
    members, rebuilt_weights, support = [], {}, []
    for population in ('full_target', 'exclude_43_pilot_units'):
        for crop in ('coffee_total', 'cacao'):
            for anchor in ('confirmation', 'onset'):
                did = _original_id(population, crop, anchor)
                candidates = [p for p in pairs if p['crop'] == crop and p[anchor + '_2y_computable']
                              and (population == 'full_target' or not p['pilot'])]
                grouped = defaultdict(list)
                for row in candidates:
                    grouped[row['territory']].append(row)
                overlap = {town: rows for town, rows in grouped.items()
                           if {row['contrast'] for row in rows} == {'short', 'long'}}
                rows = []
                for town, events in sorted(overlap.items()):
                    counts = Counter(row['contrast'] for row in events)
                    for pair in sorted(events, key=lambda row: row['pair_uid']):
                        year = pair['confirmation_year'] if anchor == 'confirmation' else pair['onset_year']
                        rows.append(dict(
                            design_id=did, population=population, anchor=anchor,
                            crop=crop, territory=town, state=pair['state'], pair_uid=pair['pair_uid'],
                            contrast=pair['contrast'], current_start_month=pair['current_start_month'],
                            current_end_month=pair['current_end_month'], previous_start_month=pair['previous_start_month'],
                            previous_end_month=pair['previous_end_month'], anchor_year=year,
                            post_first_year=year + 1, post_last_year=year + 2,
                            benchmark_first_year=pair['benchmark_first_year'], benchmark_last_year=pair['benchmark_last_year'],
                            prebenchmark_contains_prior_drought=pair['prebenchmark_contains_prior_drought'],
                            further_onset_through_target_year=pair[anchor + '_further_onset'],
                            current_episode_resolved=pair['current_episode_resolved'],
                            n_events_in_unit_group=counts[pair['contrast']], municipalities_in_contrast=len(overlap),
                            unsigned_event_weight=1 / (len(overlap) * counts[pair['contrast']]),
                            previous_duration_months=pair['previous_duration_months'],
                            previous_severity_z_months=pair['previous_severity_z_months'],
                            current_duration_months=pair['current_duration_months'],
                            current_severity_z_months=pair['current_severity_z_months'],
                            current_mean_tmax_z=pair['current_mean_tmax_z'],
                            pilot_previously_explored=pair['pilot'], endpoint_value_computed=False))
                weights = event_weights(rows, 'original')[0] if rows else {}
                members.extend(rows)
                rebuilt_weights[did] = weights
                support.append(dict(design_id=did, municipalities=len(overlap), eligible_events=len(rows),
                                    short_pairs=sum(r['contrast'] == 'short' for r in rows),
                                    long_pairs=sum(r['contrast'] == 'long' for r in rows),
                                    status='COMPLETE' if rows else 'NO_WITHIN_UNIT_OVERLAP'))
    return dict(episode_rows=ledgers, pair_rows=pairs, membership_rows=members,
                weights=rebuilt_weights, support_rows=support)


def compare_frozen(rebuilt, designs):
    """Verify exact original memberships and weights; matched identities are separate."""
    originals = [d for d in designs if d['kind'] in ('original_confirmation', 'original_onset')]
    _require(len(originals) == 8, 'Frozen source must contain eight original event designs')
    rows = []
    numeric = {'previous_severity_z_months', 'current_severity_z_months',
               'current_mean_tmax_z', 'unsigned_event_weight'}
    boolean = {'prebenchmark_contains_prior_drought', 'further_onset_through_target_year',
               'current_episode_resolved', 'pilot_previously_explored', 'endpoint_value_computed'}
    grouped = defaultdict(list)
    for row in rebuilt['membership_rows']:
        grouped[row['design_id']].append(row)
    for design in originals:
        did = design['design_id']
        actual = {(r['territory'], r['pair_uid']): r for r in grouped[did]}
        expected = {(r['territory'], r['pair_uid']): r for r in design['events']}
        _require(len(actual) == len(grouped[did]) and len(expected) == len(design['events']), 'Duplicate event membership')
        _require(set(actual) == set(expected), 'Detected event membership differs from frozen design: ' + did)
        for key, record in actual.items():
            for field, value in record.items():
                if field not in expected[key]:
                    raise ValueError('Frozen event field missing: ' + field)
                preserved = expected[key][field]
                if field in numeric:
                    _require(math.isclose(float(value), float(preserved), rel_tol=1e-10, abs_tol=1e-8),
                             'Detected numeric event attribute differs: ' + did + '/' + field)
                elif field in boolean:
                    _require(value == _truth(preserved), 'Detected event flag differs: ' + did + '/' + field)
                else:
                    _require(str(value) == str(preserved), 'Detected event identity/window differs: ' + did + '/' + field)
        _require(rebuilt['weights'][did] == design['weights_exact'], 'Detected exact weights differ: ' + did)
        rows.append(dict(design_id=did, event_members=len(actual), exact_annual_weight_rows=len(design['weights_exact']),
                         status='PASS'))
    return rows


def run(input_dir, panel, designs, output_dir, log=print):
    """Public pipeline closure from raw monthly climate to frozen event membership."""
    directory, out = Path(input_dir), Path(output_dir)
    monthly = directory / 'climate_monthly.csv.gz'
    manifest = json.loads((directory / 'input_manifest.json').read_text(encoding='utf-8'))
    expected = manifest['sha256']['climate_monthly.csv.gz']
    _require(sha256(monthly) == expected, 'Monthly source hash changed before event reconstruction')
    errors = []
    def series():
        with gzip.open(monthly, 'rt', encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            _require(reader.fieldnames == MONTH_COLUMNS, 'Monthly event-source schema changed')
            for territory, group in itertools.groupby(reader, key=lambda r: r['territory']):
                rows = list(group)
                _require([r['month'] for r in rows] == list(MONTH_LABELS), 'Incomplete monthly event calendar')
                balance = [finite_number(r['balance_mm'], missing=False) for r in rows]
                z6 = standardize(balance, 6)
                tz = [finite_number(r['tmax_calendar_month_z'], missing=False) for r in rows]
                maximum = 0.0
                for index, (row, value) in enumerate(zip(rows, z6)):
                    saved = finite_number(row['balance_z_6m'])
                    _require((saved is None) == (value is None), 'Original z6 missingness differs from recomputation')
                    if value is not None:
                        error = abs(saved - value)
                        maximum = max(maximum, error)
                        _require(error <= 1e-8, 'Original z6 differs from recomputed rolling water-balance index')
                errors.append(dict(territory=territory, months=len(rows), z6_max_abs_error=maximum))
                if len(errors) % 100 == 0:
                    log('Monthly event reconstruction: ' + str(len(errors)) + ' municipalities')
                yield territory, z6, tz
    rebuilt = reconstruct_from_series(panel, series())
    _require(sha256(monthly) == expected, 'Monthly source changed during event reconstruction')
    checks = compare_frozen(rebuilt, designs)
    for filename, rows in [('episode_ledger.csv.gz', rebuilt['episode_rows']),
                           ('pair_registry.csv.gz', rebuilt['pair_rows']),
                           ('event_membership.csv.gz', rebuilt['membership_rows']),
                           ('design_support.csv', rebuilt['support_rows']),
                           ('monthly_reconstruction.csv', errors),
                           ('frozen_design_reconciliation.csv', checks)]:
        write_table(out / filename, rows)
    summary = dict(status='COMPLETE_EVENT_RECONSTRUCTION', municipalities=len(errors),
                   episodes=len(rebuilt['episode_rows']), adjacent_pairs=len(rebuilt['pair_rows']),
                   original_designs_reproduced=len(checks), monthly_source_sha256=expected,
                   max_z6_abs_error=max((r['z6_max_abs_error'] for r in errors), default=0.0),
                   onset_rule='First of two z6 <= -1.5 months',
                   exit_rule='End before three z6 > -0.5 months; confirmation is the third month',
                   production_magnitudes_used_for_selection=False, production_availability_used=True,
                   source_registry_used_as_reconciliation_target=True,
                   historical_geography_harmonized=False, causal_identification_validated=False)
    write_json(out / 'event_reconstruction_summary.json', summary)
    return summary
