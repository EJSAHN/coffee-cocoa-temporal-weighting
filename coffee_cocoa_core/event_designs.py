"""Rebuild the 16 frozen descriptive event designs from portable input tables.

No historical archive, saved effect estimate, network source, or outcome values
are required. Exact Fraction weights, explicit zeros and required eligibility
windows are preserved independently of the production model.
"""
from __future__ import annotations
import csv
import gzip
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

YEARS = tuple(range(2002, 2026))
FILES = {
    'registry': 'legacy/design_registry.csv',
    'weights': 'legacy/consolidated_annual_weights.csv.gz',
    'events': 'legacy/design_event_membership.csv.gz',
    'units': 'legacy/design_unit_registry.csv',
}

class BridgeError(ValueError):
    pass

def _require(ok, message):
    if not ok:
        raise BridgeError(message)

def _fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def _window(row, prefix, length):
    first, last = int(row[prefix + '_first_year']), int(row[prefix + '_last_year'])
    years = tuple(range(first, last + 1))
    if len(years) != length or any(year not in YEARS for year in years):
        raise BridgeError('Invalid frozen window: ' + prefix)
    return years

def event_weights(events, kind):
    """Return exact weights, required outcomes, and fixed municipality/group counts.

    Original events have post_first_year/post_last_year. Matched events have
    onset_post_* and confirmation_post_*. Benchmark requirements are retained
    even for confirmation-minus-onset, for which they cancel algebraically.
    """
    if kind not in ('original', 'matched_onset', 'matched_confirmation', 'matched_anchor_change'):
        raise BridgeError('Unknown linear design kind')
    groups = defaultdict(list)
    identities = set()
    for row in events:
        identity = (row['territory'], row['pair_uid'])
        if identity in identities:
            raise BridgeError('Duplicate event membership')
        identities.add(identity)
        if row['contrast'] not in ('short', 'long'):
            raise BridgeError('Unexpected gap group')
        groups[row['territory']].append(row)
    if not groups:
        raise BridgeError('Empty frozen design')
    weights = defaultdict(Fraction)
    required = set()
    counts = {}
    n = len(groups)
    for territory, unit_events in sorted(groups.items()):
        count = Counter(row['contrast'] for row in unit_events)
        if set(count) != {'short', 'long'}:
            raise BridgeError('Frozen unit is missing a gap group')
        counts[territory] = dict(count)
        for row in unit_events:
            sign = 1 if row['contrast'] == 'short' else -1
            event_weight = Fraction(sign, n * count[row['contrast']])
            benchmark = _window(row, 'benchmark', 4)
            required.update((territory, year) for year in benchmark)
            if kind == 'original':
                post = _window(row, 'post', 2)
                parts = ((benchmark, -1), (post, 1))
                required.update((territory, year) for year in post)
            else:
                onset = _window(row, 'onset_post', 2)
                confirmation = _window(row, 'confirmation_post', 2)
                required.update((territory, year) for year in onset + confirmation)
                if kind == 'matched_onset':
                    parts = ((benchmark, -1), (onset, 1))
                elif kind == 'matched_confirmation':
                    parts = ((benchmark, -1), (confirmation, 1))
                else:
                    parts = ((onset, -1), (confirmation, 1))
            for years, window_sign in parts:
                for year in years:
                    weights[territory, year] += event_weight * Fraction(window_sign, len(years))
    for territory in groups:
        if sum((v for (t, y), v in weights.items() if t == territory), Fraction()) != 0:
            raise BridgeError('Unit coefficients fail exact zero-sum identity')
    return dict(weights), required, counts

def _original_id(population, crop, anchor):
    return f'm6_original__{anchor}2y__{population}__{crop}'

def _expected_ids():
    return {
        (_original_id(pop, crop, kind) if kind in ('confirmation', 'onset')
         else f'stage3m__{kind}__{pop}__{crop}')
        for pop in ('full_target', 'exclude_43_pilot_units')
        for crop in ('coffee_total', 'cacao')
        for kind in ('confirmation', 'onset', 'matched_onset', 'matched_anchor_change')
    }

def _close(actual, expected, message):
    _require(math.isfinite(float(actual)) and math.isfinite(float(expected)) and
             math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-15), message)

def _tables_to_designs(tables):
    """Reconstruct frozen rational coefficients from recorded event windows."""
    registry, events, units, annual_rows = (tables[name] for name in ('registry', 'events', 'units', 'weights'))
    recorded = {row['design_id']: row for row in registry}
    _require(len(recorded) == len(registry) == 16 and set(recorded) == _expected_ids(),
             'Legacy registry does not contain exactly the frozen 16 designs')
    event_groups, unit_groups, preserved = defaultdict(list), defaultdict(list), defaultdict(dict)
    for row in events:
        event_groups[row['design_id']].append(row)
    for row in units:
        unit_groups[row['design_id']].append(row)
    original_ids = {did for did in recorded if did.startswith('m6_original__')}
    _require(set(event_groups) == set(unit_groups) == original_ids, 'Original event/unit design inventory changed')
    for row in annual_rows:
        did = row['design_id']
        _require(did in recorded, 'Unexpected preserved weight design')
        key = (row['territory'], int(row['year']))
        _require(key not in preserved[did] and key[1] in YEARS, 'Duplicate/out-of-range annual weight')
        numerator, denominator = int(row['numerator']), int(row['denominator'])
        _require(denominator > 0, 'Nonpositive exact weight denominator')
        weight = Fraction(numerator, denominator)
        _require((weight.numerator, weight.denominator) == (numerator, denominator), 'Noncanonical exact weight')
        _require(row['crop'] == recorded[did]['crop'], 'Weight crop differs from registry')
        _require(row['exact_zero'] in ('True', 'False') and (row['exact_zero'] == 'True') == (weight == 0),
                 'Exact-zero flag disagrees with rational weight')
        _close(float(weight), row['coefficient'], 'Floating weight differs from exact weight')
        preserved[did][key] = weight
    _require(set(preserved) == set(recorded), 'Missing annual weight design')
    designs, by_id = [], {}
    # First reconstruct all original designs so matched designs can be checked
    # against both original confirmation and onset memberships.
    ordered_ids = sorted(original_ids) + sorted(set(recorded) - original_ids)
    for did in ordered_ids:
        record = recorded[did]
        pop, crop, kind = record['population'], record['crop'], record['kind']
        if did in original_ids:
            anchor = 'confirmation' if kind == 'original_confirmation' else 'onset'
            _require(kind in ('original_confirmation', 'original_onset') and
                     did == _original_id(pop, crop, anchor), 'Original design identity mismatch')
            member_rows = [dict(row) for row in event_groups[did]]
            metadata = unit_groups[did]
            for row in member_rows + metadata:
                _require(row['population'] == pop and row['crop'] == crop and row['anchor'] == anchor,
                         'Original membership metadata mismatch')
            weights, required, counts = event_weights(member_rows, 'original')
        else:
            _require(kind in ('matched_onset', 'matched_anchor_change') and
                     did == f'stage3m__{kind}__{pop}__{crop}', 'Matched design identity mismatch')
            confirmation_id, onset_id = (_original_id(pop, crop, anchor) for anchor in ('confirmation', 'onset'))
            confirmations, onsets = event_groups[confirmation_id], event_groups[onset_id]
            onset_map = {(row['territory'], row['pair_uid']): row for row in onsets}
            _require(len(onset_map) == len(onsets), 'Duplicate original onset event')
            metadata = unit_groups[confirmation_id]
            member_rows = []
            for row in confirmations:
                key = row['territory'], row['pair_uid']
                _require(key in onset_map, 'Matched confirmation event absent from original onset support')
                onset = onset_map[key]
                for field in ('contrast', 'state', 'crop', 'population', 'benchmark_first_year',
                              'benchmark_last_year', 'current_start_month'):
                    _require(row[field] == onset[field], 'Matched onset/confirmation identity/window mismatch')
                member_rows.append(dict(
                    row, design_id=did,
                    onset_post_first_year=onset['post_first_year'],
                    onset_post_last_year=onset['post_last_year'],
                    confirmation_post_first_year=row['post_first_year'],
                    confirmation_post_last_year=row['post_last_year'],
                    onset_anchor_year=onset['anchor_year'], confirmation_anchor_year=row['anchor_year']))
            weights, required, counts = event_weights(member_rows, kind)
        meta_by_unit = {row['territory']: row for row in metadata}
        _require(len(meta_by_unit) == len(metadata) and set(meta_by_unit) == set(counts),
                 'Unit registry differs from event support')
        for row in member_rows:
            meta = meta_by_unit[row['territory']]
            _require(row['state'] == meta['state'], 'Event state differs from preserved unit state')
            start = int(row['current_start_month'][:4])
            _require(2007 <= start <= 2025 and
                     int(row['benchmark_first_year']) == start - 4 and
                     int(row['benchmark_last_year']) == start - 1, 'Frozen B4 window changed')
            if did in original_ids:
                anchor_year = int(row['anchor_year'])
                _require(int(row['post_first_year']) == anchor_year + 1 and
                         int(row['post_last_year']) == anchor_year + 2, 'Frozen post window changed')
                if kind == 'original_onset':
                    _require(anchor_year == start, 'Frozen onset year changed')
                _require(int(row['n_events_in_unit_group']) == counts[row['territory']][row['contrast']] and
                         int(row['municipalities_in_contrast']) == len(counts), 'Event nesting count changed')
                _close(row['unsigned_event_weight'], 1 / (len(counts) * counts[row['territory']][row['contrast']]),
                       'Preserved nested event weight changed')
        for territory, count in counts.items():
            meta = meta_by_unit[territory]
            _require(int(meta['n_short']) == count['short'] and int(meta['n_long']) == count['long'],
                     'Unit group count differs from events')
        _require(weights == preserved[did], 'Event-derived exact annual coefficients disagree: ' + did)
        semantic = [[t, y, w.numerator, w.denominator] for (t, y), w in sorted(weights.items())]
        _require(_fingerprint(semantic) == record['coefficient_fingerprint'], 'Exact coefficient fingerprint mismatch')
        _require(_fingerprint(sorted([list(key) for key in required])) == record['required_outcomes_fingerprint'],
                 'Required-outcome fingerprint mismatch')
        support_counts = dict(municipalities=len(counts), short_pairs=sum(c['short'] for c in counts.values()),
                              long_pairs=sum(c['long'] for c in counts.values()), annual_weight_rows=len(weights))
        for name, value in support_counts.items():
            _require(value == int(record[name]), 'Frozen design support count mismatch: ' + did + '/' + name)
        design = dict(record, **{name: value for name, value in support_counts.items()})
        design.update(territories=sorted(counts), years=list(YEARS),
                      units=[dict(meta_by_unit[t]) for t in sorted(counts)],
                      weights_exact=weights, required_outcomes=required, events=member_rows)
        designs.append(design)
        by_id[did] = design
    for pop in ('full_target', 'exclude_43_pilot_units'):
        for crop in ('coffee_total', 'cacao'):
            confirmation = by_id[_original_id(pop, crop, 'confirmation')]['weights_exact']
            onset = by_id[f'stage3m__matched_onset__{pop}__{crop}']['weights_exact']
            change = by_id[f'stage3m__matched_anchor_change__{pop}__{crop}']['weights_exact']
            for key in set(confirmation) | set(onset) | set(change):
                _require(confirmation.get(key, Fraction()) == onset.get(key, Fraction()) + change.get(key, Fraction()),
                         'Exact matched anchor identity failed')
    return [by_id[row['design_id']] for row in registry]

def load_event_designs(input_dir, *, verify=True):
    """Return reconciled designs, registry and outcome-free source validation.

    A caller that already validated input_manifest.json may pass verify=False
    to avoid rehashing the complete monthly climate file. Every event/weight
    arithmetic, identity and fingerprint check still executes.
    """
    root = Path(input_dir)
    if verify:
        from .data_import import verify_manifest
        manifest = verify_manifest(root, 'input_manifest.json', FILES.values())
    else:
        manifest = json.loads((root / 'input_manifest.json').read_text(encoding='utf-8'))
    tables = {}
    for label, filename in FILES.items():
        path = root / filename
        op = gzip.open if filename.endswith('.gz') else open
        with op(path, 'rt', encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            _require(fields and len(fields) == len(set(fields)), 'Invalid event table header')
            tables[label] = list(reader)
            _require(all(None not in row and None not in row.values() for row in tables[label]),
                     'Malformed portable event table: ' + label)
    designs = _tables_to_designs(tables)
    provenance = dict(
        event_input_sha256={label: manifest['sha256'][filename] for label, filename in FILES.items()},
        designs=len(designs), annual_weight_rows=sum(len(d['weights_exact']) for d in designs),
        all_exact_coefficients_reproduced=True, matched_anchor_identity_verified=True,
        outcome_values_parsed=False, requires_historical_report_archive=False)
    return dict(designs=designs, registry=tables['registry'], provenance=provenance)
