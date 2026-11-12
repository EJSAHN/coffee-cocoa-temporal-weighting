"""Fixed expansion of the preserved synthetic benchmark across independent designs.

This module redraws complete synthetic designs, not only weather. Unit attributes,
climate, state-year backgrounds, parity and error loadings vary across designs.
Within each design, outcome errors are independent replicates, shared across the
nine original scenarios and model methods. No real-data estimates are computed.

Completed fields can be resumed only after byte-level verification. A failed or
interrupted attempt remains on disk. Unsupported event contrasts are recorded;
they are never replaced with more favorable designs.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback
import uuid

import numpy as np

from coffee_cocoa_core.benchmark import runner as engine
from coffee_cocoa_core.benchmark.synthetic_dgp import DGP_SPEC

SEED = 20261005
DESIGN_IDS = tuple(range(1000, 1100))
UNITS = 120
INNER_REPLICATES = 40
SCENARIOS = tuple(truth + '__' + background
                  for truth in ('null', 'additive', 'additive_interaction')
                  for background in ('stable', 'linear_unit_drift', 'nonlinear_area_expansion'))
ANCHORS = ('confirmation2y', 'onset2y')
METHODS = ('levels', 'levels_unit_trend', 'first_difference')
FUNCTIONALS = ('current_dryness_0p1', 'interaction_gamma')
UNCERTAINTY = ('HAC_L2', 'HAC_L4')
VERSION = 'independent_design_expansion_v1'


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp_' + uuid.uuid4().hex)
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _read_json(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def _read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def _write_csv(path, rows, columns=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = columns or list(dict.fromkeys(k for row in rows for k in row))
    temporary = path.with_name(path.name + '.tmp_' + uuid.uuid4().hex)
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def source_hashes():
    """Hash the preserved engine and this orchestration/aggregation implementation."""
    base = Path(__file__).resolve().parent
    names = [Path('expanded_benchmark.py')]
    names += [p.relative_to(base) for p in sorted((base / 'coffee_cocoa_core' / 'benchmark').glob('*.py'))]
    result = {p.as_posix(): _sha(base / p) for p in names}
    _require(len(result) >= 7, 'Expected complete preserved benchmark engine')
    return result


def expansion_protocol(smoke=False):
    _require(isinstance(smoke, bool), 'smoke must be an explicit boolean')
    ids = DESIGN_IDS[:2] if smoke else DESIGN_IDS
    replicates = 4 if smoke else INNER_REPLICATES
    spec = {
        'version': VERSION,
        'analysis': 'independent_synthetic_design_expansion',
        'seed': SEED,
        'design_ids': list(ids),
        'independent_designs_planned': len(ids),
        'units_per_design': UNITS,
        'inner_response_replicates': replicates,
        'scenarios_per_design': 9,
        'response_datasets_planned': len(ids) * replicates * 9,
        'scenario_ids': list(SCENARIOS),
        'regression_methods': list(METHODS),
        'functionals': list(FUNCTIONALS),
        'uncertainty_methods': list(UNCERTAINTY),
        'event_anchors': list(ANCHORS),
        'annual_years': [2002, 2025],
        'regression_endpoint_years': [2007, 2025],
        'states': 6,
        'original_reference_design_ids': [0, 1, 2],
        'original_reference_recomputed': False,
        'redrawn_between_designs': list(DGP_SPEC['design_rng_children']),
        'fixed_between_designs': ['DGP distributions and parameters', '120 units in six equal states',
                                 'calendar windows', 'nine truth/background conditions',
                                 'model formulas', 'event definitions and nested weights rules',
                                 'HAC L2/L4', 'normal 1.959963984540054 critical value'],
        'within_design': 'One frozen complete design; independent outcome-noise replicates, common random numbers across scenarios and methods.',
        'climate_only_redraw': False,
        'climate_redrawn_per_inner_replicate': False,
        'outcome_based_seed_selection': False,
        'response_clipping': False,
        'replacement_of_unsupported_designs': False,
        'all_planned_designs_in_estimability_denominator': True,
        'nonestimable_fields': 'Retain in registries; conditional performance summaries use only estimable fields and report both numerator and full planned denominator.',
        'MCSE': 'Independent-design blocks: sample SD of one scalar summary per design divided by sqrt(number of estimable designs); not pooled inner replicates.',
        'RMSE': 'sqrt(mean across designs of within-design mean squared error); delta-method MCSE from design-level MSE.',
        'quantiles': 'Across-design empirical quantiles of finite-inner-replicate summaries; not confidence intervals or pure between-design variance estimates.',
        'coverage_targets': ['DGP structural functional', 'within-design noiseless model projection'],
        'event_target': 'Own exact annual-weight functional, split into climate and nonclimate background; never compared with interaction gamma.',
        'estimability_warning_threshold_units': 10,
        'warning_threshold_changes_eligibility': False,
        'real_data_fits': 0,
        'network_downloads': False,
        'figures': False,
        'development_run': smoke,
        'preregistered': False,
        'protocol_fixed_before_this_expansion': True,
        'design_choice_informed_by_previous_benchmark': True,
        'stopping_rule': 'Complete fixed design IDs once. Numerical or integrity failures stop the run; preserve attempts. No stopping or redesign by significance, sign, coverage or eligibility.',
        'scientific_scope': 'Synthetic operating characteristics averaged over this fixed DGP distribution; not evidence of actual causal effects or actual-data interval validity.',
        'preserved_engine_development_flag': 'The inherited engine labels all non-original menus as development runs. This outer protocol is authoritative: only --smoke is a development run; its per-field engine flags are retained unchanged.',
        'dgp': DGP_SPEC,
        'source_sha256': source_hashes(),
    }
    return spec


def _lock_protocol(out, spec):
    path = out / 'expansion_protocol.json'
    digest = hashlib.sha256(_canonical(spec)).hexdigest()
    value = {'protocol': spec, 'sha256': digest}
    if path.exists():
        existing = _read_json(path)
        _require(existing == value, 'Existing expansion protocol differs; use a new output directory. No previous results were overwritten.')
    else:
        _require(not (out / 'fields').exists() and not (out / 'results').exists(),
                 'Unidentified results exist without a matching protocol; use a new output directory')
        _atomic_json(path, value)
    return digest


def _manifest_attempt(attempt, protocol_hash, design_id, summary):
    files = {p.relative_to(attempt).as_posix(): _sha(p)
             for p in sorted(attempt.rglob('*')) if p.is_file() and p.name != 'field_manifest.json'}
    value = {'protocol_sha256': protocol_hash, 'design_id': design_id,
             'engine_status': summary['status'], 'files_sha256': files}
    _atomic_json(attempt / 'field_manifest.json', value)
    return _sha(attempt / 'field_manifest.json')


def _verified_completed(field, protocol_hash, design_id):
    marker = field / 'COMPLETE.json'
    if not marker.exists():
        return None
    record = _read_json(marker)
    _require(record.get('protocol_sha256') == protocol_hash and record.get('design_id') == design_id,
             'Completed field has a different protocol or design identity: ' + str(field))
    relative = Path(record['attempt'])
    _require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe completed attempt path')
    attempt = field / relative
    manifest_path = attempt / 'field_manifest.json'
    _require(manifest_path.is_file() and _sha(manifest_path) == record['manifest_sha256'],
             'Completed field manifest hash differs: ' + str(field))
    manifest = _read_json(manifest_path)
    _require(manifest.get('protocol_sha256') == protocol_hash and manifest.get('design_id') == design_id,
             'Attempt manifest identity differs: ' + str(field))
    expected = manifest['files_sha256']
    actual = {p.relative_to(attempt).as_posix() for p in attempt.rglob('*')
              if p.is_file() and p.name != 'field_manifest.json'}
    _require(actual == set(expected), 'Completed field inventory differs: ' + str(field))
    for name, checksum in expected.items():
        item = Path(name)
        _require(not item.is_absolute() and '..' not in item.parts, 'Unsafe manifest path')
        _require(_sha(attempt / item) == checksum, 'Completed field hash differs: ' + str(attempt / item))
    return attempt


def _float(row, name):
    value = float(row[name])
    _require(math.isfinite(value), 'Nonfinite field summary: ' + name)
    return value


def _summarize_field_scalars(values, prefix):
    """Summary of independent design blocks; one block has no estimable MCSE."""
    values = np.asarray(values, dtype=float)
    _require(values.ndim == 1 and bool(len(values)) and np.all(np.isfinite(values)), 'Invalid field metrics')
    n = len(values)
    result = {prefix: float(np.mean(values)),
              prefix + '_mcse': float(np.std(values, ddof=1) / np.sqrt(n)) if n >= 2 else '',
              prefix + '_across_design_min': float(np.min(values)),
              prefix + '_across_design_q025': float(np.quantile(values, .025)),
              prefix + '_across_design_median': float(np.median(values)),
              prefix + '_across_design_q975': float(np.quantile(values, .975)),
              prefix + '_across_design_max': float(np.max(values))}
    return result


def _rmse_from_field_mse(mses, prefix):
    values = np.asarray(mses, dtype=float)
    _require(values.ndim == 1 and len(values) and np.all(np.isfinite(values)) and np.all(values >= 0),
             'Invalid field mean squared errors')
    value = float(np.sqrt(np.mean(values)))
    if len(values) < 2:
        mcse = ''
    elif value:
        mcse = float(np.std(values, ddof=1) / (2 * value * np.sqrt(len(values))))
    else:
        mcse = 0.0
    return {prefix: value, prefix + '_mcse_delta': mcse}


def aggregate_panel(rows, planned_designs, inner_replicates):
    """Equal-design averages; RMSE aggregates squared errors before the square root."""
    _require(planned_designs >= 1 and inner_replicates >= 2, 'Invalid planned replication counts')
    output = []
    for scenario in SCENARIOS:
        for method in METHODS:
            for functional in FUNCTIONALS:
                for uncertainty in UNCERTAINTY:
                    matched = [r for r in rows if (r['scenario'], r['method'], r['functional'], r['uncertainty']) ==
                               (scenario, method, functional, uncertainty)]
                    n = len(matched)
                    _require(len({int(r['design_id']) for r in matched}) == n, 'Duplicate panel design summary')
                    _require(n <= planned_designs, 'More panel designs than planned')
                    result = dict(scenario=scenario, method=method, functional=functional, uncertainty=uncertainty,
                                  designs_planned=planned_designs, designs_estimable=n,
                                  estimable_fraction_of_planned=n / planned_designs,
                                  outcome_replicates_used=n * inner_replicates,
                                  inner_replicates_per_design=inner_replicates,
                                  status='ESTIMABLE' if n else 'NONESTIMABLE_ALL_DESIGNS',
                                  performance_conditional_on_estimable_designs=True,
                                  quantiles_are_confidence_intervals=False,
                                  mcse_unit='independent_complete_design')
                    if matched:
                        for row in matched:
                            _require(int(row['replicates_used']) == inner_replicates and int(row['replicates_failed']) == 0,
                                     'Field summary silently lost outcome replicates')
                        metrics = ('mean_estimate', 'structural_truth', 'projection_truth', 'projection_minus_structural',
                                   'bias_structural', 'bias_projection', 'coverage_structural', 'coverage_projection',
                                   'mean_estimated_se', 'mean_interval_width')
                        for metric in metrics:
                            result.update(_summarize_field_scalars([_float(r, metric) for r in matched], metric))
                        result.update(_summarize_field_scalars([_float(r, 'empirical_sd') for r in matched],
                                                               'mean_within_design_empirical_sd'))
                        for target in ('structural', 'projection'):
                            mses = [_float(r, 'rmse_' + target) ** 2 for r in matched]
                            result.update(_rmse_from_field_mse(mses, 'rmse_' + target))
                            result.update(_summarize_field_scalars(mses, 'mse_' + target))
                        gap = np.asarray([_float(r, 'projection_minus_structural') for r in matched])
                        bias_projection = np.asarray([_float(r, 'bias_projection') for r in matched])
                        result.update(_summarize_field_scalars(gap ** 2, 'squared_projection_displacement'))
                        result.update(_summarize_field_scalars(2 * gap * bias_projection, 'projection_noise_cross_term'))
                        result['mse_decomposition_abs_error'] = abs(result['mse_structural'] -
                            (result['mse_projection'] + result['squared_projection_displacement'] + result['projection_noise_cross_term']))
                        _require(result['mse_decomposition_abs_error'] <= 1e-8 * max(1, result['mse_structural']),
                                 'Structural/projection MSE decomposition failed')
                        if scenario.startswith('null__') or (scenario.startswith('additive__') and functional == 'interaction_gamma'):
                            result.update(_summarize_field_scalars(
                                [_float(r, 'false_zero_exclusion_when_structural_null') for r in matched],
                                'false_zero_exclusion_when_structural_null'))
                    output.append(result)
    return output


def aggregate_event_support(rows, planned_designs):
    output = []
    for anchor in ANCHORS:
        selected = [r for r in rows if r['anchor'] == anchor]
        _require(len({int(r['design_id']) for r in selected}) == len(selected) <= planned_designs,
                 'Duplicate or excessive event support designs')
        counts = np.asarray([int(r['eligible_units']) for r in selected], dtype=int)
        _require(np.all(counts >= 0), 'Negative event eligibility')
        for r in selected:
            _require((int(r['eligible_units']) > 0) == (r['status'] == 'ESTIMABLE'), 'Event support status/count mismatch')
        result = dict(anchor=anchor, designs_planned=planned_designs, designs_evaluated=len(selected),
                      designs_estimable=int(np.sum(counts > 0)), designs_nonestimable=int(np.sum(counts == 0)),
                      estimable_fraction_of_planned=float(np.sum(counts > 0) / planned_designs),
                      nonestimable_fraction_of_planned=float(np.sum(counts == 0) / planned_designs),
                      designs_1_to_9_units=int(np.sum((counts >= 1) & (counts <= 9))),
                      designs_at_least_10_units=int(np.sum(counts >= 10)),
                      eligible_units_zero_designs=int(np.sum(counts == 0)),
                      eligible_units_one_designs=int(np.sum(counts == 1)),
                      reporting_threshold_changes_eligibility=False)
        if len(counts):
            result.update(_summarize_field_scalars(counts, 'eligible_units'))
        output.append(result)
    return output


def aggregate_events(rows, planned_designs, inner_replicates):
    output = []
    for scenario in SCENARIOS:
        for anchor in ANCHORS:
            selected = [r for r in rows if r['scenario'] == scenario and r['anchor'] == anchor]
            n = len(selected)
            _require(len({int(r['design_id']) for r in selected}) == n <= planned_designs,
                     'Duplicate or excessive event summary designs')
            result = dict(scenario=scenario, anchor=anchor, designs_planned=planned_designs,
                          designs_estimable=n, estimable_fraction_of_planned=n / planned_designs,
                          outcome_replicates_used=n * inner_replicates,
                          status='ESTIMABLE' if n else 'NONESTIMABLE_ALL_DESIGNS',
                          performance_conditional_on_estimable_designs=True,
                          event_compared_with_interaction_gamma=False, event_confidence_intervals_computed=False,
                          quantiles_are_confidence_intervals=False, mcse_unit='independent_complete_design')
            if selected:
                for row in selected:
                    _require(int(row['replicates_used']) == inner_replicates and int(row['replicates_failed']) == 0,
                             'Event summary silently lost outcome replicates')
                metrics = ('climate_signal_contrast', 'background_contrast', 'full_mean_contrast',
                           'mean_estimate', 'bias_own_full_mean', 'mean_minus_climate_signal')
                for metric in metrics:
                    result.update(_summarize_field_scalars([_float(r, metric) for r in selected], metric))
                result.update(_rmse_from_field_mse([_float(r, 'rmse_own_full_mean') ** 2 for r in selected],
                                                   'rmse_own_full_mean'))
                result.update(_summarize_field_scalars([_float(r, 'empirical_sd') for r in selected],
                                                       'mean_within_design_empirical_sd'))
                for metric in ('background_contrast', 'climate_signal_contrast', 'full_mean_contrast'):
                    result[metric + '_fraction_positive'] = sum(_float(r, metric) > 0 for r in selected) / n
                    result[metric + '_fraction_negative'] = sum(_float(r, metric) < 0 for r in selected) / n
            output.append(result)
    return output


def _validate_field_tables(tables, design_id, replicates):
    registry = tables['simulation_design_registry']
    support = tables['simulation_event_support']
    panel = tables['simulation_panel_summary']
    event = tables['simulation_event_decomposition']
    _require(len(registry) == 3 and {r['method'] for r in registry} == set(METHODS), 'Missing field model geometry')
    _require(len(support) == 2 and {r['anchor'] for r in support} == set(ANCHORS), 'Missing field event support')
    for name, rows in tables.items():
        _require(all(int(r['design_id']) == design_id for r in rows), 'Mixed field identity in ' + name)
    estimable_methods = {r['method'] for r in registry if r['status'] == 'ESTIMABLE'}
    _require(all(r['status'] in ('ESTIMABLE', 'NONESTIMABLE') or r['status'].startswith('NONESTIMABLE')
                 for r in registry), 'Unexpected model geometry status')
    expected_panel = {(s, m, f, u) for s in SCENARIOS for m in estimable_methods for f in FUNCTIONALS for u in UNCERTAINTY}
    observed_panel = [(r['scenario'], r['method'], r['functional'], r['uncertainty']) for r in panel]
    _require(len(observed_panel) == len(set(observed_panel)) and set(observed_panel) == expected_panel,
             'Missing/duplicate planned field panel summaries')
    estimable_anchors = {r['anchor'] for r in support if r['status'] == 'ESTIMABLE'}
    expected_events = {(s, a) for s in SCENARIOS for a in estimable_anchors}
    observed_events = [(r['scenario'], r['anchor']) for r in event]
    _require(len(observed_events) == len(set(observed_events)) and set(observed_events) == expected_events,
             'Missing/duplicate planned field event summaries')
    for row in panel + event:
        _require(int(row['replicates_used']) == replicates and int(row['replicates_failed']) == 0,
                 'Failed replicate must not be discarded')
    failures = tables['simulation_nonestimable']
    expected_failures = {(s, m) for s in SCENARIOS for m in set(METHODS) - estimable_methods}
    expected_failures |= {(s, 'event_' + a) for s in SCENARIOS for a in set(ANCHORS) - estimable_anchors}
    observed_failures = [(r['scenario'], r['method']) for r in failures]
    _require(len(observed_failures) == len(set(observed_failures)) and set(observed_failures) == expected_failures,
             'Unsupported field conditions must all be recorded')


def _load_field(attempt, design_id, replicates):
    names = ('simulation_design_registry', 'simulation_event_support', 'simulation_panel_summary',
             'simulation_event_decomposition', 'simulation_nonestimable', 'simulation_projection_oracles')
    tables = {name: _read_csv(attempt / 'results' / (name + '.csv')) for name in names}
    _validate_field_tables(tables, design_id, replicates)
    return tables


def run_expansion(outdir, *, smoke=False, log_fn=print):
    """Run/resume fixed outer-design expansion; independent engine checks run upstream.

    Output is self-contained below ``outdir``. ``results`` contains review tables;
    ``fields/design_ID/attempts`` retains every generated array and replicate.
    This function never downloads inputs or computes actual-data estimates.
    ``smoke=True`` is explicitly non-scientific and cannot resume a full run.
    """
    out = Path(outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    spec = expansion_protocol(smoke)
    digest = _lock_protocol(out, spec)
    # NumPy algorithms and RNG implementation are part of a resumable numerical run.
    runtime_path = out / 'numerical_runtime.json'
    runtime = {'numpy': np.__version__}
    if runtime_path.exists():
        _require(_read_json(runtime_path) == runtime, 'NumPy version differs from saved expansion; use a new output directory')
    else:
        _atomic_json(runtime_path, runtime)
    results = out / 'results'
    results.mkdir(exist_ok=True)
    all_tables = {}
    progress = []
    newly_elapsed = []
    ids = spec['design_ids']
    replicates = spec['inner_response_replicates']
    started = time.monotonic()
    current_id = None
    attempt = None
    try:
        for position, design_id in enumerate(ids, 1):
            current_id = design_id
            field = out / 'fields' / ('design_' + str(design_id))
            attempt = None
            attempt = _verified_completed(field, digest, design_id)
            cached = attempt is not None
            field_started = time.monotonic()
            if cached:
                log_fn('VERIFY completed independent design ' + str(position) + '/' + str(len(ids)) + ': ' + str(design_id))
            else:
                attempt = field / 'attempts' / (time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8])
                attempt.mkdir(parents=True, exist_ok=False)
                _atomic_json(attempt / 'attempt_identity.json', {'design_id': design_id, 'protocol_sha256': digest,
                                                                'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
                log_fn('START independent design ' + str(position) + '/' + str(len(ids)) + ': ' + str(design_id))
                with (attempt / 'field_run.log').open('w', encoding='utf-8') as field_log:
                    def detail(message):
                        field_log.write(time.strftime('%H:%M:%S') + ' ' + str(message) + '\n')
                        field_log.flush()
                    summary = engine._run_benchmark(attempt, attempt / 'results', seed=SEED, replicates=replicates,
                                                    n_units=UNITS, design_ids=(design_id,), log_fn=detail)
                _require(summary['completed_scenarios'] == 9 and summary['numerical_failed_replicates_dropped'] == 0,
                         'Field ended before all conditions completed')
                _atomic_json(attempt / 'field_summary.json', summary)
            tables = _load_field(attempt, design_id, replicates)
            if not cached:
                manifest_hash = _manifest_attempt(attempt, digest, design_id, summary)
                _atomic_json(field / 'COMPLETE.json', {'design_id': design_id, 'protocol_sha256': digest,
                                                       'attempt': attempt.relative_to(field).as_posix(),
                                                       'manifest_sha256': manifest_hash})
                newly_elapsed.append(time.monotonic() - field_started)
            for name, rows in tables.items():
                all_tables.setdefault(name, []).extend(rows)
            progress.append({'design_id': design_id, 'status': 'COMPLETE', 'reused_verified_field': cached,
                             'seconds_this_invocation': round(time.monotonic() - field_started, 3),
                             'attempt': attempt.relative_to(out).as_posix()})
            _write_csv(results / 'field_completion_registry.csv', progress)
            remaining = len(ids) - position
            eta = remaining * float(np.mean(newly_elapsed)) if newly_elapsed else None
            _atomic_json(results / 'expansion_progress.json', {'status': 'RUNNING', 'designs_completed': position,
                         'designs_planned': len(ids), 'protocol_sha256': digest,
                         'elapsed_seconds_this_invocation': round(time.monotonic() - started, 2),
                         'rough_remaining_seconds': None if eta is None else round(eta, 1)})
            log_fn('COMPLETE independent designs ' + str(position) + '/' + str(len(ids)) +
                   ('; rough remaining %.1f min' % (eta / 60) if eta is not None else '; verified saved design'))
        log_fn('START independent-design aggregation; MCSE uses outer designs, not pooled outcome replicates')
        for name, rows in all_tables.items():
            _write_csv(results / ('by_design_' + name + '.csv'), rows)
        panel = aggregate_panel(all_tables['simulation_panel_summary'], len(ids), replicates)
        event_support = aggregate_event_support(all_tables['simulation_event_support'], len(ids))
        events = aggregate_events(all_tables['simulation_event_decomposition'], len(ids), replicates)
        _write_csv(results / 'panel_performance_across_designs.csv', panel)
        _write_csv(results / 'event_support_across_designs.csv', event_support)
        _write_csv(results / 'event_decomposition_across_designs.csv', events)
        geometry = []
        for method in METHODS:
            selected = [r for r in all_tables['simulation_design_registry'] if r['method'] == method]
            n = sum(r['status'] == 'ESTIMABLE' for r in selected)
            geometry.append(dict(method=method, designs_planned=len(ids), designs_evaluated=len(selected),
                                 designs_estimable=n, designs_nonestimable=len(selected)-n,
                                 estimable_fraction_of_planned=n / len(ids)))
        _write_csv(results / 'panel_estimability_across_designs.csv', geometry)
        warnings = []
        for row in event_support:
            if row['designs_1_to_9_units'] or row['designs_nonestimable']:
                warnings.append(row['anchor'] + ': ' + str(row['designs_nonestimable']) + ' nonestimable designs and ' +
                                str(row['designs_1_to_9_units']) + ' designs with 1-9 eligible units; all retained; no threshold-based exclusion.')
        if any(r['designs_nonestimable'] for r in geometry):
            warnings.append('Some panel geometries are nonestimable; performance summaries condition on estimability.')
        summary = {
            'status': 'COMPLETE_EXPANSION_SMOKE_TEST' if smoke else 'COMPLETE_INDEPENDENT_DESIGN_EXPANSION',
            'development_run': smoke, 'protocol_sha256': digest,
            'designs_planned': len(ids), 'designs_completed': len(ids),
            'completed_fields': len(ids),
            'designs_reused_verified': sum(r['reused_verified_field'] for r in progress),
            'designs_new_this_invocation': len(newly_elapsed),
            'units_per_design': UNITS, 'inner_replicates_per_design_scenario': replicates,
            'scenario_design_cells': len(ids) * 9,
            'synthetic_response_datasets': len(ids) * 9 * replicates,
            'panel_performance_rows': len(panel), 'event_performance_rows': len(events),
            'panel_summary_design_rows': len(all_tables['simulation_panel_summary']),
            'event_summary_design_rows': len(all_tables['simulation_event_decomposition']),
            'nonestimable_scenario_methods': len(all_tables['simulation_nonestimable']),
            'numerical_failed_replicates_dropped': 0,
            'elapsed_seconds_this_invocation': round(time.monotonic() - started, 2),
            'mean_seconds_new_design': round(float(np.mean(newly_elapsed)), 3) if newly_elapsed else None,
            'support_warnings': warnings,
            'outer_independent_designs_are_mcse_units': True,
            'structural_and_projection_truths_separated': True,
            'RMSE_aggregated_from_MSE': True,
            'inner_and_outer_replicates_not_treated_as_independent': True,
            'performance_conditioned_on_estimability': True,
            'event_confidence_intervals_computed': False,
            'independent_engine_software_validation': 'Performed once by the calling local worker before expansion; engine sources hash-locked here.',
            'real_data_fits': 0, 'causal_effect_identified': False,
            'scientific_scope': spec['scientific_scope'],
        }
        _atomic_json(results / 'expansion_summary.json', summary)
        _atomic_json(results / 'expansion_progress.json', summary)
        return summary
    except BaseException as exc:
        error = {'status': 'FAILED_INDEPENDENT_DESIGN_EXPANSION', 'protocol_sha256': digest,
                 'current_design_id': current_id, 'completed_designs': len(progress),
                 'planned_designs': len(ids), 'error': type(exc).__name__ + ': ' + str(exc),
                 'traceback': traceback.format_exc(), 'completed_fields_and_failed_attempts_retained': True,
                 'failed_attempt': str(attempt) if attempt else None}
        # An integrity failure while reusing a completed attempt must not mutate it.
        _atomic_json(results / ('failed_invocation_' + uuid.uuid4().hex[:12] + '.json'), error)
        _atomic_json(results / 'expansion_progress.json', error)
        raise
