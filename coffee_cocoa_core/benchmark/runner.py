"""Conditional benchmark using fixed generated climate and known response laws.

The default menu has three climate designs, nine conditions, and 400 response
replicates. Only generated arrays enter the estimators. All failed or unsupported
conditions remain explicit and no result-dependent resampling is performed.
"""
from __future__ import annotations
import csv
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .synthetic_dgp import build_design, scenarios, error_replicates, DGP_SPEC
from .legacy_events import build_weights
from .simulation_models import prepare_geometry, evaluate_batch, run_independent_checks

METHODS = ('levels', 'levels_unit_trend', 'first_difference')
CRITICAL = 1.959963984540054


def log(message):
    print(time.strftime('%H:%M:%S') + ' ' + str(message), flush=True)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def table(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    opener = gzip.open if path.name.endswith('.gz') else open
    with opener(path, 'wt', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='raise')
        w.writeheader()
        w.writerows(rows)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def metrics(estimates, variances, structural_truth, projection_truth):
    """Monte Carlo summaries conditional on one fixed climate design.

    No failed replicate is dropped. Projection truth includes omitted background;
    structural truth is the known DGP coefficient/linear function of coefficients.
    """
    est, var = np.asarray(estimates, float), np.asarray(variances, float)
    require(est.ndim == var.ndim == 1 and len(est) == len(var) and len(est) >= 2,
            'Invalid Monte Carlo vector')
    require(np.all(np.isfinite(est)) and np.all(np.isfinite(var)), 'Nonfinite replicate; do not discard')
    tolerance = 1e-10 * max(1.0, float(np.max(np.abs(var))))
    require(float(np.min(var)) >= -tolerance, 'Negative HAC variance beyond floating-point tolerance')
    numerical_zero_count = int(np.sum(var < 0))
    # A PSD Bartlett covariance can be very slightly negative due to rounding.
    se = np.sqrt(np.maximum(var, 0.0))
    lower, upper = est - CRITICAL * se, est + CRITICAL * se
    r = len(est)
    result = dict(replicates_planned=r, replicates_used=r, replicates_failed=0,
                  mean_estimate=float(np.mean(est)), empirical_sd=float(np.std(est, ddof=1)),
                  mcse_mean=float(np.std(est, ddof=1) / np.sqrt(r)),
                  mean_estimated_se=float(np.mean(se)), mean_interval_width=float(np.mean(upper-lower)),
                  mcse_mean_interval_width=float(np.std(upper-lower, ddof=1) / np.sqrt(r)),
                  structural_truth=float(structural_truth), projection_truth=float(projection_truth),
                  projection_minus_structural=float(projection_truth - structural_truth),
                  negative_variances_rounded_to_zero=numerical_zero_count)
    for label, truth in (('structural', structural_truth), ('projection', projection_truth)):
        error = est - truth
        square = error ** 2
        rmse = float(np.sqrt(np.mean(square)))
        coverage = float(np.mean((lower <= truth) & (truth <= upper)))
        result.update({
            'bias_' + label: float(np.mean(error)),
            'mcse_bias_' + label: float(np.std(error, ddof=1) / np.sqrt(r)),
            'rmse_' + label: rmse,
            'mcse_rmse_' + label: float(np.std(square, ddof=1) / (2 * rmse * np.sqrt(r))) if rmse else 0.0,
            'coverage_' + label: coverage,
            'mcse_coverage_' + label: float(np.sqrt(coverage * (1-coverage) / r)),
        })
    result['false_zero_exclusion_when_structural_null'] = (
        float(np.mean((lower > 0) | (upper < 0))) if structural_truth == 0 else '')
    return result, lower, upper


def protocol(seed, replicates, n_units=120, design_ids=(0, 1, 2)):
    return dict(
        analysis='conditional_simulation_benchmark', version=1, seed=int(seed), replicates=int(replicates),
        development_run=replicates < 400 or n_units != 120 or tuple(design_ids) != (0, 1, 2),
        seed_was_default=seed == 20261005, outcome_based_seed_selection=False,
        design_ids=list(design_ids), fixed_climate_designs=len(design_ids), units_per_design=n_units,
        annual_years=[2002, 2025], regression_endpoint_years=[2007, 2025],
        first_difference_underlying_years=[2006, 2025],
        scenarios=9, truth_regimes=['null', 'additive', 'additive_interaction'],
        background_regimes=['stable', 'linear_unit_drift', 'nonlinear_area_expansion'],
        regression_methods=list(METHODS), regression_functionals=['current_dryness_0p1', 'interaction_gamma'],
        regression_truths=['DGP structural function', 'noiseless full-mean model projection'],
        fixed_reference='Mean history H over all fixed regression endpoints, before transformation',
        event_anchors=['confirmation2y', 'onset2y'],
        event_truths=['exact frozen event weight times climate component',
                      'exact frozen event weight times nonclimate component',
                      'exact frozen event weight times full noiseless mean'],
        event_theta_compared_with_gamma=False, event_confidence_intervals_computed=False,
        covariance={'methods': ['calendar_score_HAC_L2', 'calendar_score_HAC_L4'],
                    'kernel': 'Bartlett', 'critical_value': CRITICAL,
                    'calendar_scores': 'Sum over all municipalities per endpoint year',
                    'finite_sample_correction': False},
        common_random_numbers_across_methods_and_scenarios=True,
        climate_fields_fixed_within_replicates=True, climate_fields_randomized_per_replicate=False,
        outcome_noise_calibrated_to_real_data=False, parameter_search=False, clipping=False,
        event_eligibility_seed_search=False, nonestimable_designs_preserved=True,
        real_data_models_refitted=False, all_outcomes_positive_or_stop=True,
        exploratory_design=True, preregistered=False,
        stopping_rule='Stop after the fixed scenario/design menu. Numerical or preservation failure aborts; unsupported designs are recorded. No stop by sign or significance.',
        excluded_from_simulation=['missingness', 'boundary changes', 'reporting breaks', 'farm decisions',
                                  'nonlinear biological drought response', 'unconditional climate variability'],
        interpretation='Conditional synthetic operating characteristics, not proof of actual causal effects or actual confidence-interval coverage.',
        dgp=DGP_SPEC)


def _run_benchmark(out, results, seed=20261005, replicates=400, n_units=120,
                  design_ids=(0, 1, 2), log_fn=log):
    out, results = Path(out), Path(results)
    out.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    panel_rows, event_rows, design_registry, oracle_rows, failures = [], [], [], [], []
    event_registry = []
    completed_scenarios = 0
    max_identity_error = 0.0
    for design_id in design_ids:
        label = 'design_' + str(design_id)
        log_fn('START fixed synthetic climate ' + label + '; no observed outcome calibration')
        design = build_design(seed, design_id, n_units=n_units)
        years = np.asarray(design['years'])
        endpoint = (years >= 2007) & (years <= 2025)
        history = float(np.mean(design['H'][:, endpoint]))
        contrasts = {'current_dryness_0p1': np.array([.1, 0, 0, 0, 0, 0, .1 * history]),
                     'interaction_gamma': np.array([0., 0, 0, 0, 0, 0, 1.])}
        geometries = {method: prepare_geometry(design['X'], design['units'], design['state'], years, method)
                      for method in METHODS}
        legacy = build_weights(design['z6'], design['units'], years)
        directory = out / label
        directory.mkdir()
        arrays = {k: v for k, v in design.items() if isinstance(v, np.ndarray)}
        np.savez_compressed(directory / 'fixed_design.npz', **arrays)
        dump(results / 'designs' / (label + '.json'), dict(
            design_id=design_id, design_metadata=design['metadata'], reference_history_mean=history,
            event_support=legacy['diagnostics'], regression_geometry={
                method: {'status': geometry['status'], 'diagnostics': geometry['diagnostics']}
                for method, geometry in geometries.items()}))
        table(directory / 'legacy_pair_registry.csv.gz', legacy['pairs'])
        for anchor, weights in legacy['weights'].items():
            np.save(directory / ('event_weights_' + anchor + '.npy'), weights, allow_pickle=False)
            event_registry.append(dict(design_id=design_id, anchor=anchor,
                                       weight_sum=float(np.sum(weights)),
                                       weight_absolute_sum=float(np.sum(np.abs(weights))),
                                       units_with_nonzero_weight=int(np.sum(np.any(weights.reshape(n_units, -1) != 0, axis=1))),
                                       eligible_units=legacy['diagnostics']['anchors'][anchor]['eligible_units'],
                                       status='ESTIMABLE' if legacy['diagnostics']['anchors'][anchor]['estimable'] else 'NONESTIMABLE'))
        noise = error_replicates(design, seed, design_id, replicates)
        require(noise.shape == (n_units, len(years), replicates), 'Noise shape differs')
        require(np.all(np.isfinite(noise)) and np.max(np.abs(noise)) <= 120 + 1e-10, 'Error bound violated')
        np.savez_compressed(directory / 'outcome_noise.npz', error=noise)
        for method, geometry in geometries.items():
            design_registry.append(dict(design_id=design_id, method=method, status=geometry['status'],
                                        units=n_units, endpoint_years=19, reference_history_mean=history))
        scenario_list = scenarios(design)
        require(len(scenario_list) == 9 and len({s['id'] for s in scenario_list}) == 9, 'Scenario menu differs')
        for scenario in scenario_list:
            sid = scenario['id']
            log_fn(label + ' scenario ' + str(completed_scenarios + 1) + '/' + str(len(design_ids)*9) + ': ' + sid)
            mean, signal, background = (scenario[k] for k in ('mean', 'climate_component', 'background_component'))
            require(np.allclose(mean, signal + background, rtol=0, atol=1e-10), 'DGP additive identity failed')
            response = mean[:, :, None] + noise
            require(np.all(np.isfinite(response)) and float(np.min(response)) > 0, 'Nonpositive outcome; no clipping or reseeding allowed')
            # Check the normalized production / quantity / harvested-area identity.
            quantity_tonnes = design['A0'][:, None] * mean / 1000
            harvested_ha = design['A0'][:, None] * scenario['area_ratio']
            yield_kg = mean / scenario['area_ratio']
            identity_error = float(np.max(np.abs(quantity_tonnes * 1000 - harvested_ha * yield_kg)))
            require(np.allclose(quantity_tonnes * 1000, harvested_ha * yield_kg, rtol=2e-14, atol=1e-8), 'Production accounting identity failed')
            max_identity_error = max(max_identity_error, identity_error)
            np.savez_compressed(directory / (sid + '_mean_components.npz'),
                                mean=mean, climate_component=signal, background_component=background,
                                area_ratio=scenario['area_ratio'], beta=scenario['beta'])
            prefix = dict(design_id=design_id, scenario=sid, climate_truth=scenario['truth'],
                          background=scenario['background'])
            replicate_records, coefficient_records = [], []
            for method, geometry in geometries.items():
                if geometry['status'] != 'ESTIMABLE':
                    failures.append(dict(prefix, method=method, status='NONESTIMABLE',
                                         replicates_planned=replicates, replicates_used=0,
                                         reason='Fixed design geometry is not estimable; no reseeding.'))
                    continue
                oracle = evaluate_batch(geometry, mean[:, :, None], contrasts)
                fitted = evaluate_batch(geometry, response, contrasts)
                for rep in range(replicates):
                    row = dict(prefix, method=method, replicate=rep)
                    row.update({'beta_' + str(k): float(fitted['coef'][k, rep]) for k in range(7)})
                    coefficient_records.append(row)
                for name, vector in contrasts.items():
                    structural = float(vector @ scenario['beta'])
                    projection = float(oracle['estimates'][name][0])
                    oracle_rows.append(dict(prefix, method=method, functional=name,
                                            structural_truth=structural, projection_truth=projection,
                                            projection_minus_structural=projection-structural))
                    estimates = fitted['estimates'][name]
                    for lag in ('L2', 'L4'):
                        summary, lower, upper = metrics(estimates, fitted['variances'][name][lag], structural, projection)
                        panel_rows.append(dict(prefix, method=method, functional=name, uncertainty='HAC_' + lag,
                                               status='COMPLETE', **summary))
                        for rep in range(replicates):
                            replicate_records.append(dict(prefix, method=method, functional=name, uncertainty='HAC_' + lag,
                                replicate=rep, estimate=float(estimates[rep]), lower=float(lower[rep]), upper=float(upper[rep]),
                                structural_truth=structural, projection_truth=projection))
            table(directory / (sid + '_panel_replicates.csv.gz'), replicate_records)
            table(directory / (sid + '_coefficient_replicates.csv.gz'), coefficient_records)
            event_replicates = []
            for anchor, weights in legacy['weights'].items():
                if not legacy['diagnostics']['anchors'][anchor]['estimable']:
                    failures.append(dict(prefix, method='event_' + anchor, status='NONESTIMABLE',
                                         replicates_planned=replicates, replicates_used=0,
                                         reason='No supported short/long nested event contrast in fixed climate design.'))
                    continue
                signal_truth = float(weights @ signal.ravel())
                background_truth = float(weights @ background.ravel())
                full_truth = float(weights @ mean.ravel())
                require(np.isclose(full_truth, signal_truth+background_truth, rtol=1e-12, atol=1e-9), 'Event oracle decomposition failed')
                estimates = weights @ response.reshape(n_units * len(years), replicates)
                discrepancy = estimates - full_truth
                event_rows.append(dict(prefix, anchor=anchor, status='COMPLETE', replicates_planned=replicates,
                    replicates_used=replicates, replicates_failed=0, climate_signal_contrast=signal_truth,
                    background_contrast=background_truth, full_mean_contrast=full_truth,
                    mean_estimate=float(np.mean(estimates)), mcse_mean=float(np.std(estimates, ddof=1)/np.sqrt(replicates)),
                    empirical_sd=float(np.std(estimates, ddof=1)), bias_own_full_mean=float(np.mean(discrepancy)),
                    rmse_own_full_mean=float(np.sqrt(np.mean(discrepancy**2))),
                    mean_minus_climate_signal=float(np.mean(estimates)-signal_truth),
                    sampling_distribution_q025=float(np.quantile(estimates, .025)),
                    sampling_distribution_q975=float(np.quantile(estimates, .975)),
                    quantiles_are_confidence_intervals=False, compared_with_regression_gamma=False))
                for rep, value in enumerate(estimates):
                    event_replicates.append(dict(prefix, anchor=anchor, replicate=rep, estimate=float(value),
                        climate_signal_contrast=signal_truth, background_contrast=background_truth,
                        full_mean_contrast=full_truth, noise_contrast=float(value-full_truth)))
            table(directory / (sid + '_event_replicates.csv.gz'), event_replicates)
            completed_scenarios += 1
            # Checkpoint complete scenario summaries so a timeout does not erase work.
            table(results / 'simulation_panel_summary.csv', panel_rows)
            table(results / 'simulation_event_decomposition.csv', event_rows)
            table(results / 'simulation_nonestimable.csv', failures,
                  ['design_id', 'scenario', 'climate_truth', 'background', 'method', 'status', 'replicates_planned', 'replicates_used', 'reason'])
            dump(results / 'simulation_progress.json', dict(completed_scenarios=completed_scenarios,
                  planned_scenarios=len(design_ids)*9, fixed_climate_designs=len(design_ids),
                  replicates_per_scenario=replicates, seed=seed, status='RUNNING'))
    table(results / 'simulation_projection_oracles.csv', oracle_rows)
    table(results / 'simulation_design_registry.csv', design_registry)
    table(results / 'simulation_event_support.csv', event_registry)
    summary = dict(status='COMPLETE_WITH_NONESTIMABLE_DESIGNS' if failures else 'COMPLETE_SYNTHESIS_AND_SIMULATION_BENCHMARK',
        completed_scenarios=completed_scenarios, planned_scenarios=len(design_ids)*9,
        fixed_climate_designs=len(design_ids), units_per_design=n_units, replicates_per_scenario=replicates,
        synthetic_response_datasets=completed_scenarios*replicates,
        panel_summary_rows=len(panel_rows), event_summary_rows=len(event_rows), nonestimable_scenario_methods=len(failures),
        seed=seed, development_run=protocol(seed,replicates,n_units,design_ids)['development_run'],
        max_quantity_accounting_abs_error=max_identity_error, numerical_failed_replicates_dropped=0,
        structural_and_projection_truths_separated=True, legacy_and_regression_estimands_separated=True,
        climate_fields_redrawn_per_outcome_replicate=False, real_data_fits_performed=0,
        real_world_interval_coverage_validated=False, causal_effect_identified=False)
    dump(results / 'simulation_progress.json', summary)
    return summary


def run_benchmark(outdir, replicates=400, seed=20261005, *, n_units=120,
                  design_ids=(0, 1, 2), log_fn=log):
    """Run the fixed known-truth benchmark and return metadata and output paths.

    Scientific summary CSV files are written to ``outdir/results``; generated
    arrays and replicate-level tables are retained under ``outdir/design_*``.
    Reduced dimensions are available for software tests and are recorded as
    development runs. Existing design directories are never overwritten.
    """
    for name, value, minimum in (('replicates', replicates, 2), ('seed', seed, 1),
                                 ('n_units', n_units, 6)):
        require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
                and int(value) >= minimum, name + ' must be an integer >= ' + str(minimum))
    design_ids = tuple(design_ids)
    require(bool(design_ids) and len(set(design_ids)) == len(design_ids),
            'Design IDs must be nonempty and unique')
    require(all(isinstance(d, (int, np.integer)) and not isinstance(d, (bool, np.bool_))
                and d >= 0 for d in design_ids), 'Design IDs must be nonnegative integers')
    out = Path(outdir)
    require(not any((out / ('design_' + str(d))).exists() for d in design_ids),
            'Existing design output must not be overwritten; use a new output directory')
    require(n_units % 6 == 0, 'n_units must be divisible by six')
    out.mkdir(parents=True, exist_ok=True)
    results = out / 'results'
    spec = protocol(seed, replicates, n_units, design_ids)
    encoded = json.dumps(spec, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')
    dump(out / 'benchmark_protocol.json', dict(protocol=spec, sha256=hashlib.sha256(encoded).hexdigest()))
    validation = run_independent_checks()
    dump(results / 'independent_software_validation.json', validation)
    summary = _run_benchmark(out, results, seed=seed, replicates=replicates,
                             n_units=n_units, design_ids=design_ids, log_fn=log_fn)
    table_names = ('simulation_panel_summary', 'simulation_event_decomposition',
                   'simulation_nonestimable', 'simulation_projection_oracles',
                   'simulation_design_registry', 'simulation_event_support')
    summary['output_paths'] = {name: str(results / (name + '.csv')) for name in table_names}
    summary['software_validation'] = validation
    dump(results / 'simulation_progress.json', summary)
    dump(out / 'benchmark_summary.json', summary)
    return summary
