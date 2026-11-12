"""Current-exposure derivatives and exact linear-functional contributions.

The reference design sets the averaging population, not an estimated new model.
All derivatives hold the prior exposure and temperature regressors fixed. The
reported 0.1-scaled derivative is not the finite change over a 0.1 exposure step.
Contribution weights are algebraic weights, not independent-information counts.
"""
from __future__ import annotations

import math
import numpy as np

from .fixed_effects import NORMAL_975, RANK_RELATIVE_TOL, prepare_design, residualize


def current_slope(fit, reference_X, reference_names, scale=0.1):
    """Return a fixed-reference averaged partial derivative and delta intervals.

    Supports the frozen full panel linear or quadratic specifications only. The
    gradient for history-squared and temperature terms is zero because history
    and temperature are held fixed. ``reference_X`` must already reflect any
    preceding-calendar-window shift, as returned by features.select_data.
    """
    if fit.get('status') != 'FIT_COMPLETE_EXPLORATORY':
        return {'status': 'NONESTIMABLE', 'source_fit_status': fit.get('status'),
                'reason': 'Source fit did not complete; no derivative estimated.'}
    names = list(fit['coefficient_order'])
    allowed = {'D0', 'D1', 'D2', 'T0', 'T1', 'T2', 'D0_x_history',
               'D0_squared', 'history_squared', 'T0_squared'}
    if len(set(names)) != len(names) or set(names) - allowed:
        raise ValueError('Unsupported or duplicated source coefficient names')
    required = {'D0', 'D0_x_history'}
    if not required.issubset(names):
        raise ValueError('Source fit is missing current exposure or its interaction')
    ref_names = list(reference_names)
    ref = np.asarray(reference_X, dtype=float)
    if ref.ndim != 2 or not len(ref) or ref.shape[1] != len(ref_names):
        raise ValueError('Reference design must be nonempty with matching column names')
    if len(set(ref_names)) != len(ref_names) or not {'D0', 'D1', 'D2'}.issubset(ref_names):
        raise ValueError('Reference names must be unique and contain D0, D1 and D2')
    if not np.all(np.isfinite(ref)):
        raise ValueError('Reference design contains nonfinite values')
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Derivative display scale must be finite and positive')
    d0mean = float(np.mean(ref[:, ref_names.index('D0')]))
    hmean = float(np.mean((ref[:, ref_names.index('D1')] +
                           ref[:, ref_names.index('D2')]) / 2.0))
    gradient = np.zeros(len(names))
    gradient[names.index('D0')] = 1.0
    gradient[names.index('D0_x_history')] = hmean
    if 'D0_squared' in names:
        gradient[names.index('D0_squared')] = 2.0 * d0mean
    gradient *= scale
    beta = np.asarray([fit['coefficients'][n] for n in names], dtype=float)
    if not np.all(np.isfinite(beta)):
        raise ValueError('Source fit contains nonfinite coefficients')
    estimate = float(gradient @ beta)
    intervals = {}
    for key in ('HAC_L2', 'HAC_L4'):
        cov = np.asarray(fit['uncertainty'][key]['covariance'], dtype=float)
        if cov.shape != (len(names), len(names)) or not np.all(np.isfinite(cov)):
            raise ValueError('Source covariance is nonfinite or has the wrong shape')
        magnitude = max(float(np.max(np.abs(cov))), np.finfo(float).tiny)
        if np.max(np.abs(cov - cov.T)) > 1e-10 * magnitude:
            raise ArithmeticError('Source covariance is materially asymmetric')
        cov = (cov + cov.T) / 2.0
        variance = float(gradient @ cov @ gradient)
        # Only relative floating point error is tolerated; no fixed absolute
        # floor can hide a materially negative variance in a small-scale model.
        variance_scale = max(float(np.abs(gradient) @ np.abs(cov) @ np.abs(gradient)),
                             np.finfo(float).tiny)
        if variance < -1e-10 * variance_scale:
            raise ArithmeticError('Derivative variance is materially negative')
        se = math.sqrt(max(variance, 0.0))
        intervals[key] = {'estimate': estimate, 'variance': max(variance, 0.0),
                          'unrounded_variance': variance, 'standard_error': se,
                          'lower': estimate - NORMAL_975 * se,
                          'upper': estimate + NORMAL_975 * se,
                          'critical_value': NORMAL_975,
                          'interpretation': 'Exploratory calendar-score HAC normal interval; few-year coverage is not guaranteed.'}
    return {'status': 'COMPLETE_CURRENT_SLOPE', 'estimate': estimate,
            'estimand': 'reference_averaged_partial_derivative_of_outcome_with_respect_to_current_D',
            'display_scale': scale,
            'interpretation': 'Display-scaled local derivative, not a finite-step prediction or causal effect; temperature and prior exposure held fixed.',
            'reference_n': len(ref), 'reference_D0_mean': d0mean,
            'reference_history_mean': hmean, 'coefficient_order': names,
            'scaled_gradient': gradient.tolist(),
            'gradient_by_coefficient': dict(zip(names, gradient.tolist())),
            'uncertainty': intervals}


def generate_contributions(data, spec, fit, v_scaled):
    """Return (per-row contributions, independent algebraic validation).

    If Xw = U S V', then a = U S^-1 V'v satisfies Xw'a = v and
    a'yw = v'beta. Columns are RMS-scaled before this SVD. This avoids forming
    or inverting Xw'Xw and does not square its condition number.
    """
    if fit.get('status') != 'FIT_COMPLETE_EXPLORATORY':
        raise ValueError('Contributions require a completed source fit')
    names = list(data['names'])
    if names != list(fit['coefficient_order']):
        raise ValueError('Contribution design and coefficient order differ')
    v = np.asarray(v_scaled, dtype=float)
    if v.shape != (len(names),) or not np.all(np.isfinite(v)):
        raise ValueError('Contribution gradient must match finite coefficient vector')
    if not np.any(v):
        raise ValueError('Contribution gradient must not be identically zero')
    trends = bool(spec.get('unit_trends', False))
    parity = bool(spec.get('unit_parity', False))
    if trends != bool(fit.get('unit_linear_trends', False)) or parity != bool(fit.get('unit_specific_fixed_parity', False)):
        raise ValueError('Contribution nuisance specification differs from source fit')
    xw, design = prepare_design(data['X'], data['unit'], data['state'], data['year'],
                                names, unit_trends=trends, unit_parity=parity)
    if design['status'] != 'ESTIMABLE':
        raise ArithmeticError('Previously fitted contribution design is no longer estimable')
    y = np.asarray(data['y'], dtype=float)
    yw, ycheck = residualize(y, data['unit'], data['state'], data['year'],
                             unit_trends=trends, unit_parity=parity)
    if not ycheck['converged']:
        raise ArithmeticError('Contribution response projection failed')
    scales = np.sqrt(np.mean(xw*xw, axis=0))
    z = xw / scales
    u, singular, vt = np.linalg.svd(z, full_matrices=False)
    if singular[-1] <= RANK_RELATIVE_TOL * singular[0]:
        raise ArithmeticError('Contribution design became rank deficient')
    a = u @ ((vt @ (v / scales)) / singular)
    contribution = a * yw
    raw_contribution = a * y
    beta = np.asarray([fit['coefficients'][n] for n in names], dtype=float)
    target = float(v @ beta)
    reproduced = float(np.sum(contribution))
    raw_reproduced = float(np.sum(raw_contribution))
    gradient_error = xw.T @ a - v
    residual_a, acheck = residualize(a, data['unit'], data['state'], data['year'],
                                    unit_trends=trends, unit_parity=parity)
    weight_projection_relative_error = float(np.linalg.norm(a - residual_a) /
                                             max(np.linalg.norm(a), np.finfo(float).tiny))
    # The first validation is coefficient-by-coefficient; the latter permit only
    # relative roundoff/projection error, including cancellation across rows.
    gradient_tolerance = 1e-8 * max(float(np.max(np.abs(v))), np.finfo(float).tiny)
    response_scale = max(abs(target), float(np.sum(np.abs(contribution))),
                         float(np.sum(np.abs(raw_contribution))), np.finfo(float).tiny)
    arithmetic_tolerance = 1e-8 * response_scale
    if np.max(np.abs(gradient_error)) > gradient_tolerance:
        raise ArithmeticError('Contribution weights do not reproduce requested gradient')
    if abs(reproduced - target) > arithmetic_tolerance or abs(raw_reproduced - target) > arithmetic_tolerance:
        raise ArithmeticError('Contribution sum does not reproduce source-fit derivative')
    if not acheck['converged'] or weight_projection_relative_error > 1e-8:
        raise ArithmeticError('Contribution weights are not orthogonal to nuisance effects')
    energy = a*a
    energy /= np.sum(energy)
    meta = data.get('rows')
    if meta is not None and len(meta) != len(y):
        raise ValueError('Contribution metadata length differs from model rows')
    rows = []
    for j in range(len(y)):
        source = meta[j] if meta is not None else {}
        rows.append({'model_id': spec.get('model_id', ''), 'crop': spec.get('crop', ''),
                     'outcome': spec.get('outcome', ''), 'variant': spec.get('variant', ''),
                     'territory': str(source.get('territory', data['unit'][j])),
                     'unit': str(data['unit'][j]), 'state': str(data['state'][j]),
                     'year': int(data['year'][j]), 'coefficient_weight': float(a[j]),
                     'response_within': float(yw[j]), 'response_contribution': float(contribution[j]),
                     'raw_response_contribution': float(raw_contribution[j]),
                     'contrast_weight_energy_share': float(energy[j])})
    validation = {'status': 'PASS', 'target': target, 'contribution_sum': reproduced,
                  'raw_response_contribution_sum': raw_reproduced,
                  'sum_error': reproduced - target, 'raw_sum_error': raw_reproduced - target,
                  'gradient_reproduction_max_abs_error': float(np.max(np.abs(gradient_error))),
                  'nuisance_weight_projection_relative_error': weight_projection_relative_error,
                  'contrast_weight_energy_share_sum': float(np.sum(energy)),
                  'n_rows': len(rows), 'scaled_condition_number': float(singular[0]/singular[-1]),
                  'arithmetic_tolerance': arithmetic_tolerance,
                  'gradient_tolerance': gradient_tolerance,
                  'interpretation': 'Exact linear-functional arithmetic; contributions are not causal shares and squared weight concentration does not count independent observations.'}
    return rows, validation
