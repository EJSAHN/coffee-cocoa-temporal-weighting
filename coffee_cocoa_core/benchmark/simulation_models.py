"""Fast balanced-panel estimators for fictional fixed-design simulations.

The nuisance space is identical to the original model: unit effects, state by
endpoint year effects, unit-specific fixed parity, and optionally unit trends.
On a complete balanced panel, the common time-basis projection and the state
mean projection commute. No original observations are read by this module.
"""
from __future__ import annotations

import numpy as np

NAMES = ['D0', 'D1', 'D2', 'T0', 'T1', 'T2', 'D0_x_history']
METHODS = ('levels', 'levels_unit_trend', 'first_difference')
RANK_RELATIVE_TOL = 1e-10
ABSORPTION_TOL = 1e-10


def _labels(values, n, name):
    raw = np.asarray(values)
    if raw.ndim != 1 or len(raw) != n:
        raise ValueError(name+' must have one label per unit')
    if any(v is None for v in raw.tolist()):
        raise ValueError(name+' contains missing labels')
    labels = raw.astype(str)
    if any(v.strip().lower() in ('', 'none', 'nan', 'null') for v in labels):
        raise ValueError(name+' contains missing labels')
    return labels


def _project(values, geometry):
    """Orthogonal projection of [unit, endpoint, column/replicate] arrays."""
    q = geometry['time_basis']
    projected = values - q @ (q.T @ values)
    for ids in geometry['state_indices']:
        projected[ids] -= np.mean(projected[ids], axis=0, keepdims=True)
    return projected


def prepare_geometry(X, units, state, years, method):
    x = np.asarray(X, dtype=float)
    if x.ndim != 3 or x.shape[1:] != (24, len(NAMES)) or not x.shape[0]:
        raise ValueError('X must have shape (positive units, 24 calendar years, 7 columns)')
    if not np.all(np.isfinite(x)):
        raise ValueError('Balanced simulation design cannot contain missing or nonfinite entries')
    if method not in METHODS:
        raise ValueError('Unknown frozen estimator: '+str(method))
    year_raw = np.asarray(years)
    if year_raw.ndim != 1 or len(year_raw) != 24:
        raise ValueError('years must identify exactly 2002 through 2025')
    try:
        yr = year_raw.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError('years must identify exactly 2002 through 2025') from exc
    if not np.array_equal(yr, np.arange(2002, 2026)):
        raise ValueError('years must be ordered integer years 2002 through 2025')
    u = _labels(units, len(x), 'units')
    s = _labels(state, len(x), 'state')
    if len(np.unique(u)) != len(u):
        raise ValueError('Each balanced unit identity must be unique')
    endpoint = np.arange(5, 24)
    endpoint_years = yr[endpoint].astype(int)
    columns = [np.ones(len(endpoint)), np.where(endpoint_years % 2 == 0, 1., -1.)]
    if method == 'levels_unit_trend':
        columns.append(endpoint_years-endpoint_years.mean())
    q, _ = np.linalg.qr(np.column_stack(columns), mode='reduced')
    geometry = dict(method=method, n_units=len(x), n_input_years=24,
        endpoint_indices=endpoint, endpoint_years=endpoint_years,
        time_basis=q, state_indices=[np.flatnonzero(s==label) for label in np.unique(s)],
        unit_labels=u, state_labels=s, names=list(NAMES))
    design = x[:, endpoint, :].copy()
    if method == 'first_difference':
        design -= x[:, endpoint-1, :]
    within = _project(design, geometry)
    flat = within.reshape(-1, len(NAMES))
    raw_rms = np.sqrt(np.mean(design*design, axis=(0, 1)))
    within_rms = np.sqrt(np.mean(flat*flat, axis=0))
    if not np.all(np.isfinite(raw_rms)) or not np.all(np.isfinite(within_rms)):
        raise ValueError('Simulation design magnitudes exceed supported numeric range')
    ratios = np.divide(within_rms, raw_rms, out=np.zeros_like(raw_rms), where=raw_rms>0)
    absorbed = [NAMES[j] for j in range(len(NAMES)) if raw_rms[j]==0 or ratios[j]<=ABSORPTION_TOL]
    safe = within_rms.copy(); safe[safe==0] = 1.
    z = flat/safe
    left, singular, vt = np.linalg.svd(z, full_matrices=False)
    largest = float(singular[0]) if len(singular) else 0.
    rank = int(np.sum(singular>RANK_RELATIVE_TOL*largest)) if largest else 0
    reasons = []
    if absorbed:
        reasons.append('regressors_absorbed_or_numerically_negligible_after_fixed_effects')
    if rank<len(NAMES):
        reasons.append('scaled_within_design_rank_deficient')
    status = 'NONESTIMABLE' if reasons else 'ESTIMABLE'
    diagnostics = dict(status=status, method=method, reasons=reasons, n_units=len(x),
        n_states=len(geometry['state_indices']), n_input_years=24,
        n_endpoint_years=len(endpoint), endpoint_years=endpoint_years.tolist(),
        n_observations=len(flat), n_columns=len(NAMES), coefficient_order=list(NAMES),
        rank=rank, absorbed_columns=absorbed, scaled_singular_values=singular.tolist(),
        scaled_condition_number=float(singular[0]/singular[-1]) if len(singular)==7 and singular[-1]>0 else None,
        within_to_input_rms_ratio=dict(zip(NAMES, ratios.tolist())),
        unit_fixed_effects=True, state_by_endpoint_year_fixed_effects=True,
        unit_specific_fixed_parity=True, unit_linear_trends=method=='levels_unit_trend',
        first_difference_unit_effect_interpretation='Underlying level linear drift' if method=='first_difference' else None,
        nuisance_time_basis_rank=q.shape[1], balanced_commuting_projection=True,
        rank_relative_tolerance=RANK_RELATIVE_TOL, absorption_relative_tolerance=ABSORPTION_TOL)
    geometry.update(status=status, diagnostics=diagnostics, within_X=within)
    if status == 'ESTIMABLE':
        geometry['coefficient_map'] = ((vt.T/singular) @ left.T)/safe[:, None]
        geometry['bread'] = ((vt.T/(singular*singular)) @ vt)/safe[:, None]/safe[None, :]
    return geometry


def _fit(geometry, Y):
    if geometry.get('status') != 'ESTIMABLE':
        return dict(status='NONESTIMABLE', coef=None, diagnostics=geometry.get('diagnostics', {}))
    y = np.asarray(Y, dtype=float)
    if y.ndim != 3 or y.shape[:2] != (geometry['n_units'], 24) or not y.shape[2]:
        raise ValueError('Y must have shape (geometry units, 24 calendar years, positive replicates)')
    if not np.all(np.isfinite(y)):
        raise ValueError('Balanced simulation responses cannot contain missing or nonfinite entries')
    endpoint = geometry['endpoint_indices']
    outcome = y[:, endpoint, :].copy()
    if geometry['method'] == 'first_difference':
        outcome -= y[:, endpoint-1, :]
    within = _project(outcome, geometry)
    coef = geometry['coefficient_map'] @ within.reshape(-1, y.shape[2])
    if not np.all(np.isfinite(coef)):
        raise ArithmeticError('Nonfinite simulation coefficients')
    return dict(status='FIT_COMPLETE_EXPLORATORY', coef=coef, within=within)


def fit_batch(geometry, Y):
    result = _fit(geometry, Y)
    return {k:v for k,v in result.items() if k!='within'}


def evaluate_batch(geometry, Y, contrasts):
    result = _fit(geometry, Y)
    if result['status'] != 'FIT_COMPLETE_EXPLORATORY':
        return dict(result, estimates={}, variances={})
    coefficients = result['coef']
    residual = result['within'] - np.einsum('ntp,pr->ntr', geometry['within_X'], coefficients, optimize=True)
    scores = np.einsum('ntp,ntr->tpr', geometry['within_X'], residual, optimize=True)
    if not np.all(np.isfinite(scores)):
        raise ArithmeticError('Nonfinite simulated calendar scores')
    estimates, variances = {}, {}
    for name, original in contrasts.items():
        v = np.asarray(original, dtype=float)
        if v.shape != (7,) or not np.all(np.isfinite(v)):
            raise ValueError('Each fixed contrast must be a finite vector of seven coefficients')
        estimates[name] = v @ coefficients
        projected_scores = np.einsum('p,tpr->tr', geometry['bread'] @ v, scores, optimize=True)
        variances[name] = {}
        for lag in (2, 4):
            raw = np.sum(projected_scores*projected_scores, axis=0)
            magnitude = raw.copy()
            for ell in range(1, min(lag, len(projected_scores)-1)+1):
                weight = 1.-ell/(lag+1.)
                product = projected_scores[ell:]*projected_scores[:-ell]
                raw += 2.*weight*np.sum(product, axis=0)
                magnitude += 2.*weight*np.sum(np.abs(product), axis=0)
            tolerance = 1e-10*np.maximum(magnitude, np.finfo(float).tiny)
            if not np.all(np.isfinite(raw)) or np.any(raw < -tolerance):
                raise ArithmeticError('Materially negative or nonfinite simulated functional HAC variance')
            variances[name]['L'+str(lag)] = np.maximum(raw, 0.)
    return dict(status=result['status'], coef=coefficients, estimates=estimates, variances=variances)


REFERENCE_MODEL_SHA256 = '611d4f105bcabf76ee19d82c17e477513790e9b9c7986d998fe5aaf965802def'
# The independently validated iterative reference implementation has this SHA256.


def _fictional_check_fixture():
    rng = np.random.default_rng(412811)
    n, t, r = 12, 24, 2
    years = np.arange(2002, 2026)
    unit = np.array(['fictional_'+str(j) for j in range(n)])
    state = np.array(['state_'+str(j//4) for j in range(n)])
    base = rng.normal(size=(n, t, 6))
    base[:, :, :3] += 2.
    history = (base[:, :, 1]+base[:, :, 2])/2
    x = np.concatenate([base, (base[:, :, 0]*history)[:, :, None]], axis=2)
    beta = rng.normal(size=(7, r))
    y = np.einsum('ntp,pr->ntr', x, beta)
    y += rng.normal(size=(n, 1, r))*100
    y += (years-years.mean())[None, :, None]*rng.normal(size=(n, 1, r))
    y += np.where(years%2==0, 1., -1.)[None, :, None]*rng.normal(size=(n, 1, r))
    for label in np.unique(state):
        y[state==label] += rng.normal(size=(1, t, r))*3
    y += rng.normal(size=(n, t, r))
    return x, unit, state, years, y


def run_independent_checks():
    """Tiny fictional dense-dummy and unchanged iterative-model comparisons."""
    from .reference_model import fit_model
    x, units, states, years, y = _fictional_check_fixture()
    vectors = {'current': np.array([.1, 0, 0, 0, 0, 0, .2]),
               'history_interaction': np.array([0, 0, 0, 0, 0, 0, 1.])}
    rows = []
    for method in METHODS:
        geometry = prepare_geometry(x, units, states, years, method)
        if geometry['status']!='ESTIMABLE':
            raise AssertionError('Fictional validation design is unexpectedly nonestimable')
        result = evaluate_batch(geometry, y, vectors)
        endpoint = geometry['endpoint_indices']
        xx = x[:, endpoint, :].copy()
        yy = y[:, endpoint, :].copy()
        if method=='first_difference':
            xx -= x[:, endpoint-1, :]
            yy -= y[:, endpoint-1, :]
        xf, yf = xx.reshape(-1, 7), yy.reshape(-1, y.shape[2])
        ul = np.repeat(units, len(endpoint)); sl = np.repeat(states, len(endpoint))
        yl = np.tile(years[endpoint], len(units))
        ud = np.column_stack([ul==u for u in np.unique(ul)]).astype(float)
        sy = np.array([str(s)+'_'+str(t) for s, t in zip(sl, yl)])
        syd = np.column_stack([sy==a for a in np.unique(sy)]).astype(float)
        blocks = [ud, syd, ud*np.where(yl%2==0, 1., -1.)[:, None]]
        if method=='levels_unit_trend':
            blocks.append(ud*(yl-yl.mean())[:, None])
        full = np.column_stack([xf, *blocks])
        dense_beta = np.linalg.lstsq(full, yf, rcond=1e-12)[0][:7]
        if not np.allclose(dense_beta, result['coef'], rtol=1e-7, atol=1e-8):
            raise AssertionError('Balanced simulation coefficients disagree with independent dense dummies')
        coefficient_error = variance_error = 0.
        for rep in range(y.shape[2]):
            original = fit_model(xf, yf[:, rep], ul, sl, yl, NAMES,
                                 unit_trends=method=='levels_unit_trend', unit_parity=True)
            if original['status']!='FIT_COMPLETE_EXPLORATORY':
                raise AssertionError('Unchanged iterative reference model failed fictional check')
            beta = np.array([original['coefficients'][k] for k in NAMES])
            if not np.allclose(beta, result['coef'][:, rep], rtol=1e-7, atol=1e-8):
                raise AssertionError('Balanced simulation coefficients differ from iterative reference')
            coefficient_error = max(coefficient_error, float(np.max(np.abs(beta-result['coef'][:, rep]))))
            for name, v in vectors.items():
                if not np.isclose(v@beta, result['estimates'][name][rep], rtol=1e-7, atol=1e-9):
                    raise AssertionError('Simulation functional estimate differs from reference')
                for lag in ('L2', 'L4'):
                    variance = float(v @ np.array(original['uncertainty']['HAC_'+lag]['covariance']) @ v)
                    actual = float(result['variances'][name][lag][rep])
                    if not np.isclose(variance, actual, rtol=1e-7, atol=1e-10):
                        raise AssertionError('Simulation calendar-score HAC functional variance differs from reference')
                    variance_error = max(variance_error, abs(variance-actual))
        rows.append(dict(method=method, status='PASS', fictional_replicates=y.shape[2],
                         dense_coefficient_max_absolute_error=float(np.max(np.abs(dense_beta-result['coef']))),
                         iterative_coefficient_max_absolute_error=coefficient_error,
                         functional_hac_variance_max_absolute_error=variance_error))
    return dict(status='PASS', actual_study_outcomes_used=False,
                reference_model_sha256=REFERENCE_MODEL_SHA256,
                methods=rows, comparison='Independent dense dummy least squares and independent iterative projection/HAC')
