"""Fully synthetic, frozen climate designs and known production-response laws.

Nothing in this module reads research data.  Designs, truth coefficients and
bounded errors are fixed by explicit SeedSequence inputs; there is no favorable
seed search, response clipping, model selection, or noise re-drawing.
"""
from __future__ import annotations

import math

import numpy as np

COEFFICIENT_NAMES = ['D0', 'D1', 'D2', 'T0', 'T1', 'T2', 'D0_x_history']
MAX_ERROR_AMPLITUDE = 120.0
DGP_SPEC = {
    'version': 'fixed_synthetic_dgp_v1',
    'coefficient_names': COEFFICIENT_NAMES,
    'design_seed_sequence': '[seed, design_id]',
    'error_seed_sequence': '[seed, design_id, 999]',
    'bit_generator': 'PCG64', 'default_seed': 20261005,
    'design_rng_children': ['unit_attributes', 'balance_weather', 'temperature_weather',
                            'state_year_background', 'unit_parity', 'error_factor_loadings'],
    'default_units': 120, 'states': 6, 'equal_units_per_state': True,
    'production_years': [2002, 2025], 'monthly_years': [1980, 2025],
    'calendar_reference_years': [1981, 2010], 'reference_sd_ddof': 1,
    'A0_uniform_ha': [100.0, 1000.0], 'mu_uniform_kg_per_fixed_ha': [2200.0, 2800.0],
    'climate_and_background_loading_uniform': [-1.0, 1.0],
    'monthly_AR': {'rho': 0.85, 'burnin': 100, 'innovation': 'Normal(0,1)',
                   'innovation_scale': 'sqrt(1-rho**2)',
                   'common_state_unit_weights': [0.25, 0.35, 0.4]},
    'balance': {'seasonal_sine_amplitude': 100.0, 'seasonal_cosine_amplitude': 40.0,
                'seasonal_cosine_frequency': 2, 'weather_amplitude': 80.0,
                'drying': '-20 * loading_i * (calendar_year - 2000) / 10'},
    'temperature': {'intercept_C': 24.0, 'seasonal_sine_amplitude_C': 3.0,
                    'seasonal_phase_radians': 0.5, 'weather_amplitude_C': 1.5,
                    'trend_C': '0.03 * (1 + 0.25 * loading_i) * (calendar_year - 2000)'},
    'D_definition': 'Annual mean of max(0, -calendar-month z1 water balance)',
    'T_definition': 'Annual mean of calendar-month standardized temperature',
    'H_definition': '(D_annual[t-1]+D_annual[t-2])/2',
    'z6_definition': 'Exact legacy_events.standardize(balance,6), missing first five months',
    'state_year_background': {'AR_rho': 0.5, 'burnin': 100, 'amplitude': 60.0,
                              'innovation': 'Normal(0,1)', 'innovation_scale': 'sqrt(1-rho**2)'},
    'unit_parity': {'amplitude_uniform': [100.0, 250.0], 'sign_equal_probability': [-1, 1],
                    'formula': 'unit_signed_amplitude * (-1)**calendar_year'},
    'backgrounds': {
        'stable': 'mu_i + state_year_s,t + parity_i,t',
        'linear_unit_drift': 'stable + 20*loading_i*(year-2006)',
        'nonlinear_area_expansion': 'mu_i*Arel_i,t + state_year_s,t + parity_i,t',
    },
    'area_ratio': '1 + (0.3+1.7*(loading_i+1)/2) * max((year-2006)/19,0)**2',
    'area_ratio_outside_expansion': 1.0,
    'truths': {
        'null': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'additive': [-120.0, -60.0, -30.0, -40.0, -10.0, -5.0, 0.0],
        'additive_interaction': [-120.0, -60.0, -30.0, -40.0, -10.0, -5.0, -80.0],
    },
    'mean_response': 'background_component + X @ beta; additive fixed-area normalized production',
    'not_constant_biological_yield': True,
    'errors': {'amplitude': MAX_ERROR_AMPLITUDE, 'rho': 0.5, 'burnin': 100,
               'innovation': 'Uniform(-1,1)', 'recursion': '0.5*previous+0.5*innovation',
               'weights_common_state_unit': [0.25, 0.35, 0.4],
               'common_loading_uniform': [0.5, 1.0], 'state_loading_uniform': [0.5, 1.0],
               'formula': '120*(0.25*Lcommon_i*common_t + 0.35*Lstate_i*state_s,t + 0.4*unit_i,t)',
               'same_realizations_across_scenarios': True,
               'zero_mean': 'Population expectation, not forced empirical centering'},
    'positivity': 'Every scenario mean must exceed120; every realized response must be positive; no clipping or reseeding.',
}


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < minimum:
        raise ValueError(name + ' must be an integer >= ' + str(minimum))
    return int(value)


def _normal_ar(rng, n_series, periods, rho, burnin=100):
    state = np.zeros(n_series, dtype=float)
    output = np.empty((n_series, periods), dtype=float)
    sd = math.sqrt(1.0-rho*rho)
    for t in range(burnin+periods):
        state = rho*state + sd*rng.normal(size=n_series)
        if t >= burnin:
            output[:, t-burnin] = state
    return output


def _weather(rng, n_units, state_index, months):
    common = _normal_ar(rng, 1, months, 0.85)
    states = _normal_ar(rng, 6, months, 0.85)
    units = _normal_ar(rng, n_units, months, 0.85)
    return 0.25*common + 0.35*states[state_index] + 0.4*units


def standardize_calendar_month(values, monthly_years, monthly_months, scale=1):
    """Independent vectorized equivalent of the frozen monthly standardization.

    ``values`` is units by consecutive months.  Six-month sums include the
    current month and five predecessors.  Thirty complete 1981--2010 values per
    calendar month, and sample standard deviations, are required explicitly.
    The first scale-1 output positions are NaN, never interpreted as zero.
    """
    value = np.asarray(values, dtype=float)
    years = np.asarray(monthly_years)
    months = np.asarray(monthly_months)
    scale = _integer(scale, 'scale', 1)
    if value.ndim != 2 or value.shape[1] < scale or years.shape != (value.shape[1],) or months.shape != years.shape:
        raise ValueError('Monthly values and date vectors have incompatible shapes')
    if not np.all(np.isfinite(value)) or not np.all(np.isfinite(years)) or not np.all(np.isfinite(months)):
        raise ValueError('Raw monthly values and dates must be finite')
    if not np.all(years == years.astype(int)) or not np.all(months == months.astype(int)) or not np.all((months >= 1) & (months <= 12)):
        raise ValueError('Invalid calendar month/year')
    serial = years.astype(int)*12 + months.astype(int)-1
    if not np.all(np.diff(serial) == 1):
        raise ValueError('Monthly date vector is not consecutive')
    rolling = np.full(value.shape, np.nan)
    # Add in calendar order, matching the legacy left-to-right Python sum.
    sums = np.zeros((value.shape[0], value.shape[1]-scale+1))
    for offset in range(scale):
        sums += value[:, offset:offset+sums.shape[1]]
    rolling[:, scale-1:] = sums
    result = np.full(value.shape, np.nan)
    for month in range(1, 13):
        baseline = (years >= 1981) & (years <= 2010) & (months == month)
        sample = rolling[:, baseline]
        if sample.shape[1] != 30 or not np.all(np.isfinite(sample)):
            raise ValueError('Incomplete 1981-2010 monthly climate reference period')
        mu = np.mean(sample, axis=1)
        sd = np.std(sample, axis=1, ddof=1)
        if np.any(sd <= 1e-12) or not np.all(np.isfinite(sd)):
            raise ValueError('Zero-variance monthly climate reference period')
        use = months == month
        result[:, use] = (rolling[:, use]-mu[:, None])/sd[:, None]
    return result


def build_design(seed, design_id, n_units=120):
    """Create one fixed synthetic design, with no observed outcomes or files."""
    seed = _integer(seed, 'seed')
    design_id = _integer(design_id, 'design_id')
    n_units = _integer(n_units, 'n_units', 6)
    if n_units % 6:
        raise ValueError('n_units must be divisible by the six fixed states')
    children = np.random.SeedSequence([seed, design_id]).spawn(6)
    attributes, balance_rng, temp_rng, background_rng, parity_rng, error_loading_rng = [np.random.default_rng(s) for s in children]
    years = np.arange(2002, 2026, dtype=int)
    monthly_years = np.repeat(np.arange(1980, 2026, dtype=int), 12)
    monthly_months = np.tile(np.arange(1, 13, dtype=int), 46)
    phase = 2.0*np.pi*(monthly_months-1)/12.0
    state_index = np.repeat(np.arange(6, dtype=int), n_units//6)
    units = np.asarray(['synthetic_' + str(i+1).zfill(6) for i in range(n_units)])
    state = np.asarray(['S' + str(i+1).zfill(2) for i in state_index])
    A0 = attributes.uniform(100.0, 1000.0, size=n_units)
    mu = attributes.uniform(2200.0, 2800.0, size=n_units)
    loading = attributes.uniform(-1.0, 1.0, size=n_units)
    balance = (100.0*np.sin(phase)[None, :] + 40.0*np.cos(2.0*phase)[None, :]
               + 80.0*_weather(balance_rng, n_units, state_index, len(monthly_years))
               - 20.0*loading[:, None]*(monthly_years[None, :]-2000)/10.0)
    temperature = (24.0 + 3.0*np.sin(phase+0.5)[None, :]
                   + 1.5*_weather(temp_rng, n_units, state_index, len(monthly_years))
                   + 0.03*(1.0+0.25*loading[:, None])*(monthly_years[None, :]-2000))
    z1 = standardize_calendar_month(balance, monthly_years, monthly_months, scale=1)
    temp_z1 = standardize_calendar_month(temperature, monthly_years, monthly_months, scale=1)
    # Use the actual preserved implementation for event exposures.  The separate
    # vectorized implementation above is checked against it in fictional tests.
    from .legacy_events import standardize
    z6 = np.asarray([[np.nan if v is None else v for v in standardize(r.tolist(), 6)]
                     for r in balance], dtype=float)
    D_annual = np.mean(np.maximum(0.0, -z1).reshape(n_units, 46, 12), axis=2)
    T_annual = np.mean(temp_z1.reshape(n_units, 46, 12), axis=2)
    columns = [D_annual[:, years-1980-lag] for lag in range(3)]
    columns += [T_annual[:, years-1980-lag] for lag in range(3)]
    H = (columns[1]+columns[2])/2.0
    columns.append(columns[0]*H)
    X = np.stack(columns, axis=2)
    state_year = 60.0*_normal_ar(background_rng, 6, len(years), 0.5)[state_index]
    parity_amplitude = parity_rng.uniform(100.0, 250.0, size=n_units)
    parity_sign = parity_rng.choice(np.asarray([-1.0, 1.0]), size=n_units)
    parity = (parity_amplitude*parity_sign)[:, None]*np.where(years % 2 == 0, 1.0, -1.0)[None, :]
    common_error_loading = error_loading_rng.uniform(0.5, 1.0, size=n_units)
    state_error_loading = error_loading_rng.uniform(0.5, 1.0, size=n_units)
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(z6[:, 5:])):
        raise ArithmeticError('Nonfinite synthetic climate design; no redraw attempted')
    return {
        'X': X, 'H': H, 'years': years, 'units': units, 'state': state,
        'state_index': state_index, 'z6': z6, 'A0': A0, 'mu': mu, 'loading': loading,
        'state_year': state_year, 'parity': parity,
        'error_common_loading': common_error_loading, 'error_state_loading': state_error_loading,
        'monthly_balance': balance, 'monthly_temperature': temperature,
        'monthly_years': monthly_years, 'monthly_months': monthly_months,
        'z1_balance': z1, 'z1_temperature': temp_z1,
        'D_annual': D_annual, 'T_annual': T_annual,
        'annual_climate_years': np.arange(1980, 2026, dtype=int),
        'metadata': {'seed': seed, 'design_id': design_id, 'n_units': n_units,
                     'n_states': 6, 'n_years': len(years), 'n_months': len(monthly_years),
                     'coefficient_names': list(COEFFICIENT_NAMES),
                     'source': 'Entirely generated; no observed research records',
                     'fixed_design': True, 'redraws': 0,
                     'error_loadings_vary_within_state': True},
    }


def scenarios(design):
    """Return all nine known laws, including exact background/climate parts."""
    X = np.asarray(design['X'], dtype=float)
    years = np.asarray(design['years'], dtype=int)
    n, t, p = X.shape
    if p != len(COEFFICIENT_NAMES) or t != len(years) or not np.all(np.isfinite(X)):
        raise ValueError('Invalid synthetic X design')
    mu = np.asarray(design['mu'], dtype=float)
    loading = np.asarray(design['loading'], dtype=float)
    state_year = np.asarray(design['state_year'], dtype=float)
    parity = np.asarray(design['parity'], dtype=float)
    if mu.shape != (n,) or loading.shape != (n,) or state_year.shape != (n, t) or parity.shape != (n, t):
        raise ValueError('Invalid synthetic background shapes')
    if not all(np.all(np.isfinite(a)) for a in (mu, loading, state_year, parity)) or np.any(np.abs(loading) > 1):
        raise ValueError('Invalid synthetic background values')
    stable = mu[:, None]+state_year+parity
    expansion = 1.0+(0.3+1.7*(loading[:, None]+1.0)/2.0)*np.maximum((years[None, :]-2006)/19.0, 0.0)**2
    backgrounds = {
        'stable': (stable, np.ones((n, t))),
        'linear_unit_drift': (stable+20.0*loading[:, None]*(years[None, :]-2006), np.ones((n, t))),
        'nonlinear_area_expansion': (mu[:, None]*expansion+state_year+parity, expansion),
    }
    output = []
    for truth, coefficients in DGP_SPEC['truths'].items():
        beta = np.asarray(coefficients, dtype=float)
        climate = X @ beta
        for background, (component, area_ratio) in backgrounds.items():
            mean = component+climate
            sid = truth+'__'+background
            if not np.all(np.isfinite(mean)) or float(np.min(mean)) <= MAX_ERROR_AMPLITUDE:
                raise ArithmeticError('Synthetic response positivity bound failed for '+sid+'; no clipping or redraw attempted')
            output.append({'id': sid, 'background': background, 'truth': truth,
                           'beta': beta.copy(), 'area_ratio': area_ratio.copy(),
                           'mean': mean.copy(), 'background_component': component.copy(),
                           'climate_component': climate.copy()})
    return output


def _bounded_ar(rng, n_series, periods, replicates):
    values = np.zeros((n_series, replicates), dtype=float)
    result = np.empty((n_series, periods, replicates), dtype=float)
    for t in range(100+periods):
        values = 0.5*values + 0.5*rng.uniform(-1.0, 1.0, size=values.shape)
        if t >= 100:
            result[:, t-100, :] = values
    return result


def error_replicates(design, seed, design_id, replicates):
    """Population-zero-mean, bounded errors shared across the nine scenarios."""
    seed = _integer(seed, 'seed')
    design_id = _integer(design_id, 'design_id')
    replicates = _integer(replicates, 'replicates', 1)
    metadata = design.get('metadata', {})
    if metadata.get('seed') != seed or metadata.get('design_id') != design_id:
        raise ValueError('Error seed/design identity differs from frozen design')
    n, periods = len(design['units']), len(design['years'])
    state_index = np.asarray(design['state_index'], dtype=int)
    common_loading = np.asarray(design['error_common_loading'], dtype=float)
    state_loading = np.asarray(design['error_state_loading'], dtype=float)
    if state_index.shape != (n,) or np.any(state_index < 0) or np.any(state_index >= 6):
        raise ValueError('Invalid synthetic state membership')
    if any(a.shape != (n,) or not np.all(np.isfinite(a)) or np.any((a < 0.5) | (a > 1.0)) for a in (common_loading, state_loading)):
        raise ValueError('Invalid fixed synthetic error factor loadings')
    rngs = [np.random.default_rng(s) for s in np.random.SeedSequence([seed, design_id, 999]).spawn(3)]
    common = _bounded_ar(rngs[0], 1, periods, replicates)
    states = _bounded_ar(rngs[1], 6, periods, replicates)
    unit = _bounded_ar(rngs[2], n, periods, replicates)
    error = MAX_ERROR_AMPLITUDE*(0.25*common_loading[:, None, None]*common
                                + 0.35*state_loading[:, None, None]*states[state_index]
                                + 0.4*unit)
    if not np.all(np.isfinite(error)) or np.max(np.abs(error)) > MAX_ERROR_AMPLITUDE:
        raise ArithmeticError('Synthetic error bound failed; no clipping or redraw attempted')
    return error
