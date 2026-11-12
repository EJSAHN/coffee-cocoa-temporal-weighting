"""Independent synthetic arithmetic tests; no observed study outcomes are fit."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pathlib import Path
import sys
import unittest

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
from coffee_cocoa_core.panel_analysis.contrasts import current_slope, generate_contributions
from coffee_cocoa_core.panel_analysis.fixed_effects import fit_model, NORMAL_975


NAMES = ['D0', 'D1', 'D2', 'T0', 'T1', 'T2', 'D0_x_history']


def artificial_fit(quadratic=False):
    names = NAMES + (['D0_squared', 'history_squared', 'T0_squared'] if quadratic else [])
    beta = np.arange(1, len(names) + 1, dtype=float)
    rng = np.random.default_rng(9051)
    a = rng.normal(size=(len(names), len(names)))
    cov = a @ a.T
    return {'status': 'FIT_COMPLETE_EXPLORATORY', 'coefficient_order': names,
            'coefficients': dict(zip(names, beta)),
            'uncertainty': {'HAC_L2': {'covariance': cov.tolist()},
                            'HAC_L4': {'covariance': (cov * 1.7).tolist()}}}


def reference_design():
    x = np.array([[0.2, 0.4, 0.6, 2.0, 3.0, 4.0, 0.1],
                  [0.4, 0.6, 0.8, 2.0, 3.0, 4.0, 0.28],
                  [0.9, 0.1, 0.3, 2.0, 3.0, 4.0, 0.18]])
    return x, NAMES.copy()


def fixture(trends=False, parity=False, quadratic=False):
    rng = np.random.default_rng(61493)
    units, states, years = [], [], []
    for j in range(12):
        for year in range(2001, 2014):
            if (j + 3 * year) % 13 == 0:  # A genuinely unbalanced panel.
                continue
            units.append('synthetic_' + str(j))
            states.append('state_' + str(j // 4))
            years.append(year)
    units, states, years = np.array(units), np.array(states), np.array(years)
    n = len(years)
    base = rng.normal(size=(n, 6))
    history = (base[:, 1] + base[:, 2]) / 2
    x = np.column_stack([base, base[:, 0] * history])
    names = NAMES.copy()
    if quadratic:
        names += ['D0_squared', 'history_squared', 'T0_squared']
        x = np.column_stack([x, base[:, 0]**2, history**2, base[:, 3]**2])
    ud = np.column_stack([units == u for u in np.unique(units)]).astype(float)
    sy = np.array([s + '_' + str(y) for s, y in zip(states, years)])
    syd = np.column_stack([sy == value for value in np.unique(sy)]).astype(float)
    blocks = [ud, syd]
    if trends:
        blocks.append(ud * (years - years.mean())[:, None])
    if parity:
        blocks.append(ud * np.where(years % 2 == 0, 1.0, -1.0)[:, None])
    nuisance = np.column_stack(blocks)
    beta = rng.normal(size=x.shape[1])
    y = x @ beta + nuisance @ rng.normal(size=nuisance.shape[1]) + rng.normal(size=n)
    spec = {'model_id': 'synthetic_model', 'crop': 'fictional_crop',
            'outcome': 'fictional_response', 'variant': 'synthetic',
            'unit_trends': trends, 'unit_parity': parity}
    data = {'X': x, 'y': y, 'unit': units, 'state': states, 'year': years,
            'names': names,
            'rows': [{'territory': u} for u in units]}
    fit = fit_model(x, y, units, states, years, names,
                    unit_trends=trends, unit_parity=parity)
    return data, spec, fit, nuisance


class DerivativeTests(unittest.TestCase):
    def test_linear_gradient_and_full_covariance(self):
        ref, names = reference_design()
        fit = artificial_fit()
        result = current_slope(fit, ref, names)
        h = np.mean((ref[:, 1] + ref[:, 2]) / 2)
        v = np.zeros(7); v[0] = .1; v[6] = .1 * h
        b = np.array(list(fit['coefficients'].values()))
        np.testing.assert_allclose(result['scaled_gradient'], v)
        self.assertAlmostEqual(result['estimate'], float(v @ b))
        for key in ('HAC_L2', 'HAC_L4'):
            cov = np.array(fit['uncertainty'][key]['covariance'])
            expected_var = sum(v[j] * cov[j, k] * v[k] for j in range(7) for k in range(7))
            self.assertAlmostEqual(result['uncertainty'][key]['variance'], expected_var)
            self.assertAlmostEqual(result['uncertainty'][key]['lower'],
                                   v @ b - NORMAL_975 * np.sqrt(expected_var))

    def test_quadratic_is_derivative_not_finite_step(self):
        ref, names = reference_design()
        fit = artificial_fit(quadratic=True)
        result = current_slope(fit, ref, names)
        b = fit['coefficients']
        h = (ref[:, 1] + ref[:, 2]) / 2
        derivative = .1 * np.mean(b['D0'] + b['D0_x_history'] * h + 2 * b['D0_squared'] * ref[:, 0])
        self.assertAlmostEqual(result['estimate'], derivative)
        finite_step = np.mean(.1 * b['D0'] + .1 * b['D0_x_history'] * h +
                              b['D0_squared'] * ((ref[:, 0] + .1)**2 - ref[:, 0]**2))
        self.assertAlmostEqual(finite_step - result['estimate'], .01 * b['D0_squared'])
        self.assertEqual(result['gradient_by_coefficient']['history_squared'], 0)
        self.assertEqual(result['gradient_by_coefficient']['T0_squared'], 0)

    def test_reference_column_order_and_coeff_order_are_respected(self):
        ref, names = reference_design()
        fit = artificial_fit(quadratic=True)
        result = current_slope(fit, ref, names)
        permutation = list(reversed(range(7)))
        reordered = current_slope(fit, ref[:, permutation], [names[j] for j in permutation])
        self.assertEqual(result, reordered)
        order = list(reversed(fit['coefficient_order']))
        idx = [fit['coefficient_order'].index(n) for n in order]
        fit2 = dict(fit, coefficient_order=order)
        fit2['uncertainty'] = {k: {'covariance': np.array(v['covariance'])[np.ix_(idx, idx)].tolist()}
                               for k, v in fit['uncertainty'].items()}
        result2 = current_slope(fit2, ref, names)
        self.assertAlmostEqual(result['estimate'], result2['estimate'])
        self.assertAlmostEqual(result['uncertainty']['HAC_L2']['variance'],
                               result2['uncertainty']['HAC_L2']['variance'])

    def test_preceding_window_uses_supplied_X_not_unshifted_row_metadata(self):
        ref, names = reference_design()
        shifted = ref.copy()
        shifted[:, :3] = np.array([[3, 4, 6], [5, 7, 9], [2, 8, 10]])
        result = current_slope(artificial_fit(), shifted, names)
        self.assertAlmostEqual(result['reference_history_mean'], 44 / 6)

    def test_nonestimable_fit_returns_no_estimate(self):
        result = current_slope({'status': 'NONESTIMABLE'}, None, None)
        self.assertEqual(result['status'], 'NONESTIMABLE')
        self.assertNotIn('estimate', result)

    def test_invalid_reference_rejected(self):
        ref, names = reference_design()
        for invalid, invalid_names in ((ref[:0], names), (ref, names[:-1]),
                                        (ref * np.nan, names), (ref, ['D0'] * 7)):
            with self.subTest(names=invalid_names), self.assertRaises(ValueError):
                current_slope(artificial_fit(), invalid, invalid_names)

    def test_material_negative_variance_even_at_tiny_scale_rejected(self):
        ref, names = reference_design()
        fit = artificial_fit()
        fit['uncertainty']['HAC_L2']['covariance'] = (-np.eye(7) * 1e-25).tolist()
        with self.assertRaises(ArithmeticError):
            current_slope(fit, ref, names)

    def test_bad_covariance_shape_or_asymmetry_rejected(self):
        ref, names = reference_design()
        fit = artificial_fit()
        fit['uncertainty']['HAC_L2']['covariance'] = np.eye(6).tolist()
        with self.assertRaises(ValueError):
            current_slope(fit, ref, names)
        fit = artificial_fit()
        fit['uncertainty']['HAC_L2']['covariance'][0][1] += 1
        with self.assertRaises(ArithmeticError):
            current_slope(fit, ref, names)


class ContributionTests(unittest.TestCase):
    def test_against_dense_full_design_with_all_nuisance_variants(self):
        for trends, parity in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(trends=trends, parity=parity):
                data, spec, fit, nuisance = fixture(trends, parity, quadratic=True)
                self.assertEqual(fit['status'], 'FIT_COMPLETE_EXPLORATORY')
                contrast = current_slope(fit, data['X'], data['names'])
                v = np.array(contrast['scaled_gradient'])
                rows, validation = generate_contributions(data, spec, fit, v)
                a = np.array([r['coefficient_weight'] for r in rows])
                full = np.column_stack([data['X'], nuisance])
                full_v = np.r_[v, np.zeros(nuisance.shape[1])]
                dense_weights = np.linalg.lstsq(full.T, full_v, rcond=1e-12)[0]
                dense_beta = np.linalg.lstsq(full, data['y'], rcond=1e-12)[0]
                np.testing.assert_allclose(a, dense_weights, rtol=1e-7, atol=1e-10)
                np.testing.assert_allclose(nuisance.T @ a, 0, atol=1e-9)
                self.assertAlmostEqual(validation['target'], float(full_v @ dense_beta), places=8)
                self.assertAlmostEqual(sum(r['response_contribution'] for r in rows), contrast['estimate'], places=8)
                self.assertAlmostEqual(sum(r['raw_response_contribution'] for r in rows), contrast['estimate'], places=8)
                self.assertAlmostEqual(sum(r['contrast_weight_energy_share'] for r in rows), 1)
                self.assertEqual(validation['status'], 'PASS')

    def test_contributions_preserve_all_rows_and_metadata(self):
        data, spec, fit, _ = fixture()
        derivative = current_slope(fit, data['X'], data['names'])
        rows, check = generate_contributions(data, spec, fit, derivative['scaled_gradient'])
        self.assertEqual(len(rows), len(data['y']))
        for row, unit, year, state in zip(rows, data['unit'], data['year'], data['state']):
            self.assertEqual(row['territory'], unit)
            self.assertEqual(row['unit'], unit)
            self.assertEqual(row['year'], year)
            self.assertEqual(row['state'], state)
            self.assertGreaterEqual(row['contrast_weight_energy_share'], 0)

    def test_source_fit_or_spec_mismatch_rejected(self):
        data, spec, fit, _ = fixture()
        v = current_slope(fit, data['X'], data['names'])['scaled_gradient']
        with self.assertRaises(ValueError):
            generate_contributions(data, dict(spec, unit_trends=True), fit, v)
        with self.assertRaises(ValueError):
            generate_contributions(data, spec, {'status': 'NONESTIMABLE'}, v)
        with self.assertRaises(ValueError):
            generate_contributions(dict(data, names=list(reversed(data['names']))), spec, fit, v)

    def test_modified_beta_is_detected_by_reproduction(self):
        data, spec, fit, _ = fixture()
        v = current_slope(fit, data['X'], data['names'])['scaled_gradient']
        fit['coefficients']['D0'] += 100
        with self.assertRaises(ArithmeticError):
            generate_contributions(data, spec, fit, v)

    def test_zero_nonfinite_or_misaligned_gradient_rejected(self):
        data, spec, fit, _ = fixture()
        for v in (np.zeros(7), np.ones(6), np.full(7, np.nan)):
            with self.subTest(v=v), self.assertRaises(ValueError):
                generate_contributions(data, spec, fit, v)


if __name__ == '__main__':
    unittest.main()
