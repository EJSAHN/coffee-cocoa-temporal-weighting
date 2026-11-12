"""Generated-data-only tests of the known response laws and bounded errors."""
import copy
import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coffee_cocoa_core.benchmark.legacy_events import standardize
from coffee_cocoa_core.benchmark.synthetic_dgp import (COEFFICIENT_NAMES, DGP_SPEC, MAX_ERROR_AMPLITUDE,
                           build_design, error_replicates, scenarios,
                           standardize_calendar_month)


class SyntheticDGPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.design = build_design(20261005, 1, n_units=12)

    def test_fixed_design_and_seed_reproduction(self):
        d = self.design
        again = build_design(20261005, 1, n_units=12)
        for name in ('X', 'H', 'z6', 'A0', 'mu', 'loading', 'state_year', 'parity',
                     'monthly_balance', 'monthly_temperature', 'error_common_loading',
                     'error_state_loading'):
            np.testing.assert_array_equal(d[name], again[name])
        self.assertEqual(d['X'].shape, (12, 24, 7))
        self.assertEqual(d['H'].shape, (12, 24))
        self.assertEqual(d['z6'].shape, (12, 552))
        np.testing.assert_array_equal(d['years'], np.arange(2002, 2026))
        self.assertEqual(len(set(d['state'])), 6)
        self.assertEqual(len(set(d['units'])), 12)
        self.assertEqual(d['metadata']['coefficient_names'], COEFFICIENT_NAMES)
        self.assertEqual(d['metadata']['redraws'], 0)

    def test_different_frozen_design_id_is_not_reused(self):
        other = build_design(20261005, 2, n_units=12)
        self.assertFalse(np.array_equal(self.design['X'], other['X']))

    def test_calendar_standardization_independently_matches_legacy(self):
        d = self.design
        for scale in (1, 6):
            own = standardize_calendar_month(d['monthly_balance'][:2], d['monthly_years'],
                                             d['monthly_months'], scale)
            for i in range(2):
                legacy = np.asarray([np.nan if v is None else v
                                     for v in standardize(d['monthly_balance'][i].tolist(), scale)])
                np.testing.assert_allclose(own[i], legacy, rtol=2e-13, atol=2e-13, equal_nan=True)
        self.assertTrue(np.isnan(d['z6'][:, :5]).all())
        self.assertTrue(np.isfinite(d['z6'][:, 5:]).all())
        for i in range(2):
            legacy = np.asarray([np.nan if v is None else v
                                 for v in standardize(d['monthly_balance'][i].tolist(), 6)])
            np.testing.assert_array_equal(d['z6'][i], legacy)

    def test_monthly_reference_is_sample_standardized_separately(self):
        d = self.design
        for name in ('z1_balance', 'z1_temperature', 'z6'):
            for month in range(1, 13):
                mask = ((d['monthly_years'] >= 1981) & (d['monthly_years'] <= 2010)
                        & (d['monthly_months'] == month))
                sample = d[name][:, mask]
                self.assertEqual(sample.shape[1], 30)
                np.testing.assert_allclose(np.mean(sample, axis=1), 0.0, atol=2e-13)
                np.testing.assert_allclose(np.std(sample, axis=1, ddof=1), 1.0, atol=2e-13)

    def test_annual_indices_lags_and_product_use_correct_calendar(self):
        d = self.design
        for j, year in enumerate(d['years']):
            for lag in range(3):
                mask = d['monthly_years'] == year-lag
                manual_D = np.maximum(-d['z1_balance'][:, mask], 0.0).mean(axis=1)
                manual_T = d['z1_temperature'][:, mask].mean(axis=1)
                np.testing.assert_allclose(d['X'][:, j, lag], manual_D, rtol=0, atol=2e-15)
                np.testing.assert_allclose(d['X'][:, j, 3+lag], manual_T, rtol=0, atol=2e-15)
        np.testing.assert_allclose(d['H'], (d['X'][:, :, 1]+d['X'][:, :, 2])/2)
        np.testing.assert_allclose(d['X'][:, :, 6], d['X'][:, :, 0]*d['H'])
        self.assertTrue((d['X'][:, :, :3] >= 0).all())

    def test_complete_nine_truth_background_cross(self):
        ss = scenarios(self.design)
        self.assertEqual(len(ss), 9)
        self.assertEqual(len({s['id'] for s in ss}), 9)
        self.assertEqual({s['truth'] for s in ss}, set(DGP_SPEC['truths']))
        self.assertEqual({s['background'] for s in ss}, set(DGP_SPEC['backgrounds']))
        for s in ss:
            np.testing.assert_array_equal(s['beta'], DGP_SPEC['truths'][s['truth']])
            np.testing.assert_allclose(s['climate_component'], self.design['X'] @ s['beta'])
            np.testing.assert_allclose(s['mean'], s['background_component']+s['climate_component'])
            self.assertGreater(s['mean'].min(), MAX_ERROR_AMPLITUDE)
            if s['truth'] == 'null':
                np.testing.assert_array_equal(s['climate_component'], np.zeros((12, 24)))
            elif s['truth'] == 'additive':
                self.assertEqual(s['beta'][-1], 0)
            else:
                self.assertEqual(s['beta'][-1], -80)

    def test_backgrounds_are_exact_and_not_redefined_by_climate_truth(self):
        d = self.design
        ss = {s['id']: s for s in scenarios(d)}
        stable = ss['null__stable']['background_component']
        np.testing.assert_allclose(stable, d['mu'][:, None]+d['state_year']+d['parity'])
        drift = ss['null__linear_unit_drift']['background_component']-stable
        np.testing.assert_allclose(drift, 20*d['loading'][:, None]*(d['years'][None, :]-2006), atol=5e-13)
        expansion = ss['null__nonlinear_area_expansion']
        np.testing.assert_allclose(expansion['background_component'],
                                   d['mu'][:, None]*expansion['area_ratio']+d['state_year']+d['parity'])
        for background in DGP_SPEC['backgrounds']:
            np.testing.assert_array_equal(ss['null__'+background]['background_component'],
                                          ss['additive_interaction__'+background]['background_component'])

    def test_baseline_intended_area_matches_frozen_A0(self):
        d = self.design
        baseline = (d['years'] >= 2002) & (d['years'] <= 2006)
        self.assertTrue(((d['A0'] >= 100) & (d['A0'] <= 1000)).all())
        for s in scenarios(d):
            intended = d['A0'][:, None]*s['area_ratio']
            np.testing.assert_allclose(intended[:, baseline].mean(axis=1), d['A0'], rtol=0, atol=2e-13)
            np.testing.assert_array_equal(s['area_ratio'][:, baseline], np.ones((12, 5)))
            self.assertTrue((s['area_ratio'] >= 1).all())
        expansion = next(s for s in scenarios(d) if s['id'] == 'null__nonlinear_area_expansion')
        self.assertTrue((expansion['area_ratio'][:, -1] <= 3).all())
        self.assertTrue((np.diff(expansion['area_ratio'], axis=1) >= 0).all())

    def test_known_interaction_derivative_and_additive_null(self):
        d = self.design
        j = 5
        epsilon = 1e-5
        shifted = d['X'].copy()
        shifted[:, j, 0] += epsilon
        shifted[:, j, 6] += epsilon*d['H'][:, j]
        for s in scenarios(d):
            observed = ((shifted @ s['beta'])[:, j]-s['climate_component'][:, j])/epsilon
            truth = s['beta'][0]+s['beta'][-1]*d['H'][:, j]
            np.testing.assert_allclose(observed, truth, atol=2e-8, rtol=2e-9)

    def test_bounded_zero_mean_error_process_and_reproducibility(self):
        d = self.design
        errors = error_replicates(d, 20261005, 1, 1000)
        self.assertEqual(errors.shape, (12, 24, 1000))
        self.assertLessEqual(np.abs(errors).max(), 120.0)
        self.assertLess(abs(errors.mean()), 1.0)
        self.assertNotEqual(float(errors.mean()), 0.0)  # No forced sample centering.
        np.testing.assert_array_equal(errors, error_replicates(d, 20261005, 1, 1000))
        for s in scenarios(d):
            self.assertTrue((s['mean'][:, :, None]+errors > 0).all())

    def test_fixed_heterogeneous_factor_loadings_remain_within_state(self):
        d = self.design
        for name in ('error_common_loading', 'error_state_loading'):
            a = d[name]
            self.assertTrue(((a >= 0.5) & (a <= 1)).all())
            for state in range(6):
                self.assertGreater(np.ptp(a[d['state_index'] == state]), 0.0)

    def test_errors_reject_different_design_identity_or_invalid_loadings(self):
        with self.assertRaisesRegex(ValueError, 'identity differs'):
            error_replicates(self.design, 20261005, 2, 3)
        altered = copy.deepcopy(self.design)
        altered['error_common_loading'][0] = 1.01
        with self.assertRaisesRegex(ValueError, 'factor loadings'):
            error_replicates(altered, 20261005, 1, 3)

    def test_positivity_violation_is_error_without_clipping_or_redraw(self):
        d = copy.deepcopy(self.design)
        d['mu'][:] = 10
        with self.assertRaisesRegex(ArithmeticError, 'positivity bound failed'):
            scenarios(d)
        np.testing.assert_array_equal(d['mu'], np.full(12, 10.0))

    def test_invalid_calendar_or_baseline_variance_rejected(self):
        d = self.design
        with self.assertRaisesRegex(ValueError, 'Zero-variance'):
            standardize_calendar_month(np.ones((1, 552)), d['monthly_years'], d['monthly_months'], 1)
        months = d['monthly_months'].copy()
        months[5] = 7
        with self.assertRaisesRegex(ValueError, 'not consecutive'):
            standardize_calendar_month(d['monthly_balance'], d['monthly_years'], months, 1)

    def test_protocol_is_json_and_invalid_sizes_fail(self):
        json.dumps(DGP_SPEC, allow_nan=False)
        for kwargs in ({'n_units': 11}, {'n_units': 0}, {'seed': True}):
            params = dict(seed=20261005, design_id=1, n_units=12)
            params.update(kwargs)
            with self.assertRaises(ValueError):
                build_design(**params)


if __name__ == '__main__':
    unittest.main()
