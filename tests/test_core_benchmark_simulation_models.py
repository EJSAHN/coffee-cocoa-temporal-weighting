"""Independent fictional tests of fast balanced FE/FD and functional HAC."""
from pathlib import Path
import hashlib
import json
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coffee_cocoa_core.benchmark.simulation_models import (METHODS, REFERENCE_MODEL_SHA256, _fictional_check_fixture,
    evaluate_batch, fit_batch, prepare_geometry, run_independent_checks)


from coffee_cocoa_core.benchmark import reference_model


class SimulationModelTests(unittest.TestCase):
    def setUp(self):
        self.x, self.units, self.states, self.years, self.y = _fictional_check_fixture()
        self.contrasts = {'slope':np.array([.1, 0, 0, 0, 0, 0, .2]),
                          'gamma':np.array([0, 0, 0, 0, 0, 0, 1.])}

    def test_reference_bytes_and_three_method_dense_and_hac_agreement(self):
        actual = hashlib.sha256(Path(reference_model.__file__).read_bytes()).hexdigest()
        self.assertEqual(actual, REFERENCE_MODEL_SHA256)
        check = run_independent_checks()
        self.assertEqual(check['status'], 'PASS')
        self.assertEqual(len(check['methods']), 3)
        self.assertFalse(check['actual_study_outcomes_used'])
        json.dumps(check, allow_nan=False)

    def test_batch_and_individual_replicates_match(self):
        for method in METHODS:
            geo = prepare_geometry(self.x, self.units, self.states, self.years, method)
            batch = evaluate_batch(geo, self.y, self.contrasts)
            self.assertEqual(batch['coef'].shape, (7, 2))
            np.testing.assert_allclose(fit_batch(geo, self.y)['coef'], batch['coef'])
            for j in range(2):
                single = evaluate_batch(geo, self.y[:, :, j:j+1], self.contrasts)
                np.testing.assert_allclose(single['coef'][:, 0], batch['coef'][:, j], atol=1e-10)
                for name in self.contrasts:
                    self.assertAlmostEqual(single['estimates'][name][0], batch['estimates'][name][j])
                    for lag in ('L2', 'L4'):
                        self.assertAlmostEqual(single['variances'][name][lag][0], batch['variances'][name][lag][j])

    def test_nuisance_additions_are_absorbed_and_fd_retains_level_drift_space(self):
        rng = np.random.default_rng(9)
        added = rng.normal(size=(12, 1, 2))*200
        added = added + np.where(self.years%2==0, 1., -1.)[None, :, None]*rng.normal(size=(12, 1, 2))*3
        for st in np.unique(self.states):
            added[self.states==st] += rng.normal(size=(1, 24, 2))
        for method in METHODS:
            geo = prepare_geometry(self.x, self.units, self.states, self.years, method)
            nuisance = added.copy()
            if method in ('levels_unit_trend', 'first_difference'):
                nuisance += (self.years-self.years.mean())[None, :, None]*rng.normal(size=(12, 1, 2))
            first = evaluate_batch(geo, self.y, self.contrasts)
            second = evaluate_batch(geo, self.y+nuisance, self.contrasts)
            np.testing.assert_allclose(first['coef'], second['coef'], atol=1e-10)
            for name in self.contrasts:
                for lag in ('L2', 'L4'):
                    np.testing.assert_allclose(first['variances'][name][lag], second['variances'][name][lag], atol=1e-10)

    def test_unit_reordering_preserves_estimates_and_covariance(self):
        permutation = np.array([11, 1, 7, 9, 0, 5, 4, 8, 2, 10, 3, 6])
        for method in METHODS:
            original = evaluate_batch(prepare_geometry(self.x, self.units, self.states, self.years, method), self.y, self.contrasts)
            permuted = evaluate_batch(prepare_geometry(self.x[permutation], self.units[permutation], self.states[permutation], self.years, method), self.y[permutation], self.contrasts)
            np.testing.assert_allclose(original['coef'], permuted['coef'], atol=1e-10)
            for name in self.contrasts:
                for lag in ('L2', 'L4'):
                    np.testing.assert_allclose(original['variances'][name][lag], permuted['variances'][name][lag], atol=1e-10)

    def test_scaled_functionals_scale_variance_quadratically(self):
        geo = prepare_geometry(self.x, self.units, self.states, self.years, 'levels')
        v = self.contrasts['slope']
        result = evaluate_batch(geo, self.y, {'v':v, 'minus3v':-3*v, 'zero':np.zeros(7)})
        np.testing.assert_allclose(result['estimates']['minus3v'], -3*result['estimates']['v'])
        for lag in ('L2', 'L4'):
            np.testing.assert_allclose(result['variances']['minus3v'][lag], 9*result['variances']['v'][lag])
            np.testing.assert_array_equal(result['variances']['zero'][lag], 0)

    def test_rank_deficiency_reported_without_dropping_columns(self):
        x = self.x.copy(); x[:, :, 2] = x[:, :, 1]
        geo = prepare_geometry(x, self.units, self.states, self.years, 'levels')
        self.assertEqual(geo['status'], 'NONESTIMABLE')
        self.assertEqual(geo['diagnostics']['n_columns'], 7)
        self.assertLess(geo['diagnostics']['rank'], 7)
        json.dumps(geo['diagnostics'], allow_nan=False)
        result = evaluate_batch(geo, self.y, self.contrasts)
        self.assertEqual(result['status'], 'NONESTIMABLE')
        self.assertIsNone(result['coef'])
        self.assertEqual(result['estimates'], {})

    def test_absorbed_design_and_singleton_states_are_nonestimable(self):
        geo = prepare_geometry(np.ones_like(self.x), self.units, self.states, self.years, 'levels')
        self.assertEqual(geo['status'], 'NONESTIMABLE')
        self.assertEqual(len(geo['diagnostics']['absorbed_columns']), 7)
        geo2 = prepare_geometry(self.x, self.units, self.units, self.years, 'first_difference')
        self.assertEqual(geo2['status'], 'NONESTIMABLE')

    def test_malformed_inputs_fail_explicitly(self):
        badyears = self.years.copy(); badyears[-1] += 1
        with self.assertRaises(ValueError):
            prepare_geometry(self.x, self.units, self.states, badyears, 'levels')
        with self.assertRaises(ValueError):
            prepare_geometry(self.x, np.repeat('duplicate', len(self.units)), self.states, self.years, 'levels')
        with self.assertRaises(ValueError):
            prepare_geometry(self.x, self.units, self.states, self.years, 'unknown')
        x = self.x.copy(); x[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            prepare_geometry(x, self.units, self.states, self.years, 'levels')
        geo = prepare_geometry(self.x, self.units, self.states, self.years, 'levels')
        for bad in (self.y[:, :-1], self.y[:, :, 0], np.full_like(self.y, np.nan)):
            with self.assertRaises(ValueError):
                evaluate_batch(geo, bad, self.contrasts)
        for bad in ({'bad':np.ones(6)}, {'bad':np.full(7, np.nan)}):
            with self.assertRaises(ValueError):
                evaluate_batch(geo, self.y, bad)


if __name__=='__main__':
    unittest.main()
