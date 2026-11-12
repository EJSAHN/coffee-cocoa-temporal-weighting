"""Independent fictional FE/FD/lead checks; no observed study outcomes are fit."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import copy
from pathlib import Path
import sys
import unittest

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
from coffee_cocoa_core.panel_analysis.tests.fixture_panel import fictional_inputs
from coffee_cocoa_core.panel_analysis.temporal import linear_function, run


def dense_fit(data, spec):
    unit = np.column_stack([data['unit']==u for u in np.unique(data['unit'])]).astype(float)
    sy = np.array([str(s)+'_'+str(y) for s, y in zip(data['state'], data['year'])])
    state_year = np.column_stack([sy==s for s in np.unique(sy)]).astype(float)
    blocks = [unit, state_year]
    if spec['unit_trends']:
        blocks.append(unit*(data['year']-data['year'].mean())[:, None])
    if spec['unit_parity']:
        blocks.append(unit*np.where(data['year']%2==0, 1., -1.)[:, None])
    full = np.column_stack([data['X'], *blocks])
    beta = np.linalg.lstsq(full, data['y'], rcond=1e-12)[0][:len(data['names'])]
    return beta


class TemporalAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = fictional_inputs()
        cls.original = copy.deepcopy(cls.inputs)
        cls.result = run(**cls.inputs)

    def test_all_twenty_models_reproduce_dense_nuisance_design(self):
        prepared = self.result['prepared']
        self.assertEqual(len(self.result['models']), 20)
        self.assertEqual(len(prepared['specs']), 20)
        for spec in prepared['specs']:
            with self.subTest(model=spec['model_id']):
                data = prepared['data_by_model'][spec['model_id']]
                actual = self.result['models'][spec['model_id']]
                self.assertEqual(actual['status'], 'FIT_COMPLETE_EXPLORATORY')
                expected = dense_fit(data, spec)
                beta = [actual['coefficients'][n] for n in data['names']]
                np.testing.assert_allclose(beta, expected, rtol=2e-6, atol=1e-7)

    def test_same_endpoint_reference_and_unchanged_inputs(self):
        self.assertEqual(self.inputs, self.original)
        prepared = self.result['prepared']
        for crop in ('coffee_total', 'cacao'):
            expected_keys = None
            ref = prepared['reference_by_crop'][crop]
            h = (ref['X'][:, 1]+ref['X'][:, 2])/2
            self.assertAlmostEqual(ref['history_mean'], np.mean(h))
            self.assertAlmostEqual(ref['history_q25'], np.quantile(h, .25))
            self.assertAlmostEqual(ref['history_q75'], np.quantile(h, .75))
            for spec in prepared['specs']:
                if spec['crop']!=crop:
                    continue
                data = prepared['data_by_model'][spec['model_id']]
                keys = [(r['territory'], r['year']) for r in data['rows']]
                if expected_keys is None:
                    expected_keys = keys
                self.assertEqual(keys, expected_keys)
                self.assertEqual(set(data['year']), set(range(2008, 2024)))
            self.assertEqual(len(expected_keys), 15*16)
            self.assertEqual([(r['territory'], r['year']) for r in ref['rows']], expected_keys)

    def test_first_difference_interaction_is_difference_of_original_product(self):
        panel = {(r['crop'],r['territory'],r['year']):r for r in self.inputs['panel']}
        prepared = self.result['prepared']
        for crop in ('coffee_total', 'cacao'):
            mid = crop+'__production_fixed_ha__first_difference'
            data = prepared['data_by_model'][mid]
            expected, wrong = [], []
            for r in data['rows']:
                p = panel[crop,r['territory'],r['year']-1]
                h0, h1 = (p['D1']+p['D2'])/2, (r['D1']+r['D2'])/2
                expected.append(r['D0']*h1-p['D0']*h0)
                wrong.append((r['D0']-p['D0'])*(h1-h0))
            np.testing.assert_allclose(data['X'][:,data['names'].index('D0_x_history')], expected)
            self.assertGreater(np.max(np.abs(np.array(expected)-wrong)), .01)
            self.assertFalse(next(s for s in prepared['specs'] if s['model_id']==mid)['unit_trends'])

    def test_future_columns_are_exact_calendar_joins_under_serial_climate(self):
        panel = {(r['crop'],r['territory'],r['year']):r for r in self.inputs['panel']}
        for crop in ('coffee_total', 'cacao'):
            mid = crop+'__production_fixed_ha__levels_unit_trend_future'
            data = self.result['prepared']['data_by_model'][mid]
            expected = []
            for r in data['rows']:
                a = panel[crop,r['territory'],r['year']+1]
                b = panel[crop,r['territory'],r['year']+2]
                expected.append([a['D0'],b['D0'],a['T0'],b['T0']])
            np.testing.assert_allclose(data['X'][:, -4:], expected)
            self.assertEqual(data['names'][-4:], ['D_lead1','D_lead2','T_lead1','T_lead2'])
            # Persistent climate can make leads correlated: this is a software
            # fixture, not a significance or causal-validation test.
            self.assertGreater(np.corrcoef(data['X'][:,0],data['X'][:,-4])[0,1], .3)

    def test_reported_contrasts_use_original_endpoint_reference_even_for_fd(self):
        models = self.result['models']
        refs = self.result['prepared']['reference_by_crop']
        self.assertEqual(len(self.result['contrast_rows']), 20*3*2)
        self.assertEqual(len(self.result['lead_rows']), 2*4*2)
        for row in self.result['contrast_rows']:
            fit = models[row['model_id']]
            beta = fit['coefficients']; ref = refs[row['crop']]
            expected = {
                'current_D_derivative_times_0_1': .1*(beta['D0']+beta['D0_x_history']*ref['history_mean']),
                'interaction_gamma': beta['D0_x_history'],
                'current_D_slope_difference_history_q75_minus_q25': .1*beta['D0_x_history']*(ref['history_q75']-ref['history_q25'])
            }[row['quantity']]
            self.assertAlmostEqual(row['estimate'], expected)
            self.assertAlmostEqual(row['reference_history_mean'], ref['history_mean'])
        for row in self.result['lead_rows']:
            term = row['quantity'].removesuffix('_coefficient_times_0_1')
            self.assertAlmostEqual(row['estimate'], .1*models[row['model_id']]['coefficients'][term])
            self.assertIn('not a randomized placebo', row['interpretation'])

    def test_no_consecutive_support_returns_nonestimable_without_estimates(self):
        inputs = copy.deepcopy(self.inputs)
        inputs['common_reference_rows'] = [r for r in inputs['common_reference_rows'] if r['year']%2]
        result = run(**inputs, reproduce_source=False)
        self.assertEqual(len(result['models']), 20)
        self.assertTrue(all(r['status']=='NONESTIMABLE' for r in result['models'].values()))
        self.assertTrue(all(r['status']=='NONESTIMABLE' and 'estimate' not in r for r in result['contrast_rows']))
        self.assertTrue(all('estimate' not in r for r in result['lead_rows']))
        self.assertEqual(result['prepared']['summary']['eligible_endpoints'], 0)

    def test_saved_source_fits_are_verified_and_wrong_coefficient_fails(self):
        self.assertEqual(len(self.result['reproduction_rows']), 2)
        self.assertTrue(all(r['status']=='PASS' for r in self.result['reproduction_rows']))
        bad = copy.deepcopy(self.inputs)
        first = next(iter(bad['saved_models']))
        bad['saved_models'][first]['coefficients']['D0'] += 30
        with self.assertRaises(ValueError):
            run(**bad)


class LinearFunctionTests(unittest.TestCase):
    def artificial_fit(self):
        names = ['D0','D0_x_history','T0']
        cov = np.array([[4.,-1.,.5],[-1.,9.,2.],[.5,2.,5.]])
        return dict(status='FIT_COMPLETE_EXPLORATORY', coefficient_order=names,
                    coefficients=dict(zip(names,[2.,-3.,1.])),
                    uncertainty={k:dict(covariance=(cov*f).tolist()) for k,f in [('HAC_L2',1.),('HAC_L4',1.5)]})

    def test_complete_covariance_cross_terms_and_permuted_order(self):
        fit = self.artificial_fit()
        weights = {'D0': .1, 'D0_x_history': .2}
        result = linear_function(fit, weights)
        self.assertEqual(result['status'], 'COMPLETE_LINEAR_FUNCTION')
        self.assertAlmostEqual(result['estimate'], -.4)
        v = np.array([.1,.2,0.])
        for label, factor in [('HAC_L2',1),('HAC_L4',1.5)]:
            expected = float(v @ np.array(fit['uncertainty'][label]['covariance']) @ v)
            self.assertAlmostEqual(result['uncertainty'][label]['standard_error']**2, expected)
            self.assertAlmostEqual(result['uncertainty'][label]['lower'], -.4-1.959963984540054*np.sqrt(expected))
        reversed_fit = copy.deepcopy(fit)
        reversed_fit['coefficient_order'] = list(reversed(fit['coefficient_order']))
        for label in fit['uncertainty']:
            reversed_fit['uncertainty'][label]['covariance'] = np.array(fit['uncertainty'][label]['covariance'])[::-1,::-1].tolist()
        result2 = linear_function(reversed_fit, weights)
        self.assertAlmostEqual(result2['estimate'], result['estimate'])
        self.assertAlmostEqual(result2['uncertainty']['HAC_L2']['standard_error'],
                               result['uncertainty']['HAC_L2']['standard_error'])

    def test_nonestimable_fit_does_not_fabricate_linear_estimate(self):
        result = linear_function({'status':'NONESTIMABLE'}, {'D0':.1})
        self.assertEqual(result['status'], 'NONESTIMABLE')
        self.assertNotIn('estimate', result)

    def test_material_negative_variance_at_tiny_scale_rejected(self):
        fit = self.artificial_fit()
        fit['uncertainty']['HAC_L2']['covariance'] = (-np.eye(3)*1e-25).tolist()
        with self.assertRaises(ValueError):
            linear_function(fit, {'D0':.1})

    def test_missing_nonfinite_or_unknown_inputs_rejected(self):
        for weights in ({'missing_coefficient':1.}, {'D0':float('nan')}):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                linear_function(self.artificial_fit(), weights)
        fit = self.artificial_fit()
        fit['uncertainty']['HAC_L2']['covariance'] = np.eye(2).tolist()
        with self.assertRaises(ValueError):
            linear_function(fit, {'D0':.1})


if __name__=='__main__':
    unittest.main()
