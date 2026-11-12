"""Fictional-only checks of frozen temporal support and exact transformations."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import copy
import sys
import unittest
from pathlib import Path

import numpy as np

from coffee_cocoa_core.panel_analysis.features import BASE_NAMES, OUTCOMES
from coffee_cocoa_core.panel_analysis.temporal_features import FUTURE_NAMES, prepare, specifications


def fixture():
    panel = []
    for ci, crop in enumerate(('coffee_total', 'cacao')):
        for ui in range(2):
            for year in range(2005, 2026):
                r = dict(crop=crop, territory=str(1000001+ci*100+ui), state='10',
                         year=year, A0=100.0+ui, pilot=False, source_discordant=False)
                for lag in range(4):
                    yy = year-lag-2000
                    r['D'+str(lag)] = (ci+1)*0.2 + (ui+1)*yy*yy/1000.0
                    r['T'+str(lag)] = ci*2.0 + ui + yy/10.0
                r.update(production_fixed_ha=(ci+1)*100.0+(ui+1)*(year-2000)**2,
                         intended_area_fixed_ha=1.0+(year-2000)/20.0,
                         unharvested_area_fixed_ha=(year % 3)/20.0)
                panel.append(r)
    refs = [dict(crop=r['crop'], territory=r['territory'], state=r['state'], year=r['year'],
                 D0=r['D0'], history=(r['D1']+r['D2'])/2)
            for r in panel if 2007 <= r['year'] <= 2024]
    return panel, refs


def row(panel, crop='coffee_total', territory='1000001', year=2010):
    return next(r for r in panel if (r['crop'], r['territory'], r['year']) == (crop, territory, year))


def data(result, crop='coffee_total', outcome='production_fixed_ha', variant='levels'):
    return result['data_by_model']['__'.join((crop, outcome, variant))]


class TemporalFeaturesTests(unittest.TestCase):
    def test_finite_twenty_specifications_and_fd_nuisance(self):
        specs = specifications()
        self.assertEqual(len(specs), 20)
        self.assertEqual(len({s['model_id'] for s in specs}), 20)
        leads = [s for s in specs if s['variant'].endswith('_future')]
        self.assertEqual(len(leads), 2)
        self.assertTrue(all(s['outcome'] == OUTCOMES[0] and s['unit_trends'] for s in leads))
        for s in specs:
            self.assertEqual(s['unit_parity'], s['crop'] == 'coffee_total')
            if s['variant'] == 'first_difference':
                self.assertFalse(s['unit_trends'])

    def test_fixed_common_support_for_all_twenty_models(self):
        result = prepare(*fixture())
        self.assertEqual(result['summary']['eligible_endpoints'], 64)
        self.assertEqual(result['summary']['excluded_endpoints'], 0)
        for crop in ('coffee_total', 'cacao'):
            expected = [(r['territory'], r['year']) for r in result['reference_by_crop'][crop]['rows']]
            for spec in result['specs']:
                if spec['crop'] == crop:
                    d = result['data_by_model'][spec['model_id']]
                    self.assertEqual([(r['territory'], r['year']) for r in d['rows']], expected)
            s = result['summary']['by_crop'][crop]
            self.assertEqual(s['endpoint_years'], list(range(2008, 2024)))
            self.assertEqual(s['underlying_difference_outcome_years'], list(range(2007, 2024)))

    def test_difference_is_of_complete_original_product(self):
        panel, refs = fixture()
        result = prepare(panel, refs)
        d = data(result, variant='first_difference')
        j = next(j for j, r in enumerate(d['rows']) if r['territory'] == '1000001' and r['year'] == 2010)
        now, old = row(panel), row(panel, year=2009)
        hn, ho = (now['D1']+now['D2'])/2, (old['D1']+old['D2'])/2
        self.assertAlmostEqual(d['X'][j, -1], now['D0']*hn-old['D0']*ho)
        self.assertNotAlmostEqual(d['X'][j, -1], (now['D0']-old['D0'])*(hn-ho))
        np.testing.assert_allclose(d['X'][j, :6], [now[n]-old[n] for n in BASE_NAMES[:6]])
        self.assertEqual(d['y'][j], now[OUTCOMES[0]]-old[OUTCOMES[0]])
        ref = result['reference_by_crop']['coffee_total']
        self.assertEqual(ref['X'][j, 0], now['D0'])
        self.assertEqual(ref['names'], BASE_NAMES)

    def test_future_join_uses_exact_unit_crop_year(self):
        panel, refs = fixture()
        result = prepare(panel, refs)
        d = data(result, variant='levels_unit_trend_future')
        self.assertEqual(d['names'], BASE_NAMES+FUTURE_NAMES)
        for j, r in enumerate(d['rows']):
            f1 = row(panel, territory=r['territory'], year=r['year']+1)
            f2 = row(panel, territory=r['territory'], year=r['year']+2)
            np.testing.assert_allclose(d['X'][j, -4:], [f1['D0'], f2['D0'], f1['T0'], f2['T0']])

    def test_missing_future_does_not_use_neighboring_unit_or_wrap(self):
        panel, refs = fixture()
        panel = [r for r in panel if not (r['crop'] == 'coffee_total' and r['territory'] == '1000001' and r['year'] == 2025)]
        result = prepare(panel, refs)
        self.assertEqual(result['summary']['eligible_endpoints'], 63)
        excluded = result['exclusion_rows']
        self.assertEqual(len(excluded), 1)
        self.assertEqual(excluded[0]['year'], 2023)
        self.assertEqual(excluded[0]['territory'], '1000001')
        self.assertIn('future_climate_2y_missing', excluded[0]['reasons'])

    def test_future_outcome_is_not_required(self):
        panel, refs = fixture()
        row(panel, year=2025)['production_fixed_ha'] = None
        self.assertEqual(prepare(panel, refs)['summary']['eligible_endpoints'], 64)

    def test_missing_outcome_excludes_endpoint_and_successor_in_all_models(self):
        panel, refs = fixture()
        row(panel)['intended_area_fixed_ha'] = None
        result = prepare(panel, refs)
        ex = [r for r in result['exclusion_rows'] if r['territory'] == '1000001']
        self.assertEqual([r['year'] for r in ex], [2010, 2011])
        for spec in result['specs']:
            if spec['crop'] == 'coffee_total':
                self.assertEqual(len(result['data_by_model'][spec['model_id']]['y']), 30)

    def test_zero_outcome_is_retained(self):
        panel, refs = fixture()
        for outcome in OUTCOMES:
            row(panel)[outcome] = 0.0
        result = prepare(panel, refs)
        self.assertEqual(result['summary']['eligible_endpoints'], 64)
        d = data(result)
        j = next(j for j, r in enumerate(d['rows']) if r['territory'] == '1000001' and r['year'] == 2010)
        self.assertEqual(d['y'][j], 0.0)

    def test_reference_gap_is_not_bridged(self):
        panel, refs = fixture()
        refs = [r for r in refs if not (r['territory'] == '1000001' and r['year'] == 2010)]
        result = prepare(panel, refs)
        ex = [r for r in result['exclusion_rows'] if r['territory'] == '1000001']
        self.assertEqual([r['year'] for r in ex], [2010, 2011])
        self.assertIn('endpoint_not_in_common_support_common_reference', ex[0]['reasons'])
        self.assertIn('predecessor_not_in_common_support_common_reference', ex[1]['reasons'])

    def test_panel_key_missing_from_source_registry_is_hard_error(self):
        panel, refs = fixture()
        panel = [r for r in panel if not (r['territory'] == '1000001' and r['year'] == 2010)]
        with self.assertRaisesRegex(ValueError, 'absent from panel'):
            prepare(panel, refs)

    def test_duplicate_identities_rejected(self):
        panel, refs = fixture()
        with self.assertRaisesRegex(ValueError, 'Duplicate panel identity'):
            prepare(panel+[dict(panel[0])], refs)
        with self.assertRaisesRegex(ValueError, 'Duplicate common-support'):
            prepare(panel, refs+[dict(refs[0])])

    def test_state_and_baseline_area_must_be_constant(self):
        for field, value in (('state', '11'), ('A0', 101.0)):
            panel, refs = fixture()
            row(panel)[field] = value
            with self.assertRaisesRegex(ValueError, 'changes within crop-unit'):
                prepare(panel, refs)

    def test_registry_state_or_exposure_mismatch_rejected(self):
        for field, value in (('state', '11'), ('D0', 99.0), ('history', 99.0)):
            panel, refs = fixture()
            refs[0][field] = value
            with self.assertRaisesRegex(ValueError, 'differs from panel'):
                prepare(panel, refs)

    def test_missing_future_temperature_applies_to_every_model(self):
        panel, refs = fixture()
        row(panel, year=2025)['T0'] = float('nan')
        result = prepare(panel, refs)
        for spec in result['specs']:
            if spec['crop'] == 'coffee_total':
                self.assertEqual(len(result['data_by_model'][spec['model_id']]['y']), 31)

    def test_input_order_does_not_change_design_or_mutate_inputs(self):
        panel, refs = fixture()
        before = copy.deepcopy((panel, refs))
        first = prepare(panel, refs)
        second = prepare(list(reversed(panel)), list(reversed(refs)))
        self.assertEqual(first['support_rows'], second['support_rows'])
        for spec in first['specs']:
            a, b = first['data_by_model'][spec['model_id']], second['data_by_model'][spec['model_id']]
            np.testing.assert_array_equal(a['X'], b['X'])
            np.testing.assert_array_equal(a['y'], b['y'])
        self.assertEqual((panel, refs), before)

    def test_empty_crop_support_remains_explicit_without_fallback(self):
        panel, refs = fixture()
        refs = [r for r in refs if r['crop'] == 'coffee_total']
        result = prepare(panel, refs)
        self.assertEqual(result['summary']['by_crop']['cacao']['rows'], 0)
        self.assertIsNone(result['reference_by_crop']['cacao']['history_mean'])
        self.assertEqual(result['reference_by_crop']['cacao']['X'].shape, (0, 7))
        self.assertEqual(data(result, crop='cacao', variant='levels_unit_trend_future')['X'].shape, (0, 11))

    def test_invalid_identities_and_area_fail_without_coercing_boolean(self):
        for field, value in (('year', True), ('year', 2010.5), ('A0', 0), ('crop', 'other'), ('territory', None)):
            panel, refs = fixture()
            row(panel)[field] = value
            with self.assertRaises(ValueError):
                prepare(panel, refs)


if __name__ == '__main__':
    unittest.main()
