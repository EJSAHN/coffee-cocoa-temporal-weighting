"""Independent fictional checks of path components under unchanged FE weights."""
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
from coffee_cocoa_core.panel_analysis.components import allocate_association
from coffee_cocoa_core.panel_analysis.contrasts import current_slope, generate_contributions
from coffee_cocoa_core.panel_analysis.fixed_effects import fit_model
from coffee_cocoa_core.panel_analysis.trajectories import analyse_trajectories


def panel_row(year, q, h, territory='101', crop='cacao', state='10'):
    return dict(year=year, Q_tonnes=q, Hha=h, Iha=h, A0=100.,
                territory=territory, crop=crop, state=state, pilot=False)


def weights(territory='101', crop='cacao'):
    return [dict(crop=crop, territory=territory, state='10', year=year,
                 model_id='fictional', coefficient_weight=a, raw_response_contribution=a*y)
            for year, a, y in [(2007, -0.5, 100.), (2008, 0.5, 200.)]]


class ComponentTests(unittest.TestCase):
    def test_exact_known_path_with_cancelling_component_signs(self):
        panel = [panel_row(2006, 10, 100), panel_row(2007, 10, 200), panel_row(2008, 20, 100)]
        levels = analyse_trajectories(panel)['level_component_rows']
        original = copy.deepcopy(levels)
        r = allocate_association({'cacao': weights()}, levels)
        self.assertEqual(levels, original)
        s = r['summary_rows'][0]
        # 2007->08: area -125, productivity +225; weights divide by two.
        self.assertAlmostEqual(s['cumulative_area_association_allocation'], -62.5)
        self.assertAlmostEqual(s['cumulative_productivity_association_allocation'], 112.5)
        self.assertAlmostEqual(s['cumulative_unallocated_association_allocation'], 0)
        self.assertEqual(s['baseline_constant_association_allocation'], 0)
        self.assertEqual(s['total_current_association'], 50)
        self.assertEqual(s['unavailable_reference_rows'], 0)
        self.assertEqual(s['identity_residual'], 0)

    def test_zero_harvest_and_production_gap_stay_unallocated(self):
        for middle in (panel_row(2006, 10, 0), panel_row(2006, 10, None)):
            panel = [middle, panel_row(2007, 10, 0), panel_row(2008, 20, 100)]
            levels = analyse_trajectories(panel)['level_component_rows']
            s = allocate_association({'cacao': weights()}, levels)['summary_rows'][0]
            self.assertEqual(s['cumulative_area_association_allocation'], 0)
            self.assertEqual(s['cumulative_productivity_association_allocation'], 0)
            self.assertEqual(s['cumulative_unallocated_association_allocation'], 50)
        panel = [panel_row(2006, 10, 100), panel_row(2007, None, 100), panel_row(2008, 20, 100)]
        levels = analyse_trajectories(panel)['level_component_rows']
        # Q at 2007 is truly unavailable: the model has 2008 and 2009 instead.
        panel.append(panel_row(2009, 30, 100))
        levels = analyse_trajectories(panel)['level_component_rows']
        ww = [dict(model_id='fictional', territory='101', state='10', year=y,
                   coefficient_weight=a, raw_response_contribution=a*v)
              for y, a, v in [(2008, -0.5, 200), (2009, 0.5, 300)]]
        s = allocate_association({'cacao': ww}, levels)['summary_rows'][0]
        self.assertEqual(s['cumulative_productivity_association_allocation'], 50)
        self.assertEqual(s['cumulative_unallocated_association_allocation'], 0)
        # The gap residual is 100 in BOTH years, hence absorbed in this contrast.
        self.assertTrue(all(r['cumulative_unallocated_change_kg_fixed_ha'] == 100
                            for r in levels if r['year'] in (2008, 2009)))

    def test_missing_reference_preserves_whole_unit_contribution(self):
        panel = [panel_row(2006, None, 100), panel_row(2007, 10, 100), panel_row(2008, 20, 100)]
        levels = analyse_trajectories(panel)['level_component_rows']
        result = allocate_association({'cacao': weights()}, levels)
        s = result['summary_rows'][0]
        self.assertEqual(s['unavailable_reference_rows'], 2)
        self.assertEqual(s['unavailable_reference_allocation'], 50)
        self.assertEqual(s['cumulative_area_association_allocation'], 0)
        self.assertEqual(s['cumulative_productivity_association_allocation'], 0)
        self.assertEqual(s['cumulative_unallocated_association_allocation'], 0)
        self.assertTrue(all(not r['components_available'] for r in result['row_allocations']))

    def test_missing_level_is_not_a_missing_reference(self):
        with self.assertRaises(ValueError):
            allocate_association({'cacao': weights()}, [])

    def test_wrong_observed_production_and_duplicate_identity_raise(self):
        panel = [panel_row(2006, 10, 100), panel_row(2007, 10, 100), panel_row(2008, 20, 100)]
        levels = analyse_trajectories(panel)['level_component_rows']
        with self.assertRaises(ValueError):
            allocate_association({'cacao': weights()}, levels + levels[:1])
        bad = weights(); bad[1]['raw_response_contribution'] += 1
        with self.assertRaises(ValueError):
            allocate_association({'cacao': bad}, levels)

    def test_missing_reference_still_requires_matched_observed_production(self):
        panel = [panel_row(2006, None, 100), panel_row(2007, 10, 100), panel_row(2008, 20, 100)]
        levels = analyse_trajectories(panel)['level_component_rows']
        bad = weights(); bad[1]['raw_response_contribution'] += 1
        with self.assertRaises(ValueError):
            allocate_association({'cacao': bad}, levels)

    def test_inconsistent_availability_or_state_is_rejected(self):
        panel = [panel_row(2006, 10, 100), panel_row(2007, 10, 100), panel_row(2008, 20, 100)]
        for field, value in [('state', 'wrong_state'), ('level_decomposition_available', False),
                             ('observed_production_kg_fixed_ha', None)]:
            levels = analyse_trajectories(panel)['level_component_rows']
            levels[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                allocate_association({'cacao': weights()}, levels)

    def test_full_fixed_effect_functional_matches_independent_dense_fit(self):
        rng = np.random.default_rng(84711)
        panel, rows, x, y, unit, state, years = [], [], [], [], [], [], []
        names = ['D0', 'D1', 'D2', 'T0', 'T1', 'T2', 'D0_x_history']
        for j in range(12):
            ident, st = 'unit'+str(j), 'state'+str(j//4)
            panel.append(panel_row(2006, None if j == 0 else 10+j, 100+j, ident, state=st))
            for year in range(2007, 2026):
                z = rng.normal(size=6); h = (z[1]+z[2])/2
                xx = [*z, z[0]*h]
                yy = 200+4*j+(year-2007)*2+3*z[0]-2*z[0]*h+rng.normal()
                hh = 0 if (j == 1 and year == 2011) else 100+4*j+2*(year-2007)
                q = None if (j == 2 and year == 2010) else yy/10
                panel.append(panel_row(year, q, hh, ident, state=st))
                if q is None:
                    continue
                x.append(xx); y.append(yy); unit.append(ident); state.append(st); years.append(year)
                rows.append({'territory': ident})
        data = dict(X=np.array(x), y=np.array(y), unit=np.array(unit), state=np.array(state),
                    year=np.array(years), names=names, rows=rows)
        spec = dict(model_id='fictional_fe', crop='cacao', outcome='fictional', variant='primary',
                    unit_trends=False, unit_parity=False)
        fit = fit_model(data['X'], data['y'], data['unit'], data['state'], data['year'], names)
        gradient = current_slope(fit, data['X'], names)['scaled_gradient']
        wr, check = generate_contributions(data, spec, fit, gradient)
        levels = analyse_trajectories(panel)['level_component_rows']
        result = allocate_association({'cacao': wr}, levels)
        s = result['summary_rows'][0]
        u = np.column_stack([data['unit'] == v for v in np.unique(data['unit'])])
        sy = np.array([st+'_'+str(yr) for st, yr in zip(state, years)])
        t = np.column_stack([sy == v for v in np.unique(sy)])
        full = np.column_stack([data['X'], u, t])
        v = np.r_[gradient, np.zeros(u.shape[1]+t.shape[1])]
        a_dense = np.linalg.lstsq(full.T, v, rcond=1e-12)[0]
        np.testing.assert_allclose([r['coefficient_weight'] for r in wr], a_dense, atol=1e-9)
        self.assertAlmostEqual(s['total_current_association'], float(a_dense @ data['y']), places=8)
        self.assertAlmostEqual(s['total_current_association'], check['target'], places=8)
        self.assertAlmostEqual(s['identity_residual'], 0, places=9)
        self.assertAlmostEqual(s['baseline_constant_association_allocation'], 0, places=8)
        self.assertEqual(s['unavailable_reference_rows'], 19)
        # Missing 2006 changes allocation availability only, not the fit sample.
        self.assertEqual(s['n_rows'], len(data['y']))
        self.assertFalse(s['independent_component_intervals_computed'])
        self.assertFalse(s['percent_causal_shares_computed'])


if __name__ == '__main__':
    unittest.main()
