"""Fictional arithmetic examples only; no real production values or regressions."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import copy
from pathlib import Path
import sys
import unittest

from coffee_cocoa_core.panel_analysis.trajectories import analyse_trajectories


def row(year, q, h, *, i=None, a0=100.0, territory='101', crop='coffee_total', state='10'):
    return {'year': year, 'Q_tonnes': q, 'Hha': h, 'Iha': h if i is None else i,
            'A0': a0, 'territory': territory, 'crop': crop, 'state': state, 'pilot': False}


def transition(result, territory='101', year=2007, crop='coffee_total'):
    return next(r for r in result['transition_rows'] if
                r['territory'] == territory and r['end_year'] == year and r['crop'] == crop)


def level(result, year, territory='101', crop='coffee_total'):
    return next(r for r in result['level_component_rows'] if
                r['territory'] == territory and r['year'] == year and r['crop'] == crop)


class TrajectoryTests(unittest.TestCase):
    def test_exact_midpoint_signed_allocation_and_zero_valid_yield(self):
        # h/A0:1->2; yield:100->50; fixed-denominator production:100->100.
        result = analyse_trajectories([row(2006, 10, 100), row(2007, 10, 200), row(2008, 0, 100)])
        t = transition(result)
        self.assertEqual(t['observed_delta_kg_fixed_ha'], 0)
        self.assertEqual(t['area_midpoint_allocation_kg_fixed_ha'], 75)
        self.assertEqual(t['productivity_midpoint_allocation_kg_fixed_ha'], -75)
        self.assertEqual(t['unallocated_observed_change_kg_fixed_ha'], 0)
        t2 = transition(result, year=2008)
        self.assertEqual(t2['yield_end_kg_harvested_ha'], 0)
        self.assertTrue(t2['allocation_eligible'])
        self.assertAlmostEqual(t2['area_midpoint_allocation_kg_fixed_ha'] +
                               t2['productivity_midpoint_allocation_kg_fixed_ha'], -100)

    def test_zero_harvest_keeps_change_and_never_imputes_yield(self):
        result = analyse_trajectories([row(2006, 10, 100), row(2007, 0, 0), row(2008, 20, 100)])
        t = transition(result)
        self.assertFalse(t['allocation_eligible'])
        self.assertIsNone(t['yield_end_kg_harvested_ha'])
        self.assertIsNone(t['area_midpoint_allocation_kg_fixed_ha'])
        self.assertEqual(t['unallocated_observed_change_kg_fixed_ha'], -100)
        t2 = transition(result, year=2008)
        self.assertIsNone(t2['yield_start_kg_harvested_ha'])
        self.assertEqual(t2['unallocated_observed_change_kg_fixed_ha'], 200)
        c = level(result, 2008)
        self.assertEqual(c['cumulative_area_allocation_kg_fixed_ha'], 0)
        self.assertEqual(c['cumulative_productivity_allocation_kg_fixed_ha'], 0)
        self.assertEqual(c['cumulative_unallocated_change_kg_fixed_ha'], 100)
        self.assertEqual(c['undefined_yield_observed_transitions_since_reference'], 2)

    def test_missing_production_is_not_zero_and_gap_goes_to_level_residual(self):
        result = analyse_trajectories([row(2006, 10, 100), row(2007, None, 110),
                                       row(2008, 30, 120), row(2009, 40, 120)])
        for year in (2007, 2008):
            self.assertIsNone(transition(result, year=year)['observed_delta_kg_fixed_ha'])
        self.assertIsNone(level(result, 2007)['cumulative_unallocated_change_kg_fixed_ha'])
        self.assertEqual(level(result, 2008)['cumulative_unallocated_change_kg_fixed_ha'], 200)
        c = level(result, 2009)
        self.assertEqual(c['cumulative_area_allocation_kg_fixed_ha'], 0)
        self.assertAlmostEqual(c['cumulative_productivity_allocation_kg_fixed_ha'], 100)
        self.assertAlmostEqual(c['cumulative_unallocated_change_kg_fixed_ha'], 200)
        self.assertEqual(c['missing_production_transitions_since_reference'], 2)
        self.assertAlmostEqual(sum(c[k] for k in (
            'baseline_2006_production_kg_fixed_ha', 'cumulative_area_allocation_kg_fixed_ha',
            'cumulative_productivity_allocation_kg_fixed_ha', 'cumulative_unallocated_change_kg_fixed_ha')), 400)

    def test_missing_reference_does_not_drop_rows_or_fabricate_components(self):
        result = analyse_trajectories([row(2006, None, 100), row(2007, 10, 100), row(2008, 20, 100)])
        self.assertEqual(len(result['annual_unit_rows']), 3)
        self.assertEqual(result['validation']['units_missing_2006_production'], 1)
        c = level(result, 2008)
        self.assertFalse(c['level_decomposition_available'])
        self.assertEqual(c['observed_production_kg_fixed_ha'], 200)
        for key in ('cumulative_area_allocation_kg_fixed_ha',
                    'cumulative_productivity_allocation_kg_fixed_ha',
                    'cumulative_unallocated_change_kg_fixed_ha'):
            self.assertIsNone(c[key])

    def test_mixed_aggregation_keeps_same_observed_denominator_and_abs_unallocated(self):
        result = analyse_trajectories([
            row(2006, 10, 100), row(2007, 20, 100),
            row(2006, 10, 100, territory='102'), row(2007, 0, 0, territory='102'),
            row(2006, 0, 0, territory='103'), row(2007, 10, 100, territory='103'),
            row(2006, None, 100, territory='104'), row(2007, 20, 100, territory='104')])
        s = next(r for r in result['transition_summary_rows'] if r['state'] == 'ALL' and r['end_year'] == 2007)
        self.assertEqual(s['fixed_cohort_units'], 4)
        self.assertEqual(s['production_change_observed_units'], 3)
        self.assertEqual(s['allocation_eligible_units'], 1)
        self.assertEqual(s['unallocated_observed_units'], 2)
        self.assertEqual(s['sum_unallocated_observed_change_kg_fixed_ha'], 0)
        self.assertEqual(s['sum_absolute_unallocated_observed_change_kg_fixed_ha'], 200)
        self.assertAlmostEqual(s['mean_productivity_allocation_same_observed_denominator'], 100/3)
        self.assertEqual(s['allocation_identity_residual_kg_fixed_ha'], 0)

    def test_equal_unit_and_baseline_area_weighted_targets_differ(self):
        result = analyse_trajectories([row(2007, 10, 100),
                                       row(2007, 90, 300, territory='102', a0=300)])
        s = next(r for r in result['annual_summary_rows'] if r['state'] == 'ALL' and r['year'] == 2007)
        self.assertEqual(s['common_qih_equal_unit_mean_kg_fixed_ha'], 200)
        self.assertEqual(s['common_qih_baseline_area_weighted_kg_fixed_ha'], 250)
        self.assertEqual(s['common_qih_aggregate_reported_kg_per_harvested_ha'], 250)
        self.assertEqual(s['common_qih_sum_A0'], 400)

    def test_shared_municipality_crop_separation_and_all_cohort_coverage(self):
        result = analyse_trajectories([row(2006, 10, 100), row(2007, 20, 100),
                                       row(2007, 70, 100, crop='cacao')])
        self.assertEqual(result['validation']['fixed_crop_units'], 2)
        self.assertEqual(transition(result)['observed_delta_kg_fixed_ha'], 100)
        self.assertIsNone(transition(result, crop='cacao')['observed_delta_kg_fixed_ha'])
        s = next(r for r in result['annual_summary_rows'] if
                 r['state'] == 'ALL' and r['crop'] == 'cacao' and r['year'] == 2006)
        self.assertEqual(s['fixed_cohort_units'], 1)
        self.assertEqual(s['absent_source_rows'], 1)
        self.assertIsNone(s['common_qih_sum_Q_tonnes'])

    def test_missing_area_retains_production_and_reports_common_support(self):
        a = row(2007, 10, None)
        a['Iha'] = None
        result = analyse_trajectories([a, row(2007, 0, 0, territory='102')])
        s = next(r for r in result['annual_summary_rows'] if r['state'] == 'ALL' and r['year'] == 2007)
        self.assertEqual(s['production_observed_units'], 2)
        self.assertEqual(s['common_qih_observed_units'], 1)
        self.assertEqual(s['production_observed_equal_unit_mean_kg_fixed_ha'], 50)
        self.assertEqual(s['common_qih_equal_unit_mean_kg_fixed_ha'], 0)
        self.assertIsNone(s['common_qih_aggregate_reported_kg_per_harvested_ha'])

    def test_posthoc_cases_all_supplied_years_and_input_unmodified(self):
        panel = [row(2002, 10, 100, territory='1500859', crop='cacao'),
                 row(2025, 20, 200, territory='1500859', crop='cacao'),
                 row(2025, 30, 300, territory='1500859', crop='coffee_total')]
        saved = copy.deepcopy(panel)
        result = analyse_trajectories(panel)
        self.assertEqual(panel, saved)
        self.assertEqual([r['year'] for r in result['selected_case_rows']], [2002, 2025])
        self.assertTrue(all(r['selection'].startswith('POSTHOC') for r in result['selected_case_rows']))
        self.assertEqual(len(result['annual_unit_rows']), 3)

    def test_source_anomalies_flagged_not_silently_censored(self):
        result = analyse_trajectories([row(2007, 10, 0, i=20), row(2008, 10, 30, i=20)])
        zero = result['annual_unit_rows'][0]
        self.assertTrue(zero['positive_production_zero_harvested_area'])
        self.assertIsNone(zero['yield_kg_harvested_ha'])
        self.assertTrue(result['annual_unit_rows'][1]['harvested_exceeds_intended'])
        self.assertEqual(result['annual_unit_rows'][1]['reported_unharvested_area_ratio_A0'], -0.1)

    def test_duplicate_or_changed_frozen_denominator_rejected(self):
        for panel in ([row(2007, 10, 100), row(2007, 10, 100)],
                      [row(2006, 10, 100), row(2007, 10, 100, a0=200)],
                      [row(2006, 10, 100), row(2007, 10, 100, state='11')]):
            with self.assertRaises(ValueError):
                analyse_trajectories(panel)

    def test_nonfinite_negative_invalidyear_rejected(self):
        for bad in [row(2007, float('nan'), 100), row(2007, -1, 100),
                    row(2007, 1, 100, a0=0), row(2026, 1, 100)]:
            with self.assertRaises(ValueError):
                analyse_trajectories([bad])


if __name__ == '__main__':
    unittest.main()
