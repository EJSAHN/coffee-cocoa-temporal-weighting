"""Complete fictional analysis and exported-table consistency checks."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from coffee_cocoa_core.panel_analysis.workflow import run_panel
from coffee_cocoa_core.panel_analysis.tests.fixture_panel import fictional_inputs


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.out=Path(cls.temp.name)
        cls.panel=fictional_inputs()['panel']
        cls.summary=run_panel(cls.panel,cls.out,log=lambda _:None,do_influence=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def rows(self,name):
        with (self.out/name).open(newline='',encoding='utf-8') as f:
            return list(csv.DictReader(f))

    def test_all_models_fit_from_panel_and_complete_influences(self):
        self.assertEqual(self.summary['status'],'COMPLETE_PANEL_ANALYSIS')
        self.assertEqual(self.summary['models_fitted'],56)
        self.assertFalse(self.summary['model_coefficients_read_from_previous_results'])
        self.assertTrue(self.summary['influence_diagnostics_performed'])
        self.assertEqual(len(self.rows('panel_models/model_registry.csv')),25)
        self.assertEqual(len(self.rows('temporal_comparison/model_registry.csv')),20)
        self.assertEqual(self.summary['families']['response_comparison']['common_support_models_fitted'],11)
        self.assertEqual(len(self.rows('panel_models/leave_one_group.csv')),46)
        self.assertEqual(len(self.rows('response_comparison/current_slope_leave_one_group.csv')),92)

    def test_all_exported_original_coefficients_match_full_model_files(self):
        for row in self.rows('panel_models/coefficients.csv'):
            path=self.out/'panel_models/models'/(row['model_id']+'.json')
            fit=json.loads(path.read_text())
            ci=fit['uncertainty'][row['uncertainty']]['intervals'][row['term']]
            for name in ('estimate','standard_error','lower','upper'):
                self.assertEqual(float(row[name]),ci[name])

    def test_common_slope_and_temporal_output_menus_are_complete(self):
        self.assertEqual(len(self.rows('response_comparison/original_model_current_slopes.csv')),25*2*2)
        self.assertEqual(len(self.rows('response_comparison/common_fit_current_slopes.csv')),11*2)
        self.assertEqual(len(self.rows('temporal_comparison/temporal_contrasts.csv')),20*3*2)
        self.assertEqual(len(self.rows('temporal_comparison/future_climate_diagnostics.csv')),2*4*2)
        self.assertEqual(len(list((self.out/'temporal_comparison/designs').glob('*.csv.gz'))),20)
        self.assertEqual(len(self.rows('response_comparison/panel_fit_reconciliation.csv')),25)
        self.assertEqual(len(self.rows('temporal_comparison/common_support_anchor_reconciliation.csv')),2)

    def test_component_totals_match_current_production_functional(self):
        slopes={r['crop']:r for r in self.rows('response_comparison/original_model_current_slopes.csv')
                if r['variant']=='primary' and r['outcome']=='production_fixed_ha' and
                r['evaluation_scope']=='common_evaluation_rows' and r['uncertainty']=='HAC_L2'}
        for row in self.rows('response_comparison/production_association_component_allocation.csv'):
            self.assertAlmostEqual(float(row['total_current_association']),float(slopes[row['crop']]['estimate']),places=8)
            parts=['baseline_constant_association_allocation','cumulative_area_association_allocation',
                   'cumulative_productivity_association_allocation','cumulative_unallocated_association_allocation',
                   'unavailable_reference_allocation']
            self.assertAlmostEqual(sum(float(row[k]) for k in parts),float(row['total_current_association']),places=8)

    def test_json_outputs_are_finite_and_summary_matches(self):
        for path in self.out.rglob('*.json'):
            value=json.loads(path.read_text())
            json.dumps(value,allow_nan=False)
        self.assertEqual(json.loads((self.out/'panel_analysis_summary.json').read_text()),self.summary)


if __name__=='__main__':unittest.main()
