"""Known-score metrics and tiny fictional workflow tests only (20 replicates)."""
from pathlib import Path
import csv
import gzip
import json
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coffee_cocoa_core.benchmark.runner import metrics, protocol, run_benchmark, table


class MetricTests(unittest.TestCase):
    def test_structural_and_projection_truths_remain_distinct(self):
        est = np.array([-2., 0., 2., 4.])
        summary, lower, upper = metrics(est, np.ones(4), 0., 1.)
        self.assertEqual(summary['mean_estimate'], 1.)
        self.assertEqual(summary['bias_structural'], 1.)
        self.assertEqual(summary['bias_projection'], 0.)
        self.assertAlmostEqual(summary['rmse_structural'], np.sqrt(6))
        self.assertAlmostEqual(summary['rmse_projection'], np.sqrt(5))
        self.assertEqual(summary['coverage_structural'], .25)
        self.assertEqual(summary['coverage_projection'], .5)
        self.assertEqual(summary['false_zero_exclusion_when_structural_null'], .75)
        self.assertAlmostEqual(summary['mcse_coverage_structural'], np.sqrt(.25*.75/4))
        self.assertAlmostEqual(summary['mcse_mean'], np.std(est, ddof=1)/2)
        np.testing.assert_allclose(lower, est-1.959963984540054)
        np.testing.assert_allclose(upper, est+1.959963984540054)
        json.dumps(summary, allow_nan=False)

    def test_nonnull_is_not_labeled_false_positive_and_zero_error_is_valid(self):
        summary, _, _ = metrics(np.full(4, 5.), np.zeros(4), 5., 5.)
        self.assertEqual(summary['false_zero_exclusion_when_structural_null'], '')
        self.assertEqual(summary['coverage_structural'], 1.)
        self.assertEqual(summary['rmse_structural'], 0.)
        self.assertEqual(summary['mcse_rmse_structural'], 0.)

    def test_nonfinite_or_bad_replications_are_not_dropped(self):
        for est, var in (([1., np.nan], [1., 1.]), ([1., 2.], [1., np.inf]),
                         ([1., 2.], [-1., 1.]), ([1.], [1.]), ([1., 2.], [1.])):
            with self.subTest(est=est, var=var), self.assertRaises(ValueError):
                metrics(est, var, 0., 0.)

    def test_development_protocol_is_distinguished(self):
        self.assertFalse(protocol(20261005, 400)['development_run'])
        self.assertTrue(protocol(20261005, 20, 12, (0,))['development_run'])
        self.assertFalse(protocol(20261005, 400)['event_theta_compared_with_gamma'])

    def test_empty_table_can_have_explicit_header(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'empty.csv'
            table(path, [], ['status', 'reason'])
            self.assertEqual(path.read_text().splitlines(), ['status,reason'])


class MiniatureWorkerTest(unittest.TestCase):
    def test_complete_nine_scenario_workflow_preserves_oracles_and_replicates(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/'full'; results = out/'results'
            summary = run_benchmark(out, seed=20261005, replicates=20,
                                    n_units=12, design_ids=(0,), log_fn=lambda message:None)
            self.assertIn(summary['status'], ('COMPLETE_SYNTHESIS_AND_SIMULATION_BENCHMARK', 'COMPLETE_WITH_NONESTIMABLE_DESIGNS'))
            self.assertTrue(summary['development_run'])
            self.assertEqual(summary['completed_scenarios'], 9)
            self.assertEqual(summary['synthetic_response_datasets'], 180)
            self.assertEqual(summary['real_data_fits_performed'], 0)
            self.assertEqual(summary['numerical_failed_replicates_dropped'], 0)
            self.assertEqual(set(summary['output_paths']), {'simulation_panel_summary', 'simulation_event_decomposition', 'simulation_nonestimable', 'simulation_projection_oracles', 'simulation_design_registry', 'simulation_event_support'})
            json.dumps(summary, allow_nan=False)
            progress = json.loads((results/'simulation_progress.json').read_text())
            self.assertEqual(progress, summary)
            with (results/'simulation_panel_summary.csv').open(newline='') as f:
                panel = list(csv.DictReader(f))
            self.assertEqual(len(panel), 9*3*2*2)
            self.assertTrue(all(int(r['replicates_used'])==20 for r in panel))
            self.assertTrue(all(int(r['replicates_failed'])==0 for r in panel))
            # Correctly specified noiseless models recover known coefficients;
            # this is a deterministic identity, not a Monte Carlo pass criterion.
            for row in panel:
                if row['background']=='stable' or (row['background']=='linear_unit_drift' and row['method']!='levels'):
                    self.assertLess(abs(float(row['projection_minus_structural'])), 1e-7)
            with (results/'simulation_event_decomposition.csv').open(newline='') as f:
                event = list(csv.DictReader(f))
            for row in event:
                self.assertAlmostEqual(float(row['full_mean_contrast']),
                                       float(row['climate_signal_contrast'])+float(row['background_contrast']), places=8)
                self.assertEqual(row['quantiles_are_confidence_intervals'], 'False')
                self.assertEqual(row['compared_with_regression_gamma'], 'False')
                self.assertEqual(int(row['replicates_used']), 20)
            files = sorted((out/'design_0').glob('*_coefficient_replicates.csv.gz'))
            self.assertEqual(len(files), 9)
            for path in files:
                with gzip.open(path, 'rt', newline='') as f:
                    replicates = list(csv.DictReader(f))
                self.assertEqual(len(replicates), 3*20)
            saved = np.load(out/'design_0'/'outcome_noise.npz')
            self.assertEqual(saved['error'].shape, (12, 24, 20))
            self.assertTrue(np.all(np.isfinite(saved['error'])))


if __name__=='__main__':
    unittest.main()
