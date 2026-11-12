"""Independent aggregation and preservation checks; no production expansion."""
from pathlib import Path
import math
import tempfile
import unittest

import numpy as np
import expanded_benchmark as eb


class AggregationTests(unittest.TestCase):
    def test_outer_mcse_uses_design_blocks(self):
        got = eb._summarize_field_scalars([1., 3., 5.], 'metric')
        self.assertEqual(got['metric'], 3.)
        self.assertAlmostEqual(got['metric_mcse'], 2. / math.sqrt(3.))
        self.assertEqual(eb._summarize_field_scalars([4.], 'metric')['metric_mcse'], '')

    def test_rmse_is_not_average_of_field_rmse(self):
        got = eb._rmse_from_field_mse([1., 9.], 'rmse')
        self.assertAlmostEqual(got['rmse'], math.sqrt(5.))
        self.assertNotAlmostEqual(got['rmse'], 2.)
        self.assertAlmostEqual(got['rmse_mcse_delta'], 2. / math.sqrt(5.))

    def test_event_zero_one_many_all_planned_in_denominator(self):
        rows = [dict(design_id=i, anchor='onset2y', eligible_units=count,
                     status='ESTIMABLE' if count else 'NONESTIMABLE')
                for i, count in enumerate([0, 1, 12])]
        got = [r for r in eb.aggregate_event_support(rows, 3) if r['anchor'] == 'onset2y'][0]
        self.assertEqual(got['designs_estimable'], 2)
        self.assertEqual(got['designs_nonestimable'], 1)
        self.assertEqual(got['designs_1_to_9_units'], 1)
        self.assertEqual(got['designs_at_least_10_units'], 1)
        self.assertAlmostEqual(got['estimable_fraction_of_planned'], 2/3)
        self.assertAlmostEqual(got['eligible_units'], 13/3)
        no_events = eb.aggregate_events([], 3, 4)
        self.assertEqual(len(no_events), 18)
        self.assertTrue(all(r['status'] == 'NONESTIMABLE_ALL_DESIGNS' for r in no_events))
        self.assertTrue(all(r['designs_planned'] == 3 for r in no_events))

    def test_panel_coverage_and_mse_truths_kept_separate(self):
        rows = []
        for field, noise_errors in enumerate(([1., -1.], [3., -3.])):
            # Projection displacement 2; zero mean noise means structural MSE=projection MSE+4.
            projection_mse = np.mean(np.asarray(noise_errors) ** 2)
            rows.append(dict(design_id=field, scenario='additive_interaction__stable', method='levels',
                functional='interaction_gamma', uncertainty='HAC_L2', replicates_used=2, replicates_failed=0,
                mean_estimate=-78., structural_truth=-80., projection_truth=-78.,
                projection_minus_structural=2., bias_structural=2., bias_projection=0.,
                coverage_structural=[0., 1.][field], coverage_projection=1., mean_estimated_se=1.,
                mean_interval_width=3.92, empirical_sd=math.sqrt(projection_mse * 2),
                rmse_structural=math.sqrt(projection_mse + 4), rmse_projection=math.sqrt(projection_mse)))
        got = [r for r in eb.aggregate_panel(rows, 2, 2) if r['designs_estimable']][0]
        self.assertAlmostEqual(got['coverage_structural'], .5)
        self.assertAlmostEqual(got['coverage_structural_mcse'], .5)
        self.assertAlmostEqual(got['coverage_projection'], 1.)
        self.assertAlmostEqual(got['rmse_projection'], math.sqrt(5.))
        self.assertAlmostEqual(got['rmse_structural'], 3.)
        self.assertLess(got['mse_decomposition_abs_error'], 1e-12)
        self.assertEqual(got['mcse_unit'], 'independent_complete_design')
        self.assertFalse(got['quantiles_are_confidence_intervals'])

    def test_duplicate_design_is_not_extra_information(self):
        row = dict(design_id=1, anchor='onset2y', eligible_units=1, status='ESTIMABLE')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            eb.aggregate_event_support([row, row], 3)


class PreservationTests(unittest.TestCase):
    def test_completed_attempt_reuse_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            field = Path(tmp) / 'design_1000'
            attempt = field / 'attempts' / 'first'
            attempt.mkdir(parents=True)
            (attempt / 'data.txt').write_text('original data', encoding='utf-8')
            manifest = eb._manifest_attempt(attempt, 'protocol_hash', 1000, {'status': 'COMPLETE'})
            eb._atomic_json(field / 'COMPLETE.json', dict(design_id=1000, protocol_sha256='protocol_hash',
                            attempt='attempts/first', manifest_sha256=manifest))
            self.assertEqual(eb._verified_completed(field, 'protocol_hash', 1000), attempt)
            with self.assertRaisesRegex(ValueError, 'different protocol'):
                eb._verified_completed(field, 'changed_protocol', 1000)
            (attempt / 'data.txt').write_text('changed data', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'hash differs'):
                eb._verified_completed(field, 'protocol_hash', 1000)
            self.assertTrue((attempt / 'data.txt').exists())

    def test_interrupted_attempt_remains_uncompleted(self):
        with tempfile.TemporaryDirectory() as tmp:
            field = Path(tmp) / 'design_1000'
            incomplete = field / 'attempts' / 'interrupted'
            incomplete.mkdir(parents=True)
            (incomplete / 'partial.txt').write_text('retain me', encoding='utf-8')
            self.assertIsNone(eb._verified_completed(field, 'protocol_hash', 1000))
            self.assertEqual((incomplete / 'partial.txt').read_text(), 'retain me')

    def test_protocol_change_cannot_mix_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            original = {'seed': 1, 'fields': [1, 2]}
            digest = eb._lock_protocol(out, original)
            self.assertEqual(eb._lock_protocol(out, original), digest)
            with self.assertRaisesRegex(ValueError, 'differs'):
                eb._lock_protocol(out, {'seed': 2, 'fields': [1, 2]})

    def test_smoke_protocol_cannot_be_mistaken_for_production(self):
        smoke = eb.expansion_protocol(True)
        full = eb.expansion_protocol(False)
        self.assertTrue(smoke['development_run'])
        self.assertEqual(smoke['design_ids'], [1000, 1001])
        self.assertEqual(smoke['inner_response_replicates'], 4)
        self.assertFalse(full['development_run'])
        self.assertEqual(full['independent_designs_planned'], 100)
        self.assertEqual(full['response_datasets_planned'], 36000)
        self.assertFalse(full['climate_only_redraw'])


if __name__ == '__main__':
    unittest.main()
