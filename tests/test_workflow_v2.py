"""Integration and cache-integrity checks on fictional sources only."""
from fractions import Fraction
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from coffee_cocoa_core import workflow_v2
from coffee_cocoa_core.io import sha256
from coffee_cocoa_core.event_designs import _tables_to_designs
from coffee_cocoa_core.temporal import analyse
from temporal_diagnostics import diagnose
from test_pipeline_smoke import fictional_raw
from test_event_designs_portable import fictional_tables


class UnifiedWorkflowTests(unittest.TestCase):
    def test_generic_adapter_preserves_zero_rows_and_exact_coefficients(self):
        designs = _tables_to_designs(fictional_tables())
        rows = workflow_v2.diagnostic_records(designs)
        self.assertEqual(len(rows), sum(len(d['weights_exact']) for d in designs))
        self.assertTrue(any(Fraction(r['weight']) == 0 for r in rows))
        self.assertTrue(all('outcome' not in r for r in rows))
        reference = analyse(designs, log=lambda _: None)
        result = diagnose(rows, origin=2006)
        verified = workflow_v2.validate_diagnostics(result, reference['unit_rows'], reference['design_rows'])
        self.assertEqual(verified['contrasts'], 16)
        self.assertFalse(verified['outcome_values_used'])

    def test_complete_workflow_and_verified_resume_then_tamper_rejection(self):
        with tempfile.TemporaryDirectory(prefix='unified_v2_fictional_') as temporary:
            root = Path(temporary)
            fictional_raw(root / 'raw')
            output = root / 'results'
            first = workflow_v2.run(root / 'raw', output, smoke=True, log_fn=lambda _: None)
            self.assertEqual(first['status'], 'COMPLETE_UNIFIED_DEVELOPMENT_SMOKE')
            self.assertFalse(first['analysis_reused_verified'])
            self.assertEqual(first['analysis']['panel']['models_fitted'], 56)
            self.assertEqual(first['analysis']['events']['complete_designs'], 16)
            self.assertEqual(first['expansion']['designs_completed'], 2)
            self.assertEqual(first['expansion']['synthetic_response_datasets'], 72)
            self.assertEqual(Path(first['prepared_input_dir']), output / 'analysis/inputs')
            self.assertFalse(first['scientific_results_for_manuscript'])
            complete_before = sha256(output / 'analysis/output_manifest.json')
            # A successful resume must not invoke the original empirical pipeline.
            with patch('coffee_cocoa_core.pipeline.run', side_effect=AssertionError('Unexpected refit')):
                second = workflow_v2.run(root / 'raw', output, smoke=True, log_fn=lambda _: None)
            self.assertTrue(second['analysis_reused_verified'])
            self.assertEqual(second['expansion']['designs_reused_verified'], 2)
            self.assertEqual(second['expansion']['designs_new_this_invocation'], 0)
            self.assertEqual(sha256(output / 'analysis/output_manifest.json'), complete_before)
            current = json.loads((output / 'workflow_summary.json').read_text())
            self.assertEqual(current['status'], second['status'])
            with self.assertRaisesRegex(ValueError, 'runtime differs'):
                workflow_v2.run(root / 'raw', output, smoke=False, log_fn=lambda _: None)
            target = output / 'analysis/temporal_weights/temporal_design_summary.csv'
            target.write_bytes(target.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                workflow_v2.run(root / 'raw', output, smoke=True, log_fn=lambda _: None)
            self.assertTrue(list(output.glob('failed_workflow_*.json')))
            self.assertEqual(sha256(output / 'analysis/output_manifest.json'), complete_before)

    def test_os_lock_does_not_allow_concurrent_output_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            with workflow_v2._exclusive_workflow(Path(temporary)):
                with self.assertRaisesRegex(RuntimeError, 'Another process'):
                    with workflow_v2._exclusive_workflow(Path(temporary)):
                        self.fail('Second lock acquired')
            with workflow_v2._exclusive_workflow(Path(temporary)):
                pass


if __name__ == '__main__':
    unittest.main()
