"""Export boundaries: fresh inputs, exact preservation and verified resumption."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from publication import export_v2 as exports
from coffee_cocoa_core import workflow_v2
from test_pipeline_smoke import fictional_raw


class ExportIntegrationTests(unittest.TestCase):
    def test_completed_workflow_export_resume_and_corruption(self):
        with tempfile.TemporaryDirectory(prefix='v2_export_test_') as tmp:
            root = Path(tmp)
            fictional_raw(root / 'inputs')
            workflow = root / 'workflow'
            workflow_v2.run(root / 'inputs', workflow, smoke=True, log_fn=lambda *_: None)
            output = root / 'publication'
            first = exports.export_all(workflow, output, figures=False, log_fn=lambda *_: None)
            self.assertEqual(first['status'], 'COMPLETE_PUBLICATION_EXPORT_V2')
            self.assertEqual(first['figure_status'], 'NOT_REQUESTED')
            self.assertEqual(first['plot_data_tables'], 8)
            self.assertEqual(first['extension_csv_tables'], 11)
            self.assertFalse((output / 'figures').exists())
            digest = exports.sha(output / 'Supplementary_Data_S1.zip')
            # Actual expansion resumption rewrites operational timings and cache-use flags.
            with patch('coffee_cocoa_core.pipeline.run', side_effect=AssertionError('Unexpected empirical refit')):
                resumed = workflow_v2.run(root / 'inputs', workflow, smoke=True, log_fn=lambda *_: None)
            self.assertTrue(resumed['analysis_reused_verified'])
            self.assertEqual(resumed['expansion']['designs_reused_verified'], 2)
            second = exports.export_all(workflow, output, figures=False, log_fn=lambda *_: None)
            self.assertTrue(second['reused_verified_export'])
            self.assertEqual(digest, exports.sha(output / 'Supplementary_Data_S1.zip'))
            with zipfile.ZipFile(output / 'Supplementary_Data_S1.zip') as z:
                self.assertIsNone(z.testzip())
                manifest = json.loads(z.read('manifest_sha256.json'))
                for name, expected in manifest.items():
                    self.assertEqual(hashlib.sha256(z.read(name)).hexdigest(), expected)
            # The source CSV value, not a shipped plot table, is propagated.
            tables = exports.load_fresh_tables(workflow / 'analysis', input_dir=workflow / 'analysis/inputs')
            tables['event_summary']['rows'][0]['estimate'] = '123.456789'
            selected = exports.manuscript_tables(tables, workflow / 'expanded_benchmark')
            self.assertTrue(any(r['Contrast_kg_per_baseline_ha'] == '123.456789' for r in selected['Table_1_event_contrasts']))
            target = output / 'plot_data' / (exports.NAMES[0] + '_plot_data.csv')
            target.write_text('changed', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'hash verification'):
                exports.export_all(workflow, output, figures=False, log_fn=lambda *_: None)
            self.assertEqual(target.read_text(), 'changed')

    def test_output_lock_is_released_and_reacquired(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'output'
            with exports._exclusive_export(output):
                with self.assertRaisesRegex(RuntimeError, 'Another export'):
                    with exports._exclusive_export(output):
                        self.fail('Concurrent output lock acquired')
            with exports._exclusive_export(output):
                pass

    def test_changed_input_identity_rejects_existing_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exports.write_json(root / 'EXPORT_COMPLETE.json', {'source_identity_sha256': 'old', 'files_sha256': {}})
            with self.assertRaisesRegex(ValueError, 'different inputs'):
                exports._existing_result(root, {'sha256': 'new'})

    def test_incomplete_output_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'unfinished.csv').write_text('x\n1\n')
            with self.assertRaisesRegex(FileExistsError, 'incomplete'):
                exports._existing_result(root, {'sha256': 'new'})

    def test_numerical_export_import_has_no_plotting_dependency(self):
        script = '''import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in ('matplotlib', 'PIL', 'numpy'):
   raise AssertionError('unexpected dependency: ' + fullname)
sys.meta_path.insert(0, Block())
from publication.export_v2 import export_all, make_plot_tables, summary_workbook_payload
print('PASS')
'''
        proc = subprocess.run([sys.executable, '-B', '-c', script], cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn('PASS', proc.stdout)


if __name__ == '__main__':
    unittest.main()
