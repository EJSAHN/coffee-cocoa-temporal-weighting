"""Fictional table/export checks; no empirical model is fit."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local.presentation import collect_workbook_payload, extract_model_tables, figure_tables


class PresentationTests(unittest.TestCase):
    def spec(self, directory):
        path = Path(directory) / 'spec.json'
        path.write_text(json.dumps({'schema_version':'test', 'title':'Test', 'interpretation':[],
            'table_policy':{'workbook_cell_budget':300000,'excluded':'Raw source'},
            'tables':[dict(key='toy',sheet='Toy',family='toy',filename='toy.csv',description='Fictional exact weights.',
                units='dimensionless',placement='workbook',required=True)]}))
        return path

    def test_types_and_complete_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = collect_workbook_payload({'toy':[{'territory':'00123','numerator':'12345678901234567890',
                'year':'2008','coefficient':'0','missing':'','status':'NA','flag':'False'}]},
                spec_path=self.spec(directory))
        table = payload['sheets'][-1]
        row = dict(zip(table['columns'], table['rows'][0]))
        self.assertEqual(row,{'territory':'00123','numerator':'12345678901234567890','year':2008,
            'coefficient':0,'missing':None,'status':'NA','flag':False})
        self.assertTrue(payload['summary']['complete'])

    def test_missing_required_tables_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = self.spec(directory)
            with self.assertRaises(FileNotFoundError):collect_workbook_payload({}, spec_path=spec)
            result = collect_workbook_payload({},spec_path=spec,strict=False)
        self.assertEqual(result['missing_tables'], ['toy'])
        self.assertEqual(result['sheets'][1]['rows'][0][2], 'NOT_AVAILABLE')

    def test_saved_model_values_are_copied_without_recalculation(self):
        with tempfile.TemporaryDirectory() as directory:
            models=Path(directory)/'models'; models.mkdir()
            obj={'status':'COMPLETE','specification':{'model_id':'toy','crop':'cacao'},
                'coefficient_order':['D0','D0_x_history'],'coefficients':{'D0':2,'D0_x_history':3},
                'uncertainty':{'HAC_L2':{'covariance':[[4,1],[1,9]],'intervals':{
                    'D0':{'lower':-7,'upper':11,'standard_error':2},
                    'D0_x_history':{'lower':-8,'upper':14,'standard_error':3}}}}}
            (models/'toy.json').write_text(json.dumps(obj))
            result=extract_model_tables({'panel_models':directory})
        self.assertEqual(result['all_model_coefficients']['rows'][0]['lower'],-7)
        self.assertEqual([r['covariance'] for r in result['all_model_covariances']['rows']],[4,1,1,9])
        self.assertEqual(len(result['all_model_coefficients']['rows']),2)

    def test_figure_adapter_uses_temporal_family_and_keeps_saved_values(self):
        source={'c_current':[{'estimate':'1.23'}],
                'c_contrasts':[{'quantity':'interaction_gamma','estimate':'9'},{'quantity':'other','estimate':'8'}],
                'c_future':[{'estimate':'0'}], 'simulation_panel_summary':[],
                'simulation_event_decomposition':[], 'simulation_event_support':[]}
        figures=figure_tables(source)
        self.assertEqual(figures['integrated_current_response'][0]['estimate'],'1.23')
        self.assertEqual(len(figures['integrated_history_interaction']),1)
        self.assertEqual(figures['integrated_history_interaction'][0]['analysis_family'],'temporal_comparison')
        self.assertNotIn('analysis_family',source['c_current'][0])

if __name__=='__main__': unittest.main()
