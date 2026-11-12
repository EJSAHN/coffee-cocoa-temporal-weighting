
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import unittest
from pathlib import Path
import sys
from coffee_cocoa_core.panel_analysis.features import make_panel,select_data,specifications

class FeatureTests(unittest.TestCase):
    def row(self,**kw):
        r=dict(territory='1111111',crop='coffee_total',state='11',year=2010,A0=200,Q_tonnes=0,
               Iha=0,Hha=0,pilot=False,source_discordant=False)
        r.update(kw);return r
    def climate(self):
        return {('1111111',y):dict(D_mean=y-2000,Tz_mean=(y-2000)*.1) for y in range(2005,2012)}
    def spec(self,variant='primary',outcome='production_fixed_ha'):
        return next(s for s in specifications() if s['crop']=='coffee_total' and s['variant']==variant and s['outcome']==outcome)
    def test_zero_not_missing_and_zero_area_not_yield_zero(self):
        p=make_panel([self.row()],self.climate());d=select_data(p,self.spec())
        self.assertEqual(d['y'].tolist(),[0.0]);self.assertIsNone(p[0]['yield_kg_harvested_ha'])
        self.assertIsNone(p[0]['harvest_fraction']);self.assertEqual(p[0]['unharvested_area_fixed_ha'],0)
    def test_nonoverlap_and_alternate_window(self):
        p=make_panel([self.row(Q_tonnes=10)],self.climate())
        a=select_data(p,self.spec());b=select_data(p,self.spec('preceding_calendar_window'))
        self.assertEqual(a['X'][0,:3].tolist(),[10,9,8]);self.assertEqual(a['X'][0,6],85)
        self.assertEqual(b['X'][0,:3].tolist(),[9,8,7]);self.assertEqual(b['X'][0,6],67.5)
        self.assertEqual(a['y'].tolist(),[50]);self.assertEqual(b['y'].tolist(),[50])
    def test_missing_area_does_not_remove_primary_production(self):
        p=make_panel([self.row(Iha=None,Hha=None,Q_tonnes=5)],self.climate())
        self.assertEqual(len(select_data(p,self.spec())['y']),1)
        self.assertEqual(len(select_data(p,self.spec(outcome='unharvested_area_fixed_ha'))['y']),0)
        self.assertEqual(len(select_data(p,self.spec('common_component_rows'))['y']),0)
    def test_discordance_alternative_never_mutates_primary(self):
        original=self.row(source_discordant=True)
        p=make_panel([original],self.climate())
        self.assertEqual(len(select_data(p,self.spec('discordant_cells_as_missing'))['y']),0)
        self.assertEqual(len(select_data(p,self.spec())['y']),1)
        self.assertEqual(original['Q_tonnes'],0);self.assertEqual(p[0]['Q_tonnes'],0)
    def test_negative_area_difference_retained_flagged(self):
        p=make_panel([self.row(Iha=10,Hha=12)],self.climate())
        self.assertTrue(p[0]['harvested_exceeds_intended'])
        self.assertAlmostEqual(p[0]['unharvested_area_fixed_ha'],-.01)
    def test_fixed_menu_and_local_parity(self):
        specs=specifications();self.assertEqual(len(specs),25)
        self.assertEqual(len({s['model_id'] for s in specs}),25)
        self.assertTrue(self.spec()['unit_parity']);self.assertFalse(self.spec('without_local_biennial_term')['unit_parity'])
    def test_duplicate_and_nonpositive_baseline_refused(self):
        with self.assertRaises(ValueError):make_panel([self.row(),self.row()],self.climate())
        with self.assertRaises(ValueError):make_panel([self.row(A0=0)],self.climate())

if __name__=='__main__':unittest.main()
