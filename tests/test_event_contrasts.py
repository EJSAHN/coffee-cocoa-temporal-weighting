from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from coffee_cocoa_core.event_contrasts import analyse
from fixture_core_temporal import fixture


class EventContrasts(unittest.TestCase):
    def panel(self,formula):
        return [dict(territory=t,crop=c,year=y,production_fixed_ha=formula(t,y),source_discordant=False)
                for t in ('T0','T1') for c in ('coffee_total','cacao') for y in range(2002,2026)]

    def test_constant_and_parity(self):
        result=analyse(self.panel(lambda t,y:400+(25 if t=='T0' else -18)*(-1)**y),fixture())
        self.assertEqual(len(result['summary']),16)
        for row in result['summary']:
            self.assertAlmostEqual(row['estimate'],0,places=11)

    def test_linear_onset(self):
        result=analyse(self.panel(lambda t,y:200+(3 if t=='T0' else 8)*(y-2006)),fixture())
        for row in result['summary']:
            if row['kind'] in ('original_onset','matched_onset'):
                self.assertAlmostEqual(row['estimate'],0,places=11)
        self.assertTrue(any(abs(row['estimate'])>0 for row in result['summary']))

    def test_missing_required_does_not_reweight(self):
        panel=self.panel(lambda t,y:123)
        for row in panel:
            if row['territory']=='T0' and row['year']==2006:row['production_fixed_ha']=None
        result=analyse(panel,fixture())
        full=[r for r in result['summary'] if r['population']=='full_target']
        self.assertTrue(all(r['estimate'] is None for r in full))
        self.assertTrue(all(r['status']=='UNAVAILABLE_FIXED_SUPPORT' for r in full))
        self.assertTrue(all(r['estimate'] is not None for r in result['summary'] if r['population']!='full_target'))

    def test_duplicate_identity_rejected(self):
        panel=self.panel(lambda t,y:123)
        with self.assertRaises(ValueError):analyse(panel+[panel[0]],fixture())


if __name__=='__main__':unittest.main()
