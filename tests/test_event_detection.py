"""Event reconstruction and missingness tests use fictional climate and outcomes."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coffee_cocoa_core import event_detection as ed
from coffee_cocoa_core.benchmark.legacy_events import build_weights


def month(year):
    return (year - 1980) * 12


def z_fixture():
    z = [0.] * 552
    for year in (2004, 2005, 2010, 2011, 2017, 2024):
        z[month(year):month(year) + 3] = [-2.] * 3
    return z


def panel_fixture():
    return [dict(territory=town, state='11', crop=crop, year=year,
                 pilot=town == '1199998', production_fixed_ha=0 if year == 2012 else 100)
            for crop in ('coffee_total', 'cacao') for town in ('1199998', '1199999')
            for year in range(2002, 2026)]


def source_series(z=None):
    return [(town, list(z_fixture() if z is None else z), [0.5] * 552)
            for town in ('1199998', '1199999')]


def frozen_from_rebuilt(result):
    designs = []
    for row in result['support_rows']:
        did = row['design_id']
        anchor = 'confirmation' if '__confirmation2y__' in did else 'onset'
        designs.append(dict(design_id=did, kind='original_' + anchor,
                            events=[{k: str(v) for k, v in member.items()}
                                    for member in result['membership_rows'] if member['design_id'] == did],
                            weights_exact=dict(result['weights'][did])))
    return designs


class EventReconstructionTests(unittest.TestCase):
    def test_fully_observed_source_matches_exact_benchmark_weights(self):
        result = ed.reconstruct_from_series(panel_fixture(), source_series())
        for pop, towns in [('full_target', ['1199998', '1199999']),
                           ('exclude_43_pilot_units', ['1199999'])]:
            benchmark = build_weights(np.array([z_fixture()] * len(towns)), towns)
            for crop in ('coffee_total', 'cacao'):
                for anchor in ('confirmation', 'onset'):
                    did = f'm6_original__{anchor}2y__{pop}__{crop}'
                    actual = np.zeros((len(towns), 24))
                    for (town, year), weight in result['weights'][did].items():
                        actual[towns.index(town), year - 2002] = float(weight)
                    np.testing.assert_array_equal(actual.ravel(), benchmark['weights'][anchor + '2y'])
        self.assertEqual(len(ed.compare_frozen(result, frozen_from_rebuilt(result))), 8)

    def test_zero_is_eligible_missing_is_not_and_crop_identity_separate(self):
        panel = panel_fixture()
        original = ed.reconstruct_from_series(panel, source_series())
        for row in panel:
            if row['territory'] == '1199999' and row['crop'] == 'cacao' and row['year'] == 2012:
                row['production_fixed_ha'] = None
        changed = ed.reconstruct_from_series(panel, source_series())
        for result, expected in [(original, 2), (changed, 1)]:
            row = next(r for r in result['support_rows']
                       if r['design_id'] == 'm6_original__confirmation2y__full_target__cacao')
            self.assertEqual(row['municipalities'], expected)
        row = next(r for r in changed['support_rows']
                   if r['design_id'] == 'm6_original__confirmation2y__full_target__coffee_total')
        self.assertEqual(row['municipalities'], 2)

    def test_unresolved_current_event_remains_in_onset_companion(self):
        z = z_fixture()
        z[month(2011):] = [-2.] * (552 - month(2011))
        result = ed.reconstruct_from_series(panel_fixture(), source_series(z))
        support = {r['design_id']: r for r in result['support_rows']}
        self.assertEqual(support['m6_original__confirmation2y__full_target__cacao']['municipalities'], 0)
        self.assertEqual(support['m6_original__onset2y__full_target__cacao']['municipalities'], 2)

    def test_frozen_membership_mutation_fails(self):
        result = ed.reconstruct_from_series(panel_fixture(), source_series())
        designs = frozen_from_rebuilt(result)
        designs[0]['events'][0]['benchmark_first_year'] = '2000'
        with self.assertRaisesRegex(ValueError, 'identity/window differs'):
            ed.compare_frozen(result, designs)

    def test_duplicate_or_missing_monthly_town_is_rejected(self):
        for series in [source_series()[:1], source_series() + source_series()[:1]]:
            with self.assertRaises(ValueError):
                ed.reconstruct_from_series(panel_fixture(), series)

    def test_calendar_source_rebuild_closes_to_frozen_design(self):
        # Raw balance with fixed fictional pulses; normalization and events
        # are constructed before writing any stored z6 values.
        balance = [float((i * 17) % 13) for i in range(552)]
        for year in (2004, 2005, 2010, 2011, 2017, 2024):
            balance[month(year):month(year) + 3] = [-100.] * 3
        z = ed.standardize(balance, 6)
        rebuilt = ed.reconstruct_from_series(panel_fixture(), source_series(z))
        self.assertTrue(all(r['municipalities'] > 0 for r in rebuilt['support_rows']))
        with tempfile.TemporaryDirectory() as td:
            root, out = Path(td), Path(td) / 'results'
            path = root / 'climate_monthly.csv.gz'
            with gzip.open(path, 'wt', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=ed.MONTH_COLUMNS)
                writer.writeheader()
                for town in ('1199998', '1199999'):
                    for i, date in enumerate(ed.MONTH_LABELS):
                        writer.writerow(dict(territory=town, month=date, ppt_mm=200 + balance[i], pet_mm=200,
                                             tmax_monthly_mean_degC=25, balance_mm=balance[i],
                                             balance_z_3m='', balance_z_6m=z[i], balance_z_12m='',
                                             tmax_calendar_month_z=.5, grid_lat=-10, grid_lon=-50))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            (root / 'input_manifest.json').write_text(json.dumps({'sha256': {'climate_monthly.csv.gz': digest}}))
            summary = ed.run(root, panel_fixture(), frozen_from_rebuilt(rebuilt), out, log=lambda _: None)
            self.assertEqual(summary['original_designs_reproduced'], 8)
            self.assertEqual(summary['max_z6_abs_error'], 0)
            self.assertFalse(summary['production_magnitudes_used_for_selection'])


if __name__ == '__main__':
    unittest.main()
