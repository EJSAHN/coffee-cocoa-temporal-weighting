import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coffee_cocoa_core.benchmark import legacy_events as legacy


def month(year, value=1):
    return (year - 1980) * 12 + value - 1


def climate_fixture():
    z = np.zeros(552)
    for year in (2004, 2005, 2010, 2011, 2017, 2024):
        start = month(year)
        z[start:start + 3] = -2
    return z


def direct_contrast(values, pairs, anchor):
    eligible = 'primary_2y_computable' if anchor == 'confirmation2y' else 'onset_2y_computable'
    by_unit = {}
    for row in pairs:
        if not row[eligible]:
            continue
        unit = row['territory']
        start = int(row['current_start_month'][:4])
        year = row['confirmation_year'] if anchor == 'confirmation2y' else start
        baseline = sum(values[unit][y - 2002] for y in range(start - 4, start)) / 4
        post = (values[unit][year + 1 - 2002] + values[unit][year + 2 - 2002]) / 2
        by_unit.setdefault(unit, {}).setdefault(row['contrast_group'], []).append(post - baseline)
    differences = [sum(group['short']) / len(group['short']) - sum(group['long']) / len(group['long'])
                   for group in by_unit.values() if set(group) == {'short', 'long'}]
    return sum(differences) / len(differences) if differences else None


class LegacyEventTests(unittest.TestCase):
    def test_reference_is_calendar_month_sample_sd(self):
        x = np.arange(552, dtype=float)
        balance = np.sin(x / 7.3) * 20 + (x % 12) * 2 + np.cos(x / 19) * 8
        z = legacy.standardize(list(balance), 6)
        self.assertEqual(z[:5], [None] * 5)
        for calendar_month in range(12):
            values = [z[i] for i in range(552)
                      if 1981 <= legacy.MONTHS[i] // 12 <= 2010 and i % 12 == calendar_month]
            self.assertAlmostEqual(np.mean(values), 0, places=12)
            self.assertAlmostEqual(np.std(values, ddof=1), 1, places=12)

    def test_reference_failure_is_explicit(self):
        with self.assertRaises(legacy.AuditError):
            legacy.standardize([0.] * 552, 6)
        values = [np.sin(i / 7) for i in range(552)]
        values[100] = None
        with self.assertRaises(legacy.AuditError):
            legacy.standardize(values, 6)

    def test_threshold_and_confirmation_dates(self):
        seq = legacy.detect([0, 0, 0, -1.5, -1.5, -.5, 0, 0, 0], offset=0)
        self.assertEqual(len(seq), 1)
        self.assertEqual((seq[0]['start_index'], seq[0]['end_index'], seq[0]['recovery_confirmed_index']),
                         (3, 5, 8))
        self.assertFalse(seq[0]['left_censored'])
        self.assertFalse(seq[0]['right_censored'])
        self.assertFalse(legacy.detect([0, 0, 0, -2, 0, 0, 0], offset=0))

    def test_missing_censors_and_does_not_imply_recovery(self):
        seq = legacy.detect([0, 0, 0, -2, -2, None, -2, -2, 0, 0, 0], offset=0)
        self.assertTrue(seq[0]['right_censored'])
        self.assertEqual(seq[0]['end_reason'], 'missing_climate')
        self.assertTrue(seq[1]['left_censored'])

    def test_gap_excludes_both_event_boundaries(self):
        out = legacy.build_weights(climate_fixture()[None, :], ['A'])
        pair = next(p for p in out['pairs'] if p['current_start_month'] == '2010-01')
        self.assertEqual(pair['gap_months'], 57)
        self.assertEqual(pair['contrast_group'], 'long')
        self.assertTrue(pair['primary_2y_computable'])
        short = next(p for p in out['pairs'] if p['current_start_month'] == '2011-01')
        self.assertEqual(short['gap_months'], 9)
        self.assertEqual(short['contrast_group'], 'short')

    def test_direct_nested_average_matches_exact_weights(self):
        z1, z2 = climate_fixture(), climate_fixture()
        # One extra short event changes the within-unit event denominator.
        z2[month(2018):month(2018) + 3] = -2
        out = legacy.build_weights(np.vstack([z1, z2]), ['A', 'B'])
        values = {'A': np.arange(24) ** 2 + 17., 'B': np.sin(np.arange(24)) * 400 + 800}
        vector = np.r_[values['A'], values['B']]
        for anchor, weights in out['weights'].items():
            self.assertAlmostEqual(float(weights @ vector), direct_contrast(values, out['pairs'], anchor), places=11)
            self.assertAlmostEqual(sum(weights), 0, places=14)
            self.assertEqual(out['diagnostics']['anchors'][anchor]['eligible_units'], 2)

    def test_onset_and_confirmation_have_independent_overlap(self):
        z = np.zeros(552)
        for year in (2004, 2005, 2010):
            z[month(year):month(year) + 3] = -2
        z[month(2011):] = -2  # Eligible short onset, unresolved forever.
        out = legacy.build_weights(z[None, :], ['A'])
        self.assertTrue(out['diagnostics']['anchors']['onset2y']['estimable'])
        self.assertFalse(out['diagnostics']['anchors']['confirmation2y']['estimable'])
        self.assertTrue(np.all(out['weights']['confirmation2y'] == 0))

    def test_followup_boundaries_and_onset_analysis_start(self):
        out = legacy.build_weights(climate_fixture()[None, :], ['A'])
        early = next(p for p in out['pairs'] if p['current_start_month'] == '2005-01')
        late = next(p for p in out['pairs'] if p['current_start_month'] == '2024-01')
        self.assertFalse(early['base_pair_eligible'])
        self.assertFalse(late['primary_2y_computable'])
        self.assertFalse(late['onset_2y_computable'])

    def test_confirmation_uses_year_after_third_exit_month(self):
        z = np.zeros(552)
        z[month(2005):month(2005) + 3] = -2
        z[month(2010, 11):month(2011, 1) + 1] = -2
        z[month(2012):month(2012) + 3] = -2
        out = legacy.build_weights(z[None, :], ['A'])
        pair = next(p for p in out['pairs'] if p['current_start_month'] == '2010-11')
        self.assertEqual(pair['confirmation_month'], '2011-04')
        self.assertEqual(pair['confirmation_year'], 2011)
        # Both groups present; differing anchor years must change the vector.
        self.assertTrue(out['diagnostics']['anchors']['confirmation2y']['estimable'])
        self.assertFalse(np.array_equal(out['weights']['onset2y'], out['weights']['confirmation2y']))

    def test_no_overlap_reports_nonestimable_without_seed_search(self):
        out = legacy.build_weights(np.zeros((2, 552)), ['A', 'B'])
        for anchor, weights in out['weights'].items():
            self.assertEqual(weights.shape, (48,))
            self.assertTrue(np.all(weights == 0))
            self.assertEqual(out['diagnostics']['anchors'][anchor]['status'],
                             'NONESTIMABLE_NO_WITHIN_UNIT_OVERLAP')

    def test_reject_malformed_inputs(self):
        for z, units, years in [
            (np.zeros((2, 552)), ['A', 'A'], np.arange(2002, 2026)),
            (np.zeros((1, 551)), ['A'], np.arange(2002, 2026)),
            (np.zeros((1, 552)), ['A'], np.arange(2003, 2026)),
            (np.full((1, 552), np.inf), ['A'], np.arange(2002, 2026)),
        ]:
            with self.assertRaises(ValueError):
                legacy.build_weights(z, units, years)

    def test_no_further_drought_exclusion(self):
        out = legacy.build_weights(climate_fixture()[None, :], ['A'])
        pair = next(p for p in out['pairs'] if p['current_start_month'] == '2010-01')
        self.assertEqual(pair['subsequent_onset_month'], '2011-01')
        self.assertTrue(pair['primary_2y_computable'])


if __name__ == '__main__':
    unittest.main()
