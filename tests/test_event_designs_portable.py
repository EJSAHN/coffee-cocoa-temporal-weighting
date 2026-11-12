"""Portable legacy design tests use only fictional event tables."""
import csv
import gzip
import hashlib
import io
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from unittest import TestCase, main
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coffee_cocoa_core import event_designs as bridge

def fictional_tables():
    tables = dict(registry=[], events=[], units=[], weights=[])
    for pop in ('full_target', 'exclude_43_pilot_units'):
        for crop in ('coffee_total', 'cacao'):
            territories = ['T0', 'T1'] if pop == 'full_target' else ['T1']
            by_anchor = {}
            for anchor in ('confirmation', 'onset'):
                did = bridge._original_id(pop, crop, anchor)
                events = []
                for territory in territories:
                    specs = [(2010, 2011, 'short'), (2017, 2017, 'long')]
                    if anchor == 'onset':
                        specs += [(2021, 2022, 'short')]
                    counts = Counter(x[2] for x in specs)
                    tables['units'].append(dict(
                        design_id=did, population=pop, crop=crop, anchor=anchor,
                        territory=territory, state='TEST', grid_lon='-50', grid_lat='-10',
                        n_short=str(counts['short']), n_long=str(counts['long'])))
                    for onset, confirmation, contrast in specs:
                        year = confirmation if anchor == 'confirmation' else onset
                        events.append(dict(
                            design_id=did, population=pop, crop=crop, anchor=anchor,
                            territory=territory, state='TEST', pair_uid=f'{territory}_{onset}',
                            contrast=contrast, current_start_month=f'{onset}-04',
                            benchmark_first_year=str(onset - 4), benchmark_last_year=str(onset - 1),
                            anchor_year=str(year), post_first_year=str(year + 1), post_last_year=str(year + 2),
                            n_events_in_unit_group=str(counts[contrast]), municipalities_in_contrast=str(len(territories)),
                            unsigned_event_weight=str(1 / (len(territories) * counts[contrast]))))
                tables['events'].extend(events)
                by_anchor[anchor] = events
                _add_design(tables, did, pop, crop, 'original_' + anchor, events, 'original')
            matched = []
            for row in by_anchor['confirmation']:
                onset = next(r for r in by_anchor['onset'] if r['territory'] == row['territory'] and r['pair_uid'] == row['pair_uid'])
                matched.append(dict(row, onset_post_first_year=onset['post_first_year'],
                                    onset_post_last_year=onset['post_last_year'],
                                    confirmation_post_first_year=row['post_first_year'],
                                    confirmation_post_last_year=row['post_last_year']))
            for kind in ('matched_onset', 'matched_anchor_change'):
                _add_design(tables, f'stage3m__{kind}__{pop}__{crop}', pop, crop, kind, matched, kind)
    return tables

def _add_design(tables, did, pop, crop, kind, events, weight_kind):
    weights, required, counts = bridge.event_weights(events, weight_kind)
    semantic = [[t, y, w.numerator, w.denominator] for (t, y), w in sorted(weights.items())]
    tables['registry'].append(dict(
        design_id=did, population=pop, crop=crop, kind=kind, municipalities=str(len(counts)),
        short_pairs=str(sum(c['short'] for c in counts.values())),
        long_pairs=str(sum(c['long'] for c in counts.values())), annual_weight_rows=str(len(weights)),
        coefficient_fingerprint=bridge._fingerprint(semantic),
        required_outcomes_fingerprint=bridge._fingerprint(sorted([list(k) for k in required])),
        coordinate_basis='FICTIONAL'))
    for (t, y), weight in sorted(weights.items()):
        tables['weights'].append(dict(design_id=did, territory=t, crop=crop, year=str(y),
                                      numerator=str(weight.numerator), denominator=str(weight.denominator),
                                      coefficient=str(float(weight)), exact_zero=str(weight == 0)))

class PortableEventDesignTests(TestCase):
    def test_all_sixteen_designs_and_exact_anchor_identity(self):
        designs = bridge._tables_to_designs(fictional_tables())
        self.assertEqual(len(designs), 16)
        self.assertEqual({d['kind'] for d in designs},
                         {'original_confirmation', 'original_onset', 'matched_onset', 'matched_anchor_change'})
        # For T0, one short and one long event, two municipalities: short
        # benchmark year carries -1/(2*4), its two post years each +1/(2*2).
        design = next(d for d in designs if d['design_id'] == 'm6_original__confirmation2y__full_target__coffee_total')
        self.assertEqual(design['weights_exact']['T0', 2006], -bridge.Fraction(1, 8))
        self.assertEqual(design['weights_exact']['T0', 2012], bridge.Fraction(1, 4))
        for d in designs:
            self.assertEqual(sum(d['weights_exact'].values()), 0)

    def test_cancelled_benchmark_is_still_required(self):
        designs = bridge._tables_to_designs(fictional_tables())
        design = next(d for d in designs if d['design_id'] == 'stage3m__matched_anchor_change__full_target__coffee_total')
        self.assertIn(('T0', 2006), design['required_outcomes'])
        self.assertNotIn(('T0', 2006), design['weights_exact'])
        self.assertTrue(any(w == 0 for w in design['weights_exact'].values()))

    def test_exact_zero_row_cannot_be_dropped(self):
        tables = fictional_tables()
        index = next(i for i, row in enumerate(tables['weights']) if row['exact_zero'] == 'True')
        tables['weights'].pop(index)
        with self.assertRaisesRegex(bridge.BridgeError, 'Event-derived exact annual'):
            bridge._tables_to_designs(tables)

    def test_changed_coefficient_is_rejected(self):
        tables = fictional_tables()
        tables['weights'][0]['numerator'] = '1'
        tables['weights'][0]['coefficient'] = str(abs(float(tables['weights'][0]['coefficient'])))
        with self.assertRaises(bridge.BridgeError):
            bridge._tables_to_designs(tables)

    def test_missing_or_duplicate_design_is_rejected(self):
        for duplicate in (False, True):
            tables = fictional_tables()
            tables['registry'].pop()
            if duplicate:
                tables['registry'].append(dict(tables['registry'][0]))
            with self.assertRaises(bridge.BridgeError):
                bridge._tables_to_designs(tables)

    def test_unit_state_and_event_counts_checked(self):
        for field, value in [('state', 'WRONG'), ('n_short', '99')]:
            tables = fictional_tables()
            tables['units'][0][field] = value
            with self.assertRaises(bridge.BridgeError):
                bridge._tables_to_designs(tables)

    def test_outcome_free_window_fingerprint_checked(self):
        tables = fictional_tables()
        tables['registry'][0]['required_outcomes_fingerprint'] = '0' * 64
        with self.assertRaisesRegex(bridge.BridgeError, 'Required-outcome fingerprint'):
            bridge._tables_to_designs(tables)

    def test_real_outcome_column_not_required(self):
        tables = fictional_tables()
        designs = bridge._tables_to_designs(tables)
        self.assertTrue(all('Y' not in d and 'expected_theta' not in d for d in designs))
        self.assertTrue(all('benchmark_mean' not in e for d in designs for e in d['events']))

    def test_portable_files_need_no_report_archive(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            hashes = {}
            for label, rows in fictional_tables().items():
                path = root / bridge.FILES[label]
                path.parent.mkdir(parents=True, exist_ok=True)
                op = gzip.open if path.suffix == '.gz' else open
                with op(path, 'wt', encoding='utf-8', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                hashes[bridge.FILES[label]] = hashlib.sha256(path.read_bytes()).hexdigest()
            (root / 'input_manifest.json').write_text(json.dumps(dict(sha256=hashes)))
            result = bridge.load_event_designs(root)
            self.assertEqual(len(result['designs']), 16)
            self.assertFalse(result['provenance']['requires_historical_report_archive'])


if __name__ == '__main__':
    main()
