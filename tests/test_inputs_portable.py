"""Portable input preparation tests with explicitly fictional source data."""
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coffee_cocoa_core import data_import as bridge

def make_cohort(town='1199999', crop='coffee_total'):
    base = dict(territory=town, state=town[:2], crop=crop,
                municipality_name_2025='Fictional unit', baseline_area_ha_2002_2006='200')
    registry = {**base, 'target_frame_included': 'True', 'pilot_previously_explored': 'False'}
    return bridge.cohort_records([base], [registry])

def production_rows(town='1199999', crop='coffee_total'):
    rows = []
    for year in bridge.YEARS:
        rows.append(dict(territory=town, state=town[:2], crop=crop, year=str(year),
                         production_t='120', production_t_raw='120',
                         intended_harvest_ha='200', intended_harvest_ha_raw='200',
                         harvested_ha='190', harvested_ha_raw='190'))
    return rows

def write_gz_csv(path, rows, fields=None):
    with gzip.open(path, 'wt', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)

def climate_rows(town='1199999'):
    rows = []
    for year in bridge.CLIMATE_YEARS:
        for month in range(1, 13):
            b = (year - 1995.5) * month / 12
            # Closed-form z for the linear 1981..2010 reference, independently
            # of bridge.standardize_monthly and provided legacy z arithmetic.
            z = (year - 1995.5) / math.sqrt(77.5)
            row = dict(territory=town, month=f'{year:04d}-{month:02d}',
                       ppt_mm=str(200 + b), pet_mm='200',
                       tmax_monthly_mean_degC=str(20 + month + .02 * (year - 1980)),
                       balance_mm=str(b), balance_z_3m='', balance_z_6m='', balance_z_12m='',
                       tmax_calendar_month_z=str(z), grid_lat='-10.0', grid_lon='-50.0')
            rows.append(row)
    return rows


def fictional_raw(root):
    root.mkdir()
    prod = production_rows()
    prod[5]['production_t'], prod[5]['production_t_raw'] = '0', '-'
    prod[6]['production_t'], prod[6]['production_t_raw'] = '', '...'
    prod = [r for r in prod if r['year'] != '2009']
    write_gz_csv(root / 'production_source.csv.gz', prod)
    write_gz_csv(root / 'climate_monthly.csv.gz', climate_rows(), bridge.MONTH_COLUMNS)
    cohort = dict(territory='1199999', state='11', crop='coffee_total',
                  municipality_name_2025='Fictional', baseline_area_ha_2002_2006='200')
    registry = dict(cohort, target_frame_included='True', pilot_previously_explored='False')
    for name, rows in [('baseline_cohort.csv', [cohort]), ('target_sample_registry.csv', [registry]),
                       ('geography_climate_audit.csv', [dict(territory='1199999', grid_lat='-10', grid_lon='-50', status='COMPLETE')])]:
        bridge._write_table(root / name, rows)
    (root / 'climate_source_metadata.json').write_text('{"dataset":"FICTIONAL"}\n')
    (root / 'legacy').mkdir()
    # These tests exercise data preparation only; exact legacy design content
    # has its own independent 16-design/event-window test suite.
    for name in bridge.RAW_FILES:
        if name.startswith('legacy/'):
            bridge._write_table(root / name, [dict(test_fixture='NO_REAL_LEGACY_DATA')])
    hashes = {name: bridge.sha256(root / name) for name in bridge.RAW_FILES}
    bridge._dump(root / 'source_manifest.json', dict(sha256=hashes))


class PortableInputTests(unittest.TestCase):
    def test_full_prepare_load_retains_zero_missing_and_absent(self):
        with tempfile.TemporaryDirectory() as td:
            raw, out = Path(td) / 'raw', Path(td) / 'prepared'
            fictional_raw(raw)
            bridge.prepare_inputs(raw, out, log=lambda _: None)
            result = bridge.load_inputs(out, log=lambda _: None)
            self.assertEqual(len(result['panel']), 24)
            rows = {r['year']: r for r in result['panel']}
            self.assertEqual(rows[2007]['Q_tonnes'], 0)
            self.assertEqual(rows[2007]['Q_tonnes_raw'], '-')
            self.assertIsNone(rows[2008]['Q_tonnes'])
            self.assertEqual(rows[2008]['Q_tonnes_raw'], '...')
            self.assertTrue(rows[2008]['source_row_present'])
            self.assertFalse(rows[2009]['source_row_present'])
            self.assertFalse(rows[2010]['pilot'])
            self.assertEqual(rows[2010]['production_fixed_ha'], 600)
            self.assertEqual(rows[2010]['D1'], result['climate_annual']['1199999', 2009]['D_mean'])
            self.assertFalse(result['summary']['requires_historical_report_archive'])

    def test_relocatable_without_raw_sources(self):
        with tempfile.TemporaryDirectory() as td:
            raw, out = Path(td) / 'raw', Path(td) / 'prepared'
            fictional_raw(raw)
            bridge.prepare_inputs(raw, out, log=lambda _: None)
            new = Path(td) / 'relocated'
            out.rename(new)
            raw.rename(Path(td) / 'unavailable_raw')
            self.assertEqual(bridge.load_inputs(new, log=lambda _: None)['summary']['counts']['annual_rows'], 24)
            text = (new / 'provenance.json').read_text()
            self.assertNotIn(td, text)

    def test_deterministic_prepared_files(self):
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td) / 'raw'
            fictional_raw(raw)
            first, second = Path(td) / 'a', Path(td) / 'b'
            bridge.prepare_inputs(raw, first, log=lambda _: None)
            bridge.prepare_inputs(raw, second, log=lambda _: None)
            self.assertEqual(bridge.sha256(first / 'input_manifest.json'), bridge.sha256(second / 'input_manifest.json'))

    def test_corruption_and_overwrite_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            raw, out = Path(td) / 'raw', Path(td) / 'prepared'
            fictional_raw(raw)
            bridge.prepare_inputs(raw, out, log=lambda _: None)
            with self.assertRaisesRegex(bridge.BridgeError, 'already exists'):
                bridge.prepare_inputs(raw, out, log=lambda _: None)
            with (out / 'annual_panel.csv.gz').open('ab') as stream:
                stream.write(b'corrupt')
            with self.assertRaisesRegex(bridge.BridgeError, 'SHA256 mismatch'):
                bridge.load_inputs(out, log=lambda _: None)

    def test_manifest_hash_does_not_substitute_for_arithmetic(self):
        with tempfile.TemporaryDirectory() as td:
            raw, out = Path(td) / 'raw', Path(td) / 'prepared'
            fictional_raw(raw)
            bridge.prepare_inputs(raw, out, log=lambda _: None)
            rows = bridge._read_typed_annual(out / 'annual_panel.csv.gz')
            rows[0]['production_fixed_ha'] += 1
            bridge._write_table(out / 'annual_panel.csv.gz', rows)
            manifest = json.loads((out / 'input_manifest.json').read_text())
            manifest['sha256']['annual_panel.csv.gz'] = bridge.sha256(out / 'annual_panel.csv.gz')
            bridge._dump(out / 'input_manifest.json', manifest)
            with self.assertRaisesRegex(bridge.BridgeError, 'arithmetic mismatch'):
                bridge.load_inputs(out, log=lambda _: None)


if __name__ == '__main__':
    unittest.main()
