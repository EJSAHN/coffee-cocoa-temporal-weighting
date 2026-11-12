"""End-to-end software smoke using generated fictional raw source files only.

This fixture exercises the real preparation, typed loaders, monthly detector,
fixed effects, exact event weights and benchmark. No loader or estimator is
mocked. Two Monte Carlo replicates verify wiring, not operating characteristics.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coffee_cocoa_core import data_import, event_designs, event_detection, pipeline
from coffee_cocoa_core.benchmark.synthetic_dgp import build_design
from coffee_cocoa_core.benchmark.legacy_events import standardize
from coffee_cocoa_core.io import write_table, write_json, sha256
from coffee_cocoa_core.panel_analysis.tests.fixture_panel import fictional_inputs


def _record_design(tables, did, population, crop, kind, events, weight_kind):
    weights, required, counts = event_designs.event_weights(events, weight_kind)
    semantic = [[t, y, w.numerator, w.denominator] for (t, y), w in sorted(weights.items())]
    tables['registry'].append(dict(
        design_id=did, population=population, crop=crop, kind=kind,
        municipalities=len(counts), short_pairs=sum(c['short'] for c in counts.values()),
        long_pairs=sum(c['long'] for c in counts.values()), annual_weight_rows=len(weights),
        coefficient_fingerprint=event_designs._fingerprint(semantic),
        required_outcomes_fingerprint=event_designs._fingerprint(sorted([list(k) for k in required])),
        coordinate_basis='FICTIONAL_SOFTWARE_FIXTURE'))
    for (territory, year), weight in sorted(weights.items()):
        tables['weights'].append(dict(design_id=did, territory=territory, crop=crop, year=year,
            numerator=weight.numerator, denominator=weight.denominator,
            coefficient=float(weight), exact_zero=weight == 0))


def fictional_raw(directory):
    """Write 24 fictional municipalities, two crops and all 552 climate months."""
    root = Path(directory)
    root.mkdir(parents=True)
    source = fictional_inputs(n_units=24)['panel']
    generated = build_design(20261005, 0, n_units=24)
    geography, series, climate_rows = [], [], []
    for j in range(24):
        old, town = str(9000000 + j), str(90 + j // 4) + f'{j:05d}'
        for row in source:
            if row['territory'] == old:
                row['territory'], row['state'] = town, town[:2]
        latitude, longitude = -10.0 + j * .01, -50.0 + j * .01
        geography.append(dict(territory=town, grid_lat=latitude, grid_lon=longitude, status='COMPLETE'))
        balance = generated['monthly_balance'][j].tolist()
        temperature = generated['monthly_temperature'][j].tolist()
        tz = data_import.standardize_monthly(temperature)
        z = {scale: standardize(balance, scale) for scale in (3, 6, 12)}
        series.append((town, z[6], tz))
        for k, month in enumerate(data_import.MONTHS):
            # Construct nonnegative ppt/pet with exactly the generated balance.
            pet = 500.0
            climate_rows.append(dict(territory=town, month=month,
                ppt_mm=pet + balance[k], pet_mm=pet, balance_mm=balance[k],
                tmax_monthly_mean_degC=temperature[k], tmax_calendar_month_z=tz[k],
                balance_z_3m=z[3][k], balance_z_6m=z[6][k], balance_z_12m=z[12][k],
                grid_lat=latitude, grid_lon=longitude))
    raw_production, cohorts, target = [], {}, {}
    for row in source:
        key = row['territory'], row['crop']
        cohorts[key] = dict(territory=row['territory'], crop=row['crop'], state=row['state'],
            municipality_name_2025='Fictional software unit', baseline_area_ha_2002_2006=row['A0'])
        target[key] = dict(cohorts[key], target_frame_included=True, pilot_previously_explored=row['pilot'])
        record = {field: row[field] for field in ('territory', 'crop', 'state', 'year')}
        for original, field in data_import.RAW_MAP.items():
            record[field] = row[original]
            record[field + '_raw'] = row[original]
        raw_production.append(record)
    write_table(root / 'production_source.csv.gz', raw_production)
    write_table(root / 'climate_monthly.csv.gz', climate_rows, data_import.MONTH_COLUMNS)
    write_table(root / 'baseline_cohort.csv', list(cohorts.values()))
    write_table(root / 'target_sample_registry.csv', list(target.values()))
    write_table(root / 'geography_climate_audit.csv', geography)
    write_json(root / 'climate_source_metadata.json', dict(dataset='FICTIONAL_SOFTWARE_FIXTURE', real_observations=False))
    detected = event_detection.reconstruct_from_series(source, series)
    groups = defaultdict(list)
    for row in detected['membership_rows']:
        groups[row['design_id']].append(row)
    tables = dict(registry=[], events=[], units=[], weights=[])
    coordinates = {row['territory']: row for row in geography}
    for population in ('full_target', 'exclude_43_pilot_units'):
        for crop in ('coffee_total', 'cacao'):
            anchored = {}
            for anchor in ('confirmation', 'onset'):
                did = event_designs._original_id(population, crop, anchor)
                events = groups[did]
                if not events:
                    raise AssertionError('Fixed fictional source unexpectedly has no overlap: ' + did)
                anchored[anchor] = events
                tables['events'].extend(events)
                counts = defaultdict(Counter)
                for event in events:
                    counts[event['territory']][event['contrast']] += 1
                for town, count in sorted(counts.items()):
                    tables['units'].append(dict(design_id=did, population=population, crop=crop,
                        anchor=anchor, territory=town, state=town[:2], n_short=count['short'],
                        n_long=count['long'], grid_lat=coordinates[town]['grid_lat'], grid_lon=coordinates[town]['grid_lon']))
                _record_design(tables, did, population, crop, 'original_' + anchor, events, 'original')
            onset = {(r['territory'], r['pair_uid']): r for r in anchored['onset']}
            matched = []
            for record in anchored['confirmation']:
                first = onset[record['territory'], record['pair_uid']]
                matched.append(dict(record, onset_post_first_year=first['post_first_year'],
                    onset_post_last_year=first['post_last_year'],
                    confirmation_post_first_year=record['post_first_year'],
                    confirmation_post_last_year=record['post_last_year']))
            for kind in ('matched_onset', 'matched_anchor_change'):
                did = f'stage3m__{kind}__{population}__{crop}'
                _record_design(tables, did, population, crop, kind, matched, kind)
    for key, filename in event_designs.FILES.items():
        write_table(root / filename, tables[key])
    write_json(root / 'source_manifest.json', dict(sha256={name: sha256(root / name) for name in data_import.RAW_FILES}))
    return dict(municipalities=24, crop_units=48, annual_rows=1152, climate_monthly_rows=13248,
                scientific_data=False, generated_from_fixed_seed=20261005)


class PublicPipelineSmokeTests(unittest.TestCase):
    def test_raw_sources_to_all_analysis_outputs(self):
        with tempfile.TemporaryDirectory(prefix='coffee_cocoa_fictional_') as temporary:
            root = Path(temporary)
            fixture = fictional_raw(root / 'raw')
            result = pipeline.run(root / 'raw', root / 'results', replicates=2, log_fn=lambda _: None)
            self.assertEqual(result['status'], 'COMPLETE_MODIFIED_BENCHMARK')
            self.assertEqual(result['input_counts']['annual_rows'], fixture['annual_rows'])
            self.assertEqual(result['panel']['models_planned'], 56)
            self.assertEqual(result['panel']['models_fitted'], 56)
            self.assertEqual(result['events']['complete_designs'], 16)
            self.assertEqual(result['benchmark']['completed_scenarios'], 27)
            self.assertFalse(result['causal_effect_identified'])
            self.assertFalse(result['real_world_interval_coverage_validated'])
            # This hook is required once detector integration is complete.
            detector = root / 'results/event_reconstruction/event_reconstruction_summary.json'
            self.assertTrue(detector.is_file(), 'Public pipeline omitted monthly event reconstruction')
            detector_summary = json.loads(detector.read_text())
            self.assertEqual(detector_summary['original_designs_reproduced'], 8)
            self.assertEqual(detector_summary['municipalities'], 24)
            manifest = json.loads((root / 'results/output_manifest.json').read_text())
            self.assertTrue(manifest['sha256'])
            for name, expected in manifest['sha256'].items():
                self.assertEqual(sha256(root / 'results' / name), expected)
            self.assertEqual(json.loads((root / 'results/analysis_summary.json').read_text())['status'], result['status'])


if __name__ == '__main__':
    unittest.main()
