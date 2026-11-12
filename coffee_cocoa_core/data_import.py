"""Canonical, portable input preparation and reading for coffee/cacao analysis.

No historical report archive, absolute project path, network access, saved fit,
figure-making code, or outcome imputation is required by this public module.

One-time migration creates the generic raw directory described by RAW_FILES.
prepare_inputs() independently reconciles baseline production and monthly
climate, then writes a self-contained portable dataset with a SHA256 manifest.
"""
from __future__ import annotations

import collections
import csv
import gzip
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import statistics
import tempfile

from .panel_analysis.features import make_panel

SCHEMA_VERSION = 'coffee_cocoa_portable_v1'
CROPS = ('coffee_total', 'cacao')
RAW_FILES = ('production_source.csv.gz', 'climate_monthly.csv.gz',
             'baseline_cohort.csv', 'target_sample_registry.csv',
             'geography_climate_audit.csv', 'climate_source_metadata.json',
             'legacy/consolidated_annual_weights.csv.gz', 'legacy/design_registry.csv',
             'legacy/design_event_membership.csv.gz', 'legacy/design_unit_registry.csv')
BOOLEAN_FIELDS = ('pilot', 'source_row_present', 'source_discordant', 'area_order_valid',
                  'positive_production_zero_harvest', 'harvested_exceeds_intended')
NUMERIC_FIELDS = ('A0', 'Q_tonnes', 'Iha', 'Hha', 'production_fixed_ha',
                  'intended_area_fixed_ha', 'unharvested_area_fixed_ha',
                  'harvest_fraction', 'yield_kg_harvested_ha',
                  *(f'{kind}{lag}' for kind in ('D', 'T') for lag in range(4)))
PANEL_REQUIRED = ('territory', 'crop', 'state', 'year', 'A0', 'Q_tonnes', 'Iha', 'Hha',
                  'pilot', 'source_discordant', 'Q_tonnes_raw', 'Iha_raw', 'Hha_raw',
                  'source_row_present', 'source_row_number', *NUMERIC_FIELDS[4:])

MONTH_COLUMNS = ['territory', 'month', 'ppt_mm', 'pet_mm', 'tmax_monthly_mean_degC',
                 'balance_mm', 'balance_z_3m', 'balance_z_6m', 'balance_z_12m',
                 'tmax_calendar_month_z', 'grid_lat', 'grid_lon']

YEARS = tuple(range(2002, 2026))

CLIMATE_YEARS = tuple(range(1980, 2026))

MONTHS = tuple(f'{year:04d}-{month:02d}' for year in CLIMATE_YEARS for month in range(1, 13))

RAW_MAP = {'Q_tonnes': 'production_t', 'Iha': 'intended_harvest_ha', 'Hha': 'harvested_ha'}

DISCORDANT_RECORDS = frozenset(
    [('1301902', 'cacao', 2014), ('2922854', 'coffee_total', 2020)] +
    [(town, 'coffee_total', year) for town in ('3108701', '5106240', '5221304')
     for year in range(2019, 2023)])

MONTH_AUDIT_FIELDS = ['territory', 'month', 'ppt_mm', 'pet_mm', 'balance_mm',
                      'tmax_monthly_mean_degC', 'balance_z_1m', 'D', 'Tz', 'grid_lat', 'grid_lon']

class BridgeError(ValueError):
    """Input identities, source preservation, or arithmetic failed validation."""

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def require(condition, message):
    if not condition:
        raise BridgeError(message)

def _rows(stream, required=()):
    reader = csv.DictReader(stream)
    fields = reader.fieldnames or []
    require(bool(fields) and len(set(fields)) == len(fields) and set(required) <= set(fields),
            'Missing or duplicate required CSV columns')
    for row in reader:
        require(None not in row and all(v is not None for v in row.values()), 'Malformed CSV record')
        yield row

def table_bytes(data, required=()):
    return list(_rows(io.StringIO(data.decode('utf-8-sig'), newline=''), required))

def finite_number(value, *, nonnegative=False, missing=True):
    if value is None or value == '':
        require(missing, 'Required numeric value is missing')
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise BridgeError('Non-numeric source numeric field: ' + repr(value)) from exc
    require(math.isfinite(result) and (not nonnegative or result >= 0),
            'Non-finite or negative source quantity: ' + repr(value))
    return result

def _index(rows, fields):
    result = {}
    for row in rows:
        key = tuple(row[f] for f in fields)
        require(all(v != '' for v in key) and key not in result, 'Missing/duplicate identity: ' + repr(key))
        result[key] = row
    return result

def cohort_records(cohort_rows, registry_rows):
    """Preserve baseline membership and the previously explored pilot flag."""
    cohorts = _index(cohort_rows, ('territory', 'crop'))
    registry = _index(registry_rows, ('territory', 'crop'))
    require(set(cohorts) == set(registry), 'Cohort and target registry identities differ')
    result = {}
    for key, row in cohorts.items():
        reg = registry[key]
        territory, crop = key
        require(len(territory) == 7 and territory.isdigit() and row['state'] == territory[:2],
                'Municipality/state identity is not canonical')
        require(crop in ('cacao', 'coffee_total'), 'Unexpected crop in frozen frame')
        require(reg['state'] == row['state'] and reg['target_frame_included'] == 'True',
                'Target inclusion or state changed')
        require(reg['pilot_previously_explored'] in ('True', 'False'), 'Invalid pilot flag')
        a0 = finite_number(row['baseline_area_ha_2002_2006'], nonnegative=True, missing=False)
        require(a0 >= 100, 'Frozen baseline area below 100 ha')
        require(a0 == finite_number(reg['baseline_area_ha_2002_2006'], nonnegative=True, missing=False),
                'Registry baseline area differs')
        result[key] = dict(territory=territory, municipality=territory, crop=crop, state=row['state'],
                           municipality_name=row.get('municipality_name_2025', ''), A0=a0,
                           pilot=reg['pilot_previously_explored'] == 'True')
    return result

def read_production(path, cohorts, log=lambda message: None):
    """Return complete fixed 2002-2025 grid, including zeros and absent rows."""
    fields = ('territory', 'crop', 'year', 'state', *RAW_MAP.values(),
              *(name + '_raw' for name in RAW_MAP.values()))
    selected, streamed = {}, 0
    with gzip.open(path, 'rt', encoding='utf-8-sig', newline='') as stream:
        for line, row in enumerate(_rows(stream, fields), 2):
            streamed += 1
            if streamed % 100000 == 0:
                log('Original production rows checked: ' + str(streamed))
            unit = row['territory'], row['crop']
            if unit not in cohorts:
                continue
            try:
                year = int(row['year'])
            except ValueError as exc:
                raise BridgeError('Invalid production year') from exc
            require(str(year) == row['year'] and year in YEARS, 'Noncanonical/outside production year')
            key = (*unit, year)
            require(key not in selected, 'Duplicate original production unit/year: ' + repr(key))
            require(row['state'] == cohorts[unit]['state'], 'Original source state disagreement')
            selected[key] = (row, line)
    annual = []
    for unit, identity in sorted(cohorts.items()):
        baseline = []
        for year in range(2002, 2007):
            require((*unit, year) in selected, 'Missing baseline year in original source')
            source = selected[(*unit, year)][0]
            baseline.append(finite_number(source['intended_harvest_ha'], nonnegative=True, missing=False))
        require(math.isclose(math.fsum(baseline) / 5, identity['A0'], rel_tol=1e-12, abs_tol=1e-6),
                'Five original intended-area values disagree with frozen A0: ' + repr(unit))
        for year in YEARS:
            row, line = selected.get((*unit, year), (None, None))
            values = {alias: None if row is None else finite_number(row[field], nonnegative=True)
                      for alias, field in RAW_MAP.items()}
            raw = {alias + '_raw': '' if row is None else row[field + '_raw']
                   for alias, field in RAW_MAP.items()}
            I, H, Q = values['Iha'], values['Hha'], values['Q_tonnes']
            annual.append({**identity, 'year': year, **values, **raw,
                           'source_row_present': row is not None, 'source_row_number': line,
                           'source_discordant': (*unit, year) in DISCORDANT_RECORDS,
                           'area_order_valid': None if I is None or H is None else H <= I,
                           'positive_production_zero_harvest': Q is not None and Q > 0 and H == 0})
    summary = dict(source_rows_streamed=streamed, selected_original_rows=len(selected),
                   annual_rows=len(annual), original_baseline_units_reconciled=len(cohorts),
                   structural_missing_rows=sum(not r['source_row_present'] for r in annual),
                   zero_production_rows=sum(r['Q_tonnes'] == 0 for r in annual),
                   missing_production_rows=sum(r['Q_tonnes'] is None for r in annual),
                   missing_intended_area_rows=sum(r['Iha'] is None for r in annual),
                   missing_harvested_area_rows=sum(r['Hha'] is None for r in annual),
                   source_discordant_annual_rows=sum(r['source_discordant'] for r in annual),
                   harvested_exceeds_intended_rows=sum(r['area_order_valid'] is False for r in annual),
                   positive_production_zero_harvest_rows=sum(r['positive_production_zero_harvest'] for r in annual))
    return annual, summary

def standardize_monthly(values):
    """Calendar-month 1981-2010 mean/sample-SD standardization, no rolling overlap."""
    require(len(values) == len(MONTHS), 'Climate series does not contain 552 months')
    reference, answer = {}, []
    for month0 in range(12):
        sample = [values[(year - 1980) * 12 + month0] for year in range(1981, 2011)]
        require(len(sample) == 30 and all(math.isfinite(v) for v in sample), 'Incomplete climate reference')
        sd = statistics.stdev(sample)
        require(sd > 1e-12, 'Zero-variance climate reference calendar month')
        reference[month0] = statistics.mean(sample), sd
    for i, value in enumerate(values):
        mean, sd = reference[i % 12]
        answer.append((value - mean) / sd)
    return answer

def transform_town(rows, geography=None):
    """Recompute one-month balance z and independently check legacy temperature z."""
    require([r['month'] for r in rows] == list(MONTHS), 'Missing/duplicate/out-of-order climate month')
    territory = rows[0]['territory']
    require(all(r['territory'] == territory for r in rows), 'Mixed municipality climate series')
    fields = ('ppt_mm', 'pet_mm', 'balance_mm', 'tmax_monthly_mean_degC',
              'tmax_calendar_month_z', 'grid_lat', 'grid_lon')
    values = {field: [finite_number(r[field], nonnegative=field in ('ppt_mm', 'pet_mm'), missing=False)
                      for r in rows] for field in fields}
    require(all(math.isclose(p - e, b, rel_tol=1e-12, abs_tol=1e-8)
                for p, e, b in zip(values['ppt_mm'], values['pet_mm'], values['balance_mm'])),
            'Monthly water balance differs from ppt minus pet')
    for field, lower, upper in (('grid_lat', -90, 90), ('grid_lon', -180, 180)):
        require(len(set(values[field])) == 1 and lower <= values[field][0] <= upper,
                'Grid coordinate changed inside municipality series')
        if geography is not None:
            expected = finite_number(geography[field], missing=False)
            require(math.isclose(values[field][0], expected, rel_tol=0, abs_tol=1e-8),
                    'Climate grid differs from preserved geography')
    z1 = standardize_monthly(values['balance_mm'])
    tz_check = standardize_monthly(values['tmax_monthly_mean_degC'])
    tz_error = max(abs(a - b) for a, b in zip(tz_check, values['tmax_calendar_month_z']))
    require(tz_error <= 1e-8, 'Independent calendar-month temperature z check failed')
    D = [max(0.0, -z) for z in z1]
    monthly = [dict(territory=territory, month=month, ppt_mm=values['ppt_mm'][i],
                    pet_mm=values['pet_mm'][i], balance_mm=values['balance_mm'][i],
                    tmax_monthly_mean_degC=values['tmax_monthly_mean_degC'][i],
                    balance_z_1m=z1[i], D=D[i], Tz=values['tmax_calendar_month_z'][i],
                    grid_lat=values['grid_lat'][i], grid_lon=values['grid_lon'][i])
               for i, month in enumerate(MONTHS)]
    annual = {}
    for year in CLIMATE_YEARS:
        start = (year - 1980) * 12
        part = slice(start, start + 12)
        annual[(territory, year)] = dict(territory=territory, year=year,
            D_mean=math.fsum(D[part]) / 12,
            Tz_mean=math.fsum(values['tmax_calendar_month_z'][part]) / 12,
            grid_lat=values['grid_lat'][0], grid_lon=values['grid_lon'][0], months=12)
    return annual, monthly, tz_error

def read_climate(path, towns, geography, audit_path=None, log=lambda message: None):
    annual, seen, max_error = {}, set(), 0.0
    audit_stream = gzip.open(audit_path, 'wt', encoding='utf-8', newline='') if audit_path else None
    try:
        writer = csv.DictWriter(audit_stream, fieldnames=MONTH_AUDIT_FIELDS, lineterminator='\n') if audit_stream else None
        if writer:
            writer.writeheader()
        with gzip.open(path, 'rt', encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            require(reader.fieldnames == MONTH_COLUMNS, 'Frozen monthly climate schema changed')
            for territory, group in itertools.groupby(reader, lambda r: r['territory']):
                require(territory in towns and territory not in seen, 'Unexpected/repeated climate municipality')
                rows = list(group)
                require(all(None not in r and all(v is not None for v in r.values()) for r in rows), 'Malformed climate CSV record')
                transformed, monthly, error = transform_town(rows, geography[territory])
                annual.update(transformed)
                seen.add(territory)
                max_error = max(max_error, error)
                if writer:
                    writer.writerows(monthly)
                if len(seen) % 100 == 0 or len(seen) == len(towns):
                    log(f'Climate z1 and annual exposure: {len(seen)}/{len(towns)} municipalities')
        require(seen == set(towns), 'Full monthly climate omits frozen target municipality')
    finally:
        if audit_stream:
            audit_stream.close()
    return annual, dict(climate_municipalities=len(seen), climate_monthly_rows=len(seen) * len(MONTHS),
                        climate_annual_rows=len(annual), independent_temperature_z_max_abs_error=max_error,
                        climate_monthly_first=MONTHS[0], climate_monthly_last=MONTHS[-1],
                        standardization_reference=[1981, 2010], standardization_ddof=1,
                        current_prior_windows_overlap=False)




def _json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result
    def invalid(value):
        raise BridgeError('Nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),
                      object_pairs_hook=unique, parse_constant=invalid)


def _dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


def _write_table(path, rows):
    """Deterministic gzip, original raw symbol strings, blank numeric missingness."""
    rows = list(rows)
    require(bool(rows), 'Cannot write a schema-free empty canonical table')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    require(all(set(row) == set(fields) for row in rows), 'Canonical table schema differs between rows')
    if path.suffix == '.gz':
        with path.open('wb') as binary:
            with gzip.GzipFile(filename='', mode='wb', fileobj=binary, mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding='utf-8', newline='') as stream:
                    writer = csv.DictWriter(stream, fields, lineterminator='\n')
                    writer.writeheader()
                    writer.writerows(rows)
    else:
        with path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fields, lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)


def _safe_relative(value):
    path = PurePosixPath(value)
    return (isinstance(value, str) and bool(value) and '\\' not in value and ':' not in value
            and not path.is_absolute() and '..' not in path.parts and str(path) == value)


def verify_manifest(directory, filename='input_manifest.json', required=()):
    """Validate all declared files using relative paths only; never resolve a report archive."""
    root = Path(directory)
    manifest = _json(root / filename)
    hashes = manifest.get('sha256')
    require(isinstance(hashes, dict) and bool(hashes), 'Canonical manifest has no file hashes')
    require(set(required) <= set(hashes), 'Required canonical files absent from manifest')
    for name, expected in hashes.items():
        require(_safe_relative(name) and name != filename, 'Unsafe/self-referential manifest path')
        require(isinstance(expected, str) and len(expected) == 64 and
                all(c in '0123456789abcdef' for c in expected), 'Invalid manifest SHA256')
        path = root.joinpath(*PurePosixPath(name).parts)
        require(path.is_file() and not path.is_symlink(), 'Canonical file missing or is a symlink: ' + name)
        require(sha256(path) == expected, 'Canonical file SHA256 mismatch: ' + name)
    return manifest


def prepare_inputs(raw_dir, out_dir, log=print):
    """Build a portable input directory from generic raw files and their manifest.

    raw_dir/source_manifest.json must have a ``sha256`` object covering RAW_FILES.
    The raw production schema is IBGE annual municipal production used by this
    study; original symbols are in production_t_raw/intended_harvest_ha_raw/
    harvested_ha_raw. Frozen cohort and pilot membership come from their tables.
    The destination must not already exist. Source files are never modified.
    """
    raw, destination = Path(raw_dir), Path(out_dir)
    require(raw.resolve() != destination.resolve(), 'Raw and prepared directories must differ')
    require(not destination.exists(), 'Prepared destination already exists; load and verify it or choose a new directory')
    source_manifest = verify_manifest(raw, 'source_manifest.json', RAW_FILES)
    log('Prepare canonical inputs: reconcile cohort and original production')
    cohort_source = table_bytes((raw / 'baseline_cohort.csv').read_bytes())
    registry_source = table_bytes((raw / 'target_sample_registry.csv').read_bytes())
    cohorts = cohort_records(cohort_source, registry_source)
    require(bool(cohorts), 'Frozen target frame is empty')
    towns = {unit[0] for unit in cohorts}
    geo_rows = table_bytes((raw / 'geography_climate_audit.csv').read_bytes())
    geography = {key[0]: value for key, value in _index(geo_rows, ('territory',)).items()}
    require(set(geography) == towns and all(row['status'] == 'COMPLETE' for row in geography.values()),
            'Geography table differs from the frozen complete target frame')
    annual, production_summary = read_production(raw / 'production_source.csv.gz', cohorts, log)
    log('Prepare canonical inputs: reconstruct monthly-standardized annual climate')
    climate, climate_summary = read_climate(raw / 'climate_monthly.csv.gz', towns, geography, log=log)
    panel = make_panel(annual, climate)
    require(len(panel) == len(cohorts) * len(YEARS), 'Annual panel does not preserve the complete target grid')
    require(all(row['D' + str(lag)] is not None and row['T' + str(lag)] is not None
                for row in panel for lag in range(4)), 'Annual panel omits required climate history')
    # Final source check guards against a file changing while it is streamed.
    verify_manifest(raw, 'source_manifest.json', RAW_FILES)
    counts = dict(crop_units=len(cohorts), crop_unit_counts=dict(collections.Counter(unit[1] for unit in cohorts)),
                  municipalities=len(towns), pilot_crop_units=sum(row['pilot'] for row in cohorts.values()),
                  **production_summary, **climate_summary)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=destination.name + '.preparing_', dir=destination.parent))
    try:
        _write_table(temporary / 'annual_production.csv.gz', annual)
        _write_table(temporary / 'annual_panel.csv.gz', panel)
        _write_table(temporary / 'climate_annual.csv.gz', [climate[key] for key in sorted(climate)])
        _write_table(temporary / 'cohort.csv', [cohorts[key] for key in sorted(cohorts)])
        copied = [name for name in RAW_FILES if name != 'production_source.csv.gz']
        for name in copied:
            target = temporary.joinpath(*PurePosixPath(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(raw.joinpath(*PurePosixPath(name).parts), target)
        provenance = dict(schema_version=SCHEMA_VERSION, source_sha256=source_manifest['sha256'],
                          validated_source_module_hashes=VALIDATED_SOURCE_MODULE_HASHES,
                          source_values_replaced=False, missing_values_imputed=False,
                          historical_boundaries_harmonized=False, phenotype_specific_calendar_applied=False,
                          source_discordance_policy='Flag the 14 frozen annual records; original numeric fields retained',
                          climate_reference=[1981, 2010], climate_reference_by_calendar_month=True,
                          climate_reference_ddof=1, annual_climate='Mean of 12 observed months',
                          D='max(0,-monthly balance z1); annual mean',
                          T='monthly mean daily maximum temperature z; annual mean',
                          source_paths_included=False, requires_historical_report_archive=False)
        _dump(temporary / 'provenance.json', provenance)
        hashes = {p.relative_to(temporary).as_posix(): sha256(p)
                  for p in sorted(temporary.rglob('*')) if p.is_file()}
        manifest = dict(schema_version=SCHEMA_VERSION, sha256=hashes, counts=counts,
                        production_input_contract='Blank numeric cell is missing; numeric zero is retained; raw symbols unchanged',
                        analysis_years=list(YEARS), monthly_climate_range=[MONTHS[0], MONTHS[-1]])
        _dump(temporary / 'input_manifest.json', manifest)
        # Publish only after complete construction and independent typed checks.
        load_inputs(temporary, log=lambda _: None)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    log('PASS portable inputs: ' + str(len(panel)) + ' annual rows; source zeros and missingness retained')
    return dict(input_dir=str(destination), manifest=manifest, counts=counts)


def _read_typed_annual(path):
    records = []
    with gzip.open(path, 'rt', encoding='utf-8-sig', newline='') as stream:
        for original in _rows(stream, ('territory', 'crop', 'state', 'year', 'A0', 'Q_tonnes', 'Iha', 'Hha')):
            row = dict(original)
            for field in ('year', 'source_row_number'):
                if field not in row:
                    continue
                if row[field] == '' and field == 'source_row_number':
                    row[field] = None
                else:
                    require(row[field].isdigit() and str(int(row[field])) == row[field], 'Noncanonical annual integer')
                    row[field] = int(row[field])
            for field in BOOLEAN_FIELDS:
                if field not in row:
                    continue
                value = row[field]
                require(value in ('True', 'False') or (value == '' and field == 'area_order_valid'),
                        'Invalid explicit annual boolean: ' + field)
                row[field] = None if value == '' else value == 'True'
            for field in NUMERIC_FIELDS:
                if field in row:
                    row[field] = finite_number(row[field], missing=field != 'A0',
                                               nonnegative=field in ('A0', 'Q_tonnes', 'Iha', 'Hha') or field.startswith('D'))
            require(row['crop'] in CROPS and row['territory'].isdigit() and len(row['territory']) == 7
                    and row['state'] == row['territory'][:2] and row['year'] in YEARS,
                    'Canonical annual identity changed')
            records.append(row)
    return records


def _read_annual_climate(path):
    result = {}
    with gzip.open(path, 'rt', encoding='utf-8-sig', newline='') as stream:
        for row in _rows(stream, ('territory', 'year', 'D_mean', 'Tz_mean', 'grid_lat', 'grid_lon', 'months')):
            require(row['year'].isdigit() and row['months'] == '12', 'Invalid annual climate calendar')
            row['year'], row['months'] = int(row['year']), 12
            require(row['year'] in CLIMATE_YEARS, 'Annual climate outside frozen monthly record')
            for field in ('D_mean', 'Tz_mean', 'grid_lat', 'grid_lon'):
                row[field] = finite_number(row[field], missing=False, nonnegative=field == 'D_mean')
            key = row['territory'], row['year']
            require(key not in result, 'Duplicate annual climate key')
            result[key] = row
    return result


def load_inputs(input_dir, log=print):
    """Load a self-contained canonical dataset; no stage ZIP or saved model is needed."""
    directory = Path(input_dir)
    required = ('annual_production.csv.gz', 'annual_panel.csv.gz', 'climate_annual.csv.gz',
                'climate_monthly.csv.gz', 'baseline_cohort.csv', 'target_sample_registry.csv',
                'geography_climate_audit.csv', 'climate_source_metadata.json', 'cohort.csv', 'provenance.json',
                'legacy/consolidated_annual_weights.csv.gz', 'legacy/design_registry.csv',
                'legacy/design_event_membership.csv.gz', 'legacy/design_unit_registry.csv')
    manifest = verify_manifest(directory, 'input_manifest.json', required)
    require(manifest.get('schema_version') == SCHEMA_VERSION, 'Unsupported portable input schema')
    annual = _read_typed_annual(directory / 'annual_production.csv.gz')
    panel = _read_typed_annual(directory / 'annual_panel.csv.gz')
    require(all(set(PANEL_REQUIRED) <= set(row) for row in panel), 'Canonical panel required fields absent')
    climate = _read_annual_climate(directory / 'climate_annual.csv.gz')
    cohorts = cohort_records(table_bytes((directory / 'baseline_cohort.csv').read_bytes()),
                             table_bytes((directory / 'target_sample_registry.csv').read_bytes()))
    expected = {(t, crop, year) for t, crop in cohorts for year in YEARS}
    annual_lookup = {(r['territory'], r['crop'], r['year']): r for r in annual}
    panel_lookup = {(r['territory'], r['crop'], r['year']): r for r in panel}
    require(len(annual_lookup) == len(annual) and len(panel_lookup) == len(panel)
            and set(annual_lookup) == set(panel_lookup) == expected, 'Portable annual grid is incomplete or duplicated')
    for unit, cohort in cohorts.items():
        baseline = []
        for year in YEARS:
            row = annual_lookup[(*unit, year)]
            for field in ('A0', 'pilot', 'state'):
                require(row[field] == cohort[field], 'Portable cohort identity changed: ' + field)
            require(row['source_discordant'] == ((*unit, year) in DISCORDANT_RECORDS),
                    'Frozen source discordance identity changed')
            if year <= 2006:
                require(row['Iha'] is not None, 'Portable baseline intended area missing')
                baseline.append(row['Iha'])
        require(math.isclose(math.fsum(baseline) / 5, cohort['A0'], rel_tol=1e-12, abs_tol=1e-6),
                'Portable A0 differs from five original intended areas')
    towns = {t for t, crop in cohorts}
    require(set(climate) == {(t, year) for t in towns for year in CLIMATE_YEARS},
            'Portable annual climate grid is incomplete')
    rebuilt = make_panel(annual, climate)
    for row in rebuilt:
        other = panel_lookup[row['territory'], row['crop'], row['year']]
        require(set(row) == set(other), 'Portable panel fields differ from raw annual construction')
        for field, value in row.items():
            observed = other[field]
            if isinstance(value, float):
                require(observed is not None and math.isclose(value, observed, rel_tol=1e-12, abs_tol=1e-9),
                        'Portable panel arithmetic mismatch: ' + field)
            else:
                require(value == observed, 'Portable panel value/missingness changed: ' + field)
    counts = manifest.get('counts', {})
    actual = dict(annual_rows=len(panel), crop_units=len(cohorts), municipalities=len(towns),
                  crop_unit_counts=dict(collections.Counter(unit[1] for unit in cohorts)),
                  pilot_crop_units=sum(row['pilot'] for row in cohorts.values()),
                  zero_production_rows=sum(row['Q_tonnes'] == 0 for row in annual),
                  missing_production_rows=sum(row['Q_tonnes'] is None for row in annual),
                  missing_intended_area_rows=sum(row['Iha'] is None for row in annual),
                  missing_harvested_area_rows=sum(row['Hha'] is None for row in annual),
                  source_discordant_annual_rows=sum(row['source_discordant'] for row in annual),
                  climate_annual_rows=len(climate), climate_monthly_rows=len(towns) * len(MONTHS))
    require(all(counts.get(field) == value for field, value in actual.items()), 'Portable summary counts disagree')
    summary = dict(status='PASS_PORTABLE_INPUTS', schema_version=SCHEMA_VERSION, counts=actual,
                   input_manifest_sha256=sha256(directory / 'input_manifest.json'),
                   outcome_arithmetic_checked=True, full_unit_year_grid_checked=True,
                   zero_and_missing_distinguished=True, requires_historical_report_archive=False,
                   scientific_identification_validated=False)
    log('PASS canonical input hashes and typed annual panel: ' + str(len(panel)) + ' rows')
    return dict(panel=sorted(panel, key=lambda r: (r['crop'], r['territory'], r['year'])),
                annual_rows=annual, climate_annual=climate, summary=summary, manifest=manifest)

VALIDATED_SOURCE_MODULE_HASHES = {'source_preparation': '2f38f97f0bb49e799b03ffd7e186bc7074cd53ab701af89d2b03af4fcc856d1d', 'feature_preparation': '5f0eab9bbe0280c4892f3f7b1343d87bf405d837c610dada382e28c5c69aaca4'}
