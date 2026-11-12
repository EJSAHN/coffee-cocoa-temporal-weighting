"""Local publication exports from completed scientific output tables only.

This module performs no model fitting, contrast estimation, reweighting or
uncertainty calculation. The supplementary workbook and its indexed CSV.gz
companions preserve complete rows from the completed analysis families.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re

SPEC_PATH = Path(__file__).resolve().parents[1] / 'supplementary_spec.json'
FAMILIES = ('panel_models', 'response_comparison', 'temporal_comparison',
            'benchmark', 'temporal_weights', 'event_contrasts', 'legacy')
TEXT_NAMES = {'model_id', 'design_id', 'territory', 'state', 'crop', 'population',
              'kind', 'group', 'omitted_group', 'event_id', 'pair_id', 'unit_id',
              'coefficient_i', 'coefficient_j', 'numerator', 'denominator',
              'value', 'status', 'year_id'}
NUMERIC = re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$')


def load_spec(path=None):
    return json.loads(Path(path or SPEC_PATH).read_text(encoding='utf-8'))


def read_csv(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        return {'columns': reader.fieldnames or [], 'rows': list(reader),
                'source': str(path), 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def _table(value):
    if isinstance(value, dict) and 'rows' in value:
        result = dict(value)
        result['rows'] = list(result['rows'])
        result.setdefault('columns', list(dict.fromkeys(k for row in result['rows'] for k in row)))
        return result
    rows = list(value)
    return {'rows': rows, 'columns': list(dict.fromkeys(k for row in rows for k in row))}


def collect_result_tables(family_dirs, *, input_dir=None, spec_path=None):
    """Load explicitly supplied family directories; never search old archives.

The benchmark directory contains its result CSVs directly. A fresh run usually
uses ``output/benchmark/results``; reviewed results can be passed explicitly.
Missing tables remain missing and are recorded by the workbook collector.
"""
    dirs = {key: Path(value) for key, value in family_dirs.items()}
    if input_dir is not None:
        dirs.setdefault('legacy', Path(input_dir) / 'legacy')
    tables = {}
    for entry in load_spec(spec_path)['tables']:
        family = entry['family']
        if family not in dirs or not entry['filename']:
            continue
        path = dirs[family] / entry['filename']
        if path.is_file():
            tables[entry['key']] = read_csv(path)
    tables.update(extract_model_tables(dirs))
    return tables


def fresh_family_dirs(output_dir, input_dir=None):
    root = Path(output_dir)
    result = {family: root / family for family in FAMILIES if family != 'legacy'}
    result['benchmark'] = root / 'benchmark' / 'results'
    if input_dir is not None:
        result['legacy'] = Path(input_dir) / 'legacy'
    return result


def load_fresh_tables(output_dir, *, input_dir=None):
    return collect_result_tables(fresh_family_dirs(output_dir, input_dir), input_dir=input_dir)


def _flatten(value, prefix=''):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _flatten(child, (prefix + '.' if prefix else '') + key)
    else:
        yield prefix, value


def extract_model_tables(family_dirs):
    """Export existing JSON model values; intervals and covariance are copied."""
    coefficients, covariances, diagnostics = [], [], []
    count = 0
    for family in ('panel_models', 'response_comparison', 'temporal_comparison'):
        if family not in family_dirs:
            continue
        directory = Path(family_dirs[family]) / 'models'
        for path in sorted(directory.glob('*.json')):
            # The response family also keeps copies of original fits for checks.
            if family == 'response_comparison' and not path.stem.startswith('common_fit__'):
                continue
            obj = json.loads(path.read_text(encoding='utf-8'))
            spec = obj.get('specification', {})
            meta = {'analysis_family': family, 'model_id': spec.get('model_id', path.stem),
                    'crop': spec.get('crop', ''), 'outcome': spec.get('outcome', ''),
                    'variant': spec.get('variant', ''), 'model_status': obj.get('status', '')}
            count += 1
            order = obj.get('coefficient_order', [])
            estimates = obj.get('coefficients', {})
            uncertainty = obj.get('uncertainty', {})
            for label, block in uncertainty.items():
                if not isinstance(block, dict):
                    continue
                intervals = block.get('intervals', {})
                for term in order:
                    saved = intervals.get(term, {})
                    coefficients.append(dict(meta, term=term, uncertainty=label,
                        estimate=estimates.get(term), standard_error=saved.get('standard_error'),
                        lower=saved.get('lower'), upper=saved.get('upper'),
                        critical_value=block.get('critical_value'),
                        reference_distribution=block.get('reference_distribution'),
                        n_observed_score_years=block.get('n_observed_score_years'),
                        lag_years=block.get('lag_years'), scale='raw coefficient'))
                matrix = block.get('covariance', [])
                if matrix:
                    if len(matrix) != len(order) or any(len(row) != len(order) for row in matrix):
                        raise ValueError('Saved covariance dimension mismatch: ' + str(path))
                    for i, term_i in enumerate(order):
                        for j, term_j in enumerate(order):
                            covariances.append(dict(analysis_family=family, model_id=meta['model_id'],
                                uncertainty=label, coefficient_i=term_i, coefficient_j=term_j,
                                covariance=matrix[i][j]))
            retained = {k: v for k, v in obj.items() if k not in ('coefficients', 'coefficient_order', 'uncertainty')}
            retained['uncertainty_metadata'] = {label: {k: v for k, v in block.items()
                if k not in ('covariance', 'intervals', 'standard_errors')}
                for label, block in uncertainty.items() if isinstance(block, dict)}
            for key, value in _flatten(retained):
                encoded = json.dumps(value, ensure_ascii=False, allow_nan=False) if not isinstance(value, str) else value
                diagnostics.append(dict(analysis_family=family, model_id=meta['model_id'],
                    field=key, value=encoded, value_type=type(value).__name__))
    if not count:
        return {}
    return {'all_model_coefficients': _table(coefficients),
            'all_model_covariances': _table(covariances),
            'all_model_diagnostics': _table(diagnostics)}


def _text_field(name):
    return (name in TEXT_NAMES or name.endswith(('_id', '_exact', '_numerator', '_denominator',
            '_sha256', '_fingerprint')) or 'sha256' in name or 'fingerprint' in name)


def _convert(value, field):
    if _text_field(field):
        if value is None:
            return None
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=False, allow_nan=False)
        return str(value)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError('Nonfinite numerical export value: ' + field)
        return str(value) if isinstance(value, int) and abs(value) >= 10**15 else value
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    value = str(value)
    if value == '':
        return None
    if value in ('True', 'False', 'true', 'false'):
        return value.lower() == 'true'
    if NUMERIC.fullmatch(value):
        if '.' not in value and 'e' not in value.lower():
            number = int(value)
            return number if abs(number) < 10**15 else value
        number = float(value)
        if not math.isfinite(number):
            raise ValueError('Nonfinite numerical export value: ' + field)
        return number
    return value


def _column_type(field, values):
    if _text_field(field):
        return 'text'
    observed = [value for value in values if value is not None]
    if observed and all(isinstance(value, bool) for value in observed):
        return 'boolean'
    if observed and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in observed):
        return 'number'
    if observed and all(isinstance(value, str) for value in observed):
        return 'text'
    return 'auto'


def _column_description(field, entry):
    definitions = {
        'analysis_family': ('Analysis family; supports and functionals must not be pooled across families.', 'identifier'),
        'model_id': ('Unique model specification identifier.', 'identifier'),
        'design_id': ('Fixed event or synthetic climate design identifier.', 'identifier'),
        'territory': ('IBGE municipality identifier, preserved as text.', 'identifier'),
        'state': ('IBGE state code, preserved as text.', 'identifier'),
        'crop': ('Coffee-total or cacao crop identifier.', 'category'),
        'outcome': ('Model outcome; production_fixed_ha is kg per baseline intended hectare.', 'category'),
        'variant': ('Estimator/specification label; levels, unit trends and first differences are distinct models.', 'category'),
        'estimate': ('Saved point estimate of the row-defined quantity; consult outcome, term, quantity and display_scale.', 'row-specific outcome/functional units'),
        'gamma': ('Raw current-dryness-by-history coefficient.', 'outcome units per dryness unit per history unit'),
        'uncertainty': ('HAC_L2/HAC_L4: calendar score HAC with two/four-year bandwidth; approximate pointwise intervals.', 'method'),
        'lower': ('Saved approximate pointwise 95% interval lower endpoint.', 'same as estimate'),
        'upper': ('Saved approximate pointwise 95% interval upper endpoint.', 'same as estimate'),
        'standard_error': ('Saved model standard error; real-data coverage is not established.', 'same as estimate'),
        'structural_truth': ('Generating-process value of the stated functional.', 'functional-specific'),
        'projection_truth': ('Noiseless mean projected on the stated fitted model and evaluated at its fixed reference.', 'functional-specific'),
        'projection_minus_structural': ('Difference between noiseless projection and structural generating target.', 'functional-specific'),
        'climate_signal_contrast': ('Exact frozen-event-weight contrast of the generating climate contribution.', 'simulated outcome units'),
        'background_contrast': ('Exact frozen-event-weight contrast of the generating nonclimate background.', 'simulated outcome units'),
        'full_mean_contrast': ('Exact event contrast of the full noiseless generating mean.', 'simulated outcome units'),
        'coefficient': ('Annual frozen contrast weight; numerator/denominator retain its exact rational representation.', 'dimensionless'),
        'weighted_contribution': ('Annual frozen weight multiplied by reported-unit production per baseline hectare.', 'kg per baseline intended hectare'),
        'contribution': ('Municipality contribution already includes the original design normalization.', 'row-specific functional units'),
        'year': ('Calendar production or profile year.', 'year'),
        'tau': ('Calendar year minus 2006.', 'year'),
        'reference_history_mean': ('Undifferenced endpoint history average used for the reported derivative.', 'history exposure units'),
        'status': ('Estimation/export status. Unavailable estimates are not replaced with zero.', 'status text'),
        'source_discordant': ('Official source-record discordance flag; source symbols and numeric zeros remain distinct.', 'boolean'),
    }
    if field in definitions:
        return definitions[field]
    if field.endswith(('_exact', '_numerator', '_denominator')) or field in ('numerator', 'denominator'):
        return ('Exact rational value or rational component, stored as text to avoid floating-point loss.', 'exact rational text')
    if field.startswith(('m0', 'm1', 'm2', 'mp')):
        unit = 'years squared' if field.startswith('m2') else ('years' if field.startswith('m1') else 'dimensionless')
        suffix = ('Sum of absolute municipality responses (sharp bounded-coefficient response), not observed bias.' if '_l1' in field else
                  'Signed sum over municipality responses.' if '_sum' in field else
                  'Count of municipalities with exactly nonzero response.' if '_nonzero' in field else
                  'Maximum absolute municipality response.' if '_max_abs' in field else 'Municipality exact temporal response.')
        return suffix, 'count' if '_nonzero' in field else unit
    if 'coverage' in field or field == 'false_zero_exclusion_when_structural_null':
        return ('Monte Carlo interval-coverage or null-exclusion fraction for the explicitly named target; MCSE fields give Monte Carlo uncertainty.', 'fraction [0,1]')
    if field.startswith('mcse_'):
        return ('Monte Carlo standard error of the named simulation summary, not a model confidence interval.', 'same as named summary')
    if field.startswith(('bias_', 'rmse_')):
        return ('Simulation error relative to the explicitly named structural or projection target.', 'functional-specific')
    if 'fingerprint' in field or 'sha256' in field:
        return 'SHA-256 provenance fingerprint; not a numerical measurement.', 'SHA-256 text'
    if field.startswith(('n_', 'replicates_', 'missing_', 'zero_', 'excluded_')) or field.endswith('_rows'):
        return 'Recorded count for the named support, availability or simulation field.', 'count'
    if field.endswith('_l1'):
        return 'Sum of absolute exact weights; sharp response to a unit-bounded perturbation of the indicated grouping.', 'dimensionless'
    return field.replace('_', ' ').capitalize() + '. ' + entry['description'], 'source-defined; see table description'


def collect_workbook_payload(tables, *, metadata=None, spec_path=None, strict=True):
    """Return the shared XLSX/CSV-companion JSON schema, without writing XLSX.

No rows/columns are dropped. Large detailed tables are indexed companions.
When strict=False, absent tables are explicitly indexed as NOT_AVAILABLE.
"""
    spec = load_spec(spec_path)
    metadata = dict(metadata or {})
    sheets, companions, contents, dictionary = [], [], [], []
    missing, cells = [], 0
    for entry in spec['tables']:
        key = entry['key']
        if key not in tables:
            if entry.get('required'):
                missing.append(key)
            contents.append([key, '', 'NOT_AVAILABLE', 0, 0, entry['description'], entry['family'], ''])
            continue
        value = _table(tables[key])
        columns = list(value.get('columns') or entry.get('columns') or [])
        for row in value['rows']:
            for column in row:
                if column not in columns:
                    columns.append(column)
        # Explicit column schema for genuinely empty diagnostic tables.
        if not columns:
            columns = entry.get('columns') or ['status']
        rows = [[_convert(row.get(column), column) for column in columns] for row in value['rows']]
        types = [_column_type(column, [row[i] for row in rows]) for i, column in enumerate(columns)]
        count = (len(rows) + 1) * len(columns)
        placement = entry['placement']
        if placement == 'workbook' and cells + count > spec['table_policy']['workbook_cell_budget'] - 20000:
            placement = 'companion'
        target = entry['sheet'] if placement == 'workbook' else 'tables/' + key + '.csv.gz'
        table = dict(sheet=entry['sheet'], key=key, columns=columns, rows=rows, column_types=types,
            metadata=dict(description=entry['description'], source=value.get('source', ''),
                          source_sha256=value.get('source_sha256', ''), units=entry['units'],
                          analysis_family=entry['family'], row_count=len(rows)))
        if placement == 'workbook':
            sheets.append(table)
            cells += count
        else:
            table['filename'] = key + '.csv.gz'
            companions.append(table)
        contents.append([key, target, placement.upper(), len(rows), len(columns), entry['description'],
                         entry['family'], value.get('source_sha256', '')])
        for column, datatype in zip(columns, types):
            description, units = _column_description(column, entry)
            dictionary.append([key, target, column, datatype, units, description])
    if strict and missing:
        raise FileNotFoundError('Required scientific output tables missing: ' + ', '.join(missing))
    readme = [['Title', spec['title']], ['Scope', 'Complete listed scientific result tables, with large support and contribution tables in indexed CSV.gz companions.'],
              ['How to use', 'Read Contents for table locations and Data_dictionary for fields. A missing table is explicitly NOT_AVAILABLE; an empty available table is retained with its header.'],
              ['Missing numerical cells', 'Blank means unavailable/absent; numeric zero remains 0. Explicit NA, ... and other source/status strings are preserved as text.'],
              ['Companions', 'The tables/ directory belongs with this workbook in Supplementary_Data_S1.zip. No companion table is truncated.'],
              ['Excluded material', spec['table_policy']['excluded']]]
    readme += [['Interpretation ' + str(i + 1), note] for i, note in enumerate(spec['interpretation'])]
    readme += [['Export completeness', 'COMPLETE' if not missing else 'PARTIAL: see NOT_AVAILABLE rows in Contents'],
               ['Missing table keys', ', '.join(missing)]]
    for key, value in sorted(metadata.items()):
        readme.append([key, json.dumps(value, ensure_ascii=False, allow_nan=False) if not isinstance(value, str) else value])
    def meta_sheet(name, columns, rows):
        return dict(sheet=name, key=name.lower(), columns=columns, rows=rows,
                    column_types=['auto'] * len(columns), metadata={'description': name})
    front = [meta_sheet('ReadMe', ['item', 'description'], readme),
             meta_sheet('Contents', ['table_key', 'location', 'placement', 'rows', 'columns', 'description', 'analysis_family', 'source_sha256'], contents),
             meta_sheet('Data_dictionary', ['table_key', 'location', 'column', 'storage_type', 'units', 'description'], dictionary)]
    front_cells = sum((len(s['rows']) + 1) * len(s['columns']) for s in front)
    return dict(schema_version=spec['schema_version'], title=spec['title'], sheets=front + sheets,
                companions=companions, missing_tables=missing,
                summary=dict(workbook_sheets=len(front) + len(sheets), workbook_cells=cells + front_cells,
                    companion_tables=len(companions), scientific_tables=len(sheets) + len(companions),
                    scientific_rows=sum(len(s['rows']) for s in sheets + companions), complete=not missing))


def collect_from_output(output_dir, *, input_dir=None, metadata=None, strict=True):
    return collect_workbook_payload(load_fresh_tables(output_dir, input_dir=input_dir),
                                    metadata=metadata, strict=strict)


def temporal_from_tables(tables):
    mapping = {'design_rows': 'temporal_design_summary', 'unit_rows': 'temporal_unit_moments',
               'year_rows': 'temporal_common_year_profiles', 'state_year_rows': 'temporal_state_year_profiles'}
    return {key: _table(tables[source])['rows'] for key, source in mapping.items()}


def figure_tables(tables):
    def tagged(key, quantity=None):
        rows = _table(tables[key])['rows']
        return [dict(row, analysis_family='temporal_comparison', evaluation_scope='common_transition_endpoints')
                for row in rows if quantity is None or row.get('quantity') == quantity]
    result = {'integrated_current_response': tagged('c_current'),
              'integrated_history_interaction': tagged('c_contrasts', 'interaction_gamma'),
              'future_climate_reference': tagged('c_future')}
    for key in ('simulation_panel_summary', 'simulation_event_decomposition', 'simulation_event_support'):
        result[key] = _table(tables[key])['rows']
    return result


def write_payload(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n', encoding='utf-8')
    return path
