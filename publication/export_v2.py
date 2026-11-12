"""Export manuscript tables, plot data and Supplementary Data S1 from a completed run.

Numerical exports use the standard library. Plotting dependencies are imported
only when figures are requested for a completed production benchmark. No model
is fitted, no observation is selected anew, and no saved reference plot table is
used as an input. Existing completed exports are reused only after input, code
and output hashes have been verified.
"""
from __future__ import annotations

from contextlib import contextmanager
import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
import zipfile

from local.presentation import (collect_workbook_payload, figure_tables,
    load_fresh_tables, read_csv, temporal_from_tables, _convert, _column_type)
from local.supplementary import export_companions, export_workbook

NAMES = (
    'Figure_1_exact_temporal_response', 'Figure_2_actual_temporal_specifications',
    'Figure_3_null_current_response_benchmark', 'Figure_4_event_support_and_background',
    'Figure_S1_temporal_response_excluding_pilot', 'Figure_S2_null_interaction_benchmark',
    'Figure_S3_future_climate_diagnostics', 'Figure_S4_original_event_component_benchmark',
)
PANEL_COUNTS = (4, 4, 6, 4, 4, 6, 2, 6)
BENCHMARK_TABLES = (
    'by_design_simulation_design_registry.csv', 'by_design_simulation_event_decomposition.csv',
    'by_design_simulation_event_support.csv', 'by_design_simulation_nonestimable.csv',
    'by_design_simulation_panel_summary.csv', 'by_design_simulation_projection_oracles.csv',
    'event_decomposition_across_designs.csv', 'event_support_across_designs.csv',
    'field_completion_registry.csv', 'panel_estimability_across_designs.csv',
    'panel_performance_across_designs.csv',
)
DIAGNOSTIC_TABLES = ('contrast_summary.csv', 'unit_moments.csv', 'year_weights.csv',
                     'group_year_weights.csv', 'response_bounds.csv', 'consolidated_weights.csv')
CROPS = ('coffee_total', 'cacao')
KINDS = ('original_onset', 'original_confirmation', 'matched_onset', 'matched_anchor_change')
KIND_LABELS = dict(zip(KINDS, ('Original onset', 'Original confirmation', 'Matched onset', 'Matched anchor change')))
METHODS = ('levels', 'levels_unit_trend', 'first_difference')
METHOD_LABELS = dict(zip(METHODS, ('Levels', 'Municipality trends', 'First differences')))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, rows, columns=None):
    rows = list(rows)
    fields = list(columns or dict.fromkeys(k for row in rows for k in row))
    if not fields:
        raise ValueError('An empty table needs an explicit column schema: ' + str(path))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _one(rows, **conditions):
    chosen = [r for r in rows if all(str(r.get(k)) == str(v) for k, v in conditions.items())]
    if len(chosen) != 1:
        raise ValueError('Expected one result row: ' + str(conditions))
    return chosen[0]


def _kind(row):
    if row.get('kind') in KINDS:
        return row['kind']
    design = str(row.get('design_id', ''))
    for needle, kind in [('matched_anchor_change', 'matched_anchor_change'), ('matched_onset', 'matched_onset'),
                         ('confirmation', 'original_confirmation'), ('onset', 'original_onset')]:
        if needle in design:
            return kind
    raise ValueError('Unknown temporal contrast: ' + design)


def _temporal_plot_rows(temporal, population):
    result = []
    chosen = [dict(r, display_kind=_kind(r)) for r in temporal['design_rows'] if r['population'] == population]
    for crop in CROPS:
        ordered = [_one(chosen, crop=crop, display_kind=k) for k in KINDS]
        result.extend(dict(r, plot_record='design_moments') for r in ordered)
        for design in ordered:
            series = sorted((r for r in temporal['year_rows'] if r['design_id'] == design['design_id']), key=lambda r: int(r['year']))
            if not series or len({r['year'] for r in series}) != len(series):
                raise ValueError('Missing or duplicated year profile: ' + design['design_id'])
            result.extend(dict(r, plot_record='common_year_weight', display_kind=design['display_kind']) for r in series)
    return result


def make_plot_tables(tables, expansion_dir):
    """Build all eight numerical figure inputs, without importing plotting code."""
    directory = Path(expansion_dir) / 'results'
    temporal = temporal_from_tables(tables)
    old = figure_tables(tables)
    production = lambda key: [r for r in old[key] if r.get('outcome') == 'production_fixed_ha']
    actual = production('integrated_current_response') + production('integrated_history_interaction')
    actual = [r for r in actual if r.get('variant') in METHODS]
    expanded = read_csv(directory / 'panel_performance_across_designs.csv')['rows']
    panels = lambda functional: [dict(r, displayed_in_figure=r['scenario'].startswith('null__'))
                                for r in expanded if r['functional'] == functional]
    event_rows = [dict(r, plot_record='event_decomposition', displayed_in_figure=r['scenario'].startswith('null__'))
                  for r in read_csv(directory / 'by_design_simulation_event_decomposition.csv')['rows']]
    event_rows += [dict(r, plot_record='event_support', displayed_in_figure=True)
                   for r in read_csv(directory / 'by_design_simulation_event_support.csv')['rows']]
    original_event = [dict(r, plot_record='event_decomposition') for r in old['simulation_event_decomposition']]
    original_event += [dict(r, plot_record='event_support') for r in old['simulation_event_support']]
    return dict(zip(NAMES, (
        _temporal_plot_rows(temporal, 'full_target'), actual, panels('current_dryness_0p1'), event_rows,
        _temporal_plot_rows(temporal, 'exclude_43_pilot_units'), panels('interaction_gamma'),
        production('future_climate_reference'), original_event,
    )))


def manuscript_tables(tables, expansion_dir):
    """Select the stated manuscript summaries from complete calculated tables."""
    event = tables['event_summary']['rows']
    t1 = []
    for crop in CROPS:
        for kind in KINDS:
            row = _one(event, crop=crop, population='full_target', kind=kind)
            t1.append(dict(Crop='Coffee' if crop == 'coffee_total' else 'Cacao', Contrast=KIND_LABELS[kind],
                Municipalities=row['n_units'], Short_interval_events=row['n_short_events'],
                Long_interval_events=row['n_long_events'], Contrast_kg_per_baseline_ha=row['estimate']))
    t2 = []
    for crop in CROPS:
        row = _one(tables['b_area_allocation']['rows'], crop=crop)
        t2.append(dict(Crop='Coffee' if crop == 'coffee_total' else 'Cacao', Municipality_years=row['n_rows'],
            Total_current_dryness_association=row['total_current_association'],
            Harvested_area_allocation=row['cumulative_area_association_allocation'],
            Productivity_allocation=row['cumulative_productivity_association_allocation'],
            Unallocated_component=row['cumulative_unallocated_association_allocation']))
    t3 = [r for r in read_csv(Path(expansion_dir) / 'results/panel_performance_across_designs.csv')['rows']
          if r['scenario'] == 'additive_interaction__stable']
    display = []
    for functional, label in [('current_dryness_0p1', 'Current derivative ×0.1'), ('interaction_gamma', 'History interaction γ')]:
        for method in METHODS:
            lo = _one(t3, functional=functional, method=method, uncertainty='HAC_L2')
            hi = _one(t3, functional=functional, method=method, uncertainty='HAC_L4')
            def fmt(row):
                if row['coverage_structural'] == '' or row['coverage_structural_mcse'] == '':
                    return 'Unavailable'
                return f"{100 * float(row['coverage_structural']):.2f} ({100 * float(row['coverage_structural_mcse']):.2f})"
            def num(key):
                return f"{float(lo[key]):.3f}" if lo[key] != '' else 'Unavailable'
            display.append({'Quantity': label, 'Specification': METHOD_LABELS[method],
                'Mean structural value': num('structural_truth'), 'Mean estimate': num('mean_estimate'),
                'HAC2 coverage (MCSE), %': fmt(lo), 'HAC4 coverage (MCSE), %': fmt(hi)})
    return {'Table_1_event_contrasts': t1, 'Table_2_production_component_allocation': t2,
            'Table_3_nonzero_effect_benchmark': t3, 'Table_3_nonzero_effect_benchmark_display': display}


def summary_workbook_payload(expansion_dir, summary):
    """Retain all columns from four aggregate/registry tables as a readable view."""
    specs = [('Panel_performance', 'panel_performance_across_designs.csv'),
             ('Event_decomposition', 'event_decomposition_across_designs.csv'),
             ('Event_support', 'event_support_across_designs.csv'),
             ('Design_completion', 'field_completion_registry.csv')]
    readme = [
        ['Scope', 'Values exported from a completed independent-design benchmark. Complete raw CSVs accompany this workbook.'],
        ['Development run', str(bool(summary.get('development_run')))],
        ['Designs planned', summary['designs_planned']], ['Designs completed', summary['designs_completed']],
        ['Replicates per design and scenario', summary['inner_replicates_per_design_scenario']],
        ['Monte Carlo uncertainty', 'Independent complete designs are the MCSE units; inner outcome replicates are not independent outer designs.'],
        ['Coverage scale', 'Coverage and estimable fractions are stored on [0,1].'],
        ['Targets', 'Structural targets, noiseless model projections and event-specific full-mean functionals are distinct.'],
        ['Missing values', 'Blank means unavailable or not applicable; numerical zero remains zero.'],
        ['Sparse support', 'Small-support and nonestimable designs remain in eligibility denominators. No eligibility-based redraw.'],
        ['Quantiles', 'Across-design quantiles describe empirical distributions; they are not confidence intervals.'],
        ['Interpretation', 'Synthetic operating characteristics under the recorded generating distributions.'],
    ]
    sheets = [dict(sheet='ReadMe', columns=['Item', 'Description'], rows=readme)]
    mapping = []
    for name, filename in specs:
        table = read_csv(Path(expansion_dir) / 'results' / filename)
        columns = table['columns']
        rows = [[_convert(r.get(c), c) for c in columns] for r in table['rows']]
        kinds = [_column_type(c, [r[i] for r in rows]) for i, c in enumerate(columns)]
        sheets.append(dict(sheet=name, columns=columns, rows=rows, column_types=kinds))
        mapping.extend(dict(worksheet=name, source_csv=filename, source_column=c, storage_type=t) for c, t in zip(columns, kinds))
    return dict(title='Independent-design benchmark', sheets=sheets), mapping


def _copy_evidence(source, destination, suffixes=('.csv', '.json')):
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.iterdir()):
        if path.is_file() and path.suffix in suffixes:
            shutil.copyfile(path, destination / path.name)


def _manifest(directory):
    return {p.relative_to(directory).as_posix(): sha(p) for p in sorted(directory.rglob('*'))
            if p.is_file() and p != directory / 'manifest_sha256.json'}


def _zip_tree(source, target):
    with zipfile.ZipFile(target, 'x', zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for p in sorted(source.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(source).as_posix())


def _source_identity(tables, workflow, input_dir, expansion_summary, figures):
    sources = {}
    for key, table in tables.items():
        if table.get('source'):
            sources['analysis_table/' + key] = sha(table['source'])
    # Model JSONs supply coefficient and covariance tables not represented by one CSV.
    for family in ('panel_models', 'response_comparison', 'temporal_comparison'):
        for p in sorted((workflow / 'analysis' / family / 'models').glob('*.json')):
            sources['models/' + family + '/' + p.name] = sha(p)
    for dirname in ('diagnostics', 'expanded_benchmark/results'):
        for p in sorted((workflow / dirname).glob('*.csv')):
            if p.name == 'field_completion_registry.csv':
                rows = read_csv(p)['rows']
                stable = [{k: v for k, v in row.items() if k not in ('seconds_this_invocation', 'reused_verified_field')} for row in rows]
                sources[dirname + '/' + p.name] = hashlib.sha256(json.dumps(stable, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            else:
                sources[dirname + '/' + p.name] = sha(p)
    for rel in ('analysis/analysis_configuration.json', 'expanded_benchmark/expansion_protocol.json', 'diagnostics/report.json'):
        p = workflow / rel
        if p.is_file():
            sources[rel] = sha(p)
    code_root = Path(__file__).resolve().parents[1]
    for p in sorted(Path(__file__).resolve().parent.glob('*')):
        if p.is_file() and p.suffix in ('.py', '.txt', '.json'):
            sources['export_code/' + p.name] = sha(p)
    for rel in ('local/presentation.py', 'local/supplementary.py', 'supplementary_spec.json'):
        sources['export_code/' + rel] = sha(code_root / rel)
    status = {k: expansion_summary.get(k) for k in ('status', 'development_run', 'designs_planned',
              'designs_completed', 'inner_replicates_per_design_scenario')}
    identity = dict(schema_version='2.0', figures_requested=bool(figures), expansion=status, source_sha256=sources)
    identity['sha256'] = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return identity


def _existing_result(output, identity):
    if not output.exists() or not any(output.iterdir()):
        return None
    marker = output / 'EXPORT_COMPLETE.json'
    if not marker.is_file():
        raise FileExistsError('Export directory is incomplete; retain it and choose a new output directory: ' + str(output))
    complete = json.loads(marker.read_text(encoding='utf-8'))
    if complete.get('source_identity_sha256') != identity['sha256']:
        raise ValueError('Existing export belongs to different inputs, code or figure options; choose a new output directory')
    expected = complete['files_sha256']
    actual_paths = {p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file() and p != marker}
    if actual_paths != set(expected):
        raise ValueError('Existing export file inventory changed; retain it for inspection')
    for rel, digest in expected.items():
        p = output / rel
        if not p.is_file() or sha(p) != digest:
            raise ValueError('Existing export failed hash verification: ' + rel)
    result = json.loads((output / 'publication_summary.json').read_text(encoding='utf-8'))
    result['reused_verified_export'] = True
    return result


def _export_all(workflow_dir, output_dir, *, input_dir=None, figures=False, log_fn=print):
    """Export one completed v2 workflow; reuse an identical verified export safely.

    Failed attempts remain as sibling ``.attempt_*`` directories. They are never
    mistaken for a completed output, and do not require rerunning numerical work.
    Development benchmarks produce all numerical exports but skip artwork.
    """
    workflow, output = Path(workflow_dir), Path(output_dir)
    expansion = workflow / 'expanded_benchmark'
    summary = json.loads((expansion / 'results/expansion_summary.json').read_text(encoding='utf-8'))
    if summary.get('status') not in ('COMPLETE_INDEPENDENT_DESIGN_EXPANSION', 'COMPLETE_EXPANSION_SMOKE_TEST') or summary.get('designs_completed') != summary.get('designs_planned'):
        raise ValueError('Export requires a completed independent-design benchmark')
    for name in BENCHMARK_TABLES:
        if not (expansion / 'results' / name).is_file():
            raise FileNotFoundError('Missing independent-design output: ' + name)
    for name in DIAGNOSTIC_TABLES + ('report.json',):
        if not (workflow / 'diagnostics' / name).is_file():
            raise FileNotFoundError('Missing weight-diagnostic output: ' + name)
    input_dir = Path(input_dir) if input_dir else workflow / 'analysis/inputs'
    tables = load_fresh_tables(workflow / 'analysis', input_dir=input_dir)
    payload = collect_workbook_payload(tables, metadata={
        'Source': 'Completed integrated numerical analysis. Original three-field benchmark retained separately from the independent-design extension.',
        'Development run': bool(summary.get('development_run')),
    }, strict=True)
    identity = _source_identity(tables, workflow, input_dir, summary, figures)
    reused = _existing_result(output, identity)
    if reused is not None:
        log_fn('PASS verified completed publication exports; no numerical work repeated')
        return reused
    output.parent.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix=output.name + '.attempt_', dir=str(output.parent)))
    try:
        write_json(attempt / 'export_source_identity.json', identity)
        log_fn('START numerical plot tables and manuscript tables from completed outputs')
        plot_rows = make_plot_tables(tables, expansion)
        plot_dir = attempt / 'plot_data'
        for name, rows in plot_rows.items():
            write_csv(plot_dir / (name + '_plot_data.csv'), rows)
        manuscript_dir = attempt / 'manuscript_tables'
        for name, rows in manuscript_tables(tables, expansion).items():
            write_csv(manuscript_dir / (name + '.csv'), rows)
        log_fn('START Supplementary Data S1: original results and independent-design extension')
        supplement = attempt / 'Supplementary_Data_S1'
        original = supplement / 'original_analysis'
        original.mkdir(parents=True)
        companions = export_companions(payload, original / 'tables')
        book = export_workbook(payload, original / 'Supplementary_Data_S1.xlsx')
        for record in companions:
            record['path'] = 'tables/' + record['filename']
        book['path'] = 'Supplementary_Data_S1.xlsx'
        write_json(original / 'table_inventory.json', dict(summary=payload['summary'], companions=companions, workbook=book))
        (original / 'README.txt').write_text('Read the workbook ReadMe, Contents and Data_dictionary sheets. Keep all indexed tables/ companions with the workbook. Every listed result row is retained. Blank values remain unavailable; zeros remain zero.\n', encoding='utf-8')
        extension = supplement / 'independent_design_extension'
        _copy_evidence(expansion / 'results', extension / 'benchmark_tables')
        _copy_evidence(workflow / 'diagnostics', extension / 'temporal_diagnostics')
        provenance = extension / 'protocol_and_provenance'
        provenance.mkdir(parents=True)
        for name in ('expansion_protocol.json', 'numerical_runtime.json'):
            path = expansion / name
            if path.is_file():
                shutil.copyfile(path, provenance / name)
        write_json(provenance / 'export_source_identity.json', identity)
        _copy_evidence(workflow / 'analysis/benchmark/results', supplement / 'original_three_field_reference')
        shutil.copytree(manuscript_dir, supplement / 'manuscript_tables')
        extension_payload, mapping = summary_workbook_payload(expansion, summary)
        extension_book = export_workbook(extension_payload, extension / 'Independent_design_benchmark.xlsx')
        extension_book['path'] = 'independent_design_extension/Independent_design_benchmark.xlsx'
        write_csv(extension / 'workbook_column_mapping.csv', mapping)
        readme = f'''SUPPLEMENTARY DATA S1 — TEMPORAL WEIGHTING AND MODEL DIAGNOSTICS

original_analysis/: all original result families, workbook and complete CSV.gz
companions regenerated from this run. No legacy result is pooled with the new
independent-design benchmark.
independent_design_extension/benchmark_tables/: all 11 extension CSV tables and
metadata. The adjacent workbook is a readable view of all aggregate columns.
independent_design_extension/temporal_diagnostics/: exact supplied-weight
moments, response bounds and complete annual/group-year weight tables.
independent_design_extension/protocol_and_provenance/: protocol and source hashes.
original_three_field_reference/: original benchmark, preserved separately.
manuscript_tables/: the three manuscript tables and display version of Table 3.

This run completed {summary['designs_completed']} independently generated designs,
with {summary['inner_replicates_per_design_scenario']} response replicates per scenario.
Development run: {bool(summary.get('development_run'))}.
Execution timing and cache-reuse records describe the initial successful export.
Verified export reuse preserves these records; timings alone do not create new
scientific results.
Monte Carlo standard errors use independent design summaries, not pooled inner
replicates. Structural generating targets, noiseless fitted-model projections
and event-specific full-mean contrasts are distinct. Coverage is stored on [0,1].
Across-design quantiles are descriptive distributions, not confidence intervals.
Small-support and nonestimable designs remain in planned eligibility denominators;
reported performance conditions on estimability. CSVs preserve full source text
precision. Blank numerical entries are unavailable or inapplicable, never zero.
Simulation results characterize the specified generating distributions, not
real-data interval validity or causal identification. Replicate arrays remain in
the numerical run directory; this supplement contains all design-level summaries.

manifest_sha256.json verifies every other file in this supplementary archive.
'''
        (supplement / 'README.txt').write_text(readme, encoding='utf-8')
        write_json(supplement / 'manifest_sha256.json', _manifest(supplement))
        bundle = attempt / 'Supplementary_Data_S1.zip'
        _zip_tree(supplement, bundle)
        figure_status = 'NOT_REQUESTED'
        figure_artifacts = []
        if figures and summary.get('development_run'):
            figure_status = 'SKIPPED_DEVELOPMENT_BENCHMARK'
            log_fn('Development benchmark: numerical exports retained; publication artwork skipped')
        result = dict(status='COMPLETE_PUBLICATION_EXPORT_V2', development_run=bool(summary.get('development_run')),
            source_identity_sha256=identity['sha256'], reused_verified_export=False,
            figures_requested=bool(figures), figure_status=figure_status,
            figures=8 if figure_status == 'COMPLETE' else 0, panels=36 if figure_status == 'COMPLETE' else 0,
            plot_data_tables=8, manuscript_tables=4, supplementary=payload['summary'],
            independent_designs=summary['designs_completed'], extension_csv_tables=len(BENCHMARK_TABLES),
            extension_workbook=extension_book,
            bundle=dict(filename=bundle.name, sha256=sha(bundle), bytes=bundle.stat().st_size),
            figure_artifacts=figure_artifacts, model_fitting_performed=False)
        write_json(attempt / 'publication_summary.json', result)
        write_json(attempt / 'EXPORT_COMPLETE.json', dict(source_identity_sha256=identity['sha256'], files_sha256=_manifest(attempt)))
        # Empty precreated output directories can be removed; no existing files are replaced.
        if output.exists():
            if any(output.iterdir()):
                raise FileExistsError('Output appeared during export: ' + str(output))
            output.rmdir()
        attempt.rename(output)
        log_fn('PASS v2 numerical exports and requested artwork completed')
        return result
    except Exception as exc:
        write_json(attempt / 'EXPORT_FAILED.json', dict(error=type(exc).__name__ + ': ' + str(exc),
                   completed_numerical_inputs_unchanged=True))
        raise


@contextmanager
def _exclusive_export(output):
    """OS-backed lock automatically released even if the export process is killed."""
    lock = output.with_name(output.name + '.export.lock')
    stream = lock.open('a+b')
    acquired = False
    try:
        if lock.stat().st_size == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError as exc:
            raise RuntimeError('Another export is using this destination: ' + str(output)) from exc
        yield
    finally:
        if acquired:
            stream.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def export_all(workflow_dir, output_dir, *, input_dir=None, figures=False, log_fn=print):
    """Acquire an exclusive export lock, then generate or verify the output."""
    if figures:
        raise ValueError('Figure rendering is available in the local distribution only')
    output = Path(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_export(output):
        return _export_all(workflow_dir, output, input_dir=input_dir, figures=figures, log_fn=log_fn)
