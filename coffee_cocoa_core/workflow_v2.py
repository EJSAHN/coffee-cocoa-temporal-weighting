"""Unified numerical workflow for the municipal production study.

The original scientific implementation remains in ``pipeline``. This module
connects it to the portable weight diagnostics and fixed independent-design
benchmark. A completed empirical analysis is reused only after its input,
software, numerical-runtime and output fingerprints have been verified.
"""
from __future__ import annotations

from contextlib import contextmanager
import datetime as dt
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import uuid

from .io import read_table, sha256, write_json, write_manifest


VERSION = 'coffee_cocoa_unified_v2_1'


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _stamp():
    return dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:10]


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.pending_' + uuid.uuid4().hex)
    write_json(temporary, value)
    os.replace(temporary, path)


@contextmanager
def _exclusive_workflow(output):
    """OS-backed lock, released on process exit, including an interrupted run."""
    path = output / '.workflow.lock'
    stream = path.open('a+b')
    acquired = False
    try:
        if path.stat().st_size == 0:
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
            raise RuntimeError('Another process is using this workflow output directory: ' + str(output)) from exc
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


def _analysis_identity(source, smoke):
    """Check actual input bytes, rather than trusting an unchanged manifest."""
    import numpy as np
    from .data_import import verify_manifest, RAW_FILES
    from .event_designs import FILES
    if (source / 'input_manifest.json').is_file():
        name = 'input_manifest.json'
        required = ('annual_production.csv.gz', 'annual_panel.csv.gz', 'climate_annual.csv.gz',
                    'cohort.csv', 'climate_monthly.csv.gz', *FILES.values())
    elif (source / 'source_manifest.json').is_file():
        name, required = 'source_manifest.json', RAW_FILES
    else:
        raise FileNotFoundError('Input directory requires input_manifest.json or source_manifest.json')
    manifest = verify_manifest(source, name, required)
    core = Path(__file__).resolve().parent
    hashes = {p.relative_to(core).as_posix(): sha256(p)
              for p in sorted(core.rglob('*.py'))
              if 'tests' not in p.relative_to(core).parts and p.name != 'workflow_v2.py'}
    return dict(cache_contract=VERSION, input_manifest_name=name,
                input_manifest_sha256=sha256(source / name), input_files=manifest['sha256'],
                scientific_core_sha256=hashes, numpy=np.__version__,
                python_implementation=sys.implementation.name,
                python_version=list(sys.version_info[:3]),
                original_benchmark_replicates=2 if smoke else 400,
                original_benchmark_seed=20261005, development_run=smoke)


def _verify_complete_analysis(directory, identity):
    from .data_import import verify_manifest
    completion = _read_json(directory / 'workflow_completion.json')
    if completion.get('identity') != identity:
        raise ValueError('Saved empirical analysis input, software or runtime differs; use a new output directory')
    manifest = verify_manifest(directory, 'output_manifest.json',
                               ('analysis_summary.json', 'workflow_completion.json'))
    # Nested manifests are ordinary files; only the top-level manifest is omitted.
    present = {p.relative_to(directory).as_posix() for p in directory.rglob('*')
               if p.is_file() and '__pycache__' not in p.parts
               and p.relative_to(directory).as_posix() != 'output_manifest.json'}
    if present != set(manifest['sha256']):
        raise ValueError('Saved empirical analysis file inventory differs from its completion manifest')
    summary = _read_json(directory / 'analysis_summary.json')
    if not summary.get('status', '').startswith('COMPLETE_'):
        raise ValueError('Empirical analysis does not have a completed status')
    if sha256(directory / 'analysis_summary.json') != completion['analysis_summary_sha256']:
        raise ValueError('Empirical completion marker differs from its summary')
    return summary


def _empirical_analysis(source, output, identity, smoke, log_fn):
    from .pipeline import run as run_original
    analysis = output / 'analysis'
    if analysis.exists():
        summary = _verify_complete_analysis(analysis, identity)
        log_fn('PASS verified completed empirical analysis; reuse unchanged results')
        return summary, True
    attempt = output / 'analysis_attempts' / _stamp()
    attempt.parent.mkdir(parents=True, exist_ok=True)
    log_fn('START complete empirical analyses and original three-design benchmark')
    summary = run_original(source, attempt, replicates=2 if smoke else 400,
                           seed=20261005, log_fn=log_fn)
    # Check source files a second time before publishing a reusable completion.
    if _analysis_identity(source, smoke) != identity:
        raise ValueError('Input or scientific code changed during computation; incomplete attempt retained')
    write_json(attempt / 'workflow_completion.json',
               dict(identity=identity, analysis_summary_sha256=sha256(attempt / 'analysis_summary.json')))
    write_manifest(attempt)
    _verify_complete_analysis(attempt, identity)
    os.replace(attempt, analysis)
    return summary, False


def diagnostic_records(designs):
    """Adapt freshly reconstructed rational coefficients without reading outcomes."""
    records = []
    for design in designs:
        groups = {row['territory']: row['state'] for row in design['units']}
        for (unit, year), weight in sorted(design['weights_exact'].items()):
            records.append(dict(contrast=design['design_id'], unit=unit, group=groups[unit],
                                year=year, weight=str(weight)))
    return records


def validate_diagnostics(result, unit_reference, design_reference):
    """Compare generic moment/norm arithmetic against the independent core implementation."""
    mapping = {'m0': 'constant', 'm1': 'linear', 'm2': 'quadratic', 'mp': 'parity'}
    actual_units = {(r['contrast'], r['unit']): r for r in result['unit_moments']}
    expected_units = {(r['design_id'], r['territory']): r for r in unit_reference}
    if len(actual_units) != len(result['unit_moments']) or len(expected_units) != len(unit_reference):
        raise AssertionError('Duplicate unit-level temporal diagnostic')
    if set(actual_units) != set(expected_units):
        raise AssertionError('Generic diagnostic unit support differs from regenerated core support')
    count = 0
    for key, expected in expected_units.items():
        for old, new in mapping.items():
            if Fraction(actual_units[key][new + '_exact']) != Fraction(expected[old + '_exact']):
                raise AssertionError('Generic unit moment differs: ' + str(key) + ' ' + old)
            count += 1
    summaries = {r['design_id']: r for r in design_reference}
    if len(summaries) != len(design_reference) or set(summaries) != {key[0] for key in actual_units}:
        raise AssertionError('Generic diagnostic contrast inventory differs')
    for design_id, expected in summaries.items():
        units = [row for key, row in actual_units.items() if key[0] == design_id]
        if len(units) != int(expected['n_units']):
            raise AssertionError('Generic diagnostic eligible-unit count differs')
        for old, new in mapping.items():
            values = [Fraction(row[new + '_exact']) for row in units]
            checks = dict(sum=sum(values, Fraction(0)), l1=sum(map(abs, values), Fraction(0)),
                          max_abs=max(map(abs, values), default=Fraction(0)))
            for statistic, value in checks.items():
                if value != Fraction(expected[old + '_' + statistic + '_exact']):
                    raise AssertionError('Generic design moment differs: ' + design_id + ' ' + old + ' ' + statistic)
                count += 1
    return dict(status='PASS', contrasts=len(summaries), unit_contrast_rows=len(actual_units),
                exact_fraction_comparisons=count, outcome_values_used=False,
                comparison='Generic supplied-weight diagnostics versus freshly regenerated core exact moments')


def _diagnostics(prepared, analysis, output, log_fn):
    from .event_designs import load_event_designs
    from temporal_diagnostics import diagnose, write_report
    log_fn('START portable diagnostics from reconstructed exact event coefficients')
    designs = load_event_designs(prepared, verify=False)['designs']
    result = diagnose(diagnostic_records(designs), origin=2006)
    comparison = validate_diagnostics(result,
        read_table(analysis / 'temporal_weights/temporal_unit_moments.csv.gz'),
        read_table(analysis / 'temporal_weights/temporal_design_summary.csv'))
    target = output / 'diagnostics'
    temporary = output / ('diagnostics_pending_' + uuid.uuid4().hex[:12])
    write_report(temporary, result)
    write_json(temporary / 'core_comparison.json', comparison)
    if target.exists():
        expected = {p.relative_to(temporary).as_posix(): sha256(p) for p in temporary.rglob('*') if p.is_file()}
        actual = {p.relative_to(target).as_posix(): sha256(p) for p in target.rglob('*') if p.is_file()}
        if actual != expected:
            raise ValueError('Existing diagnostic differs from reproduced report; pending files retained for inspection')
        shutil.rmtree(temporary)
    else:
        os.replace(temporary, target)
    log_fn('PASS generic diagnostics: ' + str(comparison['exact_fraction_comparisons']) + ' exact moment and norm comparisons')
    return comparison


def run(input_dir, output_dir, *, smoke=False, log_fn=print):
    """Run or resume the complete fixed numerical workflow.

    ``smoke`` uses two replicates for the three original designs and two new
    designs with four replicates. These outputs are marked development-only.
    Production settings are fixed: 3 x 9 x 400 and 100 x 9 x 40 response datasets.
    Input files are never changed. Failed attempts remain in the output tree.
    Figures and supplementary exports are performed by the calling application.
    """
    if not isinstance(smoke, bool):
        raise TypeError('smoke must be an explicit boolean')
    source, output = Path(input_dir).resolve(), Path(output_dir).resolve()
    if not source.is_dir():
        raise FileNotFoundError('Input directory does not exist: ' + str(source))
    if source == output or output in source.parents:
        raise ValueError('Output must not equal or contain the input directory')
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with _exclusive_workflow(output):
        summary = dict(status='RUNNING_UNIFIED_ANALYSIS', workflow_version=VERSION,
                       development_run=smoke, input_dir=str(source), output_dir=str(output),
                       causal_effect_identified=False, real_world_interval_coverage_validated=False,
                       historical_geography_harmonized=False)
        try:
            identity = _analysis_identity(source, smoke)
            summary['empirical_identity_sha256'] = hashlib.sha256(_canonical(identity)).hexdigest()
            analysis_summary, reused = _empirical_analysis(source, output, identity, smoke, log_fn)
            prepared = source if (source / 'input_manifest.json').is_file() else output / 'analysis/inputs'
            summary.update(analysis=analysis_summary, analysis_reused_verified=reused,
                           prepared_input_dir=str(prepared), analysis_dir=str(output / 'analysis'))
            summary['diagnostics'] = _diagnostics(prepared, output / 'analysis', output, log_fn)
            from .benchmark.simulation_models import run_independent_checks
            validation = run_independent_checks()
            _atomic_json(output / 'independent_model_software_check.json', validation)
            log_fn('PASS independent model arithmetic; START independent-design expansion')
            from expanded_benchmark import run_expansion
            summary['expansion'] = run_expansion(output / 'expanded_benchmark', smoke=smoke, log_fn=log_fn)
            incomplete = (analysis_summary['status'] == 'COMPLETE_WITH_UNAVAILABLE_ESTIMATES'
                          or summary['expansion'].get('nonestimable_scenario_methods', 0))
            summary['status'] = ('COMPLETE_UNIFIED_DEVELOPMENT_SMOKE' if smoke else
                                 'COMPLETE_UNIFIED_ANALYSIS_WITH_UNAVAILABLE_ESTIMATES' if incomplete else
                                 'COMPLETE_UNIFIED_ANALYSIS')
            summary['scientific_results_for_manuscript'] = not smoke
            log_fn(summary['status'])
            return summary
        except BaseException as exc:
            summary.update(status='INCOMPLETE_UNIFIED_ANALYSIS', error=type(exc).__name__ + ': ' + str(exc))
            failure = output / ('failed_workflow_' + _stamp() + '.json')
            _atomic_json(failure, dict(summary, traceback=traceback.format_exc()))
            raise
        finally:
            summary['elapsed_seconds_this_invocation'] = round(time.monotonic() - started, 3)
            summary['finished_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
            _atomic_json(output / 'workflow_summary.json', summary)
