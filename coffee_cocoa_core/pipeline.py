"""Reproduce the fixed municipal panel analyses and simulation benchmark."""
from __future__ import annotations
import datetime as dt
import json
from pathlib import Path
import time
import traceback
from . import __version__
from .io import sha256, write_json, write_manifest, write_table


def log(message):
    print(time.strftime('%H:%M:%S')+' '+str(message), flush=True)


def run(input_dir, output_dir, *, replicates=400, seed=20261005, log_fn=log):
    """Fit all analyses using source data or a verified prepared input directory."""
    source, out = Path(input_dir), Path(output_dir)
    if not source.is_dir():
        raise FileNotFoundError('Input directory does not exist: '+str(source))
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Output directory must be new or empty: '+str(out))
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    summary = dict(status='RUNNING',software_version=__version__,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        seed=seed,replicates=replicates,default_benchmark=(replicates==400 and seed==20261005),
        causal_effect_identified=False,real_world_interval_coverage_validated=False,
        historical_geography_harmonized=False)
    write_json(out/'analysis_configuration.json',dict(version=__version__,
        annual_years=[2002,2025],climate_years=[1980,2025],reference_years=[1981,2010],
        temporal_origin=2006,benchmark_replicates=replicates,benchmark_seed=seed,
        analysis_families=['panel_models','response_comparison','temporal_comparison',
                           'event_contrasts','temporal_weights','benchmark']))
    try:
        from .data_import import load_inputs, prepare_inputs
        if (source/'input_manifest.json').is_file():
            prepared = source
        elif (source/'source_manifest.json').is_file():
            prepared = out/'inputs'
            log_fn('START prepare annual production and climate inputs')
            prepare_inputs(source,prepared,log=log_fn)
        else:
            raise FileNotFoundError('Input directory requires source_manifest.json or input_manifest.json')
        inputs = load_inputs(prepared,log=log_fn)
        summary['input_manifest_sha256'] = sha256(prepared/'input_manifest.json')
        summary['input_counts'] = inputs['summary']['counts']
        from .event_designs import load_event_designs
        legacy = load_event_designs(prepared,verify=False)
        from .event_detection import run as reconstruct_events
        log_fn('START monthly climate and event-ledger reconstruction')
        summary['event_reconstruction'] = reconstruct_events(
            prepared,inputs['panel'],legacy['designs'],out/'event_reconstruction',log=log_fn)
        from .panel_analysis.workflow import run_panel
        log_fn('START municipal panel analyses')
        summary['panel'] = run_panel(inputs['panel'],out,log=log_fn,do_influence=True)
        from .event_contrasts import run as run_events
        log_fn('START fixed-support event contrasts')
        summary['events'] = run_events(inputs['panel'],legacy['designs'],out/'event_contrasts')
        from .temporal import analyse
        log_fn('START exact temporal weight identities')
        temporal = analyse(legacy['designs'],log=log_fn)
        for filename,key in [('temporal_design_summary.csv','design_rows'),
                             ('temporal_unit_moments.csv.gz','unit_rows'),
                             ('temporal_common_year_profiles.csv','year_rows'),
                             ('temporal_state_year_profiles.csv.gz','state_year_rows')]:
            write_table(out/'temporal_weights'/filename,temporal[key])
        write_json(out/'temporal_weights/exact_temporal_validation.json',temporal['validation'])
        write_table(out/'temporal_weights/frozen_legacy_design_registry.csv',legacy['registry'])
        summary['temporal'] = temporal['validation']
        from .benchmark import run_benchmark
        log_fn('START known-truth simulation benchmark')
        benchmark = run_benchmark(out/'benchmark',replicates=replicates,seed=seed,log_fn=log_fn)
        summary['benchmark'] = {key:value for key,value in benchmark.items() if key!='output_paths'}
        summary['status'] = 'COMPLETE_ANALYSIS' if summary['default_benchmark'] else 'COMPLETE_MODIFIED_BENCHMARK'
        if (summary['panel']['models_nonestimable']
                or summary['events']['complete_designs']!=len(legacy['designs'])
                or benchmark.get('nonestimable_scenario_methods',0)):
            summary['status'] = 'COMPLETE_WITH_UNAVAILABLE_ESTIMATES'
        return summary
    except BaseException as exc:
        summary.update(status='INCOMPLETE_ANALYSIS',error=type(exc).__name__+': '+str(exc))
        (out/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
    finally:
        summary['elapsed_seconds'] = round(time.monotonic()-started,3)
        summary['finished_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(out/'analysis_summary.json',summary)
        write_manifest(out)
