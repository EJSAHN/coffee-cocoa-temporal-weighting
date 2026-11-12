"""Run the complete fixed panel, scale, and temporal analysis from source inputs.

No model estimates or result tables are accepted as analytical inputs. Every
regression is fitted from the canonical annual panel during the current run.
"""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
import time

import numpy as np

from . import baseline
from .features import OUTCOMES, select_data, specifications, support_counts
from .response import run_models
from .components import allocate_association
from .trajectories import analyse_trajectories
from .temporal import run as run_temporal_models
from .software_checks import run_independent_checks


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def write_table(path, rows, fields=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if fields is None:fields=list(dict.fromkeys(k for row in rows for k in row))
    opener=gzip.open if path.name.endswith('.gz') else open
    with opener(path,'wt',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='raise')
        writer.writeheader();writer.writerows(rows)


def _base_models(panel, directory, log, do_influence):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    specs=specifications();write_json(directory/'model_specifications.json',specs)
    coefficients=[];excluded=[];observed=[];registry=[];marginals=[];fits={};primary_data={}
    for number,spec in enumerate(specs,1):
        log('Panel model '+str(number)+'/'+str(len(specs))+' '+spec['model_id'])
        data=select_data(panel,spec);fit=baseline.run_fit(data,spec);fit['specification']=spec
        fits[spec['model_id']]=fit
        write_json(directory/'models'/(spec['model_id']+'.json'),fit)
        coefficients.extend(baseline.coef_rows(fit,spec))
        excluded.extend(data['excluded']);observed.extend(support_counts(data,spec))
        marginals.extend(baseline.observed_marginal_summaries(data,fit,spec))
        registry.append(dict(**spec,status=fit['status'],n_observations=len(data['y']),
            n_units=len(set(data['unit'].tolist())),n_states=len(set(data['state'].tolist())),
            n_years=len(set(data['year'].tolist())),zero_outcomes_retained=int(np.sum(data['y']==0)),
            excluded_rows=len(data['excluded']),scaled_condition_number=fit.get('design',{}).get('scaled_condition_number'),
            reasons=';'.join(fit.get('design',{}).get('reasons',[]))))
        if spec['variant']=='primary' and spec['outcome']==OUTCOMES[0]:
            primary_data[spec['crop']]=(data,spec)
    contributions=[];validations=[];influence=[]
    for crop,(data,spec) in primary_data.items():
        fit=fits[spec['model_id']]
        if fit['status']!='FIT_COMPLETE_EXPLORATORY':continue
        log('Primary interaction arithmetic '+crop)
        c,v=baseline.independently_reconcile_interaction(data,fit,spec)
        contributions.extend(c);validations.append(v)
        if do_influence:influence.extend(baseline.leave_group_diagnostics(data,fit,spec,log))
    comparison=[]
    for crop in ('coffee_total','cacao'):
        for outcome in OUTCOMES:
            a=fits['__'.join((crop,outcome,'primary'))]
            b=fits['__'.join((crop,outcome,'common_component_rows'))]
            comparison.append(dict(crop=crop,outcome=outcome,
                original_gamma=a.get('coefficients',{}).get('D0_x_history'),
                common_rows_gamma=b.get('coefficients',{}).get('D0_x_history'),
                original_n=a.get('design',{}).get('projection',{}).get('n_observations'),
                common_rows_n=b.get('design',{}).get('projection',{}).get('n_observations'),
                original_status=a['status'],common_rows_status=b['status']))
    write_table(directory/'coefficients.csv',coefficients,['model_id','crop','outcome','variant','role','term','estimate','standard_error','lower','upper','uncertainty','reference_label','n_rows','n_units','n_states','n_years','causal_effect'])
    write_table(directory/'model_registry.csv',registry)
    write_table(directory/'exclusions.csv.gz',excluded,['model_id','territory','crop','state','year','reasons'])
    write_table(directory/'observed_support.csv',observed,['model_id','crop','outcome','variant','state','year','observed_rows'])
    write_table(directory/'marginal_associations.csv',marginals,['model_id','crop','outcome','summary','history_value','estimate','standard_error','lower','upper','uncertainty','meaning'])
    write_table(directory/'primary_interaction_contributions.csv',contributions,['model_id','crop','dimension','group','coefficient_contribution','identification_share','n_rows'])
    write_table(directory/'primary_arithmetic_validation.csv',validations,['model_id','status','single_regressor_fwl_estimate','multivariate_estimate','absolute_error','contribution_sum','interpret_as_arithmetic_not_causal_contribution'])
    write_table(directory/'leave_one_group.csv',influence,['model_id','crop','omitted_dimension','omitted_group','status','n_rows','gamma','primary_gamma','change_from_primary','changes_primary_sample','reasons'])
    write_table(directory/'common_component_support_comparison.csv',comparison)
    availability=[]
    for crop in ('coffee_total','cacao'):
        rows=[r for r in panel if r['crop']==crop and 2007<=r['year']<=2025]
        availability.append(dict(crop=crop,frozen_followup_rows=len(rows),
            source_discordant_rows=sum(bool(r['source_discordant']) for r in rows),
            missing_production=sum(r['Q_tonnes'] is None for r in rows),
            missing_intended_area=sum(r['Iha'] is None for r in rows),
            missing_harvested_area=sum(r['Hha'] is None for r in rows),
            harvested_exceeds_intended=sum(r['harvested_exceeds_intended'] for r in rows),
            zero_production=sum(r['Q_tonnes']==0 for r in rows),
            zero_intended_area=sum(r['Iha']==0 for r in rows),
            zero_harvested_area=sum(r['Hha']==0 for r in rows)))
    write_table(directory/'outcome_availability.csv',availability)
    completed=sum(r['status']=='FIT_COMPLETE_EXPLORATORY' for r in registry)
    summary=dict(models_planned=len(specs),models_fitted=completed,models_nonestimable=len(specs)-completed,
                 leave_group_refits=len(influence),independent_coefficient_validations=validations)
    write_json(directory/'summary.json',summary)
    return dict(fits=fits,specifications=specs,summary=summary)


def _response_models(panel, fitted, directory, log, do_influence):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    # The compared fits were produced above during this run, never loaded from a
    # prior results archive. Independent repeated fits check handoff consistency.
    result=run_models(panel,fitted['fits'],fitted['specifications'],log,do_influence)
    for mid,fit in result['models'].items():write_json(directory/'models'/(mid+'.json'),fit)
    write_table(directory/'panel_fit_reconciliation.csv',result['reproduction_rows'])
    write_table(directory/'common_reference_registry.csv',result['reference_rows'])
    write_table(directory/'original_model_current_slopes.csv',[r for r in result['slope_rows'] if r['family']=='verified_original_panel'])
    write_table(directory/'common_fit_current_slopes.csv',[r for r in result['slope_rows'] if r['family']=='common_observed_rows'])
    write_table(directory/'current_slope_group_contributions.csv',result['group_contribution_rows'])
    write_table(directory/'current_slope_row_contributions.csv.gz',result['contribution_rows'])
    write_table(directory/'current_slope_arithmetic_validation.csv',result['arithmetic_validations'])
    write_table(directory/'current_slope_leave_one_group.csv',result['influence_rows'])
    log('Production and area trajectory arithmetic')
    trajectory=analyse_trajectories(panel)
    for field,filename in [('annual_unit_rows','annual_unit_trajectories.csv.gz'),
        ('annual_summary_rows','annual_state_crop_summary.csv'),('transition_rows','unit_transitions.csv.gz'),
        ('transition_summary_rows','transition_summary.csv'),('selected_case_rows','posthoc_cacao_case_trajectories.csv'),
        ('level_component_rows','cumulative_production_components.csv.gz')]:
        write_table(directory/filename,trajectory[field])
    write_json(directory/'trajectory_arithmetic_validation.json',trajectory['validation'])
    allocated=allocate_association(result['primary_weights'],trajectory['level_component_rows'])
    write_table(directory/'production_association_component_allocation.csv',allocated['summary_rows'])
    write_table(directory/'production_association_component_row_allocations.csv.gz',allocated['row_allocations'])
    nonestimable=sum(r['status']!='FIT_COMPLETE_EXPLORATORY' for r in result['common_models'].values())
    summary=dict(common_support_models_planned=11,common_support_models_fitted=11-nonestimable,
        common_support_models_nonestimable=nonestimable,
        reference_rows_by_crop={c:len(k) for c,k in result['common_keys'].items()},
        current_slope_contributions_validated=len(result['arithmetic_validations']),
        leave_group_refits=len({(r['crop'],r['omitted_dimension'],r['omitted_group']) for r in result['influence_rows']}),
        trajectory_validation=trajectory['validation'],production_association_allocations=allocated['summary_rows'])
    write_json(directory/'summary.json',summary)
    return dict(common_models=result['common_models'],reference_rows=result['reference_rows'],summary=summary)


def _temporal_models(panel, previous, directory, log):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    result=run_temporal_models(panel,previous['reference_rows'],previous['common_models'],log)
    prepared=result['prepared']
    for mid,fit in result['models'].items():write_json(directory/'models'/(mid+'.json'),fit)
    for name,field in [('temporal_contrasts.csv','contrast_rows'),('current_slope_comparison.csv','current_slope_comparison_rows'),
        ('future_climate_diagnostics.csv','lead_rows'),('model_registry.csv','registry_rows'),
        ('common_support_anchor_reconciliation.csv','reproduction_rows')]:
        write_table(directory/name,result[field])
    write_table(directory/'transition_support_registry.csv',prepared['support_rows'])
    write_table(directory/'transition_support_exclusions.csv',prepared['exclusion_rows'])
    write_json(directory/'transition_support_summary.json',prepared['summary'])
    for mid,data in prepared['data_by_model'].items():
        rows=[]
        for j in range(len(data['y'])):
            row=dict(territory=str(data['unit'][j]),state=str(data['state'][j]),year=int(data['year'][j]),response=float(data['y'][j]))
            row.update({name:float(data['X'][j,k]) for k,name in enumerate(data['names'])});rows.append(row)
        write_table(directory/'designs'/(mid+'.csv.gz'),rows,['territory','state','year','response',*data['names']])
    reference={crop:dict({k:v for k,v in ref.items() if k not in ('X','names','rows')},reference_n=len(ref['X']))
               for crop,ref in prepared['reference_by_crop'].items()}
    write_json(directory/'fixed_evaluation_reference.json',reference)
    nonestimable=sum(r['status']!='FIT_COMPLETE_EXPLORATORY' for r in result['models'].values())
    summary=dict(models_planned=20,models_fitted=20-nonestimable,models_nonestimable=nonestimable,
        future_diagnostic_models=2,endpoint_years=[2008,2023],differenced_outcome_source_years=[2007,2023],
        same_eligible_transition_endpoints=True,identical_underlying_outcome_observations=False,
        differenced_interaction_is_difference_of_original_product=True,
        future_diagnostic_is_randomized_placebo=False,support=prepared['summary'],evaluation_reference=reference)
    write_json(directory/'summary.json',summary)
    return summary


def run_panel(panel, out_dir, log=print, do_influence=True):
    """Execute the fixed analysis on an already validated canonical panel.

    The boolean switch exists for small software fixtures. The public file-based
    entry point always executes the complete influence diagnostics.
    """
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    checks=run_independent_checks();write_json(out/'software_validation.json',checks)
    base=_base_models(panel,out/'panel_models',log,do_influence)
    response=_response_models(panel,base,out/'response_comparison',log,do_influence)
    temporal=_temporal_models(panel,response,out/'temporal_comparison',log)
    fitted=base['summary']['models_fitted']+response['summary']['common_support_models_fitted']+temporal['models_fitted']
    summary=dict(status='COMPLETE_PANEL_ANALYSIS' if fitted==56 else 'COMPLETE_WITH_NONESTIMABLE_MODELS',
        models_planned=56,models_fitted=fitted,models_nonestimable=56-fitted,
        model_coefficients_read_from_previous_results=False,all_models_fitted_from_current_input_panel=True,
        influence_diagnostics_performed=bool(do_influence),elapsed_seconds=round(time.monotonic()-started,3),
        families=dict(panel_models=base['summary'],response_comparison=response['summary'],temporal_comparison=temporal),
        output_directories=dict(panel_models='panel_models',response_comparison='response_comparison',temporal_comparison='temporal_comparison'),
        causal_effect_identified=False,real_world_interval_coverage_validated=False,
        historical_geography_harmonized=False,p_values_computed=False,software_validation=checks)
    write_json(out/'panel_analysis_summary.json',summary)
    return summary


def run(input_dir, out_dir, log=print):
    """Load hash-verified canonical inputs and fit all specified panel analyses."""
    from ..data_import import load_inputs
    inputs=load_inputs(input_dir)
    return run_panel(inputs['panel'],out_dir,log=log,do_influence=True)
