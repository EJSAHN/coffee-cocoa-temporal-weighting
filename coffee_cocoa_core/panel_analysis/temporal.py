"""Finite temporal diagnostics; no outcome-selected model or inference gate."""
from __future__ import annotations
import math
import numpy as np
from .fixed_effects import fit_model, NORMAL_975
from .features import select_data, specifications as original_specifications
from .temporal_features import prepare


def fit(data,spec):
    if not len(data['y']):
        return {'status':'NONESTIMABLE','design':{'reasons':['no_eligible_transition_endpoints']}}
    return fit_model(data['X'],data['y'],data['unit'],data['state'],data['year'],data['names'],
                     unit_trends=spec['unit_trends'],unit_parity=spec['unit_parity'])


def linear_function(result,weights):
    if result.get('status')!='FIT_COMPLETE_EXPLORATORY':
        return dict(status='NONESTIMABLE',source_fit_status=result.get('status'))
    names=list(result['coefficient_order'])
    if len(set(names))!=len(names) or set(weights)-set(names):
        raise ValueError('Invalid linear-functional coefficient identity')
    v=np.array([float(weights.get(n,0.)) for n in names])
    b=np.array([result['coefficients'][n] for n in names],dtype=float)
    if not np.all(np.isfinite(v)) or not np.all(np.isfinite(b)):
        raise ValueError('Nonfinite linear-functional input')
    point=float(v@b);intervals={}
    for label in ('HAC_L2','HAC_L4'):
        cov=np.array(result['uncertainty'][label]['covariance'],dtype=float)
        if cov.shape!=(len(names),len(names)) or not np.all(np.isfinite(cov)):
            raise ValueError('Invalid covariance dimensions or values')
        scale=max(float(np.max(np.abs(cov))),np.finfo(float).tiny)
        if np.max(np.abs(cov-cov.T))>1e-10*scale:
            raise ValueError('Materially asymmetric covariance')
        cov=(cov+cov.T)/2
        raw_variance=float(v@cov@v)
        tolerance=1e-10*max(float(np.abs(v)@np.abs(cov)@np.abs(v)),np.finfo(float).tiny)
        if raw_variance < -tolerance:raise ValueError('Materially negative functional variance')
        variance=max(0.,raw_variance);se=math.sqrt(variance)
        intervals[label]=dict(estimate=point,variance=variance,standard_error=se,
            lower=point-NORMAL_975*se,upper=point+NORMAL_975*se,critical_value=NORMAL_975)
    return dict(status='COMPLETE_LINEAR_FUNCTION',estimate=point,weights=dict(weights),uncertainty=intervals)


def reproduce_anchors(panel,reference_rows,saved_models,log):
    keys={(r['crop'],str(r['territory']),int(r['year'])) for r in reference_rows}
    results=[]
    for crop in ('coffee_total','cacao'):
        spec=next(s for s in original_specifications() if s['crop']==crop and s['outcome']=='production_fixed_ha' and s['variant']=='primary')
        mid='common_fit__'+spec['model_id']
        log('Reproduce common-support common-support anchor '+crop)
        data=select_data(panel,spec)
        keep=np.array([(crop,str(r['territory']),int(r['year'])) in keys for r in data['rows']],dtype=bool)
        restricted={k:(data[k][keep] if k in ('X','y','unit','state','year') else data[k]) for k in data}
        expected_count=sum(k[0]==crop for k in keys)
        if int(keep.sum())!=expected_count:raise ValueError('common-support anchor support did not reproduce')
        old=saved_models[mid];new=fit(restricted,spec)
        if old['status']!='FIT_COMPLETE_EXPLORATORY' or new['status']!='FIT_COMPLETE_EXPLORATORY':
            raise ValueError('Complete common-support anchor did not reproduce')
        names=old['coefficient_order']
        if names!=new['coefficient_order']:raise ValueError('Anchor coefficient order changed')
        a=np.array([old['coefficients'][n] for n in names]);b=np.array([new['coefficients'][n] for n in names])
        if not np.allclose(a,b,rtol=1e-8,atol=1e-8):raise ValueError('common-support anchor coefficients differ: '+crop)
        cov_error=0.
        for label in ('HAC_L2','HAC_L4'):
            a_cov=np.array(old['uncertainty'][label]['covariance']);b_cov=np.array(new['uncertainty'][label]['covariance'])
            if not np.allclose(a_cov,b_cov,rtol=1e-8,atol=1e-8):raise ValueError('common-support anchor covariance differs: '+crop)
            cov_error=max(cov_error,float(np.max(np.abs(a_cov-b_cov))))
        for field in ('n_observations','n_units','n_states','n_calendar_years','unit_linear_trends','unit_specific_fixed_parity'):
            if old['design']['projection'][field]!=new['design']['projection'][field]:
                raise ValueError('common-support anchor support/nuisance differs: '+field)
        results.append(dict(model_id=mid,crop=crop,status='PASS',n_rows=expected_count,
           coefficient_max_abs_error=float(np.max(np.abs(a-b))),covariance_max_abs_error=cov_error,
           relative_tolerance=1e-8,absolute_tolerance=1e-8))
    return results


def function_rows(result,weights,quantity,spec,data,reference):
    estimate=linear_function(result,weights)
    base=dict(model_id=spec['model_id'],crop=spec['crop'],outcome=spec['outcome'],variant=spec['variant'],
              quantity=quantity,status=estimate['status'],endpoint_rows=len(data['y']),
              reference_history_mean=reference.get('history_mean'),
              reference_history_q25=reference.get('history_q25'),reference_history_q75=reference.get('history_q75'),
              interpretation='Underlying level-response coefficients; conditional exploratory association, not recovery causality',
              confidence_intervals_pointwise=True,multiple_comparisons_adjusted=False)
    base['display_scale']=1. if quantity=='interaction_gamma' else .1
    base['outcome_unit']='kg_per_baseline_ha' if spec['outcome']=='production_fixed_ha' else 'area_divided_by_baseline_ha'
    base['raw_gamma_unit']='outcome_unit_per_D_squared' if quantity=='interaction_gamma' else ''
    if estimate['status']!='COMPLETE_LINEAR_FUNCTION':return [base]
    return [dict(base,uncertainty=label,**ci) for label,ci in estimate['uncertainty'].items()]


def run(panel,common_reference_rows,saved_models,log=lambda message:None,reproduce_source=True):
    anchors=reproduce_anchors(panel,common_reference_rows,saved_models,log) if reproduce_source else []
    prepared=prepare(panel,common_reference_rows)
    models={};contrasts=[];leads=[];registry=[]
    for n,spec in enumerate(prepared['specs'],1):
        log('Temporal comparison '+str(n)+'/20 '+spec['model_id'])
        data=prepared['data_by_model'][spec['model_id']]
        ref=prepared['reference_by_crop'][spec['crop']]
        result=fit(data,spec)
        models[spec['model_id']]=dict(result,specification=spec)
        registry.append(dict(spec,status=result['status'],n_endpoint_rows=len(data['y']),
           n_units=len(set(data['unit'].tolist())),n_states=len(set(data['state'].tolist())),
           n_endpoint_years=len(set(data['year'].tolist())),
           outcome_form='Y_t_minus_Y_t_minus_1' if spec['variant']=='first_difference' else 'Y_t',
           covariate_form='Difference_of_each_original_complete_column' if spec['variant']=='first_difference' else 'Original_level_columns'))
        h=ref.get('history_mean');q25=ref.get('history_q25');q75=ref.get('history_q75')
        current={'D0':.1,'D0_x_history':.1*h} if h is not None else {}
        gap={'D0_x_history':.1*(q75-q25)} if q25 is not None and q75 is not None else {}
        for quantity,weights in [('current_D_derivative_times_0_1',current),
             ('interaction_gamma',{'D0_x_history':1.}),
             ('current_D_slope_difference_history_q75_minus_q25',gap)]:
            contrasts.extend(function_rows(result,weights,quantity,spec,data,ref))
        if spec['variant']=='levels_unit_trend_future':
            for name in ('D_lead1','D_lead2','T_lead1','T_lead2'):
                rr=function_rows(result,{name:.1},name+'_coefficient_times_0_1',spec,data,ref)
                for row in rr:
                    row['interpretation']='Future-climate temporal diagnostic; not a randomized placebo or causal pass/fail test'
                leads.extend(rr)
    if len(models)!=20:raise ValueError('Temporal model inventory differs from fixed twenty')
    baseline={(r['crop'],r['outcome'],r.get('uncertainty')):r for r in contrasts
        if r['quantity']=='current_D_derivative_times_0_1' and r['variant']=='levels'}
    comparisons=[]
    for r in contrasts:
        if r['quantity']!='current_D_derivative_times_0_1':continue
        row=dict(r);base=baseline.get((r['crop'],r['outcome'],r.get('uncertainty')),{})
        if 'estimate' in r and 'estimate' in base:
            row.update(levels_reference_estimate=base['estimate'],descriptive_change_from_levels=r['estimate']-base['estimate'],
                       interval_for_between_model_difference_computed=False)
        comparisons.append(row)
    return dict(models=models,contrast_rows=contrasts,current_slope_comparison_rows=comparisons,
       lead_rows=leads,reproduction_rows=anchors,prepared=prepared,registry_rows=registry)
