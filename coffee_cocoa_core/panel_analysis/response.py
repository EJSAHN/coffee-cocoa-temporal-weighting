"""Finite common-support comparisons using unchanged full panel model definitions."""
from __future__ import annotations
from collections import defaultdict
import math
import numpy as np
from .features import select_data, OUTCOMES
from .fixed_effects import fit_model
from .contrasts import current_slope,generate_contributions


def key(row):return str(row['territory']),int(row['year'])


def subset(data,mask):
    mask=np.asarray(mask,dtype=bool)
    return dict(X=data['X'][mask],y=data['y'][mask],names=list(data['names']),
        unit=data['unit'][mask],state=data['state'][mask],year=data['year'][mask],
        rows=[r for r,m in zip(data['rows'],mask) if m],excluded=[])


def fit(data,spec):
    if not len(data['y']):return {'status':'NONESTIMABLE','design':{'reasons':['no_observed_rows']}}
    return fit_model(data['X'],data['y'],data['unit'],data['state'],data['year'],data['names'],
                     unit_trends=spec['unit_trends'],unit_parity=spec['unit_parity'])


def reproduce(saved,new,spec):
    if saved['status']!='FIT_COMPLETE_EXPLORATORY' or new['status']!='FIT_COMPLETE_EXPLORATORY':
        raise ValueError('Previously complete full panel model did not reproduce: '+spec['model_id'])
    if saved['coefficient_order']!=new['coefficient_order']:raise ValueError('Coefficient order changed')
    order=saved['coefficient_order']
    a=np.array([saved['coefficients'][n] for n in order]);b=np.array([new['coefficients'][n] for n in order])
    if not np.allclose(a,b,rtol=1e-8,atol=1e-8):raise ValueError('full panel coefficients differ: '+spec['model_id'])
    cov_error=0.
    for label in ('HAC_L2','HAC_L4'):
        x=np.array(saved['uncertainty'][label]['covariance']);y=np.array(new['uncertainty'][label]['covariance'])
        if not np.allclose(x,y,rtol=1e-8,atol=1e-8):raise ValueError('full panel covariance differs: '+spec['model_id'])
        cov_error=max(cov_error,float(np.max(np.abs(x-y))))
    for field in ('n_observations','n_units','n_states','n_calendar_years','unit_linear_trends','unit_specific_fixed_parity'):
        if saved['design']['projection'][field]!=new['design']['projection'][field]:raise ValueError('full panel support or nuisance changed: '+field)
    return dict(model_id=spec['model_id'],status='PASS',coefficient_max_abs_error=float(np.max(np.abs(a-b))),
                covariance_max_abs_error=cov_error,relative_tolerance=1e-8,absolute_tolerance=1e-8)


def common_references(all_data,specs):
    common={};registry=[]
    for crop in ('coffee_total','cacao'):
        eligible=[s for s in specs if s['crop']==crop and s['variant']!='preceding_calendar_window']
        support=[{key(r) for r in all_data[s['model_id']]['rows']} for s in eligible]
        keys=set.intersection(*support)
        if not keys:raise ValueError('No common evaluation rows for '+crop)
        common[crop]=keys
        primary=next(s for s in specs if s['crop']==crop and s['variant']=='primary' and s['outcome']==OUTCOMES[0])
        for r in all_data[primary['model_id']]['rows']:
            if key(r) in keys:
                registry.append(dict(crop=crop,territory=r['territory'],state=r['state'],year=r['year'],
                    D0=r['D0'],history=(r['D1']+r['D2'])/2,source_discordant=r['source_discordant'],pilot=r['pilot']))
    return common,registry


def common_fit_specs(specs):
    chosen=[]
    for s in specs:
        primary=s['variant']=='primary'
        structural=s['outcome']==OUTCOMES[0] and s['variant'] in ('nonlinear_main_effects','unit_linear_trend','without_local_biennial_term')
        if primary or structural:
            chosen.append(dict(s,source_model_id=s['model_id'],model_id='common_fit__'+s['model_id'],
                               estimation_family='common_observed_rows'))
    return chosen


def slope_rows(slope,spec,data,scope,family):
    if slope['status']!='COMPLETE_CURRENT_SLOPE':
        return [dict(model_id=spec['model_id'],crop=spec['crop'],outcome=spec['outcome'],variant=spec['variant'],
                     family=family,evaluation_scope=scope,status=slope['status'])]
    rows=[]
    for label,ci in slope['uncertainty'].items():
        rows.append(dict(model_id=spec['model_id'],crop=spec['crop'],outcome=spec['outcome'],variant=spec['variant'],
            family=family,evaluation_scope=scope,status=slope['status'],estimate=slope['estimate'],
            lower=ci['lower'],upper=ci['upper'],standard_error=ci['standard_error'],uncertainty=label,
            fit_rows=len(data['y']),fit_units=len(set(data['unit'].tolist())),fit_years=len(set(data['year'].tolist())),
            evaluation_rows=slope['reference_n'],reference_D0_mean=slope['reference_D0_mean'],
            reference_history_mean=slope['reference_history_mean'],display_scale=slope['display_scale'],
            climate_window='preceding_year_t_minus_1' if spec['variant']=='preceding_calendar_window' else 'same_calendar_year_t',
            interpretation='0.1_scaled_local_partial_derivative_not_finite_change_or_causal_effect'))
    return rows


def grouped_contributions(rows):
    grouped=[]
    for dimension in ('territory','state','year'):
        groups=defaultdict(list)
        for r in rows:groups[str(r[dimension])].append(r)
        for label,part in sorted(groups.items()):
            grouped.append(dict(model_id=part[0]['model_id'],crop=part[0]['crop'],outcome=part[0]['outcome'],
                dimension=dimension,group=label,n_rows=len(part),
                response_contribution=math.fsum(r['response_contribution'] for r in part),
                absolute_row_contribution_sum=math.fsum(abs(r['response_contribution']) for r in part),
                raw_response_contribution=math.fsum(r['raw_response_contribution'] for r in part),
                contrast_weight_energy_share=math.fsum(r['contrast_weight_energy_share'] for r in part)))
    return grouped


def influence(data,spec,reference,primary_slope,log):
    rows=[]
    for dimension in ('state','year'):
        values=sorted(set(data[dimension].tolist()))
        for n,value in enumerate(values,1):
            reduced=subset(data,data[dimension]!=value);r=fit(reduced,spec)
            slope=current_slope(r,reference['X'],reference['names'])
            base=dict(model_id=spec['model_id'],crop=spec['crop'],omitted_dimension=dimension,omitted_group=str(value),
                      primary_estimate=primary_slope['estimate'],fit_rows=len(reduced['y']),
                      evaluation_reference_unchanged=True,original_primary_model_overwritten=False,
                      diagnostic_estimation_sample_changed=True,status=r['status'])
            if slope['status']=='COMPLETE_CURRENT_SLOPE':
                for label,ci in slope['uncertainty'].items():
                    rows.append(dict(base,estimate=slope['estimate'],change_from_primary=slope['estimate']-primary_slope['estimate'],
                                     uncertainty=label,lower=ci['lower'],upper=ci['upper']))
            else:rows.append(base)
            if n==1 or n%5==0 or n==len(values):log('Current-slope influence '+spec['crop']+' '+dimension+' '+str(n)+'/'+str(len(values)))
    return rows


def run_models(panel,saved_models,specs,log=lambda message:None,do_influence=True):
    all_data={};verified={};reconciliations=[];slopes=[];derivatives={};models={}
    for n,spec in enumerate(specs,1):
        log('Reproduce full panel '+str(n)+'/'+str(len(specs))+' '+spec['model_id'])
        data=select_data(panel,spec);r=fit(data,spec)
        reconciliations.append(reproduce(saved_models[spec['model_id']],r,spec))
        all_data[spec['model_id']]=data;verified[spec['model_id']]=r
        models[spec['model_id']]=dict(r,specification=spec,analysis_family='verified_original_panel')
    common,reference_registry=common_references(all_data,specs)
    for spec in specs:
        mid=spec['model_id'];data=all_data[mid];r=verified[mid]
        ref=subset(data,[key(row) in common[spec['crop']] for row in data['rows']])
        for scope,reference in [('own_estimation_rows',data),('common_evaluation_rows',ref)]:
            slope=current_slope(r,reference['X'],reference['names'])
            slopes.extend(slope_rows(slope,spec,data,scope,'verified_original_panel'))
            derivatives[(mid,scope)]=slope
    cfits=common_fit_specs(specs)
    if len(cfits)!=11:raise ValueError('Unexpected common-support fixed model inventory')
    common_models={}
    for n,spec in enumerate(cfits,1):
        log('Common estimation rows '+str(n)+'/'+str(len(cfits))+' '+spec['model_id'])
        data=all_data[spec['source_model_id']]
        restricted=subset(data,[key(row) in common[spec['crop']] for row in data['rows']])
        r=fit(restricted,spec);common_models[spec['model_id']]=r
        models[spec['model_id']]=dict(r,specification=spec,analysis_family='common_observed_rows')
        slope=current_slope(r,restricted['X'],restricted['names'])
        slopes.extend(slope_rows(slope,spec,restricted,'common_estimation_and_evaluation_rows','common_observed_rows'))
        derivatives[(spec['model_id'],'common_estimation_and_evaluation_rows')]=slope
    row_contributions=[];group_contributions=[];arithmetic=[];influences=[];primary_weights={}
    for spec in specs:
        if spec['variant']!='primary':continue
        mid=spec['model_id'];data=all_data[mid];r=verified[mid]
        slope=derivatives[mid,'common_evaluation_rows']
        rows,check=generate_contributions(data,spec,r,slope['scaled_gradient'])
        arithmetic.append(dict(model_id=mid,**check));row_contributions.extend(rows);group_contributions.extend(grouped_contributions(rows))
        if spec['outcome']==OUTCOMES[0]:
            primary_weights[spec['crop']]=rows
            if do_influence:
                ref=subset(data,[key(row) in common[spec['crop']] for row in data['rows']])
                influences.extend(influence(data,spec,ref,slope,log))
    return dict(models=models,derivatives=derivatives,slope_rows=slopes,reproduction_rows=reconciliations,
                reference_rows=reference_registry,common_keys=common,common_models=common_models,
                contribution_rows=row_contributions,group_contribution_rows=group_contributions,
                arithmetic_validations=arithmetic,influence_rows=influences,primary_weights=primary_weights)
