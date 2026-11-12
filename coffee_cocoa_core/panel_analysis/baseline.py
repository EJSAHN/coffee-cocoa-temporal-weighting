"""Full-panel model fitting and independent coefficient diagnostics."""
from __future__ import annotations
import math
import numpy as np
from .fixed_effects import fit_model, prepare_design, residualize

def run_fit(data,spec):
    if len(data['y'])==0:
        return dict(status='NONESTIMABLE',design=dict(status='NONESTIMABLE',reasons=['no_observed_rows']))
    return fit_model(data['X'],data['y'],data['unit'],data['state'],data['year'],data['names'],
                     unit_trends=spec['unit_trends'],unit_parity=spec['unit_parity'])

def coef_rows(fit,spec):
    rows=[]
    if fit['status']!='FIT_COMPLETE_EXPLORATORY':return rows
    meta=fit['design']['projection']
    for method,u in fit['uncertainty'].items():
        for name,ci in u['intervals'].items():
            rows.append(dict(model_id=spec['model_id'],crop=spec['crop'],outcome=spec['outcome'],
                 variant=spec['variant'],role=spec['role'],term=name,**ci,
                 uncertainty=method,reference_label='approximate_normal_95_percent',
                 n_rows=meta['n_observations'],n_units=meta['n_units'],n_states=meta['n_states'],
                 n_years=meta['n_calendar_years'],causal_effect=False))
    return rows

def independently_reconcile_interaction(data,fit,spec):
    """FWL single-regressor identity and exhaustive contribution arithmetic."""
    xw,design=prepare_design(data['X'],data['unit'],data['state'],data['year'],data['names'],
                           unit_trends=spec['unit_trends'],unit_parity=spec['unit_parity'])
    yw,ycheck=residualize(data['y'],data['unit'],data['state'],data['year'],
                         unit_trends=spec['unit_trends'],unit_parity=spec['unit_parity'])
    j=data['names'].index('D0_x_history')
    other=np.delete(xw,j,axis=1)
    scales=np.sqrt(np.mean(other*other,axis=0))
    other=other/scales
    partial=xw[:,j]-other@np.linalg.lstsq(other,xw[:,j],rcond=1e-10)[0]
    ss=float(partial@partial)
    if ss<=0:raise ValueError('No partial interaction information after successful fit')
    contributions=partial*yw/ss
    gamma=math.fsum(contributions.tolist())
    recorded=fit['coefficients']['D0_x_history']
    error=abs(gamma-recorded)
    if error>1e-7*max(1.,abs(recorded)):
        raise ValueError('Independent FWL coefficient reconciliation failed: '+spec['model_id'])
    if fit['diagnostics']['normal_equation_relative_max_abs']>1e-8:
        raise ValueError('OLS normal equations failed: '+spec['model_id'])
    grouped=[]
    for dimension,labels in [('municipality',data['unit']),('state',data['state']),('year',data['year'])]:
        for label in sorted(set(labels.tolist())):
            mask=labels==label
            grouped.append(dict(model_id=spec['model_id'],crop=spec['crop'],dimension=dimension,group=str(label),
                     coefficient_contribution=float(np.sum(contributions[mask])),
                     identification_share=float(np.sum(partial[mask]**2)/ss),n_rows=int(np.sum(mask))))
    validation=dict(model_id=spec['model_id'],status='PASS',single_regressor_fwl_estimate=gamma,
                    multivariate_estimate=recorded,absolute_error=error,
                    contribution_sum=math.fsum(contributions.tolist()),
                    interpret_as_arithmetic_not_causal_contribution=True)
    return grouped,validation

def observed_marginal_summaries(data,fit,spec):
    if fit['status']!='FIT_COMPLETE_EXPLORATORY' or spec['variant']!='primary':return []
    h=np.array([r['history'] for r in data['rows']])
    points=np.quantile(h,[.25,.5,.75])
    order=fit['coefficient_order'];b=np.array([fit['coefficients'][n] for n in order])
    rows=[]
    for label,value in zip(('observed_history_q25','observed_history_q50','observed_history_q75'),points):
        vector=np.zeros(len(order));vector[order.index('D0')]=1;vector[order.index('D0_x_history')]=value
        for method,u in fit['uncertainty'].items():
            variance=float(vector@np.asarray(u['covariance'])@vector)
            se=math.sqrt(max(0.,variance));estimate=float(vector@b)
            rows.append(dict(model_id=spec['model_id'],crop=spec['crop'],outcome=spec['outcome'],
                summary=label,history_value=float(value),estimate=estimate,standard_error=se,
                lower=estimate-1.959963984540054*se,upper=estimate+1.959963984540054*se,
                uncertainty=method,meaning='model_derivative_per_one_unit_annual_D_not_causal_effect'))
    return rows

def leave_group_diagnostics(data,fit,spec,log_fn):
    """Influence refits never change which units enter the reported primary fit."""
    rows=[];base=fit.get('coefficients',{}).get('D0_x_history')
    for dimension in ('state','year'):
        values=sorted(set(data[dimension].tolist()))
        for i,value in enumerate(values,1):
            mask=data[dimension]!=value
            reduced={k:v[mask] for k,v in data.items() if k in ('X','y','unit','state','year')}
            reduced['names']=data['names']
            r=run_fit(reduced,spec)
            gamma=r.get('coefficients',{}).get('D0_x_history')
            rows.append(dict(model_id=spec['model_id'],crop=spec['crop'],omitted_dimension=dimension,
                omitted_group=str(value),status=r['status'],n_rows=int(np.sum(mask)),gamma=gamma,
                primary_gamma=base,change_from_primary=None if gamma is None or base is None else gamma-base,
                changes_primary_sample=False,
                reasons=';'.join(r.get('design',{}).get('reasons',[]))))
            if i==1 or i%5==0 or i==len(values):
                log_fn('Influence '+spec['crop']+' leave-one-'+dimension+' '+str(i)+'/'+str(len(values)))
    return rows
