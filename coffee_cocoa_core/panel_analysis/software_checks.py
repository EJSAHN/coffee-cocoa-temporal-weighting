"""Small fictional arithmetic self-check, never a claim about real-data coverage."""
from __future__ import annotations
import numpy as np
from .fixed_effects import fit_model, residualize, calendar_score_hac


def run_independent_checks():
    rng = np.random.default_rng(904173)
    unit = np.repeat(np.arange(9), 12)
    year = np.tile(np.arange(2007,2019), 9)
    state = unit//3
    keep = rng.random(len(unit)) > .12
    unit, year, state = unit[keep],year[keep],state[keep]
    x = rng.normal(size=(len(unit),3))
    y = 2*x[:,0]-x[:,1]+.3*x[:,2]+rng.normal(size=len(unit))
    parity = np.where(year%2==0,1.,-1.)
    dummy = []
    for u in np.unique(unit):
        indicator=(unit==u).astype(float)
        dummy.extend([indicator,indicator*(year-2012),indicator*parity])
    for s,t in sorted(set(zip(state,year))):
        dummy.append(((state==s)&(year==t)).astype(float))
    nuisance = np.column_stack(dummy)
    # Explicit dummy projection is independent of the alternating algorithm.
    x_ref = x-nuisance@np.linalg.lstsq(nuisance,x,rcond=None)[0]
    y_ref = y-nuisance@np.linalg.lstsq(nuisance,y,rcond=None)[0]
    b_ref = np.linalg.lstsq(x_ref,y_ref,rcond=None)[0]
    names=['x1','x2','x3']
    fit=fit_model(x,y,unit,state,year,names,unit_trends=True,unit_parity=True,interaction_name=None)
    if fit['status']!='FIT_COMPLETE_EXPLORATORY':
        raise AssertionError('Fictional full-rank check unexpectedly nonestimable')
    b=np.array([fit['coefficients'][n] for n in names])
    err=float(np.max(np.abs(b-b_ref)))
    if err>1e-8:raise AssertionError('Explicit dummy coefficient validation failed')
    xw,diag=residualize(x,unit,state,year,unit_trends=True,unit_parity=True)
    err_x=float(np.max(np.abs(xw-x_ref)))
    if err_x>1e-8:raise AssertionError('Explicit dummy projection validation failed')
    e=y_ref-x_ref@b_ref
    scores=x_ref*e[:,None]
    # Direct every-observation-pair kernel, not the year-score summation routine.
    distance=np.abs(year[:,None]-year[None,:])
    kernel=np.maximum(0,1-distance/3.)
    meat=scores.T@kernel@scores
    bread=np.linalg.inv(x_ref.T@x_ref)
    covariance_ref=bread@meat@bread
    covariance=np.array(calendar_score_hac(x_ref,e,year,2)['covariance'])
    covariance_error=float(np.max(np.abs(covariance-covariance_ref)))
    if covariance_error>1e-9:raise AssertionError('Direct pairwise HAC validation failed')
    return dict(status='PASS',fictional_data_only=True,seed=904173,
                explicit_dummy_coefficient_max_abs_error=err,
                explicit_dummy_projection_max_abs_error=err_x,
                direct_pairwise_hac_max_abs_error=covariance_error,
                validates='Software arithmetic for fictional unbalanced panel with trends and local parity.',
                validates_real_world_assumptions=False,validates_interval_coverage=False)
