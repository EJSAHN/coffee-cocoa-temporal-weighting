"""Exact allocation of an existing linear-functional production association.

This is arithmetic under the pathwise midpoint convention, not mediation or
causal decomposition. No new regression, yield imputation, or exclusion occurs.
"""
from __future__ import annotations
import math
from collections import defaultdict

FIELDS={
 'baseline_constant':'baseline_2006_production_kg_fixed_ha',
 'cumulative_area':'cumulative_area_allocation_kg_fixed_ha',
 'cumulative_productivity':'cumulative_productivity_allocation_kg_fixed_ha',
 'cumulative_unallocated':'cumulative_unallocated_change_kg_fixed_ha',
}


def allocate_association(primary_weights,level_rows):
    index={(r['crop'],str(r['territory']),int(r['year'])):r for r in level_rows}
    if len(index)!=len(level_rows):raise ValueError('Duplicate cumulative component identity')
    rows=[];summary=[]
    for crop,weights in primary_weights.items():
        part=[]
        for w in weights:
            k=(crop,w['territory'],int(w['year']))
            if k not in index:raise ValueError('Missing cumulative component identity: '+repr(k))
            level=index[k]
            if str(level['state'])!=str(w['state']):raise ValueError('Component state identity differs')
            available=level['level_decomposition_available']
            if not available and level['reference_production_observed']:
                raise ValueError('Observed regression row has unavailable components despite an observed baseline')
            if level['observed_production_kg_fixed_ha'] is None:
                raise ValueError('Observed regression row has missing component outcome')
            observed_contribution=w['coefficient_weight']*float(level['observed_production_kg_fixed_ha'])
            observed_tolerance=1e-9*max(1.,abs(observed_contribution),abs(w['raw_response_contribution']))
            if not math.isfinite(observed_contribution) or abs(observed_contribution-w['raw_response_contribution'])>observed_tolerance:
                raise ValueError('Observed production does not match the original weighted response')
            row=dict(crop=crop,territory=w['territory'],state=w['state'],year=w['year'],
                     model_id=w['model_id'],coefficient_weight=w['coefficient_weight'],
                     raw_response_contribution=w['raw_response_contribution'],components_available=available)
            for name,field in FIELDS.items():
                row[name+'_association_allocation']=w['coefficient_weight']*float(level[field]) if available else 0.
            row['unavailable_reference_allocation']=0. if available else w['raw_response_contribution']
            # Zeros here mean no allocated contribution to this component. They
            # are NOT filled-in production or yields; availability remains explicit.
            total=math.fsum(row[n+'_association_allocation'] for n in FIELDS)+row['unavailable_reference_allocation']
            row['identity_residual']=total-w['raw_response_contribution']
            tol=1e-9*max(1.,abs(total),abs(w['raw_response_contribution']))
            if abs(row['identity_residual'])>tol:raise ValueError('Row-level association component identity failed')
            part.append(row)
        record=dict(crop=crop,model_id=weights[0]['model_id'] if weights else '',n_rows=len(part),
                    unavailable_reference_rows=sum(not r['components_available'] for r in part),
                    total_current_association=math.fsum(r['raw_response_contribution'] for r in part),
                    allocation_convention='Cumulative adjacent-year symmetric midpoint, reference 2006; noncausal arithmetic',
                    independent_component_intervals_computed=False,percent_causal_shares_computed=False)
        for n in FIELDS:
            record[n+'_association_allocation']=math.fsum(r[n+'_association_allocation'] for r in part)
        record['unavailable_reference_allocation']=math.fsum(r['unavailable_reference_allocation'] for r in part)
        total=math.fsum(record[n+'_association_allocation'] for n in FIELDS)+record['unavailable_reference_allocation']
        record['identity_residual']=total-record['total_current_association']
        scale=max(1.,math.fsum(abs(r['raw_response_contribution']) for r in part),abs(total))
        if abs(record['identity_residual'])>1e-8*scale:raise ValueError('Aggregate association component identity failed')
        if abs(record['baseline_constant_association_allocation'])>1e-8*scale:
            raise ValueError('Unit-constant baseline not absorbed by coefficient weights')
        record['status']='PASS'
        rows.extend(part);summary.append(record)
    return dict(row_allocations=rows,summary_rows=summary)
