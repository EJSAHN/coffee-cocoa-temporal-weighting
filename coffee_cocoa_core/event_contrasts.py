"""Fixed-support descriptive production contrasts and exact annual weights."""
from __future__ import annotations
from collections import defaultdict
from fractions import Fraction
import math
from pathlib import Path
from .io import write_json, write_table


def analyse(panel, designs):
    observations = {}
    for row in panel:
        key = (str(row['territory']), row['crop'], int(row['year']))
        if key in observations:
            raise ValueError('Duplicate annual production record: '+repr(key))
        observations[key] = row
    summary, units, annual, unavailable = [], [], [], []
    for design in designs:
        did, crop = design['design_id'], design['crop']
        metadata = {key: design[key] for key in ('design_id','crop','population','kind')}
        missing = []
        for territory, year in sorted(design['required_outcomes']):
            source = observations.get((territory, crop, year))
            value = None if source is None else source.get('production_fixed_ha')
            if value is None or not math.isfinite(float(value)):
                missing.append((territory, year))
                unavailable.append(dict(metadata, territory=territory, year=year,
                                        reason='required_production_unavailable'))
        by_unit = defaultdict(list)
        source_discordant = set()
        for (territory, year), weight in sorted(design['weights_exact'].items()):
            source = observations.get((territory, crop, year))
            value = None if source is None else source.get('production_fixed_ha')
            available = value is not None and math.isfinite(float(value))
            contribution = float(weight)*float(value) if available else None
            flagged = bool(source and source.get('source_discordant'))
            if flagged and weight:
                source_discordant.add((territory, year))
            if contribution is not None:
                by_unit[territory].append(contribution)
            annual.append(dict(metadata,territory=territory,year=year,
                numerator=str(weight.numerator),denominator=str(weight.denominator),
                coefficient=float(weight),exact_zero=weight == 0,
                production_fixed_ha=value,weighted_contribution=contribution,
                source_discordant=flagged,required_outcome=(territory,year) in design['required_outcomes']))
        missing_units = {t for t, year in missing}
        all_units = []
        for unit in design['units']:
            territory = str(unit['territory'])
            contribution = None if territory in missing_units else math.fsum(by_unit[territory])
            units.append(dict(metadata,territory=territory,state=unit['state'],
                n_short=int(unit['n_short']),n_long=int(unit['n_long']),
                contribution=contribution,required_production_complete=territory not in missing_units))
            if contribution is not None:
                all_units.append(contribution)
        estimate = None if missing else math.fsum(all_units)
        summary.append(dict(metadata,n_units=len(design['units']),
            n_short_events=sum(int(x['n_short']) for x in design['units']),
            n_long_events=sum(int(x['n_long']) for x in design['units']),
            required_outcome_rows=len(design['required_outcomes']),missing_required_outcomes=len(missing),
            nonzero_weight_discordant_rows=len(source_discordant),estimate=estimate,
            units='kg per baseline intended hectare',
            status='COMPLETE' if not missing else 'UNAVAILABLE_FIXED_SUPPORT',
            causal_effect=False))
    by_id = {row['design_id']: row for row in summary}
    identities = []
    for population in ('full_target','exclude_43_pilot_units'):
        for crop in ('coffee_total','cacao'):
            rows = [row for row in summary if row['population']==population and row['crop']==crop]
            by_kind = {row['kind']: row for row in rows}
            kinds = ('original_confirmation','matched_onset','matched_anchor_change')
            if not all(k in by_kind for k in kinds):
                continue
            a,b,c = (by_kind[k]['estimate'] for k in kinds)
            complete = all(v is not None for v in (a,b,c))
            discrepancy = None if not complete else a-b-c
            if complete and not math.isclose(a,b+c,rel_tol=1e-11,abs_tol=1e-8):
                raise ValueError('Matched production contrast identity failed')
            identities.append(dict(population=population,crop=crop,
                confirmation_minus_onset_minus_anchor=discrepancy,
                status='PASS' if complete else 'UNAVAILABLE_FIXED_SUPPORT'))
    return dict(summary=summary,unit_contributions=units,annual_contributions=annual,
                unavailable=unavailable,matched_identities=identities)


def run(panel, designs, output_dir):
    result = analyse(panel, designs)
    out = Path(output_dir)
    for key, rows in result.items():
        suffix = '.csv.gz' if key in ('annual_contributions','unit_contributions') else '.csv'
        write_table(out/(key+suffix),rows)
    status = {'designs':len(result['summary']),
              'complete_designs':sum(row['status']=='COMPLETE' for row in result['summary']),
              'fixed_membership_preserved':True,'incomplete_designs_not_reweighted':True,
              'causal_effect_identified':False}
    write_json(out/'event_contrast_summary.json',status)
    return status
