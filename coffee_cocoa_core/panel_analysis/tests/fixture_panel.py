"""Fictional temporal panels only; never loads the user's study data."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from pathlib import Path
import sys

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
from coffee_cocoa_core.panel_analysis.features import make_panel, select_data, specifications
from coffee_cocoa_core.panel_analysis.fixed_effects import fit_model


def fictional_inputs(n_units=16, serial=0.85):
    """Two crops share a fictional persistent climate field, not study values.

    Full input years are 2002--2025. The saved common-support support is 2007--2024
    without the first, explicitly marked pilot unit. Its source fits are computed
    with unchanged common-support nuisance terms solely to test reproduction.
    """
    rng = np.random.default_rng(97015)
    climate = {}
    for j in range(n_units):
        latent_d = latent_t = 0.
        territory = str(9000000 + j)
        for year in range(1998, 2026):
            latent_d = serial*latent_d + rng.normal(scale=0.4)
            latent_t = 0.7*latent_t + rng.normal(scale=0.4)
            climate[territory, year] = dict(D_mean=float(np.exp(0.25*latent_d)),
                                             Tz_mean=float(latent_t))
    annual = []
    for crop_index, crop in enumerate(('coffee_total', 'cacao')):
        for j in range(n_units):
            territory = str(9000000+j)
            state = str(j//4)
            a0 = 100. + j*5
            for year in range(2002, 2026):
                cl = climate[territory, year]
                history = (climate[territory, year-1]['D_mean'] +
                           climate[territory, year-2]['D_mean'])/2
                period = year-2006
                y = (350+crop_index*50+6*j + 3*period + 0.1*j*period +
                     15*cl['D_mean'] - 8*cl['D_mean']*history +
                     3*cl['Tz_mean'] + (8 if year % 2 == 0 else -8) + rng.normal(scale=3))
                ih = a0 if year<=2006 else a0*(1+0.015*period+0.002*j*period)
                hh = ih*(0.90+0.025*np.sin(j+period))
                annual.append(dict(crop=crop, territory=territory, state=state,
                    year=year, A0=a0, Q_tonnes=float(y*a0/1000), Iha=float(ih), Hha=float(hh),
                    pilot=j==0, source_discordant=False))
    panel = make_panel(annual, climate)
    reference = [dict(crop=r['crop'], territory=r['territory'], state=r['state'],
                      year=r['year'], D0=r['D0'], history=(r['D1']+r['D2'])/2,
                      source_discordant=False, pilot=False)
                 for r in panel if 2007<=r['year']<=2024 and not r['pilot']]
    reference_keys = {(r['crop'], r['territory'], r['year']) for r in reference}
    saved = {}
    for spec in specifications():
        if spec['outcome']!='production_fixed_ha' or spec['variant']!='primary':
            continue
        model_id = 'common_fit__'+spec['model_id']
        data = select_data(panel, spec)
        mask = np.array([(r['crop'], r['territory'], r['year']) in reference_keys
                         for r in data['rows']])
        fit = fit_model(data['X'][mask], data['y'][mask], data['unit'][mask],
                        data['state'][mask], data['year'][mask], data['names'],
                        unit_trends=spec['unit_trends'], unit_parity=spec['unit_parity'])
        if fit['status']!='FIT_COMPLETE_EXPLORATORY':
            raise AssertionError('Fictional source fit is unexpectedly nonestimable')
        fit['specification'] = dict(spec, model_id=model_id, source_model_id=spec['model_id'],
                                    estimation_family='common_observed_rows')
        fit['analysis_family'] = 'common_observed_rows'
        saved[model_id] = fit
    return dict(panel=panel, common_reference_rows=reference, saved_models=saved)
