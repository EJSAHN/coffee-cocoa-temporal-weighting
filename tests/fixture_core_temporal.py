"""Standalone fictional event designs; no archive, project path or real values."""
from collections import Counter, defaultdict
from fractions import Fraction


def rebuild_for_test(design):
    """Construct nested weights directly from literal fictional event windows."""
    n = len(design['units'])
    counts = Counter((e['territory'], e['contrast']) for e in design['events'])
    weights = defaultdict(Fraction)
    required = set()
    for unit in design['units']:
        for group in ('short', 'long'):
            unit['n_' + group] = str(counts[unit['territory'], group])
    for event in design['events']:
        territory, group = event['territory'], event['contrast']
        factor = Fraction(1 if group == 'short' else -1, n * counts[territory, group])
        baseline = list(range(int(event['benchmark_first_year']), int(event['benchmark_last_year']) + 1))
        required.update((territory, year) for year in baseline)
        if design['kind'].startswith('original_'):
            post = list(range(int(event['post_first_year']), int(event['post_last_year']) + 1))
            terms = [(-1, baseline), (1, post)]
            required.update((territory, year) for year in post)
        else:
            onset = list(range(int(event['onset_post_first_year']), int(event['onset_post_last_year']) + 1))
            confirmation = list(range(int(event['confirmation_post_first_year']), int(event['confirmation_post_last_year']) + 1))
            required.update((territory, year) for year in onset + confirmation)
            terms = [(-1, baseline), (1, onset)] if design['kind'] == 'matched_onset' else [(-1, onset), (1, confirmation)]
        for sign, years in terms:
            for year in years:
                weights[territory, year] += factor * Fraction(sign, len(years))
    design['weights_exact'] = dict(weights)
    design['required_outcomes'] = required


def fixture():
    """Two crops x two cohorts x four kinds; entirely fictional municipalities."""
    designs = []
    for population in ('full_target', 'exclude_43_pilot_units'):
        for crop in ('coffee_total', 'cacao'):
            territories = ['T0', 'T1'] if population == 'full_target' else ['T1']
            for kind in ('original_confirmation', 'original_onset', 'matched_onset', 'matched_anchor_change'):
                events = []
                for territory in territories:
                    event_specs = [(2010, 2011, 'short'), (2017, 2017, 'long')]
                    if kind == 'original_onset':
                        event_specs.append((2021, 2022, 'short'))
                    for onset, confirmation, contrast in event_specs:
                        event = dict(
                            territory=territory, state='TEST', pair_uid=f'{territory}_{onset}',
                            contrast=contrast, current_start_month=f'{onset}-04',
                            benchmark_first_year=str(onset-4), benchmark_last_year=str(onset-1))
                        if kind.startswith('original_'):
                            anchor = confirmation if kind == 'original_confirmation' else onset
                            event.update(anchor_year=str(anchor), post_first_year=str(anchor+1), post_last_year=str(anchor+2))
                        else:
                            event.update(onset_anchor_year=str(onset), confirmation_anchor_year=str(confirmation),
                                         onset_post_first_year=str(onset+1), onset_post_last_year=str(onset+2),
                                         confirmation_post_first_year=str(confirmation+1), confirmation_post_last_year=str(confirmation+2))
                        events.append(event)
                design = dict(design_id=f'fictional__{kind}__{population}__{crop}',
                              population=population, crop=crop, kind=kind, years=list(range(2002, 2026)),
                              territories=list(territories), events=events,
                              units=[dict(territory=t, state='TEST') for t in territories])
                rebuild_for_test(design)
                designs.append(design)
    return designs
