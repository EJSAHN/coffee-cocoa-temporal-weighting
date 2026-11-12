"""Exact temporal identities on standalone fictional event designs only."""
import copy
from fractions import Fraction
import json
from pathlib import Path
import sys
import unittest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(PACKAGE / 'tests'))
from coffee_cocoa_core.temporal import analyse
from coffee_cocoa_core.temporal.weights import _one_design
from fixture_core_temporal import fixture, rebuild_for_test


def row_by_kind(result,kind,crop='coffee_total',population='full_target'):
    return next(r for r in result['design_rows'] if (r['kind'],r['crop'],r['population'])==(kind,crop,population))


class TemporalWeightTests(unittest.TestCase):
    def test_sixteen_designs_constant_parity_linear_and_quadratic_closed_forms(self):
        designs=fixture(); frozen=copy.deepcopy(designs)
        result=analyse(designs,log=lambda _:None)
        self.assertEqual(designs,frozen)
        self.assertEqual(result['validation']['status'],'PASS')
        self.assertEqual(result['validation']['designs_checked'],16)
        self.assertEqual(result['validation']['matched_anchor_identity_groups'],4)
        self.assertFalse(result['validation']['observed_outcomes_read'])
        for row in result['unit_rows']:
            self.assertEqual(row['m0_exact'],'0');self.assertEqual(row['mp_exact'],'0')
            if row['kind'] in ('original_onset','matched_onset'):
                self.assertEqual(row['m1_exact'],'0')
        expected={'original_confirmation':('1','-44'),
                  'original_onset':('0','-12'),
                  'matched_onset':('0','-56'),
                  'matched_anchor_change':('1','12')}
        for kind,(linear,quadratic) in expected.items():
            row=row_by_kind(result,kind)
            self.assertEqual(row['m1_sum_exact'],linear)
            self.assertEqual(row['m2_sum_exact'],quadratic)
        selected=next(r for r in result['unit_rows'] if r['kind']=='original_confirmation' and r['population']=='full_target' and r['crop']=='coffee_total')
        self.assertEqual(selected['m1_exact'],'1/2')
        self.assertEqual(selected['m1_numerator'],'1')
        self.assertEqual(selected['m1_denominator'],'2')
        json.dumps(result,allow_nan=False)

    def test_cancelling_global_moments_do_not_hide_unit_responses(self):
        designs=fixture()
        for d in designs:
            if d['population']!='full_target':continue
            for e in d['events']:
                if e['territory']=='T1':
                    e['contrast']='long' if e['contrast']=='short' else 'short'
            rebuild_for_test(d)
        result=analyse(designs,log=lambda _:None)
        row=row_by_kind(result,'original_confirmation')
        self.assertEqual(row['m1_sum_exact'],'0')
        self.assertEqual(row['m1_l1_exact'],'1')
        self.assertEqual(row['m1_nonzero_units'],2)
        self.assertEqual(row['m1_max_abs_exact'],'1/2')
        quadratic=row_by_kind(result,'matched_onset')
        self.assertEqual(quadratic['m2_sum_exact'],'0')
        self.assertEqual(quadratic['m2_l1_exact'],'56')
        self.assertGreater(row['annual_weight_l1'],0)
        self.assertEqual(row['common_year_weight_l1_exact'],'0')

    def test_year_profiles_keep_zeros_and_match_independent_dot_products(self):
        designs=fixture(); result=analyse(designs,log=lambda _:None)
        self.assertEqual(len(result['year_rows']),16*24)
        self.assertTrue(any(r['weight_exact']=='0' for r in result['year_rows']))
        for d in designs:
            profile=[r for r in result['year_rows'] if r['design_id']==d['design_id']]
            self.assertEqual([r['year'] for r in profile],list(range(2002,2026)))
            self.assertEqual(sum((Fraction(r['weight_exact']) for r in profile),Fraction(0)),0)
            exact=sum((weight*(year-2006)**2 for (_,year),weight in d['weights_exact'].items()),Fraction(0))
            summary=next(r for r in result['design_rows'] if r['design_id']==d['design_id'])
            self.assertEqual(Fraction(summary['m2_sum_exact']),exact)
            stateprofile=[r for r in result['state_year_rows'] if r['design_id']==d['design_id']]
            self.assertEqual(sum((Fraction(r['weight_exact']) for r in stateprofile),Fraction(0)),0)

    def test_l1_is_achieved_by_a_unit_bounded_shock_with_matching_sign(self):
        designs=fixture(); result=analyse(designs,log=lambda _:None)
        for d in designs:
            row=next(r for r in result['design_rows'] if r['design_id']==d['design_id'])
            response=sum((w*(1 if w>0 else -1 if w<0 else 0) for w in d['weights_exact'].values()),Fraction(0))
            self.assertEqual(Fraction(row['annual_weight_l1_exact']),response)
            self.assertFalse(row['basis_response_is_observed_effect'])
            self.assertIn('not a confidence interval',row['l1_interpretation'])

    def test_changed_fraction_denominator_and_float_weights_fail(self):
        for replacement in (Fraction(1,123),.125):
            designs=fixture();first=designs[0];key=next(iter(first['weights_exact']))
            first['weights_exact'][key]=replacement
            with self.assertRaises(ValueError):analyse(designs,log=lambda _:None)

    def test_cancelled_zero_weight_and_required_baseline_year_remain_mandatory(self):
        designs=fixture();d=next(d for d in designs if d['kind']=='matched_anchor_change')
        zero=next(key for key,w in d['weights_exact'].items() if w==0)
        d['weights_exact'].pop(zero)
        with self.assertRaises(ValueError):analyse(designs,log=lambda _:None)
        designs=fixture();d=next(d for d in designs if d['kind']=='matched_anchor_change')
        cancelled=next(key for key in d['required_outcomes'] if key not in d['weights_exact'])
        d['required_outcomes'].remove(cancelled)
        with self.assertRaises(ValueError):analyse(designs,log=lambda _:None)

    def test_analytic_anchors_wrong_states_and_nested_counts_fail(self):
        for field,value in [('post_first_year','2010'),('state','WRONG'),('current_start_month','2010-13')]:
            designs=fixture();d=next(d for d in designs if d['kind']=='original_confirmation')
            d['events'][0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):analyse(designs,log=lambda _:None)
        designs=fixture();designs[0]['units'][0]['n_short']='999'
        with self.assertRaises(ValueError):analyse(designs,log=lambda _:None)

    def test_locally_valid_but_mismatched_anchor_identity_fails(self):
        designs=fixture();d=next(d for d in designs if d['kind']=='matched_anchor_change')
        e=d['events'][0]
        for field in ('confirmation_anchor_year','confirmation_post_first_year','confirmation_post_last_year'):
            e[field]=str(int(e[field])+1)
        rebuild_for_test(d)
        self.assertEqual(_one_design(d)['design_row']['status'],'PASS')
        with self.assertRaisesRegex(ValueError,'Confirmation != matched onset'):
            analyse(designs,log=lambda _:None)

    def test_all_cancelled_unit_remains_in_eligible_support(self):
        designs = fixture()
        for design in designs:
            for event in design['events']:
                if event['territory'] != 'T1':
                    continue
                onset = int(event['current_start_month'][:4])
                if design['kind'] == 'original_confirmation':
                    event.update(anchor_year=str(onset), post_first_year=str(onset+1), post_last_year=str(onset+2))
                elif design['kind'].startswith('matched_'):
                    event.update(confirmation_anchor_year=str(onset),
                                 confirmation_post_first_year=str(onset+1), confirmation_post_last_year=str(onset+2))
            rebuild_for_test(design)
        result = analyse(designs, log=lambda _: None)
        summary = row_by_kind(result, 'matched_anchor_change')
        self.assertEqual(summary['n_units'], 2)
        unit = next(row for row in result['unit_rows']
                    if row['kind'] == 'matched_anchor_change' and row['territory'] == 'T1'
                    and row['population'] == 'full_target' and row['crop'] == 'coffee_total')
        self.assertEqual(unit['annual_weight_l1_exact'], '0')
        self.assertEqual(unit['n_short'], 1)
        self.assertEqual(unit['n_long'], 1)

    def test_incomplete_duplicate_or_malformed_inventory_fails(self):
        designs=fixture()
        with self.assertRaises(ValueError):analyse(designs[:-1],log=lambda _:None)
        with self.assertRaises(ValueError):analyse(designs[:-1]+[designs[0]],log=lambda _:None)
        designs=fixture();designs[0]['years'][0]=2003
        with self.assertRaises(ValueError):analyse(designs,log=lambda _:None)


if __name__=='__main__':unittest.main()
