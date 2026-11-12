"""Independent numerical checks using entirely fictional synthetic panels."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np


from coffee_cocoa_core.panel_analysis import fixed_effects as model


def synthetic_panel(unbalanced=False, seed=20261005):
    rng = np.random.default_rng(seed)
    unit = np.repeat(np.array([f"fictional_{j:02d}" for j in range(12)]), 10)
    year = np.tile(np.arange(2007, 2017), 12)
    state = np.repeat(np.array(["Fictional_A"] * 6 + ["Fictional_B"] * 6), 10)
    x = rng.normal(size=(len(unit), 4))
    x[:, 3] = x[:, 0] * x[:, 1]
    names = ["D0", "D1", "T0", "D0_x_history"]
    nuisance = explicit_nuisance(unit, state, year, True, True)
    y = x @ np.array([2.0, -0.8, 0.3, 1.1]) + nuisance @ rng.normal(size=nuisance.shape[1])
    y += rng.normal(scale=0.4, size=len(y))
    keep = rng.random(len(y)) > 0.24 if unbalanced else np.ones(len(y), dtype=bool)
    return x[keep], y[keep], unit[keep], state[keep], year[keep], names


def explicit_nuisance(unit, state, year, trends, parity):
    """Full, deliberately redundant dummies fitted independently by dense SVD."""
    unit_levels = sorted(set(unit))
    cells = sorted(set(zip(state, year)))
    columns = [(unit == u).astype(float) for u in unit_levels]
    columns += [np.array([(s, int(t)) == cell for s, t in zip(state, year)], dtype=float)
                for cell in cells]
    if trends:
        columns += [(unit == u) * (year - np.mean(year)) for u in unit_levels]
    if parity:
        columns += [(unit == u) * np.where(year % 2 == 0, 1.0, -1.0) for u in unit_levels]
    return np.column_stack(columns)


class FixedEffectTests(unittest.TestCase):
    def test_balanced_and_unbalanced_match_explicit_dummy_ols(self):
        for unbalanced in (False, True):
            for trends in (False, True):
                for parity in (False, True):
                    with self.subTest(unbalanced=unbalanced, trends=trends, parity=parity):
                        x, y, unit, state, year, names = synthetic_panel(unbalanced)
                        nuisance = explicit_nuisance(unit, state, year, trends, parity)
                        full = np.column_stack([x, nuisance])
                        expected = np.linalg.lstsq(full, y, rcond=1e-12)[0][:x.shape[1]]
                        actual = model.fit_model(x, y, unit, state, year, names,
                                                 unit_trends=trends, unit_parity=parity)
                        self.assertEqual(actual["status"], "FIT_COMPLETE_EXPLORATORY")
                        got = np.array([actual["coefficients"][name] for name in names])
                        np.testing.assert_allclose(got, expected, rtol=2e-9, atol=2e-9)
                        self.assertLess(actual["diagnostics"]["normal_equation_relative_max_abs"], 1e-12)
                        json.dumps(actual, allow_nan=False)

    def test_projection_matches_independent_dense_projection(self):
        for trends, parity in ((False, False), (True, False), (False, True), (True, True)):
            x, _, unit, state, year, _ = synthetic_panel(True)
            nuisance = explicit_nuisance(unit, state, year, trends, parity)
            expected = x - nuisance @ np.linalg.lstsq(nuisance, x, rcond=1e-12)[0]
            actual, diag = model.residualize(x, unit, state, year,
                                            unit_trends=trends, unit_parity=parity)
            self.assertTrue(diag["converged"])
            np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=2e-9)

    def test_parity_trend_rank_deficiency_with_short_unit_histories(self):
        x, _, unit, state, year, _ = synthetic_panel(False)
        # Keep only two years for one unit. Unit parity/trend are collinear there.
        keep = (unit != "fictional_00") | (year < 2009)
        x, unit, state, year = x[keep], unit[keep], state[keep], year[keep]
        nuisance = explicit_nuisance(unit, state, year, True, True)
        expected = x - nuisance @ np.linalg.lstsq(nuisance, x, rcond=1e-12)[0]
        actual, diag = model.residualize(x, unit, state, year, unit_trends=True, unit_parity=True)
        self.assertTrue(diag["converged"])
        np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=2e-9)

    def test_no_silent_dropping_of_absorbed_or_collinear_columns(self):
        x, y, unit, state, year, names = synthetic_panel()
        bad = np.column_stack([x, np.ones(len(x))])
        r = model.fit_model(bad, y, unit, state, year, names + ["constant"])
        self.assertEqual(r["status"], "NONESTIMABLE")
        self.assertIn("constant", r["design"]["absorbed_columns"])
        self.assertNotIn("coefficients", r)
        bad = np.column_stack([x, 2 * x[:, 0]])
        r = model.fit_model(bad, y, unit, state, year, names + ["duplicate_D0"])
        self.assertEqual(r["status"], "NONESTIMABLE")
        self.assertLess(r["design"]["rank"], 5)
        self.assertNotIn("coefficients", r)

    def test_single_year_design_is_nonestimable(self):
        x, y, unit, state, year, names = synthetic_panel()
        keep = year == year[0]
        r = model.fit_model(x[keep], y[keep], unit[keep], state[keep], year[keep], names)
        self.assertEqual(r["status"], "NONESTIMABLE")

    def test_zero_outcomes_retained_and_zero_series_supported(self):
        x, y, unit, state, year, names = synthetic_panel(True)
        y[0] = 0.0
        r = model.fit_model(x, y, unit, state, year, names)
        self.assertEqual(r["diagnostics"]["n_observations"], len(y))
        self.assertEqual(r["outcome_zero_rows_retained"], 1)
        z = model.fit_model(x, np.zeros(len(y)), unit, state, year, names)
        self.assertEqual(z["status"], "FIT_COMPLETE_EXPLORATORY")
        self.assertEqual(z["outcome_zero_rows_retained"], len(y))
        self.assertIsNone(z["diagnostics"]["within_r_squared"])
        self.assertEqual(z["coefficients"], dict.fromkeys(names, 0.0))
        json.dumps(z, allow_nan=False)

    def test_measurement_unit_rescaling_preserves_estimates_and_covariance(self):
        x, y, unit, state, year, names = synthetic_panel(True)
        factor = np.array([1e-8, 1e5, 3.0, 1e-3])
        a = model.fit_model(x, y, unit, state, year, names)
        b = model.fit_model(x * factor, y, unit, state, year, names)
        ba = np.array(list(a["coefficients"].values()))
        bb = np.array(list(b["coefficients"].values()))
        np.testing.assert_allclose(bb * factor, ba, rtol=1e-9, atol=1e-9)
        for lag in (2, 4):
            ca = np.array(a["uncertainty"][f"HAC_L{lag}"]["covariance"])
            cb = np.array(b["uncertainty"][f"HAC_L{lag}"]["covariance"])
            np.testing.assert_allclose(cb * factor[:, None] * factor[None, :], ca, rtol=1e-8, atol=1e-10)

    def test_outcome_free_support_invariant_to_outcome(self):
        x, y, unit, state, year, names = synthetic_panel(True)
        a = model.fit_model(x, y, unit, state, year, names)
        b = model.fit_model(x, y[::-1], unit, state, year, names)
        self.assertEqual(a["design"], b["design"])
        info = a["design"]["interaction_identification"]
        self.assertGreater(info["partial_to_within_sum_squares_ratio"], 0)
        self.assertLessEqual(info["partial_to_within_sum_squares_ratio"], 1.0 + 1e-12)

    def test_explicit_missing_mask_required(self):
        x, y, unit, state, year, names = synthetic_panel()
        y[0] = np.nan
        with self.assertRaisesRegex(ValueError, "explicit mask"):
            model.fit_model(x, y, unit, state, year, names)

    def test_inconsistent_state_mapping_rejected(self):
        x, y, unit, state, year, names = synthetic_panel()
        state[0] = "Fictional_B"
        with self.assertRaisesRegex(ValueError, "more than one state"):
            model.fit_model(x, y, unit, state, year, names)


class HACTests(unittest.TestCase):
    def test_hac_matches_direct_row_pair_formula_including_calendar_gaps(self):
        rng = np.random.default_rng(143)
        year = np.tile(np.array([2001, 2003, 2004, 2008]), 5)
        x = rng.normal(size=(len(year), 3))
        e = rng.normal(size=len(year))
        bread = np.linalg.inv(x.T @ x)
        scores = x * e[:, None]
        for lag in (0, 2, 4):
            meat = np.zeros((x.shape[1], x.shape[1]))
            for i in range(len(year)):
                for j in range(len(year)):
                    distance = abs(int(year[i]) - int(year[j]))
                    if distance <= lag:
                        meat += (1 - distance / (lag + 1.0)) * np.outer(scores[i], scores[j])
            expected = bread @ meat @ bread
            result = model.calendar_score_hac(x, e, year, lag)
            actual = np.asarray(result["covariance"])
            np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
            self.assertEqual(result["absent_years_zero_scores"], [2002, 2005, 2006, 2007])
            self.assertEqual(result["n_observed_score_years"], 4)

    def test_cross_section_score_products_are_not_dropped(self):
        # Two municipalities with identical scores in the same two years.
        # Correct year-cluster meat includes their off-diagonal products.
        x = np.ones((4, 1))
        e = np.array([1., 1., -1., -1.])
        year = np.array([2001, 2001, 2002, 2002])
        cov = model.calendar_score_hac(x, e, year, 0)["covariance"][0][0]
        self.assertAlmostEqual(cov, 0.5)
        independent_row_variance = np.sum(e ** 2) / 16.0
        self.assertNotAlmostEqual(cov, independent_row_variance)

    def test_intervals_use_declared_reference_and_no_pvalues(self):
        x, y, unit, state, year, names = synthetic_panel()
        fit = model.fit_model(x, y, unit, state, year, names)
        for lag in (2, 4):
            h = fit["uncertainty"][f"HAC_L{lag}"]
            self.assertEqual(h["finite_sample_correction"], "none")
            for interval in h["intervals"].values():
                self.assertAlmostEqual(interval["upper"] - interval["estimate"],
                                       model.NORMAL_975 * interval["standard_error"])
        self.assertFalse(fit["p_values_computed"])


if __name__ == "__main__":
    unittest.main()
