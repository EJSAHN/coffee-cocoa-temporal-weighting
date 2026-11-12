"""Analytic and input-contract tests for public temporal diagnostics."""

import csv
from fractions import Fraction
import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from temporal_diagnostics import DiagnosticInputError, diagnose, write_report


def row(unit="u", year=2006, weight="1", **values):
    return {"unit": unit, "year": year, "weight": weight, **values}


def onset_rows(short=2012, long=2018, outcome_function=None):
    rows = []
    for onset, sign in ((short, 1), (long, -1)):
        for year in range(onset - 4, onset):
            values = {} if outcome_function is None else {"outcome": outcome_function(year)}
            rows.append(row(year=year, weight=str(Fraction(-sign, 4)), **values))
        for year in (onset + 1, onset + 2):
            values = {} if outcome_function is None else {"outcome": outcome_function(year)}
            rows.append(row(year=year, weight=str(Fraction(sign, 2)), **values))
    return rows


class DiagnosticsTests(unittest.TestCase):
    def test_onset_exact_cancellation_and_surviving_quadratic(self):
        result = diagnose(onset_rows(outcome_function=lambda year: (year - 2006) ** 2))
        unit = result["unit_moments"][0]
        for basis in ("constant", "linear", "parity"):
            self.assertEqual(unit[basis + "_exact"], "0")
            self.assertTrue(unit[basis + "_cancelled"])
        self.assertEqual(unit["quadratic_exact"], "-48")
        self.assertEqual(result["contrast_summary"][0]["reconstructed_contrast_exact"], "-48")
        self.assertEqual(result["contrast_summary"][0]["duplicate_rows_consolidated"], 1)

    def test_arbitrary_linear_and_fixed_parity_outcome_reconstruct_zero(self):
        values = lambda year: 73 + 19 * (year - 1963) + 7 * (-1) ** year
        result = diagnose(onset_rows(outcome_function=values), origin=1963)
        self.assertEqual(result["contrast_summary"][0]["reconstructed_contrast_exact"], "0")

    def test_irregular_mapping_does_not_claim_linear_cancellation(self):
        result = diagnose([row(year=1991, weight="-1"), row(year=1994)], origin=1991)
        unit = result["unit_moments"][0]
        self.assertEqual(unit["constant_exact"], "0")
        self.assertEqual(unit["linear_exact"], "3")
        self.assertEqual(unit["quadratic_exact"], "9")
        self.assertEqual(unit["parity_exact"], "-2")

    def test_exact_decimal_and_zero_are_preserved(self):
        result = diagnose([row(weight="0.1", outcome="0"), row(weight="0.2", outcome=0),
                           row(weight="-0.3", outcome="0.0")])
        summary = result["contrast_summary"][0]
        self.assertEqual(summary["weight_sum_exact"], "0")
        self.assertEqual(summary["fully_cancelled_units"], ["u"])
        self.assertEqual(summary["zero_rows_retained"], 1)
        self.assertEqual(summary["outcome_status"], "complete")
        self.assertEqual(summary["reconstructed_contrast_exact"], "0")

    def test_duplicate_outcomes_must_agree(self):
        with self.assertRaisesRegex(DiagnosticInputError, "conflicting outcomes"):
            diagnose([row(outcome="2"), row(weight="-1", outcome="3")])

    def test_missing_zero_weight_outcome_is_explicit(self):
        result = diagnose([row(weight="0", outcome=None), row(year=2007, outcome="2")])
        summary = result["contrast_summary"][0]
        self.assertEqual(summary["outcome_status"], "incomplete_missing_outcomes_on_zero_weight_rows")
        self.assertIsNone(summary["reconstructed_contrast_exact"])

    def test_missing_duplicate_is_not_filled(self):
        result = diagnose([row(outcome=None), row(outcome="3")])
        self.assertEqual(result["contrast_summary"][0]["outcome_status"], "incomplete_missing_outcomes")
        self.assertIsNone(result["consolidated_weights"][0]["outcome_exact"])

    def test_missing_outcomes_on_cancelled_rows_remain_missing(self):
        result = diagnose([row(weight="1", outcome="NA"), row(weight="-1", outcome="NA")])
        summary = result["contrast_summary"][0]
        self.assertEqual(summary["fully_cancelled_units"], ["u"])
        self.assertEqual(summary["outcome_status"], "incomplete_missing_outcomes")

    def test_outcome_not_requested(self):
        self.assertEqual(diagnose([row()])["contrast_summary"][0]["outcome_status"], "not_requested")

    def test_bounds_distinguish_common_and_unit_specific_coefficients(self):
        result = diagnose([row(unit="a", group="A"), row(unit="b", group="B", weight="-1")])
        bounds = {(r["basis"], r["coefficient_scope"]): r["response_per_unit_amplitude_exact"]
                  for r in result["response_bounds"]}
        self.assertEqual(bounds[("constant", "single_common_coefficient")], "0")
        self.assertEqual(bounds[("constant", "independently_bounded_unit_coefficients")], "2")
        self.assertEqual(bounds[("arbitrary_common_year_values", "one_bounded_value_per_year")], "0")
        self.assertEqual(bounds[("arbitrary_group_year_values", "one_bounded_value_per_group_year")], "2")
        self.assertEqual(bounds[("arbitrary_unit_year_values", "one_bounded_value_per_unit_year")], "2")

    def test_group_identity_consistent_across_contrasts(self):
        with self.assertRaisesRegex(DiagnosticInputError, "inconsistent group"):
            diagnose([row(group="A", contrast="one"), row(group="B", contrast="two")])

    def test_incomplete_group_mapping_does_not_report_partial_aggregate(self):
        result = diagnose([row(unit="a", group="A"), row(unit="b")])
        self.assertEqual(result["group_year_weights"], [])
        self.assertEqual(result["contrast_summary"][0]["group_year_status"], "unavailable_incomplete_group_mapping")

    def test_contrasts_remain_separate(self):
        result = diagnose([row(contrast="two", weight="2"), row(contrast="one")])
        self.assertEqual([r["weight_sum_exact"] for r in result["contrast_summary"]], ["1", "2"])

    def test_invalid_inputs(self):
        for bad in ([], [row(year="2006.5")], [row(year=True)], [row(weight="inf")],
                    [row(weight="nan")], [row(weight="1/0")], [row(unit="")],
                    [row(outcome="...")], [row(contrast="")]):
            with self.subTest(bad=bad):
                with self.assertRaises(DiagnosticInputError):
                    diagnose(bad)

    def test_float_overflow_keeps_exact_result(self):
        large = "1" + "0" * 400
        result = diagnose([row(weight=large)])
        self.assertEqual(result["contrast_summary"][0]["weight_sum_exact"], large)
        self.assertIsNone(result["contrast_summary"][0]["weight_sum_value"])
        json.dumps(result, allow_nan=False)

    def test_report_export_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            result = diagnose([row()])
            files = write_report(temp, result)
            self.assertEqual(len(files), 7)
            self.assertEqual(json.loads((Path(temp) / "report.json").read_text()), result)
            with self.assertRaises(FileExistsError):
                write_report(temp, result)

    def test_cli_csv_and_gzip_in_isolated_mode(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            for compressed in (False, True):
                source = temp / ("input.csv.gz" if compressed else "input.csv")
                opener = gzip.open if compressed else open
                with opener(source, "wt", encoding="utf-8", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["place", "year", "weight", "outcome"])
                    writer.writerow(["001", "1990", "1/3", "6"])
                destination = temp / ("compressed" if compressed else "plain")
                command = [sys.executable, "-I", "-B", str(ROOT / "diagnose_weights.py"),
                           "--input", str(source), "--output", str(destination), "--origin", "1990",
                           "--unit-column", "place"]
                process = subprocess.run(command, text=True, capture_output=True)
                self.assertEqual(process.returncode, 0, process.stderr)
                result = json.loads((destination / "report.json").read_text())
                self.assertEqual(result["unit_moments"][0]["unit"], "001")
                self.assertEqual(result["contrast_summary"][0]["reconstructed_contrast_exact"], "2")


if __name__ == "__main__":
    unittest.main()
