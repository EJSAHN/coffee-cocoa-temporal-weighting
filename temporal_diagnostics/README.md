# Exact temporal weight diagnostics

This dependency-free Python module describes a linear contrast

`theta = sum(weight[unit, year] * outcome[unit, year])`.

It accepts arbitrary unit identifiers, integer years and signed weights. It
does not depend on a particular crop, country, event definition or operating
system. Supply the actual observation weights, including all averaging and
matching factors. The module diagnoses those weights; it does not construct a
causal design or validate the error covariance.

## Run the example

From the extracted code directory:

```text
python -I -B diagnose_weights.py --input examples/onset_weights.csv --output example_report --origin 2006
```

The example subtracts two event contrasts for the same unit. Each event uses
the mean of years `S+1, S+2` minus the mean of `S-4, ..., S-1`. Constant, linear
and fixed even/odd-year backgrounds cancel. A quadratic background remains.
The two event onsets in this constructed example are 2012 and 2018. They are
illustrative inputs, not observations from the empirical application.

The CLI reads CSV or CSV.gz in UTF-8. Required columns are `unit`, `year` and
`weight`; optional columns are `contrast`, `group` and `outcome`. Rename column
mappings with arguments such as `--unit-column municipality`. An absent optional
column means that component was not supplied. A supplied blank `contrast` is
invalid. A missing `contrast` column assigns all rows to the name `contrast`.

## Python API

```python
from temporal_diagnostics import diagnose, write_report

report = diagnose([
    {"unit": "site-1", "year": 2010, "weight": "-1/2"},
    {"unit": "site-1", "year": 2012, "weight": "0.5"},
], origin=2010)
write_report("my_report", report)
```

Python 3.9 or later and the standard library are sufficient. Exact rational
arithmetic is used throughout. Supply strings for exact decimal/fraction
weights. A Python float is interpreted through its displayed decimal string;
precision already lost before the function call cannot be recovered. The
`*_exact` fields are authoritative. Parallel `*_value` fields are convenient
floating-point values, or null if conversion exceeds the finite float range.

## Input semantics

- Identifiers are converted to stripped strings; leading zeros are preserved
  when supplied as strings. A `group` is a fixed unit attribute. Its identity,
  including whether it is supplied, must be consistent across input rows.
- Duplicate `(contrast, unit, year)` rows are consolidated by adding their
  weights. This treats rows as additive contributions to the same outcome. If
  they describe different outcomes or observations, give them different unit
  identifiers. Conflicting nonmissing outcomes at one key are rejected.
- Zero weights and entirely cancelled units are retained. Input support,
  nonzero support and fully cancelled units are reported separately.
- Optional outcomes reconstruct the contrast only if every original input row
  has an outcome, even a row with zero weight. Missing values are never filled
  from another duplicated row. Missing tokens are null/None, an empty string,
  NA, N/A, NULL, NONE and NAN (case insensitive). Other symbols are rejected.
  Numeric zero is an observed value, not missingness.
- Group/year diagnostics are emitted for a contrast only when every included
  unit has a group. Incomplete mappings are explicitly reported; a partial
  group/year calculation is not substituted.

## Outputs

`report.json` contains all settings and tables. Each table also has its own CSV:

| Table | Contents |
| --- | --- |
| `contrast_summary` | Support, cancellation, missingness and optional reconstruction |
| `unit_moments` | Exact unit-specific responses to constant, linear, quadratic and parity bases |
| `year_weights` | Signed weights aggregated across units by year |
| `group_year_weights` | Signed weights aggregated within group/year |
| `response_bounds` | Deterministic absolute response bounds with explicit coefficient scope |
| `consolidated_weights` | Input contributions combined by unit/year, including zero sums |

All time bases use `u = year - origin`: `1`, `u`, `u^2`, and `(-1)^u`.
For example, a zero linear moment in every unit means that arbitrary
unit-specific linear backgrounds cancel exactly. A zero total linear moment
alone establishes cancellation only for a single common linear coefficient.

For a unit-specific basis moment `m_i` and coefficients bounded by
`abs(a_i) <= M`, the maximum absolute background response is
`M * sum_i abs(m_i)`. For a single common coefficient, it is
`M * abs(sum_i m_i)`. The output distinguishes these scopes. Bounds for arbitrary
common-year, group/year and unit/year backgrounds similarly sum the absolute
corresponding aggregated weights. These are sharp algebraic bounds under the
stated independent coefficient limits, not probability or confidence bounds.

Centering changes the meaning and units of coefficient bounds. Always retain
the origin alongside a reported bound. Parity is a fixed alternating pattern;
it does not cover variable-amplitude biennial cycles or phase changes. No
signed-weight effective sample size is reported. A count of units or years
does not determine statistical independence.

The interface is restricted to linear contrasts with supplied fixed weights.
Ratios, estimated data-dependent weights and nonlinear recovery indices need
their own derivation. If weights were selected using observed outcomes, these
diagnostics are conditional on that selection and cannot remove selection bias.

Report files are never overwritten. Use a new output directory for a new run.
An unavailable empty table has an empty CSV and an empty list in `report.json`.

## Tests

```text
python -m unittest discover -s tests -p test_diagnostics.py -v
```

Tests cover analytic cancellation, surviving curvature, irregular calendars,
coefficient-scope distinctions, exact decimals, duplicated observations,
missing and zero outcomes, CSV/gzip input, report export and input failures.
