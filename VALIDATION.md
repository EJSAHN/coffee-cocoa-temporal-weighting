# Validation scope

The software checks use generated fictional inputs, including the full monthly input schema. They exercise preparation, event reconstruction, fixed effects, both synthetic benchmark families, exact rational diagnostics, supplementary export, output verification, resumption and corruption rejection.

The original scientific Python implementations and expanded benchmark implementation were checked against the executed source versions. Integration adds workflow, export and checkpoint handling around those calculations.

The local numerical-reference checker was applied to the previously completed analysis tables: 81 model records, 587 table rows and 22,813 scalar comparisons agreed. The expanded checker agreed on 128 aggregate rows and 13,930 numeric cells. These comparisons validate the recorded-reference integration; they are not a new full-data execution.

Fresh figure-table adapters matched all eight manuscript v2 plot-data tables in populated values and row order. The local renderer produced eight figures with 36 validated panel labels. Figure byte identity is not required across rendering environments; numerical source values are preserved.

Python 3.10 syntax is checked. Software execution and artifact verification during packaging used Linux. The Windows launcher retains extended-path handling and isolated Python invocation; the final full-data run in the user's Windows environment is performed locally. The supplied tests do not establish causal identification, observed-data interval coverage, historical boundary harmonization or the validity of every generating process.
