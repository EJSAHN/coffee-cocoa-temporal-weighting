# Methods and interpretation

## Cohort and outcomes

The target frame contains municipality–crop units with mean intended harvest area of at least 100 ha across 2002–2006. All five baseline years are required. Membership is fixed from that baseline and does not require positive subsequent production. The study frame contains 1,104 coffee and 160 cacao units; a municipality represented in both crops contributes two distinct units.

For unit \(i\), let \(A_{i0}\) denote baseline mean intended harvest area, \(Q_{it}\) annual production in tonnes, \(I_{it}\) intended harvest area, and \(H_{it}\) harvested area. The analyzed outcomes are:

\[
Y_{it}=1000Q_{it}/A_{i0},\qquad I_{it}/A_{i0},\qquad (I_{it}-H_{it})/A_{i0}.
\]

The first outcome has units kg per baseline ha. The area outcomes have units ha per baseline ha. Observed zeros are retained. Numeric missing values remain missing, and their source and inclusion flags are preserved. Harvested area exceeding intended area is flagged; the resulting negative difference is not censored. Yield is defined only where harvested area is positive.

The fixed cohort reduces dependence of membership on subsequent cultivation. It does not remove all selection, missing-data, or geographic comparability problems. The intended-minus-harvested area measure does not independently identify drought damage, tree mortality, pruning, or crop abandonment.

## Climate exposures

Monthly precipitation and reference evapotranspiration define water balance as precipitation minus evapotranspiration. Each calendar month's balance is standardized against 1981–2010 using the sample standard deviation. Monthly dry exposure is \(D=\max(0,-z_1)\). Monthly temperature is standardized separately against the same calendar-month reference period. Annual exposures average the 12 monthly values.

Current annual dry exposure is \(D_{it}\). Earlier exposure is \(R_{it}=(D_{i,t-1}+D_{i,t-2})/2\). These annual windows do not overlap. The panel also includes the separate lagged dry-exposure and temperature terms. These standardized water-balance measures are not SPI or SPEI.

Climate values are assigned from the preserved gridded extraction and its municipality–grid mapping. They do not represent crop-area-weighted weather over harmonized historical boundaries. Monthly temperature cannot resolve individual frost events. Calendar-year exposures do not establish crop-specific physiological timing.

## Panel specifications

The primary model includes municipality effects, state-by-year effects, current and two lagged annual dry and temperature exposures, and the current-exposure–history interaction:

\[
Y_{it}=\alpha_i+\lambda_{s(i),t}
 +\sum_{\ell=0}^{2}\beta_\ell D_{i,t-\ell}
 +\sum_{\ell=0}^{2}\eta_\ell T_{i,t-\ell}
 +\gamma D_{it}R_{it}+\varepsilon_{it}.
\]

Coffee specifications additionally include a municipality-specific fixed parity term, except for the explicitly labeled sensitivity without that term. A common odd/even-year indicator would be absorbed by state-by-year effects. The local parity term describes a fixed alternating component; it does not capture every form of biennial fruiting.

The finite specification menu includes nonlinear exposure terms, municipality-specific linear trends, previously examined pilot-unit exclusion, final-year omission, an alternative treatment of source-discordant records, and a preceding-calendar exposure window. All specified results are retained. The analysis is exploratory; it is not a prospectively registered confirmatory study.

The interaction coefficient \(\gamma\) describes how the current dry-exposure association varies with previous exposure. It is different from a current-response slope. In the linear specification, the current slope at history \(r\), scaled by 0.1, is \(0.1(\beta_0+\gamma r)\). Nonlinear specifications include their corresponding derivative terms. Evaluation conditions and estimation samples are reported separately.

## Production scale and temporal comparisons

Production changes are partitioned using harvested-area and yield arithmetic where those quantities are defined. Undefined yield at zero harvested area and missing transitions are retained as unallocated components. Cumulative components use a fixed reference and an explicit sequence of annual changes. Applying production contrast weights to these components is an arithmetic allocation, not causal mediation analysis.

Temporal comparisons use the same eligible transition endpoints in 2008–2023. The level response is \(Y_t\); the first-difference response is \(Y_t-Y_{t-1}\), which additionally uses predecessor observations beginning in 2007. Identical endpoint sets therefore do not imply identical underlying outcome observations.

First differences are taken after constructing each original regressor. In particular, \(\Delta(DR)\) is not \(\Delta D\,\Delta R\). Municipality effects in the differenced model allow unit-specific linear drift in levels. Future-climate diagnostics append one- and two-year dry and temperature exposures to the designated production model with unit trends. Persistence, omitted lags, temporal alignment, anticipation, or other misspecification can produce those associations; a nonzero future term does not identify which explanation applies.

## Uncertainty and original benchmark

Uncertainty calculations aggregate estimating-equation scores over municipalities within each calendar year, then apply Bartlett-weighted HAC calculations with lag lengths 2 and 4. Both choices are reported. The intervals use a normal approximation. Limited time support remains important even with many municipalities, and state-by-year effects do not eliminate all residual dependence.

The benchmark crosses three climate-response laws with three background processes in each of three fixed generated climate designs. The default is 400 response replicates per condition. It reports Monte Carlo uncertainty and distinguishes the known structural coefficient from the projection of the full noiseless mean onto each fitted model. These quantities can differ under background misspecification.

The same generated errors are shared across methods and conditions within a climate design. Replicates are not independent climate fields. Unsupported designs remain reported and are not replaced by repeated seed searches. Event contrasts are evaluated against their own climate, background, and total-mean components, not against a regression interaction coefficient. Results are conditional operating characteristics of the specified generated processes.

## Event-weight diagnostics

The frozen event contrasts distinguish short intervals of 3–24 months from long intervals of 48–72 months. Events use the recorded six-month standardized water-balance rule: two consecutive months at or below −1.5 establish onset, and three consecutive months above −0.5 establish termination confirmation. The interval counts months strictly between the previous event's end and the next onset.

For each event, the four annual production values before its onset form the reference mean. Two annual values after onset or termination confirmation form the endpoint mean. Events are averaged within interval group and municipality, then municipalities receive equal weight. Matched contrasts retain the designated common event memberships. The pre-event mean may already include earlier drought responses and is not assumed to be an undamaged baseline.

Exact rational annual weights are reconstructed from the event membership records. For each municipality the workflow evaluates those weights against \(1\), \(t-2006\), \((t-2006)^2\), and \((-1)^t\). Eligible units whose consolidated weights cancel remain in the registry.

Constant and fixed-parity components cancel within each unit. Onset contrasts also cancel unit-specific linear components. Confirmation contrasts can retain linear components because confirmation delays differ. Quadratic moments and year-specific weight profiles quantify the response to hypothetical temporal components. Their deterministic bounds are not statistical confidence intervals or estimates of observed confounding.

## Independent-design expansion

The expanded benchmark uses 100 independently generated designs with identifiers 1000–1099 and master seed 20261005. Each design contains 120 units in six states and 19 endpoint years. Forty response replicates are evaluated for each of nine combinations of response law and background process. Designs regenerate climate fields and the associated unit attributes, temporal background, parity component, and error loadings; the original three fixed designs remain a separate reference.

Methods share generated observations within a design. Performance is first summarized within each design and then aggregated across designs. Monte Carlo standard errors use the independent outer designs, not pooled outcome replicates. Event support and nonestimable conditions are retained in the planned denominator; low-support designs are not redrawn. The benchmark reports structural targets and model-specific noiseless projections separately. Its scope is the specified generating processes.

The default workflow reproduces the original benchmark with 400 replicates and the expanded benchmark with 100 designs and 40 replicates. `--smoke` is a development mode: it uses two original replicates and two independent designs with four replicates. Smoke outputs are explicitly labeled and cannot be resumed as production results.

## Portable exact-weight diagnostics

`temporal_diagnostics` accepts observation-level rational weights independently of the study data. It reports contrast- and unit-level temporal moments, year and group-by-year weights, and deterministic response bounds. Consolidated zero coefficients and registered units are retained. The integrated workflow reconstructs these input weights from the frozen event memberships and checks agreement against the study-specific exact arithmetic.
