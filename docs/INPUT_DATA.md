# Input data and reproducibility

The command accepts either a raw canonical directory containing `source_manifest.json` or a prepared directory containing `input_manifest.json`. When raw inputs are supplied, it prepares and validates a portable dataset before fitting the models. Prepared inputs are verified and their outcome and exposure arithmetic is reconstructed before use.

The data bundle must be obtained separately. No download service or access credentials are required by the analysis command. The original production extraction, climate extraction, cohort, geographic mapping, and event membership are substantive inputs; completed model estimates are not accepted in their place.

## Published input archive

Download [`coffee_cocoa_empirical_inputs_v2.zip`](https://github.com/EJSAHN/coffee-cocoa-temporal-weighting/releases/download/v2.0.0/coffee_cocoa_empirical_inputs_v2.zip) from [release v2.0.0](https://github.com/EJSAHN/coffee-cocoa-temporal-weighting/releases/tag/v2.0.0).

- Size: 38,759,041 bytes.
- SHA-256: `ea0b85e3a7baa4ca95cb1df530a33bbe94c0e2513e259f562de98b95ebc4b1ed`.
- Layout after extraction: `raw_inputs/source_manifest.json` and the ten files listed below, with event-design files under `raw_inputs/legacy/`.
- Code snapshot: `v2.0.0`, commit `fadb680320df628e449d1f2e45292d40e0b62462`.

The archive uses portable ZIP paths. Every source file is byte-identical to the corresponding frozen integrated v2 input. Repacking the ZIP for portable paths changed the outer ZIP hash without changing any member file.

See the [README](../README.md#published-code-and-empirical-inputs) for archive verification, extraction and execution commands. Pass the extracted `raw_inputs` directory to `--input-dir`.

## Raw canonical directory

`source_manifest.json` contains a `sha256` object mapping each relative file name below to its lowercase SHA256 digest. Paths use forward slashes. A manifest does not hash itself. All ten required files must be present and must match their hashes.

| File | Content |
| --- | --- |
| `production_source.csv.gz` | Original annual production and area extraction, including original response strings |
| `climate_monthly.csv.gz` | Decoded monthly climate values and preserved grid assignment |
| `baseline_cohort.csv` | Frozen municipality–crop frame and baseline intended area |
| `target_sample_registry.csv` | Target inclusion and previously examined pilot membership |
| `geography_climate_audit.csv` | Municipality identity, grid coordinates, and extraction status |
| `climate_source_metadata.json` | Dataset version, variables, units, and extraction metadata |
| `legacy/consolidated_annual_weights.csv.gz` | Exact annual event-contrast coefficients |
| `legacy/design_registry.csv` | Design membership counts and coefficient fingerprints |
| `legacy/design_event_membership.csv.gz` | Event windows and within-unit averaging membership |
| `legacy/design_unit_registry.csv` | Eligible units and short-/long-interval event counts |

Files use UTF-8 text, comma delimiters, and a single header row. `.csv.gz` denotes gzip compression. Field names are case sensitive. The authoritative required-field lists and validation rules are in `coffee_cocoa_core/data_import.py` and `coffee_cocoa_core/event_designs.py`.

### Production fields

Required columns in `production_source.csv.gz` are:

```text
territory,crop,year,state,production_t,intended_harvest_ha,harvested_ha,
production_t_raw,intended_harvest_ha_raw,harvested_ha_raw
```

The displayed line break is for readability; the file has one header row. `territory` is a seven-digit municipality code stored as text. `state` is its first two digits. `crop` is `coffee_total` or `cacao`. Years are canonical integer strings in 2002–2025. A municipality–crop–year key occurs at most once in the selected frame.

Production is in tonnes, and areas are in hectares. Numeric quantities are finite and nonnegative. Empty numeric fields mean missing. A numeric zero is an observed zero. Original source symbols belong in the corresponding `_raw` fields and must not be discarded or interpreted anew by the analysis reader. The reader does not convert arbitrary source symbols to numbers.

Every selected unit must have all five baseline intended-area values. Their mean must agree with the registered baseline denominator. The prepared data retain the complete target unit–year grid, including absent source rows marked by `source_row_present=False`.

### Cohort and sample registry

`baseline_cohort.csv` requires `territory`, `crop`, `state`, and `baseline_area_ha_2002_2006`. Each denominator is at least 100 ha. `municipality_name_2025`, when present, supplies a display label and does not determine membership.

`target_sample_registry.csv` uses the same identities and denominator and additionally requires `target_frame_included` and `pilot_previously_explored`. Boolean values use `True` or `False`. The target inclusion field is `True` for the entire supplied cohort. The pilot flag records prior examination and supports an explicitly labeled sensitivity analysis.

### Monthly climate fields

The required ordered header is:

```text
territory,month,ppt_mm,pet_mm,tmax_monthly_mean_degC,balance_mm,balance_z_3m,balance_z_6m,balance_z_12m,tmax_calendar_month_z,grid_lat,grid_lon
```

Each target municipality has one continuous ordered series from `1980-01` through `2025-12`, totaling 552 months. Precipitation and reference evapotranspiration are in mm; maximum temperature is the monthly mean of daily maxima in degrees Celsius. Grid latitude and longitude are constant within municipality and agree with `geography_climate_audit.csv`. That table requires one row per target municipality with `territory`, `grid_lat`, `grid_lon`, and `status=COMPLETE`.

Monthly water balance must equal precipitation minus evapotranspiration. Preparation recomputes one-month balance standardization and independently checks the saved calendar-month temperature standardization. The one-month index and annual exposures are reconstructed from the decoded climate values. Source metadata identify the retained longer-window indices and event definitions.

### Event-design files

The annual-weight file stores `design_id`, `territory`, `crop`, `year`, `numerator`, `denominator`, `coefficient`, and `exact_zero`. Numerator and denominator define the exact rational weight; `coefficient` is its floating-point representation. Preserve exact-zero rows when present and preserve eligible units even if their full contrast cancels.

The design registry and event/unit membership files must remain mutually consistent. They specify onset and confirmation years, four-year reference windows, two-year endpoint windows, short/long interval labels, within-group event counts, and the number of equally weighted units. Required-outcome and coefficient fingerprints bind those memberships and weights. The reader independently reconstructs the annual coefficients from the event windows and checks exact agreement. These files are not substitutes for production outcomes; the event contrasts are recomputed from the prepared annual production panel.

## Prepared directory

`input_manifest.json` uses schema version `coffee_cocoa_portable_v1`. Its `sha256` mapping covers all prepared files, while `counts` records the unit, year, observation, missing-value, zero, and climate totals. The manifest also records production years 2002–2025 and the monthly climate range.

The prepared directory includes:

- `annual_production.csv.gz`: typed quantities, raw source strings, fixed denominator, and source-status flags.
- `annual_panel.csv.gz`: those records plus normalized outcomes and annual climate lags.
- `climate_annual.csv.gz`: `territory`, `year`, `D_mean`, and `Tz_mean`, with the preserved additional climate summary fields.
- `cohort.csv`: the reconciled target identities and fixed denominators.
- `provenance.json`: input hashes and the exact transformations applied.
- The original raw files listed above, except the full `production_source.csv.gz` extraction.

Together these are 14 manifested files plus `input_manifest.json`. The original annual source file's hash remains in provenance. Preserve the original source extraction separately when distributing a complete source-data archive.

Important annual-panel columns include:

| Field | Meaning |
| --- | --- |
| `A0` | Mean intended harvest area in 2002–2006, ha |
| `Q_tonnes`, `Iha`, `Hha` | Production, intended area, harvested area |
| `production_fixed_ha` | `1000 * Q_tonnes / A0` |
| `intended_area_fixed_ha` | `Iha / A0` |
| `unharvested_area_fixed_ha` | `(Iha - Hha) / A0` |
| `harvest_fraction` | `Hha / Iha`, only when intended area is positive |
| `yield_kg_harvested_ha` | `1000 * Q_tonnes / Hha`, only when harvested area is positive |
| `D0`–`D3`, `T0`–`T3` | Current through third-lag annual dry and temperature exposures |
| `pilot` | Unit previously examined in pilot work |
| `source_discordant` | Membership in the documented source-table discrepancy sensitivity set |
| `source_row_present`, `source_row_number` | Source-row presence and traceability |
| `Q_tonnes_raw`, `Iha_raw`, `Hha_raw` | Preserved production/area response strings |

The source-discordance sensitivity set is study-specific and explicitly encoded. The default keeps original numeric values. The labeled alternative treats the designated records as missing; it is not a claim that the original extraction was parsed incorrectly.

## Reproducing a result

Use the same input manifest, code version, seed, replicate count, and analysis settings. Each output directory records the analysis configuration and its file hashes. Keep result tables with these records so numerical changes can be attributed to input changes, specifications, or software changes.

Use a new output directory for a new data or software version. Repeating the identical command resumes that directory after verifying its source files, numerical implementation, runtime, completed outputs, and simulation checkpoints. Failed attempts are retained separately. An incomplete directory is not an input bundle.

Official tables and climate archives can change. Matching a source URL or nominal date range is not sufficient for an exact reproduction. The preserved bytes, decoded units, municipality mapping, missing-value treatment, and manifest establish which data entered the analysis.
