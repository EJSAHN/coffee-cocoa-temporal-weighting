# Temporal diagnostics for municipal crop-production analyses

This repository reproduces the coffee and cacao study's empirical analyses, exact observation-weight diagnostics and synthetic benchmarks. It provides a portable numerical command. The numerical implementation is shared with the local-project distribution.

The main entry point performs source verification, panel preparation, event reconstruction, fixed-effects specifications, temporal comparisons, exact-weight diagnostics, the original three-design benchmark and the independent-design expansion. Supplementary Data S1 and manuscript tables are generated directly from the completed results.

## Requirements

Python 3.10 or later and NumPy 2.2 or later in the 2.x series. Figure-rendering code and plotting dependencies are excluded from this distribution. Commands do not install packages, change an environment or download data. The requirements file describes the distribution's dependencies.

## Published code and empirical inputs

[Release v2.0.0](https://github.com/EJSAHN/coffee-cocoa-temporal-weighting/releases/tag/v2.0.0) provides the source-code snapshot at commit `fadb680320df628e449d1f2e45292d40e0b62462` and the separate [empirical-input ZIP](https://github.com/EJSAHN/coffee-cocoa-temporal-weighting/releases/download/v2.0.0/coffee_cocoa_empirical_inputs_v2.zip) (38,759,041 bytes). Download the code and data separately; GitHub's automatically generated source-code archives contain code only.

The input ZIP contains `raw_inputs/` with ten preserved production, climate, cohort, geographic and event-design files, plus `source_manifest.json`. Its ten input hashes match the frozen integrated v2 sources. The data bundle includes the empirical inputs required by the numerical workflow.

Expected SHA-256 of `coffee_cocoa_empirical_inputs_v2.zip`:

```text
ea0b85e3a7baa4ca95cb1df530a33bbe94c0e2513e259f562de98b95ebc4b1ed
```

Place the downloaded input ZIP in the code directory. From that directory, using a Python environment meeting the requirements above, verify the ZIP and extract it into a new `empirical_inputs` directory:

```bash
python -c "import hashlib,pathlib; p=pathlib.Path('coffee_cocoa_empirical_inputs_v2.zip'); actual=hashlib.sha256(p.read_bytes()).hexdigest(); print(actual); raise SystemExit(0 if actual=='ea0b85e3a7baa4ca95cb1df530a33bbe94c0e2513e259f562de98b95ebc4b1ed' else 'SHA-256 mismatch')"
python -m zipfile -e coffee_cocoa_empirical_inputs_v2.zip empirical_inputs
```

Supply `empirical_inputs/raw_inputs`, the directory containing `source_manifest.json`, to the workflow below. The analysis command verifies the individual input files before calculation.

## Portable numerical workflow

Provide either a manifested raw-source directory or a prepared input directory. The input contract is described in [docs/INPUT_DATA.md](docs/INPUT_DATA.md).

```bash
python -I -B -X utf8 -u run_analysis.py --input-dir empirical_inputs/raw_inputs --output-dir results/production
```

The default runs 400 response replicates for each of the original three generated designs and 100 independently generated designs with 40 response replicates per condition. The two benchmark families are reported separately. The command also exports the numerical manuscript tables, eight plot-data tables and Supplementary Data S1. It does not render figures.

Use the identical command to resume. Completed empirical outputs are reused after input, implementation, runtime and output checks. Completed independent designs are individually verified and reused. An interrupted design is restarted, with its earlier attempt retained. A change in inputs, numerical code, Python or NumPy requires a new output directory. Do not run two processes against one output directory.

Development-only execution:

```bash
python -I -B -X utf8 -u run_analysis.py --input-dir /path/to/inputs --output-dir /path/to/smoke --smoke
```

Smoke mode uses two original response replicates and two independent designs with four response replicates. It labels all outputs as development results. Smoke results are not manuscript estimates.

## Output layout

| Location | Contents |
| --- | --- |
| `analysis/` | Prepared data, empirical fits, event contrasts, original benchmark and exact temporal weights |
| `diagnostics/` | Portable rational-weight diagnostics and agreement checks |
| `expanded_benchmark/results/` | Independent-design performance, event support, estimability and completion tables |
| `expanded_benchmark/fields/` | Per-design simulation checkpoints and retained arrays |
| `publication/manuscript_tables/` | Regenerated Tables 1–3 |
| `publication/plot_data/` | Numerical inputs for eight manuscript and supplementary figures |
| `publication/Supplementary_Data_S1.zip` | Workbook, complete companion tables and numerical provenance |
| `workflow_summary.json` | Execution status and calculation summaries |

The output directory also retains failure records and incomplete attempts. A completed calculation can contain explicitly reported nonestimable models; status and support tables distinguish this from a software failure.

## Exact-weight diagnostics for other studies

`diagnose_weights.py` accepts CSV records containing `contrast`, `unit`, `year` and exact `weight`; `group` is optional. Weights may be rational strings such as `1/8`. Use `--help` for available output and origin settings. The module requires only the Python standard library. It evaluates the supplied weights; it does not select a scientific design or fit a causal model.

## Methods and reproducibility

[docs/METHODS.md](docs/METHODS.md) describes estimands, transformations and simulation targets. [docs/SOURCES.md](docs/SOURCES.md) identifies the preserved data products and methodological references. Model results are recomputed from source inputs; reference results are never substituted for calculations.

Run the supplied checks from the repository root:

```bash
python -B -m unittest discover -s tests
python -B -m unittest discover -s coffee_cocoa_core/panel_analysis/tests
```

The included fixtures contain generated data. The preserved empirical inputs are distributed separately through the release linked above.

The original source code developed for this project, including the code released as v2.0.0, is dedicated to the public domain under CC0 1.0 Universal. See [LICENSE](LICENSE) for the full terms.

This dedication applies to the project's original source code. IBGE and TerraClimate data, third-party software, and other third-party materials remain subject to their respective licenses and terms of use.

An archival DOI has not yet been assigned.
