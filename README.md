# Temporal diagnostics for municipal crop-production analyses

This repository reproduces the coffee and cacao study's empirical analyses, exact observation-weight diagnostics and synthetic benchmarks. It provides a portable numerical command. The numerical implementation is shared with the local-project distribution.

The main entry point performs source verification, panel preparation, event reconstruction, fixed-effects specifications, temporal comparisons, exact-weight diagnostics, the original three-design benchmark and the independent-design expansion. Supplementary Data S1 and manuscript tables are generated directly from the completed results.

## Requirements

Python 3.10 or later and NumPy 2.2 or later in the 2.x series. Figure-rendering code and plotting dependencies are excluded from this distribution. Commands do not install packages, change an environment or download data. The requirements file describes the distribution's dependencies.

## Portable numerical workflow

Provide either a manifested raw-source directory or a prepared input directory. The input contract is described in [docs/INPUT_DATA.md](docs/INPUT_DATA.md).

```bash
python -I -B -X utf8 -u run_analysis.py --input-dir /path/to/inputs --output-dir /path/to/results
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

The included fixtures contain generated data. Input data are distributed separately from code. No data archive, DOI, public repository URL or publication status is implied by this code distribution. A repository license has not been assigned in this package.
