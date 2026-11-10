# Molecular imbalanced classification experiments

A reproducible batch pipeline for comparing molecular classifiers on imbalanced binary datasets. It validates and audits raw SMILES, generates Morgan fingerprints or frozen ChemBERTa embeddings, tunes models within the training set, and exports held-out predictions and per-class metrics.

## Quick start

Run these commands from the project root on Windows or Linux. The checked-in `.python-version` selects Python 3.12, so no `--python` flag is needed:

```shell
uv sync
uv run python -m generate_config
```

Keep `uv.lock` in Git. Ordinary `uv sync` uses the lockfile when it is current and can update it when project requirements change. Use `uv sync --locked` in CI if a stale lockfile should fail the build.

Place your `.csv` or `.csv.gz` datasets directly in `data/raw/` before generating the config. Subfolders are ignored. Review the detected SMILES columns and fill in each blank `target_column` in `config.grid.json`. Targets must contain both original labels `0` and `1`. The generator never overwrites an existing config; use `--output config.new.json` to generate another.

```shell
uv run python -m molpipeline config.grid.json --plan
uv run python -m molpipeline config.grid.json --validate-only
uv run python -m molpipeline config.grid.json
```

- **Plan:** list experiments without loading datasets or fitting models.
- **Validate:** clean all selected datasets and write inspection reports. This does not generate features, make splits, or train models.
- **Run:** repeat data preparation, generate or load features, split, tune, evaluate, and save results. To generate feature exports without fitting, use `--export-features` instead.

ChemBERTa uses a pinned model revision and downloads model weights on first use. For a quicker CPU run without model downloads, set the config's `matrix.feature` to `["morgan"]` and select a small classifier grid. Keep `search_jobs` low if memory is limited.

## Configuration

New configs select both features, both random and scaffold splitting, Random Forest and CatBoost, default-threshold and ensemble M-Tune decisions, grid search, and balanced accuracy for refit. Edit `matrix` to select a subset, or supply `experiments` as a list of complete experiment specifications. The default matrix has 16 experiments per dataset and seed. The smoke preset still uses random splitting only. Example dataset entry:

```json
{
  "name": "BBBP",
  "path": "data/raw/BBBP.csv",
  "smiles_column": "smiles",
  "target_column": "p_np",
  "invalid_smiles": "skip",
  "label_conflicts": "prefer_minority",
  "minority_label": "auto"
}
```

Paths are relative to the config unless absolute. Use `matrix.split` to select `random`, `scaffold`, or both. `matrix.optimization` selects `default` (fixed hyperparameters) or `grid`. These are independent of `matrix.strategy`, which selects `default` (probability threshold 0.5) or `mtune`.

`grids` defines base-model candidates, and `mtune_grid.n_ensemble` adds ensemble-size candidates when the M-Tune method is `ensemble`. Direct M-Tune has no ensemble size. The pipeline delegates threshold learning and ensemble voting to the installed `mtune==0.0.8` package. Refit objectives are `balanced_accuracy`, `average_precision`, and `f2`; setting the objective does not restrict the metrics reported.

## Data preparation and evaluation

Every selected dataset is checked before feature generation or fitting. Raw files remain unchanged. RDKit canonicalizes valid SMILES, and same-label duplicate molecules collapse to the first retained record. Invalid SMILES are excluded only with `invalid_smiles: "skip"`. Missing labels and nonbinary labels fail validation.

The default conflict policy, `prefer_minority`, uses original input class counts to resolve opposite labels for the same canonical molecule. With `minority_label: "auto"`, the less frequent label is kept; a tie with conflicts fails validation. An explicit `0` or `1` overrides that choice. Use `label_conflicts: "error"` to stop on conflicts, `"exclude_all"` to exclude every record for unresolved molecules, or `"prefer_label"` with an explicit `preferred_label`.

**Benchmark caveat:** automatic minority preference is a whole-dataset cleaning assumption. It uses labels from records that may later be held out. Report it when comparing scores, and consider `exclude_all` or independently adjudicated labels for a stricter benchmark.

Random mode is unchanged: it uses canonical molecules as groups and selects an outer holdout with `StratifiedGroupKFold`. Scaffold mode follows the outer split from `scaffold_split.py`: ring-containing molecules sharing a Bemis–Murcko scaffold stay together, distinct acyclic canonical molecules each form their own group, and `GroupShuffleSplit` selects the outer development/test assignment. Its `test_size` is a fraction of groups, so the molecule fraction can differ and class balance is not guaranteed. Both outer partitions must contain both classes. GridSearchCV still uses `StratifiedGroupKFold` within the development set, with the same group keys and both classes required in each fold. There is no separate fixed validation partition. The selected estimator is refitted on the full development set and evaluated on the held-out test set. `split.json` records the effective `split_groups` as well as the raw scaffold keys. The standalone script is not imported.

## Outputs

Start with `results/performance_summary.csv`:

| Partition | Interpretation |
|---|---|
| `train` | Final model scored on its own fitting data; useful for diagnosing overfitting |
| `cv_validation_mean` | Selected candidate's mean score across inner validation folds |
| `test` | Final model scored on the outer holdout; use for model comparisons |

Metric suffixes `_0` and `_1` refer to the original labels. Minority-oriented metrics use the minority label from the outer training set. Precision, recall, specificity, F1, F2, average precision and ROC AUC are reported where available; ranking metrics use continuous probabilities. Partition timings distinguish final training fit/scoring, mean inner-fold fit/scoring, and held-out scoring. Overall elapsed time appears once per experiment.

- **`results/validation/<timestamp>/<dataset>/`**: `cleaned.csv`, exclusions, validation details and label conflict audit when applicable. The run's shared `run.log` and `validation_summary.json` sit in the timestamp folder. `cleaned.csv` contains `row_number,original_smiles,processed_smiles,label` for retained molecules **before** splitting. Row numbers count the original header as row 1.
- **`results/features/`**: shared feature CSVs with source rows, canonical SMILES and original labels. Numeric arrays are cached under `.cache/features/`.
- **Experiment folders**: `metrics.csv`, `cv_results.csv`, `split.json`, `cv.json`, predictions, model details, fitted model, and experiment metadata. The split and CV files record the exact cleaned-row assignments. A failed experiment records its stage in `error.json`.
- **`results/experiments.csv`**: one row per completed experiment. `results/_runs/` holds the shared run configuration. Exact reruns skip verified completed experiments; changes to code, inputs, settings, or dependency versions can create new IDs.

Refresh summaries from saved results without training:

```shell
uv run python -m molpipeline --refresh-summaries results
```

If the CSV is open and locked, the refresh writes a `.updated.csv` copy. Older unrecorded timings remain blank.

## Code organization

```text
src/
â”œâ”€â”€ generate_config.py       Dataset discovery and config generator
â””â”€â”€ molpipeline/
    â”œâ”€â”€ __init__.py
    â”œâ”€â”€ __main__.py          python -m molpipeline entry point
    â”œâ”€â”€ cli.py               Command parsing
    â”œâ”€â”€ config.py            Option validation and experiment matrix
    â”œâ”€â”€ data.py              Dataset preparation and validation reports
    â”œâ”€â”€ splits.py            Outer split and inner CV assignments
    â”œâ”€â”€ features.py          Feature generation, caching and export
    â”œâ”€â”€ training.py          Model construction, fitting and evaluation
    â”œâ”€â”€ reporting.py         Result files and summaries
    â”œâ”€â”€ pipeline.py          Stage coordination and run logging
    â”œâ”€â”€ cache.py             Cross-platform runtime cache routing
    â””â”€â”€ _artifacts.py        Shared artifact I/O and provenance
tests/
```

The installed commands `mol-experiment` and `mol-config` are also available through `uv run`. Relative runtime and feature caches resolve beside the config, and uv package downloads default to `.cache/uv`. An OS-specific launcher is unnecessary. Existing caches are not moved, and Python cannot redirect files created by external tools before it starts.

## Tests

```shell
uv sync --extra test
uv run --extra test python -m pytest tests -q
```

Tests cover cleaning and audit reports, split isolation, class-label orientation, training-only tuning, M-Tune integration, feature reuse, timing, and resume. The automated suite uses small fixtures and does not require a ChemBERTa weight download.
