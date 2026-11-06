# Molecular imbalanced classification experiments

Python 3.11 or 3.12 is required. The system Python found during setup was 3.9; use an explicit supported interpreter. Existing project directory was empty when inspected. No real datasets were supplied.

## Setup (PowerShell)

```powershell
cd C:\Users\ASUS\Documents\CSIR-NEIST\2026\imbalanced_ml_software
# Replace this with your Python 3.11/3.12 executable:
& 'C:\path\to\python.exe' -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[test]'
Copy-Item config.example.json config.json
.\.venv\Scripts\python.exe -m molpipeline config.json --plan
.\.venv\Scripts\python.exe -m molpipeline config.json
.\.venv\Scripts\python.exe -m pytest -q
```

CatBoost, Torch and Transformers are main dependencies and install with the project. Model libraries are still imported only when needed. The test extra adds pytest. ChemBERTa downloads model weights on first use; choose the model repository and replace `revision` with its immutable 40-character commit SHA before running. The frozen transformer uses CPU evaluation mode and attention-mask mean pooling (including special tokens); long SMILES are truncated at `max_length`. No fine-tuning, normalization, imputation, or resampling is performed.

## Datasets and configuration

`config.example.json` has four independent CSV dataset entries. Paths are relative to the configuration file, or may be absolute. Each CSV requires a SMILES column and binary numeric target column:

```csv
smiles,label
CCO,0
c1ccccc1,1
CCN,0
```

Configure each entry's `name`, `path`, `smiles_column`, and `target_column`. Extra columns are ignored. Missing values, invalid/empty molecules, nonbinary labels, one-class data, and conflicting labels for canonical duplicates are rejected. Identical canonical SMILES are deduplicated before splitting; reported row indexes refer to this cleaned table, not original CSV line numbers. Canonical SMILES retain stereochemistry; salts/tautomers are not standardized. Perform any study-specific chemical standardization before input.

The four dataset names are placeholders, not assumed scientific databases. Any number of entries is supported. Provide enough independent molecules/scaffolds and both classes in every holdout and inner fold. The three-line schema example is not large enough for experiments.

`matrix` chooses lists on each axis: `dataset`, `feature` (morgan/chemberta), `split` (random/scaffold), `model` (random_forest/catboost), `strategy` (default/mtune), `optimization` (default/grid), and `seed`. Omitted axes use defaults: all listed datasets, both features/splits/models/strategies, default hyperparameters, seeds from `seeds` (42 by default). The example selects both optimization modes: 128 experiments across four datasets and one seed. Remove `grid` for 64. Select one option per axis for a small run.

Instead of a Cartesian matrix, supply `experiments` as a list of complete objects, for example:

```json
{"dataset":"dataset_1","feature":"morgan","split":"scaffold","model":"random_forest","strategy":"mtune","optimization":"grid","seed":42}
```

Put that object inside an `experiments` array in your config. Explicit experiments override the matrix. `model_params` may set fixed parameters by model name; otherwise library defaults apply, apart from reproducible seeds, one worker per estimator, and disabled CatBoost logging/files. `grids` uses ordinary base estimator parameter names; M-Tune nesting prefixes are added automatically. `refit_metric` defaults to `average_precision`; alternatives are `balanced_accuracy` and `f2`. All eight CV metrics are always recorded. The old `scoring` setting is accepted as a deprecated refit alias for these three objectives; `refit_metric` takes precedence. `search_jobs` controls grid-search parallelism. `cv_folds` defaults to 5. `mtune.method` may be direct or ensemble (the generator default); `mtune.n_ensemble` defaults to 11. No silent reduction of ensemble size or CV folds occurs.

## Splitting and leakage controls

Random splitting uses shuffled stratified grouping by canonical molecule. Scaffold splitting uses RDKit Bemis-Murcko scaffolds, shared across all compounds with the same scaffold; all acyclic molecules share the empty scaffold. Neither identical molecules nor scaffold groups (in scaffold mode) cross the holdout or inner-CV boundaries. Fold sizes/classes can vary because groups remain whole. Holdout size is approximate: use `round(1/test_size)` grouped folds and select the closest-sized fold without looking at predictive performance. Invalid single-class folds fail with a clear error; adjust the protocol or data rather than silently falling back to random splitting.

GridSearchCV runs on the outer training partition only, using the same grouping rule for inner folds. Each M-Tune candidate is fitted independently inside each fold, then the winning estimator is refitted on the complete outer training partition. The outer test labels never choose parameters or thresholds. Feature caching across all rows is valid here because Morgan is stateless and ChemBERTa is frozen: neither learns from these datasets. Future learned preprocessing must be fitted within CV.

## Actual M-Tune integration

The downloaded `mtune==0.0.8` wheel was inspected, including `Mtune.__init__`, `fit`, `predict`, and `predict_proba`. The pipeline delegates to this package, not a reimplementation. API: `from mtune import Mtune`; `Mtune(base_model=..., method='direct' or 'ensemble', n_ensemble=11, random_state=42)`. It expects pandas-aligned feature and target indexes. The adapter supplies these and current sklearn classifier tags; nested grid parameters use `base_model__...`.

Direct mode fits a cloned learner and computes its minority-class threshold from that learner's training predictions. Ensemble mode partitions majority rows, pairs each part with all minority rows, and combines learner threshold decisions by voting. These training-derived thresholds are the package's algorithm, not held-out threshold optimization. Test data remain untouched until final evaluation. Ensemble probabilities average learner probabilities; final labels follow voting, so they need not match thresholding the averaged probability at 0.5. The minority class can be label 0 or 1. Metrics treat the minority label selected from the outer training partition as positive; original label 1 is used only to break equal-count ties. This mapping is fixed across inner CV and held-out evaluation, even if a fold has different proportions. The M-Tune package itself still determines its threshold class from each fitted training fold. The adapter rejects tied training class counts because this release's majority/minority identification is ambiguous for ties; this also applies inside CV.

Sources: https://pypi.org/project/mtune/0.0.8/ and https://github.com/Naga270588/M-Tune . Wheel SHA256: `5fe43717dbc56daca430c2fdcd6e73297316484b9b2d5c9ea419776f44bff764`.

## Results, reproducibility, resume

Each experiment has a content-derived directory in `output_dir`. It contains configuration/environment/source/input hashes, canonical molecule and scaffold assignments, outer split indexes, inner-CV indexes and scores when applicable, fitted `model.joblib`, held-out predictions, metrics and learned M-Tune thresholds. Metrics include accuracy, balanced accuracy, F1, F2, precision, minority recall, specificity, MCC, ROC AUC, average precision and confusion counts. Average precision is reported by that name; it is not trapezoidal PR AUC.

`summary.csv` collects completed runs in the output directory, including historical configurations. Compare run identities before aggregating different studies. A complete `result.json` is written last. Repeating the command skips verified completed runs; missing/corrupt artifacts trigger rerunning. Changed input bytes, configuration, implementation or recorded dependency versions create a new identity. Failed experiment details go to `error.json`; other experiments continue, then the command exits unsuccessfully. Dataset validation errors stop before fitting. Feature caches are keyed by molecules, settings, implementation and versions; remove cache files to recompute. Run one runner per output/cache directory at a time (concurrent writers are not supported). Resume assumes local artifacts are trusted and does not validate feature-cache contents for malicious modification.

Seeds, single-worker base learners, frozen model revision and recorded versions support reproducibility. Numerical results may still differ across hardware/library builds. Preserve `pip freeze` output alongside a study. Loading joblib artifacts requires compatible dependencies and trusted files. No supplied-data scientific experiment has been run until your four datasets and ChemBERTa revision are configured.

## Local validation

Nine automated tests passed using `.venv312\Scripts\python.exe -m pytest -q --basetemp work/pytest-final`. This prepared environment supports Morgan, Random Forest, CatBoost and M-Tune. Optional Torch/Transformers are not installed here. The standard editable-install attempts stalled; the test environment was prepared from pinned PyPI wheels directly. Use the setup command above to create your own normal installation. Verified versions and wheel identity are recorded in `work/validation.json`. The upstream M-Tune ensemble emitted pandas/NumPy deprecation warnings.

## Multi-metric search and minority-positive evaluation

```json
"refit_metric": "average_precision"
```

Use `balanced_accuracy` or `f2` instead to select a different winner. Names are lowercase (`f2`, not `F2`). GridSearchCV records `balanced_accuracy`, `recall`, `specificity`, `precision`, `f1`, `f2`, `average_precision`, and `roc_auc` for every candidate/fold, with means, standard deviations and ranks in `cv_results.csv`. `best_cv_score` belongs to the named `refit_metric`; the winner is refitted on all outer training rows. The objective is validated before execution (including `--plan`).

The positive label is inferred ONLY from the deduplicated outer training partition, never the test labels. Its original label is recorded in `result.json` under `label_mapping`. Evaluation maps that label to binary 1, and the other label to binary 0. Tied outer counts use original label 1 for reporting; M-Tune still rejects equal-count fitted training sets because of its upstream limitation. Minority means training minority, even when the test distribution differs.

Threshold metrics use the default 0.5 class-1 probability rule or M-Tune's actual direct/ensemble decisions. Recall is minority sensitivity; specificity is majority recall. Precision/F1/F2 use minority as positive with zero-division scores set to zero. AP and ROC-AUC use the continuous probability column for the fixed positive label, never hard labels. A candidate can therefore have different threshold metrics but identical ranking metrics to another candidate. Default and M-Tune grid runs are tuned separately with the same configured objective and grouped folds; their selected base parameters may differ.

Prediction CSVs preserve original `y_true`, `y_pred`, and `p_class_1` columns for compatibility, and add `y_true_minority_positive`, `y_pred_minority_positive`, and `p_minority`. Confusion counts and the `metrics` object now use the minority-positive convention. Summary rows include `positive_original_label` and `refit_metric` so historical label-1-positive runs (blank mapping fields) are distinguishable. Do not pool old and new metric conventions.

Scaffold CV keeps all shared scaffolds together; random CV groups by canonical molecule. Inner folds and refit see only the outer training rows. Threshold learning remains training-only through the real M-Tune wrapper. All cached feature logic and completion/artifact checks are retained. The source/configuration hash changes create new experiment identities instead of reusing old metric results; stateless/frozen feature caches remain reusable.

Scoring update validation (2026-10-27): 18 tests passed, including all three refit objectives, original minority label 0/1, continuous-score metric orientation, real M-Tune direct and CatBoost ensemble integration, scaffold-disjoint CV and training-call audits excluding outer-test rows, plus existing caching/resume and corruption recovery tests. Six upstream NumPy/pandas deprecation warnings remain. Command: `.venv312\Scripts\python.exe -m pytest tests -q -p no:cacheprovider --basetemp work/pytest-scoring-final-20261027`.

## Generate configuration from arbitrary downloaded CSVs

```powershell
.\.venv312\Scripts\python.exe -m generate_config --mtune-method ensemble --refit-metric balanced_accuracy --output config.discovered.json
```

The generator scans only files directly inside `data/raw` for `.csv` and `.csv.gz` files, without dataset-name lookups. Dataset names come from file stems. It inspects every row in chunks with RDKit: a column qualifies if at least 50% of all its entries parse as nonempty molecular SMILES. Missing, numeric and invalid entries count against the total. The unique highest-scoring column is selected. Ties or no qualifying column leave `smiles_column` empty with a notice; all discovered CSVs still appear in the draft. RDKit parseability is a heuristic, so review the detected column (molecular-looking identifiers may also parse). The 50% rule is for discovery only: training still validates every SMILES and does not silently drop bad rows.

Every `target_column` is empty by default, including standard MoleculeNet files. Open the generated config and fill in the binary label column for each dataset. Neither dataset names nor binary-valued column patterns decide your scientific endpoint. For example:

```json
{"name": "my_dataset", "path": "data/raw/my_dataset.csv", "smiles_column": "structure", "target_column": "YOUR_BINARY_COLUMN"}
```

If needed, also fill in a blank `smiles_column`. The runner rejects unfilled fields before preview/training. Labels must contain both numeric 0 and 1; multi-target datasets need an explicit endpoint. To supply a target intentionally during generation, the optional `--target "relative_file.csv=COLUMN"` remains supported. Repeat for different files.

The generator does not modify or download datasets and never overwrites a config. Existing completed configurations are unchanged. Paths are relative to the generated config. `--data-dir PATH` changes the scanned directory (subfolders are always ignored); `--output FILE` selects a new filename. Duplicate file stems are rejected rather than silently combining datasets.

By default, the generator writes `config.grid.json` with 8 combinations per discovered file (Morgan and ChemBERTa), random splitting only, grid-only optimization, ensemble M-Tune, balanced-accuracy refit, and one seed. Use `--optimization default` or `--optimization both` for other modes, `--mtune-method direct` for direct M-Tune, and `--refit-metric average_precision` or `f2` for other objectives. Grid-only ensemble configs omit the redundant fixed `n_ensemble`; the candidate sizes remain in `mtune_grid`. Optimization applies to either preset: `--preset smoke` now selects one Morgan/random/RandomForest/default-threshold combination per file with grid search; add `--optimization default` for a quick fixed-parameter smoke run. Both Morgan and ChemBERTa are included by default with a bundled pinned model revision. `--chemberta-revision SHA` overrides that 40-character commit; optional Torch/Transformers dependencies must also be installed.

After reviewing/filling the draft:

```powershell
.\.venv312\Scripts\python.exe -m molpipeline config.discovered.json --plan
.\.venv312\Scripts\python.exe -m molpipeline config.discovered.json
```

## Tune M-Tune ensemble size

Direct M-Tune has no ensemble-size parameter: the previous `n_ensemble: 11` entry did not affect direct-mode predictions. For ensemble experiments, generate a new configuration with:

```powershell
.\.venv312\Scripts\python.exe -m generate_config --mtune-method ensemble --output config.ensemble.generated.json
```

If that output already exists, use it or choose a different filename. For a config including fixed-parameter experiments (`--optimization both`), the generator writes these settings. Grid-only configs omit the scalar `n_ensemble`:

```json
"mtune": {"method": "ensemble", "n_ensemble": 11},
"mtune_grid": {"n_ensemble": [3, 5, 7, 9, 11]}
```

The scalar `mtune.n_ensemble` is used for optimization=default runs. For optimization=grid runs, `mtune_grid.n_ensemble` is searched jointly with the classifier grid, using the configured refit metric (balanced accuracy in newly generated configs). The scalar is overridden by each candidate and the final estimator uses the selected value. The selected `n_ensemble` appears in `best_params`, the candidate values in `cv_results.csv`, and the final model/threshold list reflect that selected ensemble size.

`mtune_grid` applies only to M-Tune, not default-threshold estimators. Supplying a nonempty `mtune_grid` for direct-mode grid runs raises a clear error instead of searching an irrelevant parameter. Existing ensemble configs without `mtune_grid` retain their fixed ensemble size. Sizes must be positive integers and no larger than the majority count in the smallest inner training fold; invalid candidates fail rather than silently being removed. Odd sizes avoid tied hard votes, though positive even sizes are supported by the package. Five ensemble candidates multiply the base parameter candidate count by five; the number of outer experiments is unchanged, but training cost increases substantially.

Ensemble training and threshold estimation still occur entirely inside each CV training fold. The winning ensemble is refitted on the full outer training set. Test rows never enter the search. Feature caching and artifact-verified resume are unchanged; source/config changes produce new run identities.

Ensemble-grid update validation: 29 tests passed, including the joint search under AP, balanced accuracy and F2 refit, actual fitted ensemble sizes, scaffold/test leakage audits, resume, invalid-size rejection, and generator output.

Generic discovery validation: 33 tests passed. Coverage includes arbitrary column names, the exact 50% boundary, missing/numeric entries, ambiguous candidates, full-file chunk scanning, blank-target run blocking, compressed files, explicit target overrides, non-overwrite behavior and existing training/search/resume tests. Actual local discovery found bace/mol, BBBP/smiles, HIV/smiles and muv/smiles; all default targets were left blank in config.discovered.json. Existing configs retain their previous selections.


Generate a grid-only ensemble config with balanced-accuracy refit:

```powershell
.\.venv312\Scripts\python.exe -m generate_config --mtune-method ensemble --refit-metric balanced_accuracy --optimization grid --output config.grid.json
```

Fill blank target columns and review detected SMILES columns before running. Existing configs are not modified by the new flag. Choose an unused output filename.


## Current default generation command

```powershell
.\.venv312\Scripts\python.exe -m generate_config
```

Equivalent to `--mtune-method ensemble --refit-metric balanced_accuracy --optimization grid --output config.grid.json`. Existing files are never overwritten; use `--output another-name.json` if `config.grid.json` already exists. Fill blank target columns and review detected SMILES before training. This changes generation defaults only; existing configuration files and the runner's fallback settings remain unchanged.

Default-settings update validation: 39 tests passed, including matching no-argument API/CLI output, overwrite protection, omission of redundant fixed ensemble size, and explicit alternatives.
# Reading experiment results

New output folders describe their contents, for example:

```text
results/
  experiments.csv
  performance_summary.csv
  summary.csv
  _runs/
    <shared-run-id>.json
  bace__morgan__random__random_forest__mtune-ensemble__grid__seed-42__<version-id>/
    metrics.csv
    model_details.json
    result.json
    manifest.json
    cv_results.csv
    predictions.csv
    model.joblib
    split.json
    cv.json
```

Start with **performance_summary.csv**: it lists all completed experiments and separately labeled train, cv_validation_mean and test rows. **experiments.csv** maps experiment choices to folder names. Both update after each completed experiment and on resume, rather than only at the end of the batch. **summary.csv** is retained as the backward-compatible held-out test table.

Within one experiment, open **metrics.csv**. The train row evaluates the final model on its own training data; it is an optimistic fitting diagnostic. The cv_validation_mean row reports the selected candidate's inner-validation means. The test row reports untouched outer-test performance. Specificity/precision/recall/F1/F2 use the training-minority-positive convention. AP and ROC-AUC use continuous minority probabilities. Model selection still uses only inner CV; adding training metrics does not influence selection.

**model_details.json** identifies the classifier, strategy, selected grid parameters and final model settings. **result.json** includes only that experiment's dataset specification, relevant feature/model/search settings, selected parameters and performance. It no longer repeats all input datasets. The old metrics key remains an alias for held-out test metrics. The complete original configuration is stored at run level under **_runs/**, with a relative reference from each experiment. Internal versions and hashes remain for reproducibility.

The previous seemingly random directory names were SHA-256-derived experiment IDs, used for deterministic identity and resume checks. New names are descriptive and retain a short version suffix so differing configurations cannot overwrite one another. The full ID remains in metadata.

Training creates these folders automatically. Existing configurations need no layout changes. Exact completed experiments still resume after artifact verification; source/config/data/environment changes create new identities as before. Do not run multiple writers against the same results directory.

To reorganize completed old result folders without retraining:

```powershell
.\.venv312\Scripts\python.exe -m molpipeline --organize-results results
```

The organizer verifies original artifact checksums, backs up original result/manifest metadata under _runs/legacy_metadata/, renames completed legacy folders and writes focused metadata/tables. It never loads/refits models or changes predictions. Incomplete or checksum-mismatched folders are left untouched. Migration is idempotent. Run organization when no training process is writing to that directory.

Historical training scores were not saved and are marked not_recorded; they are not invented or mislabeled as CV scores. New experiments save final-model training scores. Default-optimization experiments have no CV search and their CV row is also not_recorded. Empty output directories have summary headers only until experiments complete.

## Invalid SMILES in downloaded datasets

RDKit may reject structures such as metal complexes with unsupported valence. The loader remains strict when the policy is omitted; newly generated and example configs explicitly select audited skipping by default. To exclude rejected structures explicitly, add `"invalid_smiles": "skip"` to the relevant dataset entry (not the matrix). This does not change the raw CSV or weaken sanitization; invalid rows are removed before deduplication and any train/test/CV split. Missing values, invalid labels and conflicting canonical duplicate labels still fail. Both classes must remain after filtering.

Each new experiment saves `excluded_rows.csv` (original 1-based data-record position, SMILES, label, reason) and `data_quality.json` (input, invalid, duplicate and retained counts, plus class counts). CSV positions exclude the header and refer to data records, not necessarily physical lines for quoted multiline CSVs. `split.json` and `predictions.csv` also retain original source data-record positions. This makes the changed evaluation population explicit. Feature caches and result identities continue to reflect the retained molecules and configured policy.

For the reported HIV valence error, `config.grid.skip-invalid.json` copies the user's current settings and opts into skip mode; the original `config.grid.json` is preserved. Run:

```powershell
.\.venv312\Scripts\python.exe -m molpipeline config.grid.skip-invalid.json
```

This is a deliberate data-exclusion policy; report excluded counts when interpreting results. It does not repair the chemical structures. Validation: 45 tests passed, including strict rejection, the reported aluminium structure, source-file preservation, audited exclusion, class-loss/conflict rejection and saved-report checks.


Generated dataset entries now include `"invalid_smiles": "skip"` by default. The current `config.grid.json` was also updated, with its previous contents backed up under `work/config.grid.before-default-skip.json`. Invalid structures stay in the original raw CSV and are recorded in each experiment's `excluded_rows.csv`; they do not enter retained features, train/test partitions or CV. Use `"invalid_smiles": "error"` to opt into strict rejection.

## Human-readable features with labels

Normal experiment runs now automatically export shared CSVs under `results/features/` (or the configured output_dir/features). Each contains `cleaned_row` (zero-based cleaned dataset index), `source_data_row` (1-based original CSV data-record position excluding header), canonical `smiles`, the original binary `label`, and all feature columns. Morgan columns are named `morgan_bit_0`, etc.; ChemBERTa columns are named `chemberta_embedding_0`, etc. Invalid molecules and canonical duplicates are absent, consistently with training. The exported label retains the original 0/1 meaning; minority-positive remapping is recorded separately in experiment results.

An accompanying JSON records dataset/target, feature settings, environment, row counts, cleaning counts and the CSV checksum. A shared export identity includes the dataset/target specification, source hash, retained molecules, labels, source rows, feature shape/type, settings and versions. It does not include classifier, split, threshold strategy, optimizer or seed, so those experiments reuse one CSV. The `.npy` feature cache remains available for fast numeric loading. Export checksums are checked before reuse; deleted/corrupted exports are regenerated. If only an export is missing when resuming a verified completed experiment, it is rebuilt without refitting the model.

Each experiment's `feature_export.json`, manifest and result refer to its shared CSV using relative paths. The export contains the entire retained dataset before splitting; use split.json/cv.json cleaned-row assignments to identify training/test/CV partitions. No fitted preprocessing is learned from held-out data: Morgan is stateless and ChemBERTa remains frozen. Merely exporting labels does not use them for training or tuning.

To export without training, even for a dataset too small to split:

```powershell
.\.venv312\Scripts\python.exe -m molpipeline config.grid.json --export-features
```

This uses the selected datasets/features and still requires valid configured target columns and loader validation. It computes features or reuses existing `.npy` caches, does not construct splits or fit classifiers, and deduplicates dataset/feature pairs across the experiment matrix. ChemBERTa requires its model dependencies and pinned revision. CSVs are plain UTF-8 and can be large (especially Morgan's thousands of columns); they are intended for inspection/export, not for loading arbitrary edited data back into training. Floating embeddings are written with 17 significant digits. If you move the results directory, keep its shared features folder with the experiment folders.

Validation: 48 tests passed, covering Morgan and synthetic ChemBERTa embedding CSV precision/schema, identifier/label alignment, label-change invalidation, corruption repair, shared exports across experiment combinations, resume without refit, and feature-only operation without splitting/training. Actual ChemBERTa inference was not exercised by these export tests.


Generator split default: new configs now contain `"split": ["random"]` only. Scaffold splitting remains supported; explicitly edit the generated matrix to `["scaffold"]` or `["random", "scaffold"]` when desired. Existing config files and the runner fallback axes are unchanged. Default matrix size is eight experiments per dataset and seed (Morgan and ChemBERTa, random split, two classifiers, two threshold strategies, grid search).


Dataset discovery is nonrecursive: only `.csv` and `.csv.gz` files directly inside `data/raw` (or the selected `--data-dir`) are considered. Files in subdirectories are ignored. Existing configs are not modified; generate a new config to apply discovery changes. The training runner still loads exactly the paths explicitly listed in its config.


## Both feature types by default

Running `python -m generate_config` now selects `"feature": ["morgan", "chemberta"]`, including for the smoke preset. The generated ChemBERTa settings use `DeepChem/ChemBERTa-77M-MLM` pinned to commit `ed8a5374f2024ec8da53760af91a33fb8f6a15ff` (verified via the Hugging Face model API on 2026-11-03). Config generation remains offline; `--chemberta-revision SHA` is an optional explicit override. Existing configs are not changed. To regenerate without overwriting your current config:

```powershell
.\.venv312\Scripts\python.exe -m generate_config --output config.both-features.json
```

Fill blank target fields before training. ChemBERTa requires the optional dependencies; install in the same environment:

```powershell
.\.venv312\Scripts\python.exe -m pip install -e "."
```

The first ChemBERTa feature calculation downloads model/tokenizer weights. Morgan fingerprints are binary vectors; ChemBERTa features are continuous frozen-model embeddings. To select only one later, edit `matrix.feature` explicitly. No large model download or real ChemBERTa inference is performed by the generator-default tests.


## Current dependency setup (2026-11-03)

CatBoost, Torch and Transformers are now main project dependencies. Only pytest remains in the optional `test` extra. There is no package named chemberta to add: the ChemBERTa model is loaded by Transformers and Torch.

```powershell
uv sync --python 3.12
uv run python -m generate_config
uv run python -m molpipeline config.grid.json
```

Use `uv sync --extra test --python 3.12` and `uv run --extra test python -m pytest tests` for testing. Pip alternative: `python -m pip install -e .`. Old `--extra chemberta` / `--extra catboost` commands are no longer needed. uv uses the project's `.venv`; use `uv run` consistently rather than the previous `.venv312` interpreter. The first ChemBERTa inference still downloads model weights; installing Torch/Transformers alone does not download the model.

## Experiment durations in summary tables

`performance_summary.csv`, `experiments.csv`, and the backward-compatible `summary.csv` now include `elapsed_seconds` and `elapsed_minutes`, taken from the saved experiment duration. This is total recorded experiment elapsed time, not training-only time or a new overall command timer. To avoid double counting the train/CV/test rows, performance_summary places duration only on the test row; train and CV duration cells are blank. You can sum elapsed_seconds directly across that file, or use experiments.csv (one row per experiment). Historical results without a recorded duration remain blank, not zero. Summing durations from resumed/historical experiments is not the current session's wall-clock duration.

Refresh existing reports without loading datasets or training models:

```powershell
uv run python -m molpipeline --refresh-summaries results
```

New runs also update summaries after each completed experiment. Experiment JSON/model files remain unchanged by report refresh.

## Original-label class metrics and partition timings

`metrics.csv` and `performance_summary.csv` now include recall_0/1, precision_0/1, specificity_0/1, f1_0/1, f2_0/1, average_precision_0/1, and roc_auc_0/1. Suffixes always mean the original CSV label 0 or 1, not the minority/majority ordering. Training and test rows also include support_0/1 and per-class one-vs-rest confusion counts where available. Accuracy, balanced accuracy and MCC stay shared metrics rather than misleading class-specific duplicates. Specificity for class 0 equals recall for class 1, and vice versa. Binary complementary class ROC-AUC values agree; class AP values need not agree.

Bare minority-oriented score columns in the readable tables are now explicitly named minority_recall, minority_precision, minority_specificity, minority_f1, minority_f2, minority_average_precision, and minority_roc_auc. The original JSON metric aliases and grid refit objective names remain backward compatible. GridSearchCV records both original-label class scores alongside the original eight scoring objectives; refit selection still uses the configured minority-positive balanced_accuracy/AP/F2 objective. AP/AUC always use continuous probabilities for the corresponding original class.

Each partition row now has fit_seconds, score_seconds, partition_elapsed_seconds, partition_elapsed_minutes, timing_scope and timing_status:

- train: final full-training fit/refit plus prediction/metric calculation on that training set. The fit is the selected final estimator's refit, not the entire grid search.
- cv_validation_mean: mean fit plus mean scoring time per inner fold for the selected parameter candidate. This is an average, not total grid-search wall time. GridSearchCV measures these fold times.
- test: held-out predictions and metric calculation only; fit_seconds is zero because the test data are never fitted.

Total grid_search_seconds records the wall time of the complete search and its final refit. Overall elapsed_seconds/elapsed_minutes retain the complete experiment timer and appear only on the test row to avoid triple counting. These timings overlap or represent averages, so summing the partition times does not recover total experiment time. Feature generation, CSV exports, model saving and other overhead are included only in the overall experiment duration, not scoring timings. All new timing measurements use a monotonic performance counter.

Refresh existing reports without training:

```powershell
uv run python -m molpipeline --refresh-summaries results
```

Old metrics are mapped to original labels wherever mathematically recoverable from saved label mappings and confusion counts. Unknown opposite-class AP or other unavailable historical CV class means are left blank, not inferred from aggregate means. Historical final-fit/train-scoring/test-scoring durations cannot be reconstructed and remain blank with timing_status=not_recorded. Existing CV candidate mean fit/score times can be recovered. New experiments record all class-specific metrics and partition timings directly. Refresh never modifies models, predictions or original experiment artifacts.

If a summary CSV is open/locked (for example in Excel), refresh writes a `.updated.csv` sibling instead of failing or replacing the open file. Close the original and refresh again to update the standard filename. Per-class/timing update validation: 55 tests passed, including both minority-label orientations, reversed probability class ordering, all three refit objectives, ensemble tuning, exact mocked evaluation timing, historical metric mapping, unavailable historical times and locked-summary fallback. Existing upstream deprecation warnings remain.

## Validate data and save cleaned CSVs before training

```powershell
uv run python -m molpipeline config.grid.json --validate-only
```

Normal training and feature-only exports perform the same preflight before proceeding. All selected datasets are checked, with each prepared only once per run. If any dataset fails, all selected datasets still receive validation reports, then fitting stops. Raw files are never modified and conflicting labels are never silently dropped or relabeled.

Outputs use readable local-machine timestamps (with `_02`, `_03`, etc. on same-second collisions):

```text
results/
  validation/
    2026-11-05_10-45-43/
      run.log
      validation_summary.json
      BBBP/
        validation.json
        label_conflicts.csv       # only when conflicts exist
        excluded_rows.csv         # invalid rows skipped under explicit policy
      HIV/
        validation.json
        data_quality.json
        cleaned.csv               # only after this dataset passes
        excluded_rows.csv
```

The base directory follows the configured output_dir. Dataset names are sanitized for safe Windows folder names, with readable numeric suffixes for collisions. The summary maps original dataset names to the actual report folders. Old logs/validation outputs are preserved and are not automatically moved.

`validation_summary.json` lists each dataset's pass/fail status and report paths. Each dataset's `validation.json` contains only that dataset's validation details. `run.log` records validation, experiment progress and exceptions. Failing experiments also record their stage in error.json.

`label_conflicts.csv` identifies the dataset, source path, target column, original source data row, original SMILES, canonical SMILES and label. Use source_excel_row to locate the row in the original dataset opened in Excel: header=1, first data record=2. source_data_row remains the original 1-based data record position excluding the header. These are CSV records rather than physical text lines for quoted multiline values.

`cleaned.csv` is saved only for datasets that pass validation, even if another selected dataset fails. It contains exactly `row_number`, `original_smiles`, `processed_smiles` and `label`. The row_number locates the retained record in the original source CSV opened in Excel (header=1); processed_smiles is the canonical SMILES used for feature generation. Labels retain their original 0/1 values. Invalid structures are removed only under invalid_smiles=skip; consistent canonical duplicates retain the first occurrence. Their counts are recorded in data_quality.json. A dataset with conflicting labels does not get a cleaned.csv that could be mistaken as ready for fitting. Fix the reported labels or prepare an explicitly audited conflict-exclusion policy first.

The current run uses the same prepared table in memory; it does not reread the saved cleaned CSV. To reuse a cleaned file in a later config, point `path` at it and set smiles_column=processed_smiles and target_column=label. Its source row columns refer to the original raw file as saved; a later load of that cleaned file numbers source records relative to that input file. Do not confuse those provenance conventions.

`--plan` remains a matrix preview. Per-experiment elapsed times exclude shared preflight. Existing completed artifacts remain unchanged; code changes can produce new experiment IDs under existing resume rules.


The cleaned.csv inspection file represents all retained molecules before splitting, not only the training partition. Use each experiment's split.json to identify its training/test assignment. Unresolved label conflicts still prevent a dataset from producing a file marked as cleaned.

## Cross-platform cache directories

No OS-specific launcher is required. Use the same commands on Windows or Linux:

```shell
uv sync --python 3.12
uv run python -m generate_config
uv run python -m molpipeline config.grid.json
```

Python pathlib resolves `.cache/features` and `.cache/runtime` relative to the config file. Model downloads, Torch/compiler caches and temporary training files are routed beneath the runtime cache; paths are recorded in each validation run's cache_locations.json. The context restores process environment settings on exit. `MOLPIPELINE_RUNTIME_CACHE` and `MOLPIPELINE_FEATURE_CACHE` remain optional environment overrides. uv package caching defaults to `.cache/uv` via pyproject.toml.

Put the project/config on the drive where you want its caches stored. Existing C-drive caches are not deleted or migrated. Python runtime settings cannot redirect uv-managed Python downloads, operating-system files or bytecode produced before the program starts; those remain subject to the external tool/system settings. The former project.ps1 launcher has been removed.

## Conflict resolution policy (default: automatically prefer input minority)

Generated dataset entries now include:

```json
"label_conflicts": "prefer_minority",
"minority_label": "auto"
```

Missing or null minority_label also means auto for backward compatibility. The original input CSV's target counts, before invalid-SMILES removal, deduplication or conflict resolution, determine whether 0 or 1 is the minority. No label value is hardcoded. For each conflicting canonical molecule, the first source record with that minority label is retained; opposite-label records are excluded and additional matching-label duplicates collapse to one molecule. Tied input counts with conflicts fail explicitly because no automatic minority exists. An explicit minority_label=0/1 remains a manual override; prefer_label with preferred_label=0/1, error and exclude_all are available alternatives.

Each resolved conflict report records original and processed SMILES, source row numbers, label, retained/excluded action, policy, selected label, preference source and both original input class counts. data_quality.json and validation details record the chosen label, counts before/after cleaning, exclusions and dataset-level cleaning scope. Original CSVs remain unchanged. cleaned.csv contains the actual retained rows and original labels; both classes must remain after preparation.

Automatic preference is a WHOLE-DATASET cleaning assumption: it uses labels from records that later enter the test set and favors the minority label for ambiguous molecules. This departs from strict leakage-free label adjudication and can change evaluation results. It is not evidence that the retained label is scientifically correct. Report this policy when interpreting benchmark results; use error/exclude_all or an external fixed preference for stricter adjudication. Hyperparameter search, threshold fitting and metric positive-label selection still use only training partitions.
