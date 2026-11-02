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
