# Landsat Thin-Cloud Masking Over Water
Abolfazl Irani Rahaghi, Eawag, Switzerland, 2026
abolfazl.irani@eawag.ch

This repository restructures the old notebook workflow into three reusable pipelines:

1. Train, tune, and assess the Decision Tree, Random Forest, and XGBoost classifiers from prepared train/test CSV files.
2. Process one pre-downloaded Landsat-8 or Landsat-9 Collection-2 Level-1 scene over one lake polygon. The output is a cropped Fmask v3 QA layer plus RF and XGBoost thin-cloud layers.
3. Run the same masking workflow for a batch of pre-downloaded scenes.
4. Regenerate the manuscript-style `df_l1_*.csv` and `df_l2_*.csv` lake tables from pre-downloaded C2L1/C2L2 products using the selected final models in `models/general`.

The workflow follows the manuscript and old notebooks: operational Landsat `QA_PIXEL` Fmask bits are applied first, then ML classifiers are applied only to pixels that are inside the lake polygon and initially flagged as clear water.

## Install

Create and activate a Python environment, then install the package in editable mode:

```powershell
pip install -e .
```

For a standalone Windows computer, the most reliable route is usually conda-forge because `rasterio` depends on GDAL/PROJ:

```powershell
conda env create -f environment.yml
conda activate lswt-thin-cloud
pip install -e .
```

No Landsat download code is included. Scene masking expects pre-downloaded C2L1 scene folders or tar files. The operational Fmask layer is read from the Landsat `QA_PIXEL` band, so a separate Fmask executable is not required for this pipeline.

## Pipeline 1: Train, Tune, Assess

The training data are now divided into three roles:

- `df_train_80.csv`: model fitting and cross-validation (80%).
- `df_test_inscene_10.csv`: familiar-scene test pixels (10%).
- `df_test_los_10.csv`: leave-out-of-scene (LOS) test; its scenes occur nowhere else (10%).

Create or refresh these files whenever `df_train_all.csv` or `df_test_all.csv` changes:

```powershell
python scripts/create_training_splits.py --config "configs/training_split.example.json"
```

The split is deterministic, validates that every split contains every lake, all three classes,
Landsat 8/9, and all four meteorological seasons, and keeps irreplaceable lake/class examples
available for model development. The scene assignment and coverage audit are saved in
`data/scene_split_manifest.csv` and `data/scene_split_summary.json`.

Then run the full training pipeline:

```powershell
python scripts/train_tune_assess_models.py `
  --train-csv "data/df_train_80.csv" `
  --test-csv "data/df_test_inscene_10.csv" `
  --los-test-csv "data/df_test_los_10.csv" `
  --output-dir "models/general/80_10_10"
```

With the copied CSV files already under `data/`, this shorter command also works:

```powershell
python scripts/train_tune_assess_models.py --config "configs/training_config.example.json"
```

The `--output-dir` value is the destination for a standalone training run. The downstream masking and lake-table pipelines expect the selected final model files in `models/general`: `DT_best_all_general.pkl`, `RF_best_all_general.pkl`, and `XGB_best_all_general.pkl`.

Outputs include tuned `.pkl` models, one Optuna study per tuning seed, candidate-ranking CSVs, feature metadata, familiar/LOS classification reports and confusion matrices, and summary metrics. `n_trials_dt`, `n_trials_rf`, and `n_trials_xgb` are trial counts **per tuning seed**, so the default three-seed search performs three times that number for each model. Full tuning is intentionally computationally expensive; use `--no-tuning` only as a basic pipeline check, not for the reported final models.

Each model is tuned independently with the three CV seeds in `tuning_cv_seeds`. The top `top_candidates_per_run` parameter sets from every run are deduplicated and re-evaluated over all repeated scene-grouped folds. Final selection maximizes `mean grouped balanced accuracy - stability_penalty × fold standard deviation`; `selection_scoring` fixes that metric to balanced accuracy and the default penalty is `0.25`. This favors both scene-level accuracy and consistency. DT and RF use scene-grouped tuning throughout. XGBoost retains its grouped + pixel-stratified multi-objective search for candidate discovery, then uses the same repeated grouped stability criterion as DT/RF for final selection. Every XGBoost fold uses early stopping, and the final tree count is the median best iteration over the repeated grouped folds. Neither test CSV participates in tuning, early stopping, or candidate ranking. The `scene_id` column is metadata only and is excluded from model features.

`metrics_summary.json` reports train, familiar test, LOS test, scene-grouped CV, and pixel-stratified CV accuracy, balanced accuracy, and macro F1. Use LOS results as the strict unseen-scene assessment; familiar-test and pixel-CV results estimate interpolation among known scenes.

### Repeated-split sensitivity analysis

To measure sensitivity to the particular 80/10/10 allocation, run the complete split-and-training workflow for five data-split seeds:

```powershell
python scripts/run_training_sensitivity.py `
  --config "configs/training_sensitivity.example.json"
```

First verify the workflow with a short one-split smoke test:

```powershell
python scripts/run_training_sensitivity.py `
  --config "configs/training_sensitivity.example.json" `
  --smoke-test
```

Smoke runs are recomputed on each invocation, which makes it safe to edit their trial counts between checks. Full sensitivity runs continue to resume matching completed realizations.

Every realization has the same row fractions and coverage constraints but a different LOS scene allocation and familiar-test pixel allocation. Exact datasets are retained under `outputs/training_sensitivity_80_10_10/datasets/run_*`; all corresponding models, Optuna studies, candidate rankings, and detailed metrics are retained under `outputs/training_sensitivity_80_10_10/models/run_*`. Existing matching completed runs are resumed automatically.

The sensitivity-run folders are archived experiment outputs. After choosing the preferred realization, copy its three final model files from `outputs/training_sensitivity_80_10_10/models/run_*` into `models/general`; the downstream examples below use that canonical model directory.

The referenced split and training JSON files are templates. The sensitivity runner uses `df_train_all.csv` and `df_test_all.csv` as the source rows, overrides the template `train_csv`, `test_csv`, `los_test_csv`, and `output_dir` values for every realization, creates the three run-specific CSVs first, and only then starts tuning. Path overrides through `training_overrides` are rejected to prevent accidental reuse of the canonical `data/df_*_80/10.csv` files.

The two main comparison files are:

- `sensitivity_results.csv`: one row per realization and classifier, including train, familiar-test, and LOS balanced accuracy and the exact dataset/model paths.
- `sensitivity_summary_by_model.csv`: mean, sample standard deviation, minimum, and maximum balanced accuracy across realizations.

`los_scene_overlap.csv` records how much the held-out scene sets overlap between realizations. The model-training seed and inner tuning/CV procedure remain fixed across outer data splits, so the comparison isolates sensitivity to dataset allocation. The default sensitivity configuration inherits the full trial counts and three inner CV seeds from `training_config.example.json`; consequently, five complete realizations take about five times as long as one complete training run. Use `training_overrides` in `training_sensitivity.example.json` if a smaller but consistent per-realization search budget is required.

The familiar and LOS scores are never supplied to Optuna or candidate selection. Compare their distributions across runs, but do not choose the final model solely from the highest LOS score, because doing so would make LOS part of model selection.

## Pipeline 2: One C2L1 Scene

Use a scene folder or `.tar` containing the Landsat C2L1 files (`*_B1.TIF`, `*_B2.TIF`, ..., `*_B11.TIF`, `*_QA_PIXEL.TIF`, `*_MTL.txt` or `*_MTL.json`):

```powershell
python scripts/mask_single_scene.py `
  --scene "E:/Trishna/Landsat_processing/Landsat_C2/Geneva_L1/LC08_L1TP_195028_20240705_20240712_02_T1" `
  --lake-geojson "data/lake_geneva_simple.geojson" `
  --rf-model "models/general/RF_best_all_general.pkl" `
  --xgb-model "models/general/XGB_best_all_general.pkl" `
  --output-dir "outputs/geneva_20240705"
```

The script writes:

- `*_fmask_v3.tif`: operational Fmask-derived lake mask.
- `*_rf_thin_cloud.tif`: RF layer applied on top of Fmask.
- `*_xgb_thin_cloud.tif`: XGBoost layer applied on top of Fmask.
- `*_mask_stack.tif`: three-band stack in the order Fmask, RF, XGBoost.
- `*_summary.json`: pixel counts and percentages.

Layer values:

- Fmask layer: `0 = outside/nodata`, `1 = Fmask clear water`, `2 = Fmask cloud/cirrus/shadow/dilated cloud`, `3 = other`.
- RF/XGBoost layers: `0 = outside/not evaluated`, `1 = thin cloud`, `2 = cloud-affected`, `3 = water`.

## Pipeline 3: Batch Processing

Prepare a manifest like [configs/batch_scenes.example.csv](configs/batch_scenes.example.csv), then run:

```powershell
python scripts/mask_batch.py `
  --manifest "configs/batch_scenes.example.csv" `
  --output-dir "outputs/batch"
```

The batch script assumes scenes are already downloaded and never downloads Landsat products.

## Pipeline 4: Regenerate L1/L2 Lake Tables

This pipeline reproduces the old `df_l1_*.csv` and ML-enhanced `df_l2_*.csv` tables with cleaner reusable code. It expects an external-drive folder layout equivalent to the old Windows path `D:\Trishna\Landsat_processing\Landsat_C2`, for example on macOS:

```text
/Volumes/YOUR_EXTERNAL_DRIVE/Trishna/Landsat_processing/Landsat_C2/
  Geneva_L1/
  Geneva_L2_acolite/
  Geneva_L2_usgs/
```

Edit [configs/lake_tables.example.json](configs/lake_tables.example.json), especially `landsat_c2_root` and `models_dir`. Use `models/general` for the selected final models unless you intentionally want to test a run-specific folder under `outputs/training_sensitivity_80_10_10/models/run_*`. Then run one lake:

```bash
python scripts/regenerate_landsat_tables.py \
  --config configs/lake_tables.example.json \
  --lake geneva
```

Or run all configured lakes:

```bash
python scripts/regenerate_landsat_tables.py --config configs/lake_tables.example.json
```

Outputs are written to:

- `data/landsat_l1_tables/df_l1_<lake>.csv`
- `data/landsat_l2_tables/df_l2_<lake>.csv`
- `data/landsat_l2_tables/df_landsat_generation_report_<lake>.json`

The default ML LST filtering masks class `1`, matching the old `df_diff_ml_stats_*` workflow. To remove several ML classes from LST maps, pass repeated flags such as `--mask-cloud-class 1 --mask-cloud-class 2`.

## Dependencies

Core training dependencies:

- `numpy`, `pandas`
- `scikit-learn`, `xgboost`, `optuna`, `joblib`
- `matplotlib`, `seaborn` for reports/plots

Scene and batch masking dependencies:

- `rasterio`, including its GDAL/PROJ runtime stack
- `netCDF4` for ACOLITE L1R/ST files
- `scipy` for nearest-neighbor resampling between USGS and ACOLITE grids

Notebook-only dependencies:

- `jupyterlab` or `notebook`
- `ipykernel`

## Notebooks

- [notebooks/01_train_tune_assess_models.ipynb](notebooks/01_train_tune_assess_models.ipynb)
- [notebooks/02_lake_geneva_single_scene_masking.ipynb](notebooks/02_lake_geneva_single_scene_masking.ipynb)
- [notebooks/03_regenerate_landsat_tables_one_lake.ipynb](notebooks/03_regenerate_landsat_tables_one_lake.ipynb)
- [notebooks/04_regenerate_landsat_tables_batch.ipynb](notebooks/04_regenerate_landsat_tables_batch.ipynb)

The notebooks are thin wrappers around the scripts and package modules. Keep analysis and figures there, but keep reusable logic under `src/lswt_cloud_masking`.
