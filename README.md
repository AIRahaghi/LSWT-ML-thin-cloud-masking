# Landsat Thin-Cloud Masking Over Water
Abolfazl Irani Rahaghi, Eawag, Switzerland, 2026
abolfazl.irani@eawag.ch

This repository restructures the old notebook workflow into three reusable pipelines:

1. Train, tune, and assess the Decision Tree, Random Forest, and XGBoost classifiers from prepared train/test CSV files.
2. Process one pre-downloaded Landsat-8 or Landsat-9 Collection-2 Level-1 scene over one lake polygon. The output is a cropped Fmask v3 QA layer plus RF and XGBoost thin-cloud layers.
3. Run the same masking workflow for a batch of pre-downloaded scenes.

The workflow follows the manuscript and old notebooks: operational Landsat `QA_PIXEL` Fmask bits are applied first, then ML classifiers are applied only to pixels that are inside the lake polygon and initially flagged as clear water.

## Install

Create and activate a Python environment, then install the package in editable mode:

```powershell
pip install -e .
```

The geospatial steps require GDAL-compatible `rasterio`. If pip installation struggles on Windows, use a conda environment for `rasterio`, then install the rest with pip.

## Pipeline 1: Train, Tune, Assess

Edit [configs/training_config.example.json](configs/training_config.example.json) or pass paths directly:

```powershell
python scripts/train_tune_assess_models.py `
  --train-csv "E:/Trishna/Landsat_processing/processed/train_test_all_merged/df_train_all.csv" `
  --test-csv "E:/Trishna/Landsat_processing/processed/train_test_all_merged/df_test_all.csv" `
  --output-dir "models/general"
```

Outputs include tuned `.pkl` models, Optuna study files, feature metadata, classification reports, confusion matrices, and summary metrics.

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

## Notebooks

- [notebooks/01_train_tune_assess_models.ipynb](notebooks/01_train_tune_assess_models.ipynb)
- [notebooks/02_lake_geneva_single_scene_masking.ipynb](notebooks/02_lake_geneva_single_scene_masking.ipynb)

The notebooks are thin wrappers around the scripts and package modules. Keep analysis and figures there, but keep reusable logic under `src/lswt_cloud_masking`.
