"""Feature engineering shared by training and scene masking."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


RAW_FEATURE_NAMES = [
    "bluemap",
    "greenmap",
    "redmap",
    "nirmap",
    "swirmap1",
    "swirmap2",
    "cirrusmap",
    "aerosolmap",
    "btmap1",
    "btmap2",
]

SPECTRAL_INDEX_NAMES = [
    "hotmap",
    "whitenessmap",
    "ndsimap1",
    "ndsimap2",
    "ndvimap",
    "ndwimap",
    "mndwimap1",
    "mndwimap2",
    "waterratiomap1",
    "waterratiomap2",
    "brtestmap1",
    "brtestmap2",
]

MODEL_FEATURE_NAMES = RAW_FEATURE_NAMES + SPECTRAL_INDEX_NAMES

DEFAULT_DROP_COLUMNS = [
    "lakeN",
    "scene_id",
    "lst_raw",
    "lst_filt",
    "raamap",
    "vzamap",
    "szamap",
]

CLASS_LABELS = {
    1: "thin_cloud",
    2: "cloud_affected",
    3: "water",
}


def as_float_array(arr: Any) -> np.ndarray:
    """Return a float ndarray with masked values converted to NaN."""

    if np.ma.isMaskedArray(arr):
        return np.asarray(arr.filled(np.nan), dtype="float32")
    return np.asarray(arr, dtype="float32")


def safe_divide(numerator: Any, denominator: Any) -> np.ndarray:
    """Divide arrays while converting zero denominators and infinities to NaN."""

    numerator_arr = as_float_array(numerator)
    denominator_arr = as_float_array(denominator)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = numerator_arr / denominator_arr
    out = np.asarray(out, dtype="float32")
    out[~np.isfinite(out)] = np.nan
    return out


def compute_spectral_indices(feature_arrays: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Compute the spectral indices used in the paper and old notebooks."""

    blue = as_float_array(feature_arrays["bluemap"])
    green = as_float_array(feature_arrays["greenmap"])
    red = as_float_array(feature_arrays["redmap"])
    nir = as_float_array(feature_arrays["nirmap"])
    swir1 = as_float_array(feature_arrays["swirmap1"])
    swir2 = as_float_array(feature_arrays["swirmap2"])

    mean_vis = (blue + green + red) / 3.0
    whiteness = (
        np.abs(safe_divide(blue - mean_vis, mean_vis))
        + np.abs(safe_divide(green - mean_vis, mean_vis))
        + np.abs(safe_divide(red - mean_vis, mean_vis))
    )

    return {
        "hotmap": blue - 0.5 * red,
        "whitenessmap": whiteness.astype("float32"),
        "ndsimap1": safe_divide(green - swir1, green + swir1),
        "ndsimap2": safe_divide(green - swir2, green + swir2),
        "ndvimap": safe_divide(nir - red, nir + red),
        "ndwimap": safe_divide(green - nir, green + nir),
        "mndwimap1": safe_divide(green - swir1, green + swir1),
        "mndwimap2": safe_divide(green - swir2, green + swir2),
        "waterratiomap1": safe_divide(green + red, nir + swir1),
        "waterratiomap2": safe_divide(green + red, nir + swir2),
        "brtestmap1": safe_divide(swir1, nir),
        "brtestmap2": safe_divide(swir2, nir),
    }


def build_feature_arrays(
    raw_arrays: Mapping[str, Any],
    valid_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Return raw features plus spectral indices, optionally masking invalid pixels."""

    arrays = {name: as_float_array(value).copy() for name, value in raw_arrays.items()}
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        for name in arrays:
            arrays[name][~mask] = np.nan

    arrays.update(compute_spectral_indices(arrays))
    return arrays


def model_feature_columns(model: Any | None, available_columns: Sequence[str]) -> list[str]:
    """Choose feature order from a fitted model when possible."""

    if model is not None and hasattr(model, "feature_names_in_"):
        columns = [str(col) for col in model.feature_names_in_]
    else:
        columns = [col for col in MODEL_FEATURE_NAMES if col in available_columns]

    missing = [col for col in columns if col not in available_columns]
    if missing:
        raise ValueError(f"Missing feature columns required by model: {missing}")
    return columns


def feature_dataframe(
    feature_arrays: Mapping[str, Any],
    valid_mask: np.ndarray,
    columns: Sequence[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Flatten selected arrays into a DataFrame and return the valid flat indices."""

    mask = np.asarray(valid_mask, dtype=bool)
    flat_mask = mask.ravel().copy()
    for col in columns:
        arr = as_float_array(feature_arrays[col])
        flat_mask &= np.isfinite(arr.ravel())

    flat_indices = np.flatnonzero(flat_mask)
    data = {col: as_float_array(feature_arrays[col]).ravel()[flat_indices] for col in columns}
    return pd.DataFrame(data, columns=list(columns)), flat_indices


def prepare_training_frame(
    train_csv: str,
    test_csv: str,
    label_column: str = "lst_class",
    drop_columns: Sequence[str] = DEFAULT_DROP_COLUMNS,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]:
    """Load old train/test CSVs and return clean model matrices."""

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    return prepare_training_dataframes(train_df, test_df, label_column, drop_columns)


def prepare_training_dataframes(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_column: str = "lst_class",
    drop_columns: Sequence[str] = DEFAULT_DROP_COLUMNS,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]:
    """Prepare train/test DataFrames using the feature convention from the notebooks."""

    if label_column not in train_df.columns or label_column not in test_df.columns:
        raise KeyError(f"Both CSV files must contain the label column '{label_column}'.")

    def _split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        y = pd.to_numeric(df[label_column], errors="coerce")
        x = df.drop(columns=[label_column]).drop(columns=list(drop_columns), errors="ignore")
        x = x.apply(pd.to_numeric, errors="coerce")
        keep = y.notna() & x.notna().all(axis=1)
        return x.loc[keep].copy(), y.loc[keep].astype(int).copy()

    x_train, y_train = _split(train_df)
    x_test, y_test = _split(test_df)
    feature_columns = list(x_train.columns)

    missing_test = [col for col in feature_columns if col not in x_test.columns]
    if missing_test:
        raise ValueError(f"Test CSV is missing feature columns found in training CSV: {missing_test}")

    x_test = x_test[feature_columns]
    return x_train, y_train, x_test, y_test, feature_columns
