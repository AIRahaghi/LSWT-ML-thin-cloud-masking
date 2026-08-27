"""Binary total-cloud versus water evaluation for saved LOSO models."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd

from .leave_one_site_out import DATASET_FILENAMES, MODEL_FILENAMES


EVALUATION_DATASETS = {
    "train": DATASET_FILENAMES["train"],
    "test": DATASET_FILENAMES["test"],
    "loso": DATASET_FILENAMES["leave_out_site"],
}

BINARY_METRICS = [
    "overall_binary_accuracy",
    "balanced_binary_accuracy",
    "total_cloud_accuracy",
    "water_accuracy",
    "total_cloud_user_accuracy",
    "water_user_accuracy",
    "total_cloud_f1",
    "water_f1",
]


def evaluate_saved_loso_binary_models(
    loso_output_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Load completed LOSO artifacts and evaluate binary predictions without fitting."""

    loso_root = Path(loso_output_dir).resolve()
    manifest_path = loso_root / "leave_one_site_out_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"LOSO manifest was not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = manifest.get("runs", [])
    if not runs:
        raise ValueError(f"LOSO manifest contains no completed runs: {manifest_path}")

    binary_output = (
        Path(output_dir).resolve()
        if output_dir is not None
        else loso_root / "binary_evaluation"
    )
    binary_output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for run_index, run in enumerate(runs, start=1):
        run_name = str(run["run_name"])
        held_out_site = str(run["held_out_site"])
        dataset_dir = loso_root / "datasets" / run_name
        model_dir = loso_root / "models" / run_name
        metadata_path = model_dir / "model_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Model metadata was not found: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        feature_columns = [str(column) for column in metadata["feature_columns"]]
        label_column = str(metadata.get("config", {}).get("label_column", "lst_class"))

        prepared: dict[str, tuple[pd.DataFrame, pd.Series, int, Path]] = {}
        for dataset_name, filename in EVALUATION_DATASETS.items():
            csv_path = dataset_dir / filename
            frame = pd.read_csv(csv_path)
            x, y, dropped_rows = _prepare_evaluation_frame(
                frame,
                feature_columns,
                label_column,
            )
            prepared[dataset_name] = (x, y, dropped_rows, csv_path)

        for model_name, filename in MODEL_FILENAMES.items():
            model_path = model_dir / filename
            if not model_path.exists():
                raise FileNotFoundError(f"Saved model was not found: {model_path}")
            progress(
                f"[{run_index}/{len(runs)}] {held_out_site}: loading and evaluating "
                f"{model_name}."
            )
            model = joblib.load(model_path)
            _validate_model_features(model, feature_columns, model_path)
            for dataset_name, (x, y, dropped_rows, csv_path) in prepared.items():
                prediction = np.asarray(model.predict(x), dtype="int64")
                if model_name == "xgboost":
                    prediction = prediction + 1
                metrics = binary_cloud_water_metrics(y.to_numpy(), prediction)
                rows.append(
                    {
                        "run_index": run_index,
                        "run_name": run_name,
                        "held_out_site": held_out_site,
                        "model": model_name,
                        "dataset": dataset_name,
                        "n_source_rows": len(y) + dropped_rows,
                        "n_evaluated_rows": len(y),
                        "dropped_incomplete_rows": dropped_rows,
                        **metrics,
                        "dataset_csv": str(csv_path),
                        "model_path": str(model_path),
                    }
                )
            del model
            gc.collect()

    detailed = pd.DataFrame(rows).sort_values(
        ["run_index", "model", "dataset"], kind="stable"
    )
    expected_rows = len(runs) * len(MODEL_FILENAMES) * len(EVALUATION_DATASETS)
    if len(detailed) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} binary evaluations, created {len(detailed)}."
        )

    summary = _summarize_binary_results(detailed)
    comparison = _binary_comparison_table(detailed)
    detailed_path = binary_output / "binary_metrics_by_site_model_dataset.csv"
    summary_path = binary_output / "binary_metrics_summary_by_model_dataset.csv"
    comparison_path = binary_output / "binary_accuracy_comparison.csv"
    metadata_output_path = binary_output / "binary_evaluation_metadata.json"
    detailed.to_csv(detailed_path, index=False)
    summary.to_csv(summary_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    metadata_output_path.write_text(
        json.dumps(
            {
                "evaluation_only": True,
                "model_training_or_tuning_performed": False,
                "loso_output_dir": str(loso_root),
                "source_manifest": str(manifest_path),
                "n_site_runs": len(runs),
                "n_models_per_site": len(MODEL_FILENAMES),
                "n_datasets_per_model": len(EVALUATION_DATASETS),
                "n_evaluations": len(detailed),
                "class_definition": {
                    "total_cloud": "original classes 1 and 2",
                    "water": "original class 3",
                },
                "accuracy_definition": {
                    "total_cloud_accuracy": (
                        "correctly predicted total-cloud rows divided by actual "
                        "total-cloud rows (cloud producer accuracy/recall)"
                    ),
                    "water_accuracy": (
                        "correctly predicted water rows divided by actual water rows "
                        "(water producer accuracy/recall)"
                    ),
                    "overall_binary_accuracy": (
                        "all correct binary predictions divided by all evaluated rows"
                    ),
                    "balanced_binary_accuracy": (
                        "mean of total_cloud_accuracy and water_accuracy"
                    ),
                },
                "xgboost_label_handling": (
                    "Saved XGBoost predictions are shifted from 0/1/2 to 1/2/3 before "
                    "binary aggregation."
                ),
                "output_files": {
                    "detailed": str(detailed_path),
                    "summary": str(summary_path),
                    "comparison": str(comparison_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    progress(f"Binary cloud/water results written to {binary_output}.")
    return {
        "detailed": detailed,
        "summary": summary,
        "comparison": comparison,
        "metadata": json.loads(metadata_output_path.read_text(encoding="utf-8")),
        "paths": {
            "output_dir": str(binary_output),
            "detailed": str(detailed_path),
            "summary": str(summary_path),
            "comparison": str(comparison_path),
            "metadata": str(metadata_output_path),
        },
    }


def binary_cloud_water_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """Return binary total-cloud/water metrics from three-class labels."""

    truth = np.asarray(y_true, dtype="int64")
    prediction = np.asarray(y_pred, dtype="int64")
    if truth.shape != prediction.shape:
        raise ValueError("True and predicted label arrays must have the same shape.")
    if truth.ndim != 1:
        raise ValueError("True and predicted labels must be one-dimensional.")
    if truth.size == 0:
        raise ValueError("At least one label is required for binary evaluation.")
    expected_labels = {1, 2, 3}
    unexpected_truth = sorted(set(np.unique(truth)) - expected_labels)
    unexpected_prediction = sorted(set(np.unique(prediction)) - expected_labels)
    if unexpected_truth:
        raise ValueError(f"Unexpected true labels: {unexpected_truth}")
    if unexpected_prediction:
        raise ValueError(f"Unexpected predicted labels: {unexpected_prediction}")

    true_cloud = np.isin(truth, [1, 2])
    predicted_cloud = np.isin(prediction, [1, 2])
    cloud_as_cloud = int(np.sum(true_cloud & predicted_cloud))
    cloud_as_water = int(np.sum(true_cloud & ~predicted_cloud))
    water_as_cloud = int(np.sum(~true_cloud & predicted_cloud))
    water_as_water = int(np.sum(~true_cloud & ~predicted_cloud))
    actual_cloud = cloud_as_cloud + cloud_as_water
    actual_water = water_as_cloud + water_as_water
    predicted_cloud_count = cloud_as_cloud + water_as_cloud
    predicted_water_count = cloud_as_water + water_as_water

    cloud_accuracy = _safe_ratio(cloud_as_cloud, actual_cloud)
    water_accuracy = _safe_ratio(water_as_water, actual_water)
    return {
        "actual_total_cloud_rows": actual_cloud,
        "actual_water_rows": actual_water,
        "predicted_total_cloud_rows": predicted_cloud_count,
        "predicted_water_rows": predicted_water_count,
        "true_total_cloud_predicted_total_cloud": cloud_as_cloud,
        "true_total_cloud_predicted_water": cloud_as_water,
        "true_water_predicted_total_cloud": water_as_cloud,
        "true_water_predicted_water": water_as_water,
        "overall_binary_accuracy": _safe_ratio(
            cloud_as_cloud + water_as_water,
            truth.size,
        ),
        "balanced_binary_accuracy": (
            (cloud_accuracy + water_accuracy) / 2
            if cloud_accuracy is not None and water_accuracy is not None
            else None
        ),
        "total_cloud_accuracy": cloud_accuracy,
        "water_accuracy": water_accuracy,
        "total_cloud_user_accuracy": _safe_ratio(
            cloud_as_cloud,
            predicted_cloud_count,
        ),
        "water_user_accuracy": _safe_ratio(
            water_as_water,
            predicted_water_count,
        ),
        "total_cloud_f1": _safe_ratio(
            2 * cloud_as_cloud,
            2 * cloud_as_cloud + water_as_cloud + cloud_as_water,
        ),
        "water_f1": _safe_ratio(
            2 * water_as_water,
            2 * water_as_water + water_as_cloud + cloud_as_water,
        ),
    }


def _prepare_evaluation_frame(
    frame: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
) -> tuple[pd.DataFrame, pd.Series, int]:
    missing = sorted(set(feature_columns + [label_column]) - set(frame.columns))
    if missing:
        raise ValueError(f"Evaluation CSV is missing required columns: {missing}")
    x = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(frame[label_column], errors="coerce")
    keep = x.notna().all(axis=1) & y.notna()
    return x.loc[keep].copy(), y.loc[keep].astype(int).copy(), int((~keep).sum())


def _validate_model_features(
    model: Any,
    feature_columns: list[str],
    model_path: Path,
) -> None:
    if not hasattr(model, "feature_names_in_"):
        return
    model_columns = [str(column) for column in model.feature_names_in_]
    if model_columns != feature_columns:
        raise ValueError(
            f"Saved model feature order does not match metadata: {model_path}"
        )


def _summarize_binary_results(detailed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model_name, dataset_name), group in detailed.groupby(
        ["model", "dataset"], sort=True
    ):
        row: dict[str, Any] = {
            "model": model_name,
            "dataset": dataset_name,
            "n_sites": int(group["held_out_site"].nunique()),
        }
        for metric in BINARY_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def _binary_comparison_table(detailed: pd.DataFrame) -> pd.DataFrame:
    values = [
        "overall_binary_accuracy",
        "balanced_binary_accuracy",
        "total_cloud_accuracy",
        "water_accuracy",
    ]
    comparison = detailed.pivot(
        index=["held_out_site", "model"],
        columns="dataset",
        values=values,
    )
    comparison.columns = [
        f"{dataset}_{metric}" for metric, dataset in comparison.columns
    ]
    return comparison.reset_index()


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None
