"""Single-scene Fmask plus ML thin-cloud masking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .features import CLASS_LABELS, build_feature_arrays, feature_dataframe, model_feature_columns
from .landsat_scene import (
    load_l1_feature_arrays,
    load_lake_geometry,
    resolve_scene,
)
from .qa import FmaskLayerValues, decode_qa_pixel, fmask_class_layer, fmask_clear_water
from .raster_io import write_single_band, write_stack


MlLayerValues = {
    0: "outside_or_not_evaluated",
    1: "thin_cloud",
    2: "cloud_affected",
    3: "water",
}


def process_scene(
    scene_path: str | Path,
    lake_geojson: str | Path,
    output_dir: str | Path,
    *,
    lake_key: str | None = None,
    rf_model_path: str | Path | None = None,
    xgb_model_path: str | Path | None = None,
    dt_model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Process one C2L1 scene over one lake polygon."""

    scene = resolve_scene(scene_path)
    geometry = load_lake_geometry(lake_geojson, lake_key=lake_key)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    raw_arrays, qa, profile, lake_mask = load_l1_feature_arrays(scene, geometry)
    decoded = decode_qa_pixel(qa)
    fmask_layer = fmask_class_layer(qa, lake_mask)
    clear_water = fmask_clear_water(decoded, lake_mask)
    feature_arrays = build_feature_arrays(raw_arrays, valid_mask=clear_water)

    model_specs = {
        "dt": (dt_model_path, False),
        "rf": (rf_model_path, False),
        "xgb": (xgb_model_path, True),
    }

    ml_layers: dict[str, np.ndarray] = {}
    for name, (model_path, shifted_labels) in model_specs.items():
        if model_path is None:
            continue
        model = joblib.load(model_path)
        pred_layer = predict_ml_layer(
            model,
            feature_arrays,
            valid_mask=clear_water,
            output_shape=qa.shape,
            shift_zero_based_labels=shifted_labels,
        )
        ml_layers[name] = apply_on_top_of_fmask(pred_layer, fmask_layer)

    tags = {
        "scene_id": scene.scene_id,
        "lake_key": lake_key or "",
        "workflow": "Fmask v3 QA_PIXEL plus ML thin-cloud masking",
    }

    stem = scene.scene_id
    paths: dict[str, str] = {}
    paths["fmask"] = str(
        write_single_band(
            output_path / f"{stem}_fmask_v3.tif",
            fmask_layer.astype("uint8"),
            profile,
            description="fmask_v3",
            tags={**tags, "value_labels": json.dumps(FmaskLayerValues)},
        )
    )

    stack_arrays: dict[str, np.ndarray] = {"fmask_v3": fmask_layer.astype("uint8")}
    for name, layer in ml_layers.items():
        out_name = "random_forest" if name == "rf" else ("xgboost" if name == "xgb" else "decision_tree")
        paths[name] = str(
            write_single_band(
                output_path / f"{stem}_{name}_thin_cloud.tif",
                layer.astype("uint8"),
                profile,
                description=f"{out_name}_thin_cloud",
                tags={**tags, "value_labels": json.dumps(MlLayerValues)},
            )
        )
        stack_arrays[f"{name}_thin_cloud"] = layer.astype("uint8")

    paths["stack"] = str(
        write_stack(
            output_path / f"{stem}_mask_stack.tif",
            stack_arrays,
            profile,
            tags={**tags, "band_order": ",".join(stack_arrays.keys())},
        )
    )

    summary = summarize_layers({"fmask": fmask_layer, **ml_layers})
    summary_payload = {
        "scene_id": scene.scene_id,
        "scene_path": str(scene.scene_path),
        "lake_geojson": str(lake_geojson),
        "lake_key": lake_key,
        "paths": paths,
        "layers": summary,
        "fmask_values": FmaskLayerValues,
        "ml_values": MlLayerValues,
        "class_labels": CLASS_LABELS,
    }
    summary_file = output_path / f"{stem}_summary.json"
    summary_file.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    paths["summary"] = str(summary_file)
    summary_payload["paths"] = paths
    return summary_payload


def predict_ml_layer(
    model: Any,
    feature_arrays: dict[str, np.ndarray],
    *,
    valid_mask: np.ndarray,
    output_shape: tuple[int, int],
    shift_zero_based_labels: bool = False,
) -> np.ndarray:
    """Predict 1/2/3 ML labels for valid pixels and 0 elsewhere."""

    columns = model_feature_columns(model, list(feature_arrays.keys()))
    x, flat_indices = feature_dataframe(feature_arrays, valid_mask, columns)
    out = np.zeros(output_shape, dtype="uint8")
    if x.empty:
        return out

    pred = np.asarray(model.predict(x)).astype("int16")
    if shift_zero_based_labels or (pred.size and pred.min() == 0):
        pred = pred + 1
    pred = np.clip(pred, 0, 255).astype("uint8")
    out_flat = out.ravel()
    out_flat[flat_indices] = pred
    return out


def apply_on_top_of_fmask(ml_prediction: np.ndarray, fmask_layer: np.ndarray) -> np.ndarray:
    """Combine ML clear-water predictions with operational Fmask cloud detections."""

    out = np.zeros(fmask_layer.shape, dtype="uint8")
    out[fmask_layer == 2] = 2
    predicted = ml_prediction > 0
    out[predicted] = ml_prediction[predicted]
    return out


def summarize_layers(layers: dict[str, np.ndarray]) -> dict[str, Any]:
    """Return pixel counts and percentages for each layer."""

    summary: dict[str, Any] = {}
    for name, arr in layers.items():
        labels = FmaskLayerValues if name == "fmask" else MlLayerValues
        values, counts = np.unique(arr, return_counts=True)
        count_map = {int(value): int(count) for value, count in zip(values, counts)}
        evaluated = int(np.count_nonzero(arr))
        label_summary = {}
        for value, label in labels.items():
            count = count_map.get(value, 0)
            denom = arr.size if value == 0 else evaluated
            pct = (100.0 * count / denom) if denom else 0.0
            label_summary[str(value)] = {
                "label": label,
                "count": count,
                "percent": pct,
            }
        summary[name] = {
            "total_pixels_in_crop": int(arr.size),
            "evaluated_pixels": evaluated,
            "counts": label_summary,
        }
    return summary
