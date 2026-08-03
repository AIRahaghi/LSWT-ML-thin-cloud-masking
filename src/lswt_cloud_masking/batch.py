"""Batch processing for pre-downloaded Landsat scenes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .masking import process_scene


MANIFEST_COLUMNS = [
    "scene_path",
    "lake_geojson",
    "lake_key",
    "rf_model",
    "xgb_model",
    "dt_model",
    "output_dir",
]


def run_batch(
    manifest_csv: str | Path,
    *,
    output_dir: str | Path,
    default_lake_geojson: str | Path | None = None,
    default_lake_key: str | None = None,
    default_rf_model: str | Path | None = None,
    default_xgb_model: str | Path | None = None,
    default_dt_model: str | Path | None = None,
    fail_fast: bool = False,
) -> list[dict[str, Any]]:
    """Run the masking pipeline for every row in a CSV manifest."""

    manifest = pd.read_csv(manifest_csv)
    if "scene_path" not in manifest.columns:
        raise KeyError("Batch manifest must contain a 'scene_path' column.")

    base_output = Path(output_dir)
    base_output.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    for index, row in manifest.iterrows():
        scene_path = _value(row, "scene_path")
        row_output = _value(row, "output_dir") or str(base_output / Path(scene_path).stem)
        lake_geojson = _value(row, "lake_geojson") or default_lake_geojson
        lake_key = _value(row, "lake_key") or default_lake_key
        rf_model = _value(row, "rf_model") or default_rf_model
        xgb_model = _value(row, "xgb_model") or default_xgb_model
        dt_model = _value(row, "dt_model") or default_dt_model

        result_base = {
            "row": int(index),
            "scene_path": scene_path,
            "output_dir": str(row_output),
        }

        try:
            if lake_geojson is None:
                raise ValueError("No lake_geojson was provided in the row or defaults.")
            summary = process_scene(
                scene_path,
                lake_geojson,
                row_output,
                lake_key=lake_key,
                rf_model_path=rf_model,
                xgb_model_path=xgb_model,
                dt_model_path=dt_model,
            )
            results.append({**result_base, "status": "ok", "summary": summary["paths"].get("summary")})
        except Exception as exc:  # noqa: BLE001 - record row-level failures.
            results.append({**result_base, "status": "failed", "error": str(exc)})
            if fail_fast:
                raise

    pd.DataFrame(results).to_csv(base_output / "batch_summary.csv", index=False)
    (base_output / "batch_summary.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    return results


def _value(row: pd.Series, column: str) -> str | None:
    if column not in row:
        return None
    value = row[column]
    if pd.isna(value):
        return None
    value_str = str(value).strip()
    return value_str or None
