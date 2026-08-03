from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lswt_cloud_masking.masking import process_scene  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Fmask QA plus RF/XGBoost thin-cloud masking for one C2L1 scene."
    )
    parser.add_argument("--scene", required=True, help="Extracted C2L1 scene folder or .tar file.")
    parser.add_argument("--lake-geojson", required=True, help="Lake polygon GeoJSON in EPSG:4326.")
    parser.add_argument("--lake-key", help="Feature key/name when the GeoJSON has multiple lakes.")
    parser.add_argument("--rf-model", help="Random Forest .pkl model.")
    parser.add_argument("--xgb-model", help="XGBoost .pkl model.")
    parser.add_argument("--dt-model", help="Optional Decision Tree .pkl model.")
    parser.add_argument("--output-dir", required=True, help="Folder for output GeoTIFFs and summary JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = process_scene(
        args.scene,
        args.lake_geojson,
        args.output_dir,
        lake_key=args.lake_key,
        rf_model_path=args.rf_model,
        xgb_model_path=args.xgb_model,
        dt_model_path=args.dt_model,
    )
    print(json.dumps(summary["paths"], indent=2))


if __name__ == "__main__":
    main()
