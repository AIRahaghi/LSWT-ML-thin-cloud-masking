from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lswt_cloud_masking.batch import run_batch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-process pre-downloaded Landsat C2L1 scenes."
    )
    parser.add_argument("--manifest", required=True, help="CSV with at least a scene_path column.")
    parser.add_argument("--output-dir", required=True, help="Base folder for batch outputs.")
    parser.add_argument("--default-lake-geojson", help="Default lake polygon GeoJSON.")
    parser.add_argument("--default-lake-key", help="Default lake key if GeoJSON has multiple features.")
    parser.add_argument("--default-rf-model", help="Default RF model path.")
    parser.add_argument("--default-xgb-model", help="Default XGBoost model path.")
    parser.add_argument("--default-dt-model", help="Default Decision Tree model path.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first failed row.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_batch(
        args.manifest,
        output_dir=args.output_dir,
        default_lake_geojson=args.default_lake_geojson,
        default_lake_key=args.default_lake_key,
        default_rf_model=args.default_rf_model,
        default_xgb_model=args.default_xgb_model,
        default_dt_model=args.default_dt_model,
        fail_fast=args.fail_fast,
    )
    print(json.dumps({"processed_rows": len(results), "results": results}, indent=2))


if __name__ == "__main__":
    main()
