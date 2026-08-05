from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate lake-level df_l1_*.csv and ML-enhanced df_l2_*.csv tables "
            "from pre-downloaded Landsat C2L1/C2L2 folders."
        )
    )
    parser.add_argument("--config", help="JSON config. See configs/lake_tables.example.json.")
    parser.add_argument(
        "--landsat-root",
        help="Root folder containing <Lake>_L1, <Lake>_L2_acolite, and <Lake>_L2_usgs folders.",
    )
    parser.add_argument("--lake-geojson", help="GeoJSON with lake polygons in EPSG:4326.")
    parser.add_argument("--models-dir", help="Folder with tuned model .pkl files.")
    parser.add_argument("--output-l1-dir")
    parser.add_argument("--output-l2-dir")
    parser.add_argument(
        "--lake",
        action="append",
        help="Lake output key to process. Repeat to process multiple lakes. Defaults to all configured lakes.",
    )
    parser.add_argument(
        "--mask-cloud-class",
        type=int,
        action="append",
        default=None,
        help=(
            "ML class label to mask from LST. Repeat for several classes. "
            "Default is 1, matching the old df_diff_ml_stats workflow."
        ),
    )
    parser.add_argument(
        "--include-cirrus-as-cloud",
        action="store_true",
        help="Treat QA_PIXEL cirrus bit as operational cloud before ML classification.",
    )
    parser.add_argument(
        "--include-lake-metadata-in-l1",
        action="store_true",
        help="Add lakeN and pathrow columns to df_l1_*.csv.",
    )
    parser.add_argument("--limit-scenes", type=int, help="Debug option: process only the first N paired scenes per lake.")
    parser.add_argument("--continue-on-error", action="store_true", help="Write successful rows even if some scenes fail.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_config(args.config) if args.config else {}

    landsat_root = args.landsat_root or config.get("landsat_c2_root")
    if not landsat_root:
        raise SystemExit("Provide --landsat-root or set landsat_c2_root in the config JSON.")

    from lswt_cloud_masking.lake_tables import default_lake_configs, generate_all_lake_tables

    lake_geojson = args.lake_geojson or config.get("lake_geojson") or str(ROOT / "data" / "lakes_cloud_masking.geojson")
    models_dir = args.models_dir or config.get("models_dir") or str(ROOT / "models" / "general")
    output_l1_dir = args.output_l1_dir or config.get("output_l1_dir") or str(ROOT / "data" / "landsat_l1_tables")
    output_l2_dir = args.output_l2_dir or config.get("output_l2_dir") or str(ROOT / "data" / "landsat_l2_tables")
    lakes = config.get("lakes") or default_lake_configs()
    mask_cloud_classes = tuple(args.mask_cloud_class or config.get("mask_cloud_classes", [1]))

    reports = generate_all_lake_tables(
        landsat_root=landsat_root,
        lake_geojson=_resolve_repo_path(lake_geojson),
        model_dir=_resolve_repo_path(models_dir),
        output_l1_dir=_resolve_repo_path(output_l1_dir),
        output_l2_dir=_resolve_repo_path(output_l2_dir),
        lakes=lakes,
        only_lakes=args.lake or config.get("only_lakes"),
        mask_cloud_classes=mask_cloud_classes,
        include_cirrus_as_cloud=bool(args.include_cirrus_as_cloud or config.get("include_cirrus_as_cloud", False)),
        include_lake_metadata_in_l1=bool(
            args.include_lake_metadata_in_l1 or config.get("include_lake_metadata_in_l1", False)
        ),
        limit_scenes=args.limit_scenes or config.get("limit_scenes"),
        continue_on_error=bool(args.continue_on_error or config.get("continue_on_error", False)),
    )
    print(json.dumps(reports, indent=2))


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _load_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
