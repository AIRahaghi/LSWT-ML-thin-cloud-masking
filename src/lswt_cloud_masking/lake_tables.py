"""Regenerate lake-level Landsat L1 and L2 CSV tables.

The old notebooks mixed three jobs in long cells: scene discovery, raster/NetCDF
reading, and point-statistic extraction. This module keeps those jobs separate
while preserving the table names and column conventions used by the paper
workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from .features import build_feature_arrays, feature_dataframe, model_feature_columns
from .qa import decode_qa_pixel


DEFAULT_LAKES: list[dict[str, Any]] = [
    {
        "output_key": "geneva",
        "lake_key": "geneva",
        "lakeN": "geneva",
        "folder_name": "Geneva",
        "stations": [
            {"name": "LXP", "lat": 46.5003, "lon": 6.6609},
            {"name": "BUC", "lat": 46.4540, "lon": 6.3990},
        ],
        "numpix_xy": 5,
    },
    {
        "output_key": "aegeri",
        "lake_key": "ageri",
        "lakeN": "ageri",
        "folder_name": "Aegeri",
        "stations": [{"name": "Buoy", "lat": 47.12486, "lon": 8.60855}],
        "numpix_xy": 5,
    },
    {
        "output_key": "bianco",
        "lake_key": "bianco",
        "lakeN": "bianco",
        "folder_name": "Bianco",
        "stations": [{"name": "Buoy", "lat": 46.411859, "lon": 10.014917}],
        "numpix_xy": 5,
    },
    {
        "output_key": "greifensee",
        "lake_key": "greifensee",
        "lakeN": "greifensee",
        "folder_name": "Greifensee",
        "stations": [{"name": "Platform", "lat": 47.366, "lon": 8.665}],
        "numpix_xy": 5,
    },
    {
        "output_key": "mendota",
        "lake_key": "mendota",
        "lakeN": "mendota",
        "folder_name": "Mendota",
        "stations": [{"name": "Buoy", "lat": 43.0995, "lon": -89.4045}],
        "numpix_xy": 5,
    },
    {
        "output_key": "venice",
        "lake_key": "venice",
        "lakeN": "venice",
        "folder_name": "Venice",
        "stations": [{"name": "Platform", "lat": 45.3142467, "lon": 12.5082483}],
        "numpix_xy": 5,
    },
]

L1_STAT_NAMES = [
    "nirmap",
    "swirmap1",
    "swirmap2",
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
    "btmap1",
    "btmap2",
    "raamap",
    "vzamap",
    "szamap",
]

L2_STAT_NAMES = [
    "lst_class_dt",
    "lst_class_rf",
    "lst_class_xgb",
    "lstmap_usgs_on_tact",
    "lstmap_dt_usgs",
    "lstmap_rf_usgs",
    "lstmap_xgb_usgs",
    "lstmap1",
    "lstmap_dt_acolite1",
    "lstmap_rf_acolite1",
    "lstmap_xgb_acolite1",
    "lstmap2",
    "lstmap_dt_acolite2",
    "lstmap_rf_acolite2",
    "lstmap_xgb_acolite2",
]

MODEL_FILENAMES = {
    "dt": "DT_best_all_general.pkl",
    "rf": "RF_best_all_general.pkl",
    "xgb": "XGB_best_all_general.pkl",
}

REFLECTANCE_TARGETS = {
    "aerosolmap": 443,
    "bluemap": 482,
    "greenmap": 561,
    "redmap": 655,
    "nirmap": 865,
    "swirmap1": 1609,
    "swirmap2": 2201,
    "cirrusmap": 1373,
}

REFLECTANCE_CANDIDATES = {
    "LANDSAT_8": {
        "aerosolmap": ["rhot_443"],
        "bluemap": ["rhot_483", "rhot_482"],
        "greenmap": ["rhot_561"],
        "redmap": ["rhot_655", "rhot_654"],
        "nirmap": ["rhot_865"],
        "swirmap1": ["rhot_1609", "rhot_1608"],
        "swirmap2": ["rhot_2201"],
        "cirrusmap": ["rhot_1373", "rhot_1374"],
    },
    "LANDSAT_9": {
        "aerosolmap": ["rhot_443"],
        "bluemap": ["rhot_482", "rhot_483"],
        "greenmap": ["rhot_561"],
        "redmap": ["rhot_654", "rhot_655"],
        "nirmap": ["rhot_865"],
        "swirmap1": ["rhot_1608", "rhot_1609"],
        "swirmap2": ["rhot_2201"],
        "cirrusmap": ["rhot_1374", "rhot_1373"],
    },
}


@dataclass(frozen=True)
class SceneInfo:
    """Metadata parsed from a Landsat product folder name."""

    path: Path
    scene_id: str
    sensor: str
    time_utc: str
    pathrow: str

    @property
    def match_key(self) -> tuple[str, str, str]:
        return self.sensor, self.time_utc, self.pathrow


@dataclass(frozen=True)
class SceneBundle:
    """Matched L1, ACOLITE L2, and USGS L2 folders for one acquisition."""

    info: SceneInfo
    l1_scene: Path
    acolite_scene: Path
    usgs_scene: Path | None


@dataclass(frozen=True)
class Station:
    """Point where lake statistics are extracted."""

    name: str
    lat: float
    lon: float

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "Station":
        return cls(
            name=str(value.get("name", "station")),
            lat=float(value["lat"]),
            lon=float(value["lon"]),
        )


def load_generation_config(path: str | Path) -> dict[str, Any]:
    """Read a JSON config for table generation."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def default_lake_configs() -> list[dict[str, Any]]:
    """Return editable defaults for the six lakes used in the paper workflow."""

    return json.loads(json.dumps(DEFAULT_LAKES))


def load_general_models(model_dir: str | Path) -> dict[str, Any]:
    """Load the tuned DT, RF, and XGBoost models from a model directory."""

    import joblib

    model_path = Path(model_dir)
    models = {}
    for name, filename in MODEL_FILENAMES.items():
        path = model_path / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing {name.upper()} model: {path}")
        models[name] = joblib.load(path)
    return models


def _normalize_table_selection(tables: str) -> str:
    value = str(tables).strip().lower().replace("-", "_")
    aliases = {
        "all": "both",
        "l1_l2": "both",
        "l1+l2": "both",
        "l2_only": "l2",
        "l1_only": "l1",
    }
    value = aliases.get(value, value)
    if value not in {"both", "l1", "l2"}:
        raise ValueError("tables must be one of: 'both', 'l1', or 'l2'.")
    return value


def generate_all_lake_tables(
    *,
    landsat_root: str | Path,
    lake_geojson: str | Path,
    model_dir: str | Path,
    output_l1_dir: str | Path,
    output_l2_dir: str | Path,
    lakes: list[dict[str, Any]] | None = None,
    only_lakes: list[str] | None = None,
    mask_cloud_classes: tuple[int, ...] = (1,),
    include_cirrus_as_cloud: bool = False,
    include_lake_metadata_in_l1: bool = False,
    limit_scenes: int | None = None,
    continue_on_error: bool = False,
    tables: str = "both",
) -> list[dict[str, Any]]:
    """Generate L1 and L2 tables for several lakes."""

    tables = _normalize_table_selection(tables)
    selected_lakes = lakes or default_lake_configs()
    if only_lakes:
        wanted = {name.lower() for name in only_lakes}
        selected_lakes = [
            lake
            for lake in selected_lakes
            if str(lake.get("output_key", lake.get("lake_key", ""))).lower() in wanted
            or str(lake.get("lake_key", "")).lower() in wanted
        ]

    models = load_general_models(model_dir) if tables in {"both", "l2"} else None
    return [
        generate_lake_tables(
            lake=lake,
            landsat_root=landsat_root,
            lake_geojson=lake_geojson,
            model_dir=model_dir,
            models=models,
            output_l1_dir=output_l1_dir,
            output_l2_dir=output_l2_dir,
            mask_cloud_classes=mask_cloud_classes,
            include_cirrus_as_cloud=include_cirrus_as_cloud,
            include_lake_metadata_in_l1=include_lake_metadata_in_l1,
            limit_scenes=limit_scenes,
            continue_on_error=continue_on_error,
            tables=tables,
        )
        for lake in selected_lakes
    ]


def generate_lake_tables(
    *,
    lake: dict[str, Any],
    landsat_root: str | Path,
    lake_geojson: str | Path,
    model_dir: str | Path,
    output_l1_dir: str | Path,
    output_l2_dir: str | Path,
    models: dict[str, Any] | None = None,
    mask_cloud_classes: tuple[int, ...] = (1,),
    include_cirrus_as_cloud: bool = False,
    include_lake_metadata_in_l1: bool = False,
    limit_scenes: int | None = None,
    continue_on_error: bool = False,
    tables: str = "both",
) -> dict[str, Any]:
    """Regenerate one lake's `df_l1_*.csv` and `df_l2_*.csv` files."""

    tables = _normalize_table_selection(tables)
    run_l1 = tables in {"both", "l1"}
    run_l2 = tables in {"both", "l2"}

    output_key = str(lake.get("output_key", lake["lake_key"])).lower()
    lake_key = str(lake.get("lake_key", output_key)).lower()
    lake_name_value = str(lake.get("lakeN", lake_key)).lower()
    geometry = _load_lake_geometry(lake_geojson, lake_key=lake_key)
    stations = [Station.from_mapping(value) for value in lake["stations"]]
    numpix_xy = int(lake.get("numpix_xy", 5))

    folders = discover_lake_folders(landsat_root, lake)
    bundles = pair_scene_folders(
        folders["l1"],
        folders["l2_acolite"],
        folders.get("l2_usgs"),
        keep_landsat_8_9_only=True,
    )
    if limit_scenes is not None:
        bundles = bundles[:limit_scenes]

    if models is None and run_l2:
        models = load_general_models(model_dir)

    out_l1 = Path(output_l1_dir)
    out_l2 = Path(output_l2_dir)
    df_l1 = None
    df_l2 = None
    l1_csv = None
    l2_csv = None
    l1_failures: list[dict[str, str]] = []
    l2_failures: list[dict[str, str]] = []

    if run_l1:
        df_l1, l1_failures = generate_l1_table(
            bundles,
            lake_name=lake_name_value,
            geometry=geometry,
            stations=stations,
            numpix_xy=numpix_xy,
            include_cirrus_as_cloud=include_cirrus_as_cloud,
            include_lake_metadata=include_lake_metadata_in_l1,
            continue_on_error=continue_on_error,
        )
        out_l1.mkdir(parents=True, exist_ok=True)
        l1_csv = out_l1 / f"df_l1_{output_key}.csv"
        df_l1.to_csv(l1_csv, index=True, index_label="scene_id")

    if run_l2:
        if models is None:
            raise ValueError("L2 table generation requires loaded models.")
        df_l2, l2_failures = generate_l2_table(
            bundles,
            lake_name=lake_name_value,
            geometry=geometry,
            stations=stations,
            numpix_xy=numpix_xy,
            models=models,
            mask_cloud_classes=mask_cloud_classes,
            include_cirrus_as_cloud=include_cirrus_as_cloud,
            continue_on_error=continue_on_error,
        )
        out_l2.mkdir(parents=True, exist_ok=True)
        l2_csv = out_l2 / f"df_l2_{output_key}.csv"
        df_l2.to_csv(l2_csv, index=True, index_label="scene_id")

    report = {
        "lake": output_key,
        "lake_key": lake_key,
        "lakeN": lake_name_value,
        "tables": tables,
        "folders": {key: str(value) for key, value in folders.items() if value is not None},
        "l1_csv": str(l1_csv) if l1_csv is not None else None,
        "l2_csv": str(l2_csv) if l2_csv is not None else None,
        "l1_rows": int(len(df_l1)) if df_l1 is not None else None,
        "l2_rows": int(len(df_l2)) if df_l2 is not None else None,
        "l1_failures": l1_failures,
        "l2_failures": l2_failures,
    }
    report_dir = out_l2 if run_l2 else out_l1
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"df_landsat_generation_report_{output_key}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_json"] = str(report_path)
    return report


def generate_l1_table(
    bundles: list[SceneBundle],
    *,
    lake_name: str,
    geometry: dict[str, Any],
    stations: list[Station],
    numpix_xy: int,
    include_cirrus_as_cloud: bool = False,
    include_lake_metadata: bool = False,
    continue_on_error: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Build the L1 spectral-feature table for one lake."""

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for bundle in bundles:
        try:
            row = _l1_row(
                bundle,
                lake_name=lake_name,
                geometry=geometry,
                stations=stations,
                numpix_xy=numpix_xy,
                include_cirrus_as_cloud=include_cirrus_as_cloud,
                include_lake_metadata=include_lake_metadata,
            )
            rows.append(row)
        except Exception as exc:  # noqa: BLE001 - row-level report for batch processing.
            failures.append({"scene_id": bundle.info.scene_id, "error": str(exc)})
            if not continue_on_error:
                raise

    return _rows_to_dataframe(rows), failures


def generate_l2_table(
    bundles: list[SceneBundle],
    *,
    lake_name: str,
    geometry: dict[str, Any],
    stations: list[Station],
    numpix_xy: int,
    models: dict[str, Any],
    mask_cloud_classes: tuple[int, ...] = (1,),
    include_cirrus_as_cloud: bool = False,
    continue_on_error: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Build the ML-enhanced L2 LSWT table for one lake."""

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for bundle in bundles:
        try:
            if bundle.usgs_scene is None:
                raise FileNotFoundError(f"No matched USGS L2 folder for {bundle.info.scene_id}")
            row = _l2_row(
                bundle,
                lake_name=lake_name,
                geometry=geometry,
                stations=stations,
                numpix_xy=numpix_xy,
                models=models,
                mask_cloud_classes=mask_cloud_classes,
                include_cirrus_as_cloud=include_cirrus_as_cloud,
            )
            rows.append(row)
        except Exception as exc:  # noqa: BLE001 - row-level report for batch processing.
            failures.append({"scene_id": bundle.info.scene_id, "error": str(exc)})
            if not continue_on_error:
                raise

    return _rows_to_dataframe(rows), failures


def discover_lake_folders(landsat_root: str | Path, lake: dict[str, Any]) -> dict[str, Path | None]:
    """Find the L1, ACOLITE L2, and USGS L2 folders for one lake."""

    root = Path(landsat_root).expanduser()
    folder_name = str(lake.get("folder_name", lake.get("output_key", lake["lake_key"])))

    explicit_l1 = lake.get("l1_dir")
    explicit_l2_acolite = lake.get("l2_acolite_dir")
    explicit_l2_usgs = lake.get("l2_usgs_dir")
    explicit_l2 = lake.get("l2_dir")

    l1 = Path(explicit_l1).expanduser() if explicit_l1 else _first_existing_dir(_folder_variants(root, folder_name, "L1"))

    l2_base = Path(explicit_l2).expanduser() if explicit_l2 else _first_existing_dir(_folder_variants(root, folder_name, "L2"))
    if explicit_l2_acolite:
        l2_acolite = Path(explicit_l2_acolite).expanduser()
    else:
        l2_acolite = _first_existing_dir(
            _folder_variants(root, folder_name, "L2_acolite")
            + _nested_l2_variants(l2_base, ["acolite", "ACOLITE", "l2_acolite"])
            + ([l2_base] if l2_base is not None else [])
        )

    if explicit_l2_usgs:
        l2_usgs = Path(explicit_l2_usgs).expanduser()
    else:
        l2_usgs = _first_existing_dir(
            _folder_variants(root, folder_name, "L2_usgs")
            + _nested_l2_variants(l2_base, ["usgs", "USGS", "l2_usgs"])
            + ([l2_base] if l2_base is not None else [])
        )

    if l1 is None:
        raise FileNotFoundError(f"Could not find L1 folder for {folder_name} under {root}.")
    if l2_acolite is None:
        raise FileNotFoundError(f"Could not find ACOLITE L2 folder for {folder_name} under {root}.")

    return {"l1": l1, "l2_acolite": l2_acolite, "l2_usgs": l2_usgs}


def pair_scene_folders(
    l1_dir: str | Path,
    l2_acolite_dir: str | Path,
    l2_usgs_dir: str | Path | None,
    *,
    keep_landsat_8_9_only: bool = True,
) -> list[SceneBundle]:
    """Pair scene folders by sensor, acquisition date, and path/row."""

    l1_index = _scene_index(Path(l1_dir), keep_landsat_8_9_only=keep_landsat_8_9_only)
    acolite_index = _scene_index(Path(l2_acolite_dir), keep_landsat_8_9_only=keep_landsat_8_9_only)
    usgs_index = (
        _scene_index(Path(l2_usgs_dir), keep_landsat_8_9_only=keep_landsat_8_9_only)
        if l2_usgs_dir is not None and Path(l2_usgs_dir).exists()
        else {}
    )

    bundles = []
    for key in sorted(acolite_index):
        acolite = acolite_index[key]
        l1 = l1_index.get(key)
        if l1 is None:
            continue
        bundles.append(
            SceneBundle(
                info=acolite,
                l1_scene=l1.path,
                acolite_scene=acolite.path,
                usgs_scene=usgs_index.get(key).path if key in usgs_index else None,
            )
        )
    return bundles


def parse_scene_info(path: str | Path) -> SceneInfo:
    """Parse sensor, date, and WRS path/row from a Landsat product path."""

    path_obj = Path(path)
    name = path_obj.name
    match = re.search(
        r"^(?P<prefix>L[COTEM]0(?P<sat>[789]))_[A-Z0-9]+_(?P<pathrow>\d{6})_(?P<date>\d{8})_",
        name,
    )
    if not match:
        raise ValueError(f"Could not parse Landsat scene metadata from: {path_obj}")
    return SceneInfo(
        path=path_obj,
        scene_id=name,
        sensor=f"LANDSAT_{match.group('sat')}",
        time_utc=match.group("date"),
        pathrow=match.group("pathrow"),
    )


def _l1_row(
    bundle: SceneBundle,
    *,
    lake_name: str,
    geometry: dict[str, Any],
    stations: list[Station],
    numpix_xy: int,
    include_cirrus_as_cloud: bool,
    include_lake_metadata: bool,
) -> dict[str, Any]:
    scene_arrays = load_scene_arrays(
        bundle,
        geometry,
        include_cirrus_as_cloud=include_cirrus_as_cloud,
    )
    row = _base_row(bundle.info, lake_name=lake_name, include_lake_metadata=include_lake_metadata)
    _append_l1_stats(row, scene_arrays["features"], scene_arrays["lon"], scene_arrays["lat"], stations, numpix_xy)
    return row


def _l2_row(
    bundle: SceneBundle,
    *,
    lake_name: str,
    geometry: dict[str, Any],
    stations: list[Station],
    numpix_xy: int,
    models: dict[str, Any],
    mask_cloud_classes: tuple[int, ...],
    include_cirrus_as_cloud: bool,
) -> dict[str, Any]:
    scene_arrays = load_scene_arrays(
        bundle,
        geometry,
        include_cirrus_as_cloud=include_cirrus_as_cloud,
        include_l2=True,
    )
    feature_arrays = scene_arrays["features"]
    clear_mask = scene_arrays["clear_mask"]

    class_maps = {
        "dt": predict_class_map(models["dt"], feature_arrays, clear_mask, shift_zero_based_labels=False),
        "rf": predict_class_map(models["rf"], feature_arrays, clear_mask, shift_zero_based_labels=False),
        "xgb": predict_class_map(models["xgb"], feature_arrays, clear_mask, shift_zero_based_labels=True),
    }

    lstmap1 = scene_arrays["lstmap1"]
    lstmap2 = scene_arrays["lstmap2"]
    lstmap_usgs_on_tact = scene_arrays["lstmap_usgs_on_tact"]

    arrays = {
        "lst_class_dt": _class_for_lst(class_maps["dt"], lstmap1),
        "lst_class_rf": _class_for_lst(class_maps["rf"], lstmap1),
        "lst_class_xgb": _class_for_lst(class_maps["xgb"], lstmap1),
        "lstmap_usgs_on_tact": lstmap_usgs_on_tact,
        "lstmap_dt_usgs": filter_lst_with_classes(lstmap_usgs_on_tact, class_maps["dt"], mask_cloud_classes),
        "lstmap_rf_usgs": filter_lst_with_classes(lstmap_usgs_on_tact, class_maps["rf"], mask_cloud_classes),
        "lstmap_xgb_usgs": filter_lst_with_classes(lstmap_usgs_on_tact, class_maps["xgb"], mask_cloud_classes),
        "lstmap1": lstmap1,
        "lstmap_dt_acolite1": filter_lst_with_classes(lstmap1, class_maps["dt"], mask_cloud_classes),
        "lstmap_rf_acolite1": filter_lst_with_classes(lstmap1, class_maps["rf"], mask_cloud_classes),
        "lstmap_xgb_acolite1": filter_lst_with_classes(lstmap1, class_maps["xgb"], mask_cloud_classes),
        "lstmap2": lstmap2,
        "lstmap_dt_acolite2": filter_lst_with_classes(lstmap2, class_maps["dt"], mask_cloud_classes),
        "lstmap_rf_acolite2": filter_lst_with_classes(lstmap2, class_maps["rf"], mask_cloud_classes),
        "lstmap_xgb_acolite2": filter_lst_with_classes(lstmap2, class_maps["xgb"], mask_cloud_classes),
    }

    row = _base_row(bundle.info, lake_name=lake_name, include_lake_metadata=True)
    _append_l2_stats(row, arrays, scene_arrays["lon"], scene_arrays["lat"], stations, numpix_xy)
    return row


def load_scene_arrays(
    bundle: SceneBundle,
    geometry: dict[str, Any],
    *,
    include_cirrus_as_cloud: bool = False,
    include_l2: bool = False,
) -> dict[str, Any]:
    """Load ACOLITE L1R features and optional L2 LSWT arrays on the ACOLITE grid."""

    l1r_path = _find_file(bundle.acolite_scene, ["*_L1R.nc"])
    raw, lon, lat, lake_mask = read_acolite_l1r(l1r_path, geometry, bundle.info.sensor)
    qa_path = _find_file(bundle.l1_scene, ["*_QA_PIXEL.TIF", "*_QA_PIXEL.tif"])
    clear_mask = clear_water_mask_on_grid(
        qa_path,
        geometry,
        lon,
        lat,
        include_cirrus_as_cloud=include_cirrus_as_cloud,
    )
    clear_mask &= lake_mask

    masked_raw = {name: _masked_copy(value, clear_mask) for name, value in raw.items()}
    features = build_feature_arrays(masked_raw, valid_mask=clear_mask)

    out: dict[str, Any] = {
        "lon": lon,
        "lat": lat,
        "lake_mask": lake_mask,
        "clear_mask": clear_mask,
        "features": features,
    }
    if include_l2:
        st_path = _find_file(bundle.acolite_scene, ["*_ST.nc"])
        lst_arrays = read_acolite_l2_st(st_path, geometry, bundle.info.sensor)
        for name in ["lstmap1", "lstmap2"]:
            out[name] = _clean_zero_as_nan(_masked_copy(lst_arrays[name], clear_mask))

        if bundle.usgs_scene is None:
            raise FileNotFoundError(f"No USGS L2 scene paired with {bundle.info.scene_id}")
        usgs_st_path = _find_file(bundle.usgs_scene, ["*_ST_B*.tif", "*_ST_B*.TIF"])
        usgs_lst, usgs_lon, usgs_lat = read_usgs_lst(usgs_st_path, geometry)
        lstmap_usgs_on_tact = nearest_resample(usgs_lon, usgs_lat, usgs_lst, lon, lat)
        out["lstmap_usgs_on_tact"] = _clean_zero_as_nan(_masked_copy(lstmap_usgs_on_tact, clear_mask))

    return out


def read_acolite_l1r(
    path: str | Path,
    geometry: dict[str, Any],
    sensor: str,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Read ACOLITE L1R TOA reflectance, BT, and angle maps clipped to a lake."""

    Dataset = _netcdf_dataset()
    with Dataset(path, "r") as dataset:
        lon = _read_netcdf_array(dataset, "lon")
        lat = _read_netcdf_array(dataset, "lat")
        lake_mask = lake_mask_from_lonlat(geometry, lon, lat)

        arrays: dict[str, np.ndarray] = {}
        candidates = REFLECTANCE_CANDIDATES.get(sensor, REFLECTANCE_CANDIDATES["LANDSAT_8"])
        for name, names in candidates.items():
            arrays[name] = _read_reflectance_variable(dataset, names, REFLECTANCE_TARGETS[name])

        arrays["btmap1"] = _read_optional_netcdf_array(dataset, ["bt10"], lon.shape) - 273.15
        arrays["btmap2"] = _read_optional_netcdf_array(dataset, ["bt11"], lon.shape) - 273.15
        arrays["raamap"] = _read_optional_netcdf_array(dataset, ["raa"], lon.shape)
        arrays["vzamap"] = _read_optional_netcdf_array(dataset, ["vza"], lon.shape)
        arrays["szamap"] = _read_optional_netcdf_array(dataset, ["sza"], lon.shape)

    for name in arrays:
        arrays[name] = _masked_copy(arrays[name], lake_mask)
    return arrays, lon, lat, lake_mask


def read_acolite_l2_st(
    path: str | Path,
    geometry: dict[str, Any],
    sensor: str,
) -> dict[str, np.ndarray]:
    """Read ACOLITE ST variables and return Celsius LSWT maps."""

    Dataset = _netcdf_dataset()
    with Dataset(path, "r") as dataset:
        lon = _read_netcdf_array(dataset, "lon")
        lat = _read_netcdf_array(dataset, "lat")
        lake_mask = lake_mask_from_lonlat(geometry, lon, lat)
        if sensor == "LANDSAT_7":
            st1 = _read_netcdf_array(dataset, "st6_vcid_1")
            st2 = _read_netcdf_array(dataset, "st6_vcid_2")
        else:
            st1 = _read_netcdf_array(dataset, "st10")
            st2 = _read_netcdf_array(dataset, "st11")

    return {
        "lstmap1": _masked_copy(st1 - 273.15, lake_mask),
        "lstmap2": _masked_copy(st2 - 273.15, lake_mask),
    }


def read_usgs_lst(path: str | Path, geometry: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a USGS C2L2 ST_B raster, cropped to the lake, in Celsius."""

    raster, lon, lat, lake_mask = read_raster_crop(path, geometry, filled_nodata=0)
    lst = raster.astype("float32") * 0.00341802 + 149.0 - 273.15
    lst[(raster <= 0) | ~lake_mask] = np.nan
    return lst, lon, lat


def read_raster_crop(
    path: str | Path,
    geometry: dict[str, Any],
    *,
    filled_nodata: int | float = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Crop a raster to a lon/lat geometry and return values plus lon/lat grids."""

    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.mask import mask as rio_mask
    from rasterio.warp import transform as warp_transform
    from rasterio.warp import transform_geom

    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"Raster has no CRS: {path}")
        geom_raster = transform_geom("EPSG:4326", src.crs, geometry, precision=8)
        nodata = src.nodata if src.nodata is not None else filled_nodata
        out_image, out_transform = rio_mask(src, [geom_raster], crop=True, filled=True, nodata=nodata)
        raster = out_image[0]
        height, width = raster.shape
        rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        xs_arr = (
            out_transform.c
            + out_transform.a * (cols + 0.5)
            + out_transform.b * (rows + 0.5)
        ).astype("float64")
        ys_arr = (
            out_transform.f
            + out_transform.d * (cols + 0.5)
            + out_transform.e * (rows + 0.5)
        ).astype("float64")
        if src.crs.to_string() != "EPSG:4326":
            lon_flat, lat_flat = warp_transform(src.crs, "EPSG:4326", xs_arr.ravel(), ys_arr.ravel())
            lon = np.asarray(lon_flat, dtype="float64").reshape(raster.shape)
            lat = np.asarray(lat_flat, dtype="float64").reshape(raster.shape)
        else:
            lon = xs_arr
            lat = ys_arr
        lake_mask = geometry_mask([geom_raster], out_shape=raster.shape, transform=out_transform, invert=True)
    return raster, lon, lat, lake_mask


def clear_water_mask_on_grid(
    qa_path: str | Path,
    geometry: dict[str, Any],
    lon_target: np.ndarray,
    lat_target: np.ndarray,
    *,
    include_cirrus_as_cloud: bool = False,
) -> np.ndarray:
    """Return the Fmask-clear water mask resampled to the ACOLITE grid."""

    qa, lon_qa, lat_qa, lake_mask_qa = read_raster_crop(qa_path, geometry, filled_nodata=0)
    decoded = decode_qa_pixel(qa.astype("uint32"))
    cloud_like = decoded["dilated_cloud"] | decoded["cloud"] | decoded["cloud_shadow"]
    if include_cirrus_as_cloud:
        cloud_like |= decoded["cirrus"]

    cloud_free = ~cloud_like & ~decoded["fill"] & ~decoded["snow"] & lake_mask_qa
    water = decoded["water"] & lake_mask_qa
    cloud_free_on_target = nearest_resample(lon_qa, lat_qa, cloud_free.astype("uint8"), lon_target, lat_target) > 0
    water_on_target = nearest_resample(lon_qa, lat_qa, water.astype("uint8"), lon_target, lat_target) > 0
    target_lake = lake_mask_from_lonlat(geometry, lon_target, lat_target)
    return cloud_free_on_target & water_on_target & target_lake


def lake_mask_from_lonlat(geometry: dict[str, Any], lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Rasterize a GeoJSON geometry against curvilinear lon/lat arrays."""

    try:
        from matplotlib.path import Path as MplPath
    except ImportError:  # pragma: no cover - exercised only in lean runtimes.
        MplPath = None

    points = np.column_stack([lon.ravel(), lat.ravel()])
    mask = np.zeros(points.shape[0], dtype=bool)
    for outer, holes in _iter_polygon_rings(geometry):
        if MplPath is None:
            polygon_mask = _points_in_ring(points, outer)
        else:
            polygon_mask = MplPath(np.asarray(outer, dtype="float64")).contains_points(points)
        for hole in holes:
            if MplPath is None:
                polygon_mask &= ~_points_in_ring(points, hole)
            else:
                polygon_mask &= ~MplPath(np.asarray(hole, dtype="float64")).contains_points(points)
        mask |= polygon_mask
    return mask.reshape(lon.shape)


def nearest_resample(
    lon_src: np.ndarray,
    lat_src: np.ndarray,
    values: np.ndarray,
    lon_target: np.ndarray,
    lat_target: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbor sample one lon/lat grid onto another lon/lat grid."""

    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:  # pragma: no cover - dependency message.
        raise ImportError("scipy is required for nearest-grid resampling.") from exc

    source_points = np.column_stack([lon_src.ravel(), lat_src.ravel()])
    values_arr = np.asarray(values)
    source_values = values_arr.ravel()
    valid_source = np.isfinite(source_points).all(axis=1)
    if not np.any(valid_source):
        return np.full(lon_target.shape, np.nan, dtype="float32")

    query_points = np.column_stack([lon_target.ravel(), lat_target.ravel()])
    valid_query = np.isfinite(query_points).all(axis=1)
    out_dtype = values_arr.dtype if np.issubdtype(values_arr.dtype, np.floating) else np.float32
    out = np.full(query_points.shape[0], np.nan, dtype=out_dtype)
    tree = cKDTree(source_points[valid_source])
    _, index = tree.query(query_points[valid_query])
    out[valid_query] = source_values[valid_source][index]
    return out.reshape(lon_target.shape)


def predict_class_map(
    model: Any,
    feature_arrays: dict[str, np.ndarray],
    valid_mask: np.ndarray,
    *,
    shift_zero_based_labels: bool = False,
) -> np.ndarray:
    """Predict classifier labels as a float map with NaN outside valid pixels."""

    columns = model_feature_columns(model, list(feature_arrays.keys()))
    x, flat_indices = feature_dataframe(feature_arrays, valid_mask, columns)
    out = np.full(valid_mask.shape, np.nan, dtype="float32")
    if x.empty:
        return out

    pred = np.asarray(model.predict(x)).astype("int16")
    if shift_zero_based_labels or (pred.size and pred.min() == 0):
        pred = pred + 1
    out_flat = out.ravel()
    out_flat[flat_indices] = pred.astype("float32")
    return out


def filter_lst_with_classes(
    lst: np.ndarray,
    class_map: np.ndarray,
    mask_cloud_classes: tuple[int, ...] = (1,),
) -> np.ndarray:
    """Mask an LST map where the ML class map indicates a cloud class."""

    out = np.asarray(lst, dtype="float32").copy()
    out[np.isin(class_map, list(mask_cloud_classes))] = np.nan
    return out


def window_stats(
    lon: np.ndarray,
    lat: np.ndarray,
    values: np.ndarray,
    station: Station,
    numpix_xy: int,
    *,
    min_valid: int = 4,
) -> dict[str, float]:
    """Return median, mean, and std around the nearest pixel to a station."""

    pixlag = int(np.floor((numpix_xy - 1) / 2))
    pixlead = int(np.floor((numpix_xy + 1) / 2))
    distance = np.sqrt(np.square(lat - station.lat) + np.square(lon - station.lon))
    if not np.isfinite(distance).any():
        return {"median": np.nan, "mean": np.nan, "std": np.nan}

    center_flat = int(np.nanargmin(distance))
    row, col = np.unravel_index(center_flat, distance.shape)
    row_start = max(row - pixlag, 0)
    row_stop = min(row + pixlead, values.shape[0])
    col_start = max(col - pixlag, 0)
    col_stop = min(col + pixlead, values.shape[1])
    window = np.asarray(values[row_start:row_stop, col_start:col_stop], dtype="float32")
    valid = np.isfinite(window)
    if int(valid.sum()) < min_valid:
        return {"median": np.nan, "mean": np.nan, "std": np.nan}
    return {
        "median": float(np.nanmedian(window)),
        "mean": float(np.nanmean(window)),
        "std": float(np.nanstd(window)),
    }


def _append_l1_stats(
    row: dict[str, Any],
    arrays: dict[str, np.ndarray],
    lon: np.ndarray,
    lat: np.ndarray,
    stations: list[Station],
    numpix_xy: int,
) -> None:
    for station_index, station in enumerate(stations):
        for name in L1_STAT_NAMES:
            stats = window_stats(lon, lat, arrays[name], station, numpix_xy)
            row[f"{name}_mean{station_index}"] = stats["mean"]
            row[f"{name}_median{station_index}"] = stats["median"]
            row[f"{name}_std{station_index}"] = stats["std"]


def _append_l2_stats(
    row: dict[str, Any],
    arrays: dict[str, np.ndarray],
    lon: np.ndarray,
    lat: np.ndarray,
    stations: list[Station],
    numpix_xy: int,
) -> None:
    for station_index, station in enumerate(stations):
        for name in L2_STAT_NAMES:
            stats = window_stats(lon, lat, arrays[name], station, numpix_xy)
            row[f"{name}_median{station_index}"] = stats["median"]
            row[f"{name}_mean{station_index}"] = stats["mean"]
            row[f"{name}_std{station_index}"] = stats["std"]
            if station_index == 0:
                row[f"{name}_mean_all_pixels"] = _nanmean(arrays[name])
                row[f"{name}_median_all_pixels"] = _nanmedian(arrays[name])


def _base_row(info: SceneInfo, *, lake_name: str, include_lake_metadata: bool) -> dict[str, Any]:
    row: dict[str, Any] = {
        "scene_id": info.scene_id,
        "sensor": info.sensor,
        "time_utc": info.time_utc,
    }
    if include_lake_metadata:
        row["lakeN"] = lake_name
        row["pathrow"] = info.pathrow
    return row


def _rows_to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.set_index("scene_id")


def _class_for_lst(class_map: np.ndarray, lst: np.ndarray) -> np.ndarray:
    out = class_map.copy()
    out[~np.isfinite(lst)] = np.nan
    return out


def _masked_copy(arr: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype="float32").copy()
    out[~np.asarray(valid_mask, dtype=bool)] = np.nan
    return out


def _clean_zero_as_nan(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype="float32").copy()
    out[out == 0] = np.nan
    out[~np.isfinite(out)] = np.nan
    return out


def _nanmean(arr: np.ndarray) -> float:
    valid = np.isfinite(arr)
    return float(np.nanmean(arr)) if valid.any() else np.nan


def _nanmedian(arr: np.ndarray) -> float:
    valid = np.isfinite(arr)
    return float(np.nanmedian(arr)) if valid.any() else np.nan


def _scene_index(folder: Path, *, keep_landsat_8_9_only: bool) -> dict[tuple[str, str, str], SceneInfo]:
    scenes: dict[tuple[str, str, str], SceneInfo] = {}
    for child in sorted(path for path in folder.iterdir() if path.is_dir()):
        try:
            info = parse_scene_info(child)
        except ValueError:
            continue
        if keep_landsat_8_9_only and info.sensor not in {"LANDSAT_8", "LANDSAT_9"}:
            continue
        scenes.setdefault(info.match_key, info)
    return scenes


def _find_file(folder: Path, patterns: list[str]) -> Path:
    for pattern in patterns:
        matches = sorted(path for path in folder.glob(pattern) if path.is_file())
        if not matches:
            matches = sorted(path for path in folder.rglob(pattern) if path.is_file())
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No file matching {patterns} found under {folder}.")


def _folder_variants(root: Path, folder_name: str, suffix: str) -> list[Path]:
    bases = {
        folder_name,
        folder_name.lower(),
        folder_name.upper(),
        folder_name.capitalize(),
    }
    return [root / f"{base}_{suffix}" for base in bases]


def _nested_l2_variants(l2_base: Path | None, names: list[str]) -> list[Path]:
    if l2_base is None:
        return []
    return [l2_base / name for name in names]


def _first_existing_dir(paths: list[Path]) -> Path | None:
    for path in paths:
        if path is not None and path.exists() and path.is_dir():
            return path
    return None


def _read_reflectance_variable(dataset: Any, candidates: list[str], target_nm: int) -> np.ndarray:
    for name in candidates:
        if name in dataset.variables:
            return _read_netcdf_array(dataset, name)

    wavelength_matches = []
    for name in dataset.variables:
        match = re.fullmatch(r"rhot_(\d+)", name)
        if match:
            wavelength_matches.append((abs(int(match.group(1)) - target_nm), name))
    if wavelength_matches:
        _, best_name = min(wavelength_matches)
        return _read_netcdf_array(dataset, best_name)
    raise KeyError(f"Could not find reflectance variable near {target_nm} nm.")


def _read_optional_netcdf_array(dataset: Any, names: list[str], shape: tuple[int, int]) -> np.ndarray:
    for name in names:
        if name in dataset.variables:
            return _read_netcdf_array(dataset, name)
    return np.full(shape, np.nan, dtype="float32")


def _read_netcdf_array(dataset: Any, name: str) -> np.ndarray:
    if name not in dataset.variables:
        raise KeyError(f"Variable '{name}' was not found in {getattr(dataset, 'filepath', lambda: 'NetCDF')()}.")
    arr = dataset.variables[name][:]
    if np.ma.isMaskedArray(arr):
        arr = arr.filled(np.nan)
    out = np.asarray(arr, dtype="float32")
    out[~np.isfinite(out)] = np.nan
    return out


def _netcdf_dataset() -> Any:
    try:
        from netCDF4 import Dataset
    except ImportError as exc:  # pragma: no cover - dependency message.
        raise ImportError("netCDF4 is required to read ACOLITE NetCDF outputs.") from exc
    return Dataset


def _iter_polygon_rings(geometry: dict[str, Any]) -> list[tuple[list[list[float]], list[list[list[float]]]]]:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])
    if geometry_type == "Polygon":
        return [(coordinates[0], coordinates[1:])]
    if geometry_type == "MultiPolygon":
        return [(polygon[0], polygon[1:]) for polygon in coordinates]
    raise ValueError(f"Unsupported geometry type for lake mask: {geometry_type}")


def _load_lake_geometry(geojson_path: str | Path, lake_key: str | None = None) -> dict[str, Any]:
    path = Path(geojson_path)
    data = json.loads(path.read_text(encoding="utf-8"))

    if data.get("type") == "FeatureCollection":
        features = data.get("features", [])
        if lake_key:
            lake_key_lower = lake_key.lower()
            for feature in features:
                props = feature.get("properties", {})
                candidates = [
                    props.get("key"),
                    props.get("name"),
                    props.get("lake"),
                    props.get("lakeN"),
                ]
                if any(str(value).lower() == lake_key_lower for value in candidates if value is not None):
                    return feature["geometry"]
            raise KeyError(f"Lake key '{lake_key}' was not found in {path}.")
        if len(features) != 1:
            raise ValueError("GeoJSON has multiple features; provide lake_key.")
        return features[0]["geometry"]

    if data.get("type") == "Feature":
        return data["geometry"]

    if data.get("type") in {"Polygon", "MultiPolygon"}:
        return data

    raise ValueError(f"Unsupported GeoJSON object type in {path}: {data.get('type')}")


def _points_in_ring(points: np.ndarray, ring: list[list[float]]) -> np.ndarray:
    """Vectorized point-in-polygon fallback for environments without matplotlib."""

    ring_arr = np.asarray(ring, dtype="float64")
    if ring_arr.shape[0] < 3:
        return np.zeros(points.shape[0], dtype=bool)
    if not np.allclose(ring_arr[0], ring_arr[-1]):
        ring_arr = np.vstack([ring_arr, ring_arr[0]])

    x = points[:, 0]
    y = points[:, 1]
    inside = np.zeros(points.shape[0], dtype=bool)
    x0 = ring_arr[:-1, 0]
    y0 = ring_arr[:-1, 1]
    x1 = ring_arr[1:, 0]
    y1 = ring_arr[1:, 1]

    for start_x, start_y, end_x, end_y in zip(x0, y0, x1, y1):
        crosses = (start_y > y) != (end_y > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_intersect = (end_x - start_x) * (y - start_y) / (end_y - start_y) + start_x
        inside ^= crosses & (x < x_intersect)
    return inside
