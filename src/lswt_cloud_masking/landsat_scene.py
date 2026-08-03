"""Landsat Collection-2 Level-1 scene loading and scaling."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import tarfile
from typing import Any

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.mask import mask as rio_mask
from rasterio.warp import transform_geom


REFLECTIVE_BANDS = {
    1: "aerosolmap",
    2: "bluemap",
    3: "greenmap",
    4: "redmap",
    5: "nirmap",
    6: "swirmap1",
    7: "swirmap2",
    9: "cirrusmap",
}

THERMAL_BANDS = {
    10: "btmap1",
    11: "btmap2",
}


@dataclass(frozen=True)
class SceneFiles:
    """Resolved file access for one Landsat scene."""

    scene_path: Path
    scene_id: str
    mtl: dict[str, Any]
    is_tar: bool = False

    def raster_path(self, suffix: str) -> str:
        """Return a raster path usable by rasterio."""

        if self.is_tar:
            member = _find_tar_member(self.scene_path, suffix)
            return f"/vsitar/{self.scene_path.as_posix()}/{member}"
        return str(_find_folder_file(self.scene_path, suffix))


def load_lake_geometry(geojson_path: str | Path, lake_key: str | None = None) -> dict[str, Any]:
    """Load a Polygon or MultiPolygon geometry from a GeoJSON file."""

    path = Path(geojson_path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

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
            raise ValueError("GeoJSON has multiple features; provide --lake-key.")
        return features[0]["geometry"]

    if data.get("type") == "Feature":
        return data["geometry"]

    if data.get("type") in {"Polygon", "MultiPolygon"}:
        return data

    raise ValueError(f"Unsupported GeoJSON object type in {path}: {data.get('type')}")


def resolve_scene(scene_path: str | Path) -> SceneFiles:
    """Resolve scene ID, metadata, and folder/tar access."""

    path = Path(scene_path)
    if not path.exists():
        raise FileNotFoundError(path)

    is_tar = path.is_file() and path.suffix.lower() in {".tar", ".tgz", ".gz"}
    if is_tar:
        mtl = _read_mtl_from_tar(path)
        scene_id = _scene_id_from_mtl(mtl, path.stem)
    else:
        mtl = _read_mtl_from_folder(path)
        scene_id = _scene_id_from_mtl(mtl, path.name)
    return SceneFiles(scene_path=path, scene_id=scene_id, mtl=mtl, is_tar=is_tar)


def crop_raster_to_geometry(
    raster_path: str,
    geometry_epsg4326: dict[str, Any],
    *,
    filled_nodata: int | float = 0,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    """Crop one raster to a lon/lat geometry and return data, profile, and lake mask."""

    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError(f"Raster has no CRS: {raster_path}")
        geom_raster = transform_geom("EPSG:4326", src.crs, geometry_epsg4326, precision=8)
        nodata = src.nodata if src.nodata is not None else filled_nodata
        out_image, out_transform = rio_mask(
            src,
            [geom_raster],
            crop=True,
            filled=True,
            nodata=nodata,
        )
        profile = src.profile.copy()

    arr = out_image[0]
    profile.update(
        height=arr.shape[0],
        width=arr.shape[1],
        transform=out_transform,
        count=1,
    )
    lake_mask = geometry_mask(
        [geom_raster],
        out_shape=arr.shape,
        transform=out_transform,
        invert=True,
    )
    return arr, profile, lake_mask


def load_l1_feature_arrays(
    scene: SceneFiles,
    geometry_epsg4326: dict[str, Any],
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any], np.ndarray]:
    """Load QA plus the scaled raw feature arrays used by the ML models."""

    qa, profile, lake_mask = crop_raster_to_geometry(
        scene.raster_path("_QA_PIXEL.TIF"),
        geometry_epsg4326,
        filled_nodata=0,
    )
    arrays: dict[str, np.ndarray] = {}

    for band, name in REFLECTIVE_BANDS.items():
        dn, _, _ = crop_raster_to_geometry(scene.raster_path(f"_B{band}.TIF"), geometry_epsg4326)
        arrays[name] = scale_reflectance(dn, band, scene.mtl)

    for band, name in THERMAL_BANDS.items():
        dn, _, _ = crop_raster_to_geometry(scene.raster_path(f"_B{band}.TIF"), geometry_epsg4326)
        arrays[name] = scale_brightness_temperature_c(dn, band, scene.mtl)

    return arrays, qa.astype("uint32"), profile, lake_mask


def scale_reflectance(dn: np.ndarray, band: int, mtl: dict[str, Any]) -> np.ndarray:
    """Scale Level-1 DN to top-of-atmosphere reflectance."""

    mult = _mtl_float(mtl, f"REFLECTANCE_MULT_BAND_{band}")
    add = _mtl_float(mtl, f"REFLECTANCE_ADD_BAND_{band}")
    sun_elevation = _mtl_float(mtl, "SUN_ELEVATION", default=None)

    dn_float = dn.astype("float32")
    out = dn_float * mult + add
    if sun_elevation is not None:
        sun_sin = math.sin(math.radians(sun_elevation))
        if sun_sin > 0:
            out = out / sun_sin

    out = out.astype("float32")
    out[dn_float <= 0] = np.nan
    out[~np.isfinite(out)] = np.nan
    return out


def scale_brightness_temperature_c(dn: np.ndarray, band: int, mtl: dict[str, Any]) -> np.ndarray:
    """Scale Level-1 thermal DN to at-sensor brightness temperature in Celsius."""

    radiance_mult = _mtl_float(mtl, f"RADIANCE_MULT_BAND_{band}")
    radiance_add = _mtl_float(mtl, f"RADIANCE_ADD_BAND_{band}")
    k1 = _mtl_float(mtl, f"K1_CONSTANT_BAND_{band}")
    k2 = _mtl_float(mtl, f"K2_CONSTANT_BAND_{band}")

    dn_float = dn.astype("float32")
    radiance = dn_float * radiance_mult + radiance_add
    with np.errstate(divide="ignore", invalid="ignore"):
        bt_kelvin = k2 / np.log((k1 / radiance) + 1.0)
    bt_celsius = (bt_kelvin - 273.15).astype("float32")
    bt_celsius[(dn_float <= 0) | (radiance <= 0)] = np.nan
    bt_celsius[~np.isfinite(bt_celsius)] = np.nan
    return bt_celsius


def _read_mtl_from_folder(path: Path) -> dict[str, Any]:
    json_files = sorted(path.glob("*_MTL.json"))
    if json_files:
        return _flatten_mtl_json(json.loads(json_files[0].read_text(encoding="utf-8")))

    txt_files = sorted(path.glob("*_MTL.txt"))
    if txt_files:
        return _parse_mtl_text(txt_files[0].read_text(encoding="utf-8", errors="ignore"))

    raise FileNotFoundError(f"No *_MTL.json or *_MTL.txt found in {path}.")


def _read_mtl_from_tar(path: Path) -> dict[str, Any]:
    with tarfile.open(path, "r:*") as archive:
        members = archive.getmembers()
        json_members = [member for member in members if member.name.endswith("_MTL.json")]
        if json_members:
            with archive.extractfile(json_members[0]) as handle:
                if handle is None:
                    raise FileNotFoundError(json_members[0].name)
                return _flatten_mtl_json(json.loads(handle.read().decode("utf-8")))

        txt_members = [member for member in members if member.name.endswith("_MTL.txt")]
        if txt_members:
            with archive.extractfile(txt_members[0]) as handle:
                if handle is None:
                    raise FileNotFoundError(txt_members[0].name)
                return _parse_mtl_text(handle.read().decode("utf-8", errors="ignore"))

    raise FileNotFoundError(f"No *_MTL.json or *_MTL.txt found in {path}.")


def _parse_mtl_text(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"')
        if key and key not in {"GROUP", "END_GROUP"}:
            out[key] = value
    return out


def _flatten_mtl_json(data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}

    def _walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if not isinstance(child, dict):
                    out[key] = child
                _walk(child)

    _walk(data)
    return out


def _mtl_float(mtl: dict[str, Any], key: str, default: float | None = None) -> float:
    if key not in mtl:
        if default is not None:
            return default
        raise KeyError(f"Missing MTL key: {key}")
    return float(mtl[key])


def _scene_id_from_mtl(mtl: dict[str, Any], fallback: str) -> str:
    for key in ["LANDSAT_PRODUCT_ID", "LANDSAT_SCENE_ID"]:
        value = mtl.get(key)
        if value:
            return str(value).strip('"')
    return fallback


def _find_folder_file(scene_dir: Path, suffix: str) -> Path:
    matches = sorted(path for path in scene_dir.glob(f"*{suffix}") if path.is_file())
    if not matches:
        matches = sorted(path for path in scene_dir.rglob(f"*{suffix}") if path.is_file())
    if not matches:
        raise FileNotFoundError(f"No file ending with {suffix} found under {scene_dir}.")
    return matches[0]


def _find_tar_member(tar_path: Path, suffix: str) -> str:
    with tarfile.open(tar_path, "r:*") as archive:
        for member in archive.getmembers():
            if member.name.endswith(suffix):
                return member.name
    raise FileNotFoundError(f"No tar member ending with {suffix} found in {tar_path}.")


def scene_pathrow(scene_id: str) -> str | None:
    """Extract WRS path/row from a Landsat product ID when present."""

    match = re.search(r"L[COTEM]0[89]_[A-Z0-9]+_(\d{6})_", scene_id)
    return match.group(1) if match else None
