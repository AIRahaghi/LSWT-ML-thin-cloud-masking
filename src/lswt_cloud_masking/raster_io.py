"""Raster writing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import rasterio


def output_profile(template_profile: Mapping, *, count: int = 1, dtype: str = "uint8") -> dict:
    """Return a compact GeoTIFF profile based on a cropped source profile."""

    profile = dict(template_profile)
    profile.update(
        driver="GTiff",
        count=count,
        dtype=dtype,
        nodata=0,
        compress="deflate",
        tiled=False,
    )
    return profile


def write_single_band(
    path: str | Path,
    array: np.ndarray,
    template_profile: Mapping,
    *,
    description: str,
    tags: Mapping[str, str] | None = None,
) -> Path:
    """Write one mask array to GeoTIFF."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = output_profile(template_profile, count=1, dtype=str(array.dtype))

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(array, 1)
        dst.set_band_description(1, description)
        if tags:
            dst.update_tags(**dict(tags))
    return out_path


def write_stack(
    path: str | Path,
    arrays: Mapping[str, np.ndarray],
    template_profile: Mapping,
    *,
    tags: Mapping[str, str] | None = None,
) -> Path:
    """Write a multi-band GeoTIFF where keys become band descriptions."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    names = list(arrays.keys())
    first = arrays[names[0]]
    profile = output_profile(template_profile, count=len(names), dtype=str(first.dtype))

    with rasterio.open(out_path, "w", **profile) as dst:
        for index, name in enumerate(names, start=1):
            dst.write(arrays[name].astype(first.dtype), index)
            dst.set_band_description(index, name)
        if tags:
            dst.update_tags(**dict(tags))
    return out_path
