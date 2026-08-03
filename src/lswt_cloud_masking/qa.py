"""Landsat C2 QA_PIXEL decoding helpers."""

from __future__ import annotations

import numpy as np


FmaskLayerValues = {
    0: "outside_or_nodata",
    1: "fmask_clear_water",
    2: "fmask_cloud_cirrus_shadow_or_dilated",
    3: "other_inside_crop",
}


def qa_bit(qa: np.ndarray, bit_index: int) -> np.ndarray:
    """Return a boolean array for one QA_PIXEL bit."""

    return ((qa.astype("uint32") >> bit_index) & 1).astype(bool)


def decode_qa_pixel(qa: np.ndarray) -> dict[str, np.ndarray]:
    """Decode the Landsat Collection-2 QA_PIXEL flags used by Fmask."""

    return {
        "fill": qa_bit(qa, 0),
        "dilated_cloud": qa_bit(qa, 1),
        "cirrus": qa_bit(qa, 2),
        "cloud": qa_bit(qa, 3),
        "cloud_shadow": qa_bit(qa, 4),
        "snow": qa_bit(qa, 5),
        "clear": qa_bit(qa, 6),
        "water": qa_bit(qa, 7),
    }


def fmask_cloud_flags(decoded: dict[str, np.ndarray]) -> np.ndarray:
    """Return Fmask cloud-like flags: cloud, cirrus, shadow, or dilated cloud."""

    return (
        decoded["dilated_cloud"]
        | decoded["cirrus"]
        | decoded["cloud"]
        | decoded["cloud_shadow"]
    )


def fmask_clear_water(decoded: dict[str, np.ndarray], lake_mask: np.ndarray | None = None) -> np.ndarray:
    """Return pixels that are water and free of Fmask cloud-like flags."""

    clear_water = decoded["water"] & ~decoded["fill"] & ~decoded["snow"] & ~fmask_cloud_flags(decoded)
    if lake_mask is not None:
        clear_water &= np.asarray(lake_mask, dtype=bool)
    return clear_water


def fmask_class_layer(qa: np.ndarray, lake_mask: np.ndarray) -> np.ndarray:
    """Build a compact operational Fmask layer for the cropped lake area."""

    decoded = decode_qa_pixel(qa)
    lake = np.asarray(lake_mask, dtype=bool)
    out = np.zeros(qa.shape, dtype="uint8")

    cloud_like = fmask_cloud_flags(decoded) & lake
    clear_water = fmask_clear_water(decoded, lake)
    other = lake & ~(cloud_like | clear_water)

    out[clear_water] = 1
    out[cloud_like] = 2
    out[other] = 3
    return out
