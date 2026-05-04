"""Shared fixtures.

Right now: a tiny, deterministic local Zarr v3 store written via
``zarr-python`` so the rustytree walk has a known shape to assert against.
Lives at a per-session ``tmp_path`` so tests are isolated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr


@pytest.fixture
def tiny_zarr_store(tmp_path: Path) -> Path:
    """Write a 2-array Zarr v3 store at ``tmp_path/store.zarr`` and return the path.

    Layout::

        store.zarr/                     attrs: {"title": "tiny"}
            temp     (lat=4, lon=3)     dtype: float64, attrs: {"units": "K"}
            mask     (lat=4, lon=3)     dtype: int8

    Both arrays advertise dimension names so the walk surfaces them
    verbatim.
    """
    path = tmp_path / "store.zarr"
    root = zarr.create_group(store=str(path), zarr_format=3)
    root.attrs["title"] = "tiny"

    temp = root.create_array(
        "temp",
        shape=(4, 3),
        dtype="float64",
        chunks=(2, 3),
        dimension_names=("lat", "lon"),
    )
    temp[:] = np.arange(12, dtype=np.float64).reshape(4, 3)
    temp.attrs["units"] = "K"

    mask = root.create_array(
        "mask",
        shape=(4, 3),
        dtype="int8",
        chunks=(4, 3),
        dimension_names=("lat", "lon"),
    )
    mask[:] = np.zeros((4, 3), dtype=np.int8)

    return path
