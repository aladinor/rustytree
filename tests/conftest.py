"""Shared fixtures.

A vanilla Zarr v3 store and an icechunk repository, both built fresh per
test from ``tmp_path`` so tests are isolated. Each fixture writes the same
two arrays via :func:`_write_tiny_layout` so the walk assertions stay
shared between them.
"""

from __future__ import annotations

from pathlib import Path

import icechunk
import numpy as np
import pytest
import zarr


def _write_tiny_layout(root: zarr.Group) -> None:
    """Write the canonical 2-array layout used by both fixtures.

    Layout::

        <root>/                         attrs: {"title": "tiny"}
            temp     (lat=4, lon=3)     dtype: float64, attrs: {"units": "K"}
            mask     (lat=4, lon=3)     dtype: int8

    Both arrays advertise dimension names so the walk surfaces them
    verbatim.
    """
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


def _write_multilevel_layout(root: zarr.Group) -> None:
    """Write the 3-level layout shared by `multilevel_zarr_store` and
    `multilevel_icechunk_repo`. See ``multilevel_zarr_store``'s
    docstring for the full shape.
    """
    root.attrs["title"] = "multilevel"
    root.create_array("x", shape=(4,), dtype="float64", chunks=(4,), dimension_names=("x",))[:] = (
        np.arange(4, dtype=np.float64)
    )

    volume_a = root.create_group("volume_a")
    volume_a.attrs["id"] = "A"
    volume_a.create_array(
        "temp", shape=(4,), dtype="float64", chunks=(4,), dimension_names=("x",)
    )[:] = np.arange(4, dtype=np.float64)

    for i, angle in enumerate([0.5, 1.5]):
        sweep = volume_a.create_group(f"sweep_{i}")
        sweep.attrs["angle"] = angle
        sweep.create_array(
            "dbz",
            shape=(8, 16),
            dtype="float32",
            chunks=(8, 16),
            dimension_names=("azimuth", "range"),
        )[:] = np.zeros((8, 16), dtype=np.float32)


@pytest.fixture
def tiny_zarr_store(tmp_path: Path) -> Path:
    """Vanilla Zarr v3 store at ``tmp_path/store.zarr``."""
    path = tmp_path / "store.zarr"
    root = zarr.create_group(store=str(path), zarr_format=3)
    _write_tiny_layout(root)
    return path


@pytest.fixture
def multilevel_zarr_store(tmp_path: Path) -> Path:
    """A 3-level vanilla Zarr v3 store for exercising the recursive walk.

    Layout::

        store.zarr/                                         attrs: {"title": "multilevel"}
            x        (n=4)        float64
            volume_a/                                       attrs: {"id": "A"}
                volume_a/temp     (n=4)  float64
                volume_a/sweep_0/                           attrs: {"angle": 0.5}
                    volume_a/sweep_0/dbz   (a=8, r=16)  float32
                volume_a/sweep_1/                           attrs: {"angle": 1.5}
                    volume_a/sweep_1/dbz   (a=8, r=16)  float32

    Five groups total: `/`, `/volume_a`, `/volume_a/sweep_0`,
    `/volume_a/sweep_1` (and the implicit array containers — those don't
    show up as groups). The walk should surface 4 group nodes.
    """
    path = tmp_path / "store.zarr"
    root = zarr.create_group(store=str(path), zarr_format=3)
    _write_multilevel_layout(root)
    return path


@pytest.fixture
def tiny_icechunk_repo(tmp_path: Path) -> Path:
    """Fresh icechunk repository with the same layout as ``tiny_zarr_store``."""
    path = tmp_path / "repo"
    storage = icechunk.local_filesystem_storage(str(path))
    repo = icechunk.Repository.create(storage)
    session = repo.writable_session("main")
    root = zarr.create_group(store=session.store, zarr_format=3)
    _write_tiny_layout(root)
    session.commit("init")
    return path


@pytest.fixture
def multilevel_icechunk_repo(tmp_path: Path) -> Path:
    """Multilevel icechunk repo with the same layout as ``multilevel_zarr_store``."""
    path = tmp_path / "repo"
    storage = icechunk.local_filesystem_storage(str(path))
    repo = icechunk.Repository.create(storage)
    session = repo.writable_session("main")
    root = zarr.create_group(store=session.store, zarr_format=3)
    _write_multilevel_layout(root)
    session.commit("init")
    return path
