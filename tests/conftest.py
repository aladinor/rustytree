"""Shared fixtures.

A vanilla Zarr v3 store and an icechunk repository, both built fresh per
test from ``tmp_path`` so tests are isolated. Each fixture writes the same
two arrays via :func:`_write_tiny_layout` so the walk assertions stay
shared between them.
"""

from __future__ import annotations

import os
from pathlib import Path

import icechunk
import numpy as np
import pytest
import zarr

# ---- KTWX opt-in smoke repo ----------------------------------------
#
# A few tests smoke rustytree against a real radar icechunk repo on the
# maintainer's machine. They must skip cleanly everywhere else — including
# CI, and including the case where the directory exists but is empty.

KTWX_PATH = Path("/home/alfonso-ladino/python/raw2zarr/zarr/KTWX")


def ktwx_repo_available() -> bool:
    """Whether `KTWX_PATH` holds a usable icechunk repository.

    Mirrors rustytree's own detector (`looks_like_icechunk_repo` in
    `src/icechunk_store.rs`): a `repo` manifest file plus a `snapshots/`
    directory. Testing `Path.exists()` alone is not enough — an empty
    leftover directory passes that check, so the guard doesn't fire and
    the test fails with `KeyError: group / not found in store` instead of
    skipping.
    """
    if os.environ.get("RUSTYTREE_SKIP_KTWX") == "1":
        return False
    return (KTWX_PATH / "repo").is_file() and (KTWX_PATH / "snapshots").is_dir()


#: Reason string shared by the KTWX `skipif` marks.
KTWX_SKIP_REASON = f"no icechunk repo at {KTWX_PATH} (set RUSTYTREE_SKIP_KTWX=1 to skip explicitly)"


def vars_by_name(tree: dict, group: str = "/") -> dict[str, dict]:
    """Index one walked group's vars by name.

    The walk returns `vars` as a list; nearly every test wants it keyed.
    Lives here because the var-dict shape has already changed once (when
    `data` was added for eager fetch) and three modules were carrying
    their own copy.
    """
    return {var["name"]: var for var in tree[group]["vars"]}


@pytest.fixture
def track_raw_indexing_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    """Record every `(shape, dtype)` seen by
    `RustyBackendArray._raw_indexing_method`, so a test can assert an
    eager-fetched var never round-trips through the lazy backend. Shared by
    the numeric (`test_eager_fetch.py`) and string (`test_string_arrays.py`)
    "no lazy read" regression tests, which were previously duplicating this
    monkeypatch setup verbatim.
    """
    from rustytree._array import RustyBackendArray

    touched: list[tuple] = []
    original = RustyBackendArray._raw_indexing_method

    def tracking(self: RustyBackendArray, key: tuple) -> np.ndarray:
        touched.append((self.shape, self.dtype))
        return original(self, key)

    monkeypatch.setattr(RustyBackendArray, "_raw_indexing_method", tracking)
    return touched


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
    volume_a.create_array("temp", shape=(4,), dtype="float64", chunks=(4,), dimension_names=("x",))[
        :
    ] = np.arange(4, dtype=np.float64)

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
