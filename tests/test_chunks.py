"""``chunks={}`` correctness via `Variable.encoding["preferred_chunks"]`.

xarray's chunking pass reads `Variable.encoding["preferred_chunks"]`
to decide dask chunk shapes when the user passes `chunks={}` to
``xr.open_datatree``. Without it the pass collapses to a single
chunk per dim, which is wrong for any array that actually has chunks
on disk (e.g. radar data with `(1, azimuth, range)` chunks in a
multi-time arrays).

This module verifies:
  - `ZarrsArrayHandle.chunks` exposes the on-disk chunk shape.
  - `_node_to_dataset` populates `var.encoding["chunks"]` (tuple) and
    `var.encoding["preferred_chunks"]` (dim->size dict) for every var.
  - End-to-end via `xr.open_datatree(..., chunks={})` produces dask
    arrays whose chunks match the on-disk shape — multi-chunk along
    chunked dimensions, not one big chunk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr

from rustytree._rustytree import open_datatree


@pytest.fixture
def chunked_zarr_store(tmp_path: Path) -> Path:
    """Vanilla v3 store with non-trivial chunk layout.

    Layout::

        store.zarr/
            t       shape=(8,)        chunks=(4,)        — coord
            field   shape=(8, 12, 6)  chunks=(1, 12, 6)  — multi-chunk along t
    """
    path = tmp_path / "chunked.zarr"
    root = zarr.create_group(store=str(path), zarr_format=3)

    t = root.create_array("t", shape=(8,), dtype="int64", chunks=(4,), dimension_names=("t",))
    t[:] = np.arange(8, dtype=np.int64)

    field = root.create_array(
        "field",
        shape=(8, 12, 6),
        dtype="float32",
        chunks=(1, 12, 6),
        dimension_names=("t", "y", "x"),
    )
    field[:] = np.arange(8 * 12 * 6, dtype=np.float32).reshape(8, 12, 6)
    return path


def test_handle_exposes_chunks(chunked_zarr_store: Path) -> None:
    tree = open_datatree(str(chunked_zarr_store))
    by_name = {var["name"]: var for var in tree["/"]["vars"]}
    assert tuple(by_name["t"]["handle"].chunks) == (4,)
    assert tuple(by_name["field"]["handle"].chunks) == (1, 12, 6)


def test_variable_encoding_carries_preferred_chunks(chunked_zarr_store: Path) -> None:
    """End-to-end via xarray entrypoint: every Variable's encoding
    includes `chunks` (tuple) and `preferred_chunks` (dim->size dict)
    matching the on-disk chunk shape."""
    dt = xr.open_datatree(str(chunked_zarr_store), engine="rustytree")

    field = dt["/"].dataset["field"]
    assert field.encoding["chunks"] == (1, 12, 6)
    assert field.encoding["preferred_chunks"] == {"t": 1, "y": 12, "x": 6}

    t = dt["/"].dataset["t"]
    assert t.encoding["chunks"] == (4,)
    assert t.encoding["preferred_chunks"] == {"t": 4}


def test_chunks_empty_uses_preferred(chunked_zarr_store: Path) -> None:
    """With `chunks={}`, dask should chunk according to the on-disk
    shape — so `field` (chunks=(1,12,6) over shape (8,12,6)) gets
    8 chunks along t, not a single big chunk."""
    pytest.importorskip("dask")
    dt = xr.open_datatree(str(chunked_zarr_store), engine="rustytree", chunks={})
    field = dt["/"].dataset["field"]
    # field.chunks is a tuple of tuples, one per dim, of chunk lengths
    assert field.chunks == ((1, 1, 1, 1, 1, 1, 1, 1), (12,), (6,))


def test_chunks_empty_round_trips_values(chunked_zarr_store: Path) -> None:
    """Materialise via dask and verify values match the underlying
    numpy round-trip."""
    pytest.importorskip("dask")
    dt = xr.open_datatree(str(chunked_zarr_store), engine="rustytree", chunks={})
    actual = dt["/"].dataset["field"].values
    expected = np.arange(8 * 12 * 6, dtype=np.float32).reshape(8, 12, 6)
    np.testing.assert_array_equal(actual, expected)
