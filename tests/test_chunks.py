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


# ---- empty selections (issue #65) ----


@pytest.fixture
def ragged_zarr_store(tmp_path: Path) -> Path:
    """Store whose shape is NOT a multiple of its chunk shape.

    The ragged final chunk is what made an empty read at the very end of
    the array panic rather than merely return the wrong length.
    """
    path = tmp_path / "ragged.zarr"
    root = zarr.create_group(store=str(path), zarr_format=3)
    v = root.create_array("v", shape=(13,), dtype="float64", chunks=(4,), dimension_names=("n",))
    v[:] = np.arange(13, dtype=np.float64)
    # Self-named dim coord over the same ragged dim. The walk pre-fetches
    # these (`should_eager_fetch`), so it is served from `var["data"]`
    # and never calls `read_subset` -- the eager and lazy halves must
    # still agree on length for an empty selection.
    n = root.create_array("n", shape=(13,), dtype="int64", chunks=(4,), dimension_names=("n",))
    n[:] = np.arange(13, dtype=np.int64)
    c = root.create_array(
        "c", shape=(5, 4, 3), dtype="float32", chunks=(2, 3, 2), dimension_names=("z", "cy", "cx")
    )
    c[:] = np.arange(60, dtype=np.float32).reshape(5, 4, 3)
    g = root.create_array(
        "g", shape=(7, 5), dtype="float64", chunks=(3, 2), dimension_names=("y", "x")
    )
    g[:] = np.arange(35, dtype=np.float64).reshape(7, 5)
    return path


@pytest.mark.parametrize("start", [0, 1, 4, 5, 11, 12, 13])
def test_empty_selection_returns_no_elements(ragged_zarr_store: Path, start: int) -> None:
    """`read_subset` with `start == stop` must yield zero elements.

    Regression test for #65. Offsets that are not a multiple of the
    chunk size (1, 5, 11, 13) used to return one bogus element; `13` --
    the end of the ragged final chunk -- raised `PanicException`, which
    derives from `BaseException` and so slips past `except Exception`.
    """
    tree = open_datatree(str(ragged_zarr_store))
    handle = {var["name"]: var["handle"] for var in tree["/"]["vars"]}["v"]
    out = handle.read_subset([(start, start)])
    assert len(out) == 0
    assert out.dtype == np.float64


@pytest.mark.parametrize(
    "sel",
    [
        {"n": slice(1, 1)},
        {"n": slice(13, 13)},
        {"n": slice(0, 0)},
        {"n": slice(2, 5)},  # non-empty control
    ],
)
def test_empty_selection_matches_zarr_engine_1d(ragged_zarr_store: Path, sel: dict) -> None:
    rusty = xr.open_dataset(str(ragged_zarr_store), engine="rustytree")
    zarr_ds = xr.open_dataset(str(ragged_zarr_store), engine="zarr", consolidated=False)
    np.testing.assert_array_equal(rusty.v.isel(**sel).values, zarr_ds.v.isel(**sel).values)


@pytest.mark.parametrize(
    "sel",
    [
        {"y": slice(1, 1)},  # one axis empty, the other full
        {"x": slice(3, 3)},
        {"y": slice(1, 1), "x": slice(2, 4)},  # empty axis + non-aligned offset
        {"y": slice(7, 7)},  # end of a ragged dimension
        {"y": slice(1, 1), "x": slice(3, 3)},  # every axis empty at once
        {"y": slice(5, 2)},  # reversed slice: empty for numpy, must not raise
        {"y": slice(1, 4), "x": slice(1, 4)},  # non-empty control
    ],
)
def test_empty_selection_matches_zarr_engine_2d(ragged_zarr_store: Path, sel: dict) -> None:
    """An empty axis must not disturb the other axes' extents."""
    rusty = xr.open_dataset(str(ragged_zarr_store), engine="rustytree")
    zarr_ds = xr.open_dataset(str(ragged_zarr_store), engine="zarr", consolidated=False)
    np.testing.assert_array_equal(rusty.g.isel(**sel).values, zarr_ds.g.isel(**sel).values)


@pytest.mark.parametrize(
    "sel",
    [
        {"z": slice(1, 1)},
        {"cx": slice(3, 3)},
        {"z": slice(1, 1), "cy": slice(1, 3), "cx": slice(1, 3)},
        {"z": slice(1, 4), "cy": slice(1, 3), "cx": slice(1, 3)},  # non-empty control
    ],
)
def test_empty_selection_matches_zarr_engine_3d(ragged_zarr_store: Path, sel: dict) -> None:
    """3-D exercises `slice_nd`'s per-axis stride arithmetic; reading a
    3-D array whole takes the identity fast path and never does."""
    rusty = xr.open_dataset(str(ragged_zarr_store), engine="rustytree")
    zarr_ds = xr.open_dataset(str(ragged_zarr_store), engine="zarr", consolidated=False)
    np.testing.assert_array_equal(rusty.c.isel(**sel).values, zarr_ds.c.isel(**sel).values)


def test_empty_selection_on_eager_dim_coord(ragged_zarr_store: Path) -> None:
    """The issue's own reproducer shape: an empty selection over a
    self-named dim coord, which the walk pre-fetches eagerly, alongside a
    lazily-read data var on the same dim."""
    rusty = xr.open_dataset(str(ragged_zarr_store), engine="rustytree")
    zarr_ds = xr.open_dataset(str(ragged_zarr_store), engine="zarr", consolidated=False)
    for sel in ({"n": slice(1, 1)}, {"n": slice(13, 13)}):
        np.testing.assert_array_equal(rusty.n.isel(**sel).values, zarr_ds.n.isel(**sel).values)
        np.testing.assert_array_equal(rusty.v.isel(**sel).values, zarr_ds.v.isel(**sel).values)


@pytest.mark.parametrize("dtype", ["bool", "int8", "uint16", "float32", "float64"])
def test_empty_selection_preserves_dtype(tmp_path: Path, dtype: str) -> None:
    """The short-circuit runs its own dtype dispatch, so every supported
    dtype must come back with the right one -- not just float64."""
    path = tmp_path / f"dt_{dtype}.zarr"
    root = zarr.create_group(store=str(path), zarr_format=3)
    a = root.create_array("a", shape=(13,), dtype=dtype, chunks=(4,), dimension_names=("n",))
    a[:] = np.ones(13, dtype=dtype)
    handle = {var["name"]: var["handle"] for var in open_datatree(str(path))["/"]["vars"]}["a"]
    out = handle.read_subset([(1, 1)])
    assert len(out) == 0
    assert out.dtype == np.dtype(dtype)


def test_empty_selection_does_not_fetch_the_chunk(tmp_path: Path) -> None:
    """The short-circuit's whole point is skipping the read.

    Truncate the ragged final chunk on disk: an empty selection landing
    in it must still succeed (nothing is fetched), while a one-element
    read of the same chunk must fail. Without the short-circuit the empty
    read would chunk-align, fetch the corrupt chunk, and raise -- so this
    is what distinguishes it from the `slice_nd` fix, which would return
    an empty array either way.
    """
    path = tmp_path / "corrupt.zarr"
    root = zarr.create_group(store=str(path), zarr_format=3)
    v = root.create_array(
        "v", shape=(13,), dtype="float64", chunks=(4,), dimension_names=("n",), compressors=None
    )
    v[:] = np.arange(13, dtype=np.float64)
    (path / "v" / "c" / "3").write_bytes(b"\x00\x00")  # truncate chunk 3 (elements 12..13)

    handle = {var["name"]: var["handle"] for var in open_datatree(str(path))["/"]["vars"]}["v"]
    assert len(handle.read_subset([(13, 13)])) == 0
    assert len(handle.read_subset([(12, 12)])) == 0
    with pytest.raises(ValueError, match="zarrs read failed"):
        handle.read_subset([(12, 13)])


def test_read_subset_still_validates_ranges(ragged_zarr_store: Path) -> None:
    """The short-circuit sits *after* validation; hoisting it above would
    silently turn these into empty successes."""
    handle = {
        var["name"]: var["handle"] for var in open_datatree(str(ragged_zarr_store))["/"]["vars"]
    }["v"]
    with pytest.raises(IndexError, match="exceeds dim size"):
        handle.read_subset([(14, 14)])
    with pytest.raises(IndexError, match="expected 1 ranges"):
        handle.read_subset([])
    with pytest.raises(ValueError, match="start 5 > stop 3"):
        handle.read_subset([(5, 3)])


@pytest.mark.xfail(
    reason="xarray's _decompose_outer_indexer cannot handle an empty fancy index at "
    "IndexingSupport.BASIC (indexing.py, `slice(np.min(k), np.max(k) + 1)`); "
    "engine='zarr' escapes it by declaring VECTORIZED",
    raises=ValueError,
)
def test_empty_fancy_index_matches_zarr_engine(ragged_zarr_store: Path) -> None:
    """Tracks a known divergence from `engine="zarr"` for empty *fancy*
    indexes (`isel(n=[])`, all-False masks) -- a different mechanism from
    the empty *slices* fixed here, and not fixable in this layer."""
    rusty = xr.open_dataset(str(ragged_zarr_store), engine="rustytree")
    zarr_ds = xr.open_dataset(str(ragged_zarr_store), engine="zarr", consolidated=False)
    sel = {"n": np.array([], dtype=np.intp)}
    np.testing.assert_array_equal(rusty.v.isel(**sel).values, zarr_ds.v.isel(**sel).values)
