"""Lazy chunk reads via ``ZarrsArrayHandle`` + ``RustyBackendArray``.

The walk eagerly opens every array (so shape/dtype/dims/attrs are
cached), but data does NOT travel from the store until the test asks
for it. These tests verify that contract by:

  - Asserting each var dict carries a ``handle``.
  - Asserting ``shape``/``dtype`` round-trip from the handle.
  - Reading the full array via ``RustyBackendArray`` and matching
    ``xr.open_zarr`` element-for-element.
  - Reading sliced sub-rectangles and matching the same slice on the
    reference array.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from xarray.core import indexing

from rustytree._array import RustyBackendArray
from rustytree._rustytree import ZarrsArrayHandle, open_datatree


def _vars_by_name(tree: dict, group: str) -> dict[str, dict]:
    return {var["name"]: var for var in tree[group]["vars"]}


def test_handle_present_and_typed(tiny_zarr_store: Path) -> None:
    tree = open_datatree(str(tiny_zarr_store))
    by_name = _vars_by_name(tree, "/")
    for name, var in by_name.items():
        assert "handle" in var, f"{name}: missing handle"
        assert isinstance(var["handle"], ZarrsArrayHandle), f"{name}: wrong handle type"


def test_handle_shape_and_dtype(tiny_zarr_store: Path) -> None:
    tree = open_datatree(str(tiny_zarr_store))
    by_name = _vars_by_name(tree, "/")

    temp_handle = by_name["temp"]["handle"]
    assert tuple(temp_handle.shape) == (4, 3)
    assert temp_handle.dtype == "float64"

    mask_handle = by_name["mask"]["handle"]
    assert tuple(mask_handle.shape) == (4, 3)
    assert mask_handle.dtype == "int8"


def test_full_read_matches_reference(tiny_zarr_store: Path) -> None:
    tree = open_datatree(str(tiny_zarr_store))
    by_name = _vars_by_name(tree, "/")

    # The fixture writes np.arange(12, dtype=float64).reshape(4, 3).
    temp = RustyBackendArray(by_name["temp"]["handle"])
    full_key = indexing.BasicIndexer((slice(None), slice(None)))
    actual = temp[full_key]
    expected = np.arange(12, dtype=np.float64).reshape(4, 3)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float64


def test_full_read_int8_zeros(tiny_zarr_store: Path) -> None:
    tree = open_datatree(str(tiny_zarr_store))
    by_name = _vars_by_name(tree, "/")

    mask = RustyBackendArray(by_name["mask"]["handle"])
    actual = mask[indexing.BasicIndexer((slice(None), slice(None)))]
    np.testing.assert_array_equal(actual, np.zeros((4, 3), dtype=np.int8))
    assert actual.dtype == np.int8


def test_basic_slice_indexing(tiny_zarr_store: Path) -> None:
    tree = open_datatree(str(tiny_zarr_store))
    by_name = _vars_by_name(tree, "/")
    temp = RustyBackendArray(by_name["temp"]["handle"])
    expected = np.arange(12, dtype=np.float64).reshape(4, 3)

    # Mid-range two-dim slice.
    sliced = temp[indexing.BasicIndexer((slice(1, 3), slice(0, 2)))]
    np.testing.assert_array_equal(sliced, expected[1:3, 0:2])


def test_integer_indexing_squeezes_axis(tiny_zarr_store: Path) -> None:
    tree = open_datatree(str(tiny_zarr_store))
    by_name = _vars_by_name(tree, "/")
    temp = RustyBackendArray(by_name["temp"]["handle"])
    expected = np.arange(12, dtype=np.float64).reshape(4, 3)

    # Single row.
    row = temp[indexing.BasicIndexer((2, slice(None)))]
    assert row.shape == (3,)
    np.testing.assert_array_equal(row, expected[2, :])

    # Single column.
    col = temp[indexing.BasicIndexer((slice(None), 1))]
    assert col.shape == (4,)
    np.testing.assert_array_equal(col, expected[:, 1])

    # Single element.
    elem = temp[indexing.BasicIndexer((1, 2))]
    assert elem.shape == ()
    np.testing.assert_array_equal(elem, expected[1, 2])


def test_negative_int_index_resolves(tiny_zarr_store: Path) -> None:
    tree = open_datatree(str(tiny_zarr_store))
    by_name = _vars_by_name(tree, "/")
    temp = RustyBackendArray(by_name["temp"]["handle"])
    expected = np.arange(12, dtype=np.float64).reshape(4, 3)

    last_row = temp[indexing.BasicIndexer((-1, slice(None)))]
    np.testing.assert_array_equal(last_row, expected[-1, :])


def test_handle_repr_is_informative(tiny_zarr_store: Path) -> None:
    tree = open_datatree(str(tiny_zarr_store))
    handle = _vars_by_name(tree, "/")["temp"]["handle"]
    rep = repr(handle)
    assert "shape=" in rep
    assert "float64" in rep


def test_recursive_walk_handles_present(multilevel_zarr_store: Path) -> None:
    tree = open_datatree(str(multilevel_zarr_store))
    # Every var across every node must carry a handle.
    for path, node in tree.items():
        for var in node["vars"]:
            assert isinstance(var["handle"], ZarrsArrayHandle), (
                f"{path}/{var['name']}: handle not present or wrong type"
            )


def test_data_round_trip_against_xarray(tiny_zarr_store: Path) -> None:
    """Hard parity check: rustytree's read_subset must match what
    `xr.open_zarr` produces on the same store, byte for byte."""
    tree = open_datatree(str(tiny_zarr_store))
    by_name = _vars_by_name(tree, "/")
    ref = xr.open_zarr(tiny_zarr_store, consolidated=False)

    for name, expected in (("temp", ref["temp"]), ("mask", ref["mask"])):
        rusty = RustyBackendArray(by_name[name]["handle"])
        actual = rusty[indexing.BasicIndexer((slice(None), slice(None)))]
        np.testing.assert_array_equal(actual, expected.values)


def test_read_unsupported_dtype_raises_clearly(tmp_path: Path) -> None:
    """ZarrsArrayHandle today supports the common primitives; an
    unsupported dtype must surface a NotImplementedError that names
    what's missing rather than panic across the FFI boundary."""
    import zarr

    path = tmp_path / "store.zarr"
    root = zarr.create_group(store=str(path), zarr_format=3)
    # complex64 is in the dtype-string map but not in the read_subset
    # dispatch — we deliberately leave it unimplemented for now to
    # keep the surface honest.
    arr = root.create_array(
        "z",
        shape=(2,),
        dtype="complex64",
        chunks=(2,),
        dimension_names=("n",),
    )
    arr[:] = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)

    tree = open_datatree(str(path))
    handle = _vars_by_name(tree, "/")["z"]["handle"]
    rusty = RustyBackendArray(handle)
    with pytest.raises(NotImplementedError, match="dtype"):
        rusty[indexing.BasicIndexer((slice(None),))]
