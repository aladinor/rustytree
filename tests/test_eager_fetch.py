"""Phase C eager-fetch correctness.

The Rust walk's Phase C identifies "decoder-trigger" vars (1-D
self-named dim coords + CF time-likes) and pre-fetches their full
contents in parallel during the open. The Python entrypoint uses the
resident numpy data directly so xarray's per-node CF decoders and
Index construction don't trigger lazy chunk reads through our backend.

These tests exercise the predicate boundaries:

  - 1-D self-named coords are present as `"data"` on the var dict.
  - CF time-like vars (`units` containing `" since "`) are present
    even when not 1-D self-named.
  - 2-D ordinary data variables are absent (lazy fallback).
  - Variables larger than the 1 M-element cap are absent (lazy
    fallback) regardless of coord/time shape.
  - End-to-end: opening a tree with time vars triggers zero calls
    through `RustyBackendArray._raw_indexing_method` for the time vars
    under `decode_times=True`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr

from rustytree._array import RustyBackendArray
from rustytree._rustytree import open_datatree


@pytest.fixture
def time_zarr_store(tmp_path: Path) -> Path:
    """Vanilla v3 store with a self-named 1-D dim coord, a CF time-like
    var (not self-named), and an ordinary 2-D data var.

    Layout::

        store.zarr/                                attrs: {"title": "tz"}
            x       (x=4)         float64          self-named dim coord
            time    (m=3)         int64            units="seconds since 2020-01-01"
            grid    (x=4, m=3)    float64          ordinary 2-D data var
    """
    path = tmp_path / "store.zarr"
    root = zarr.create_group(store=str(path), zarr_format=3)
    root.attrs["title"] = "tz"

    # Self-named 1-D dim coord (var name == only dim name).
    x = root.create_array("x", shape=(4,), dtype="float64", chunks=(4,), dimension_names=("x",))
    x[:] = np.array([0.0, 1.0, 2.0, 3.0])

    # CF time-like, not self-named. Triggers eager fetch via units arm.
    t = root.create_array("time", shape=(3,), dtype="int64", chunks=(3,), dimension_names=("m",))
    t[:] = np.array([100, 200, 300])
    t.attrs["units"] = "seconds since 2020-01-01"
    t.attrs["calendar"] = "proleptic_gregorian"

    # 2-D ordinary data var. Should stay lazy.
    g = root.create_array(
        "grid",
        shape=(4, 3),
        dtype="float64",
        chunks=(4, 3),
        dimension_names=("x", "m"),
    )
    g[:] = np.arange(12, dtype=np.float64).reshape(4, 3)

    return path


def _vars_by_name(tree: dict, group: str = "/") -> dict[str, dict]:
    return {var["name"]: var for var in tree[group]["vars"]}


def test_eager_predicate_self_named_dim_coord(time_zarr_store: Path) -> None:
    """Self-named 1-D dim coords must carry `data` — they're what xarray's
    `_maybe_create_default_indexes` reads in its post-pass to build
    pandas Index objects, and pre-fetching them in parallel from Rust
    avoids the N×serial RTT cold-cache cost."""
    tree = open_datatree(str(time_zarr_store))
    by_name = _vars_by_name(tree)

    assert "data" in by_name["x"], "self-named 1-D coord should be eager-fetched"
    assert by_name["x"]["data"].shape == (4,)
    np.testing.assert_array_equal(by_name["x"]["data"], np.array([0.0, 1.0, 2.0, 3.0]))


def test_eager_predicate_skips_cf_time_like(time_zarr_store: Path) -> None:
    """CF time-likes (not self-named) stay LAZY — the metadata-only
    datetime patch in `backend.py` infers their dtype without reading
    any chunks, so we don't need to materialise them up front. Pulling
    them eagerly would over-fetch on radar repos where time arrays are
    multi-MB per node."""
    tree = open_datatree(str(time_zarr_store))
    by_name = _vars_by_name(tree)
    assert "data" not in by_name["time"], (
        "CF time-likes must stay lazy; metadata-only datetime decode handles them"
    )


def test_eager_predicate_skips_ordinary_data_var(time_zarr_store: Path) -> None:
    """2-D ordinary data variable must remain lazy."""
    tree = open_datatree(str(time_zarr_store))
    by_name = _vars_by_name(tree)
    assert "data" not in by_name["grid"], "2-D data var should stay lazy (no `data` key)"


def test_eager_data_round_trip(time_zarr_store: Path) -> None:
    """Eager-fetched arrays carry the *raw* on-disk values (no CF
    decoding yet). CF decoding happens later in the Python entrypoint
    via `decode_cf_variables`."""
    tree = open_datatree(str(time_zarr_store))
    by_name = _vars_by_name(tree)
    ref_undecoded = xr.open_zarr(time_zarr_store, consolidated=False, decode_times=False)

    np.testing.assert_array_equal(by_name["x"]["data"], ref_undecoded["x"].values)


def test_oversized_coord_stays_lazy(tmp_path: Path) -> None:
    """A self-named 1-D coord larger than 1 M elements must NOT be
    eager-fetched — coverage of the size cap. Two-million-long int8
    is 2 MB, cheap to allocate but past the 1 << 20 element cap, so
    it stays lazy."""
    path = tmp_path / "big.zarr"
    root = zarr.create_group(store=str(path), zarr_format=3)
    big_n = 2_000_000
    arr = root.create_array(
        "big",
        shape=(big_n,),
        dtype="int8",
        chunks=(big_n,),
        dimension_names=("n",),
    )
    arr[:] = np.zeros((big_n,), dtype=np.int8)

    tree = open_datatree(str(path))
    by_name = _vars_by_name(tree)
    assert "data" not in by_name["big"], (
        f"vars with > 1 M elements should stay lazy; got `data` of shape "
        f"{by_name['big']['data'].shape if 'data' in by_name['big'] else None}"
    )


def test_engine_rustytree_does_not_trigger_reads_for_eager_vars(
    time_zarr_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: with eager-fetch in place, opening via
    `engine="rustytree"` must NOT route the eager-fetched vars through
    the lazy backend. We track which `RustyBackendArray` instances saw
    a `_raw_indexing_method` call and check the eager vars (`x`,
    `time`) are absent. Lazy vars (`grid`) may or may not get touched
    by xarray internals — we don't pin that.
    """
    touched: list[tuple] = []
    original = RustyBackendArray._raw_indexing_method

    def tracking(self: RustyBackendArray, key: tuple) -> np.ndarray:
        touched.append((self.shape, self.dtype.name))
        return original(self, key)

    monkeypatch.setattr(RustyBackendArray, "_raw_indexing_method", tracking)

    dt = xr.open_datatree(str(time_zarr_store), engine="rustytree")

    # Eager-fetch should have produced numpy data inline; both `x` and
    # `time` decode without any backend call.
    np.testing.assert_array_equal(dt["x"].values, np.array([0.0, 1.0, 2.0, 3.0]))
    assert "time" in dt.dataset.coords or "time" in dt.dataset.data_vars

    # No lazy read should have happened against a 1-D float64 (`x`) or
    # 1-D int64 (`time`); only the 2-D float64 (`grid`) is allowed.
    eager_shapes = {((4,), "float64"), ((3,), "int64")}
    for shape, dtype in touched:
        assert (shape, dtype) not in eager_shapes, (
            f"unexpected lazy read on eager-fetched var: shape={shape} dtype={dtype}"
        )
