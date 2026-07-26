"""Zarr v3 string arrays (issue #70).

rustytree opens both zarr v3 string flavours and mirrors ``engine="zarr"``:

  - **variable-length ``string``** (the production flavour — raw2zarr's FM301
    retrofit writes ``sweep_mode``/``prt_mode``/``follow_mode`` this way, as the
    registered, warning-free zarr v3 string type): the *variable* is declared
    ``object`` while its *values* materialise as numpy-2 ``StringDType``. That
    split is exactly what ``engine="zarr"`` produces — xarray's CF-decode
    flattens a ``StringDType``-declared variable's values to ``object``, so
    declaring ``object`` up-front is the only way to keep ``StringDType`` values.
  - **``fixed_length_utf32``** (numpy ``<U…``, xarray's default ``str`` encoding):
    declared and read as ``<U{n}``.

The strongest assertion here is ``xr.testing.assert_identical`` against
``engine="zarr"``: the two backends are identical for all valid-Unicode content
(one documented exception — lone surrogate code points in ``fixed_length_utf32``,
see ``test_utf32_lone_surrogate_dropped``). Both the vanilla-Zarr and icechunk
walkers are exercised.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr
import zarr
from conftest import vars_by_name
from xarray.core import indexing

from rustytree._array import RustyBackendArray
from rustytree._rustytree import open_datatree

SCALAR_VALUE = "azimuth_surveillance"
LABELS = ["azimuth_surveillance", "rhi", "vp"]
STRING_DTYPE = np.dtypes.StringDType()

# `to_zarr` with a numpy `<U` scalar writes zarr v3 `fixed_length_utf32`, which
# zarr-python flags as an unstable spec. That's expected here — silence it.
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _write_vlen_layout(root: zarr.Group) -> None:
    """A vlen-``string`` scalar + 1-D array (the registered, warning-free type)."""
    root.create_array("sweep_mode", shape=(), dtype="string")[()] = SCALAR_VALUE
    labels = root.create_array(
        "labels", shape=(len(LABELS),), dtype="string", dimension_names=("n",)
    )
    labels[:] = LABELS


@pytest.fixture
def vlen_zarr_store(tmp_path: Path) -> Path:
    path = tmp_path / "vlen.zarr"
    _write_vlen_layout(zarr.create_group(store=str(path), zarr_format=3))
    return path


@pytest.fixture
def vlen_icechunk_repo(tmp_path: Path) -> Path:
    """icechunk repo with the vlen layout (exercises the snapshot walker)."""
    icechunk = pytest.importorskip("icechunk")
    path = tmp_path / "repo"
    repo = icechunk.Repository.create(icechunk.local_filesystem_storage(str(path)))
    session = repo.writable_session("main")
    _write_vlen_layout(zarr.create_group(store=session.store, zarr_format=3))
    session.commit("init")
    return path


@pytest.fixture
def utf32_zarr_store(tmp_path: Path) -> Path:
    """``fixed_length_utf32`` scalar + 1-D (xarray's default ``str`` encoding)."""
    path = tmp_path / "utf32.zarr"
    ds = xr.Dataset(
        {
            "sweep_mode": ((), SCALAR_VALUE),
            "modes": ("n", np.array(LABELS)),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds.to_zarr(path, zarr_format=3, consolidated=False, mode="w")
    return path


@pytest.fixture
def utf32_icechunk_repo(tmp_path: Path) -> Path:
    """icechunk repo with a `fixed_length_utf32` layout (its zarr.json carries a
    capacity config field the vlen `string` one lacks)."""
    icechunk = pytest.importorskip("icechunk")
    path = tmp_path / "repo"
    repo = icechunk.Repository.create(icechunk.local_filesystem_storage(str(path)))
    session = repo.writable_session("main")
    ds = xr.Dataset({"sweep_mode": ((), SCALAR_VALUE), "modes": ("n", np.array(LABELS))})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds.to_zarr(session.store, zarr_format=3, consolidated=False, mode="w")
    session.commit("init")
    return path


@pytest.fixture
def string_2d_zarr_store(tmp_path: Path) -> Path:
    """2-D vlen `string` with real chunking — exercises the multi-axis
    `slice_nd` stride path + object-array reshape/squeeze (1-D can't)."""
    path = tmp_path / "s2d.zarr"
    root = zarr.create_group(store=str(path), zarr_format=3)
    arr = root.create_array(
        "grid", shape=(4, 3), chunks=(2, 2), dtype="string", dimension_names=("y", "x")
    )
    arr[:] = np.array([[f"r{i}c{j}" for j in range(3)] for i in range(4)], dtype=object)
    return path


@pytest.fixture
def string_edgecases_zarr_store(tmp_path: Path) -> Path:
    """vlen `string` with tricky-but-valid content: interior NUL (must survive
    the trailing-only trim), a non-BMP emoji, and an empty string."""
    path = tmp_path / "edge.zarr"
    root = zarr.create_group(store=str(path), zarr_format=3)
    arr = root.create_array("odd", shape=(4,), dtype="string", dimension_names=("n",))
    arr[:] = ["a\x00b", "🦀 rust", "", "plain"]
    return path


@pytest.fixture
def string_partial_zarr_store(tmp_path: Path) -> Path:
    """vlen `string` with an unwritten trailing chunk — reading the whole array
    must supply the fill value and span the chunk boundary."""
    path = tmp_path / "partial.zarr"
    root = zarr.create_group(store=str(path), zarr_format=3)
    arr = root.create_array("half", shape=(4,), chunks=(2,), dtype="string", dimension_names=("n",))
    arr[0:2] = ["a", "bb"]  # elements [2:4] left unwritten -> fill value
    return path


def _handle(store: Path, name: str) -> Any:
    """The `ZarrsArrayHandle` for var ``name`` in a freshly-walked ``store``."""
    return vars_by_name(open_datatree(str(store)))[name]["handle"]


# --------------------------------------------------------------------------
# variable-length `string`
# --------------------------------------------------------------------------


def test_vlen_handle_dtypes(vlen_zarr_store: Path) -> None:
    """Declared dtype is ``object``; the read materialises ``StringDType``."""
    handles = {n: v["handle"] for n, v in vars_by_name(open_datatree(str(vlen_zarr_store))).items()}
    for name in ("sweep_mode", "labels"):
        assert handles[name].dtype == "object", name
        assert handles[name].read_dtype == "T", name


def test_vlen_scalar_read(vlen_zarr_store: Path) -> None:
    """0-D read (the production flavour): the empty-``ranges`` path returns the
    single string, materialised as ``StringDType`` before numpy collapses it."""
    handle = _handle(vlen_zarr_store, "sweep_mode")
    out = RustyBackendArray(handle)[indexing.BasicIndexer(())]
    assert out.shape == ()
    assert out.dtype == STRING_DTYPE
    assert str(out.item()) == SCALAR_VALUE


def test_vlen_1d_read_full_and_slice(vlen_zarr_store: Path) -> None:
    handle = _handle(vlen_zarr_store, "labels")
    rusty = RustyBackendArray(handle)

    full = rusty[indexing.BasicIndexer((slice(None),))]
    assert full.dtype == STRING_DTYPE
    assert [str(x) for x in full] == LABELS

    # A non-chunk-aligned sub-rectangle exercises slice_nd on a non-Copy type.
    sliced = rusty[indexing.BasicIndexer((slice(1, 3),))]
    assert [str(x) for x in sliced] == LABELS[1:3]

    single = rusty[indexing.BasicIndexer((2,))]  # integer index collapses the axis
    assert single.shape == ()
    assert str(single.item()) == LABELS[2]


def test_vlen_open_datatree(vlen_zarr_store: Path) -> None:
    """Issue #70 MRE: the full ``engine="rustytree"`` open no longer raises,
    and mirrors ``engine="zarr"`` — ``object`` variable, ``StringDType`` values."""
    dt = xr.open_datatree(vlen_zarr_store, engine="rustytree")

    sweep = dt["sweep_mode"]
    assert sweep.variable.dtype == np.dtype("object")
    assert str(sweep.item()) == SCALAR_VALUE

    labels = dt["labels"]
    assert labels.variable.dtype == np.dtype("object")
    assert np.asarray(labels.values).dtype == STRING_DTYPE
    assert [str(x) for x in labels.values] == LABELS


def test_vlen_icechunk(vlen_icechunk_repo: Path) -> None:
    """The icechunk snapshot walker parses a ``string`` array's zarr.json."""
    dt = xr.open_datatree(vlen_icechunk_repo, engine="rustytree")
    assert dt["sweep_mode"].variable.dtype == np.dtype("object")
    assert str(dt["sweep_mode"].item()) == SCALAR_VALUE
    assert [str(x) for x in dt["labels"].values] == LABELS


# --------------------------------------------------------------------------
# fixed_length_utf32
# --------------------------------------------------------------------------


def test_utf32_open_datatree(utf32_zarr_store: Path) -> None:
    """``fixed_length_utf32`` → fixed-width ``<U`` (zarrs 0.23 opens it; the
    UTF-32 → ``str`` decode + ``<U`` construction is rustytree's)."""
    dt = xr.open_datatree(utf32_zarr_store, engine="rustytree")

    sweep = dt["sweep_mode"]
    assert sweep.variable.dtype.kind == "U"
    assert str(sweep.item()) == SCALAR_VALUE

    modes = dt["modes"]
    assert modes.variable.dtype.kind == "U"
    assert [str(x) for x in modes.values] == LABELS


# --------------------------------------------------------------------------
# exact parity with the stock zarr backend
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fixture",
    [
        "vlen_zarr_store",
        "utf32_zarr_store",
        "string_2d_zarr_store",
        "string_edgecases_zarr_store",
        "string_partial_zarr_store",
    ],
)
def test_identical_to_zarr_engine(fixture: str, request: pytest.FixtureRequest) -> None:
    """The strongest guarantee: identical to ``engine="zarr"`` — variables and
    values alike — across both flavours, 2-D chunking, tricky-but-valid content
    (interior NUL / emoji / empty), and an unwritten-chunk fill value.

    (One documented exception, not covered here: lone surrogate code points in
    ``fixed_length_utf32`` — see ``test_utf32_lone_surrogate_dropped``.)"""
    store = request.getfixturevalue(fixture)
    ref = xr.open_datatree(store, engine="zarr", consolidated=False)
    rst = xr.open_datatree(store, engine="rustytree")
    xr.testing.assert_identical(ref, rst)


def test_2d_chunked_string_slicing(string_2d_zarr_store: Path) -> None:
    """Multi-axis `slice_nd` + object-array reshape/squeeze on a 2-D string
    array: a non-chunk-aligned sub-rectangle and an integer-index squeeze."""
    rusty = RustyBackendArray(_handle(string_2d_zarr_store, "grid"))

    sub = rusty[indexing.BasicIndexer((slice(1, 4), slice(1, 3)))]
    assert sub.shape == (3, 2)
    assert [[str(x) for x in row] for row in sub] == [
        [f"r{i}c{j}" for j in (1, 2)] for i in (1, 2, 3)
    ]

    row = rusty[indexing.BasicIndexer((2, slice(None)))]  # int index collapses axis 0
    assert row.shape == (3,)
    assert [str(x) for x in row] == [f"r2c{j}" for j in range(3)]


def test_selfnamed_string_dim_coord(tmp_path: Path) -> None:
    """A self-named 1-D `string` dim coord now triggers eager-fetch (Phase C
    extended to `object`-dtype scalars/self-named coords) — it must be
    eager, read correctly, and still work as a pandas Index."""
    path = tmp_path / "coord.zarr"
    root = zarr.create_group(store=str(path), zarr_format=3)
    cat = root.create_array("category", shape=(3,), dtype="string", dimension_names=("category",))
    cat[:] = ["low", "medium", "high"]

    # The walk now eager-fetches it: self-named 1-D coord, `object` dtype,
    # under both the element-count and byte-size caps.
    assert "data" in vars_by_name(open_datatree(str(path)))["category"]

    ref = xr.open_datatree(path, engine="zarr", consolidated=False)
    rst = xr.open_datatree(path, engine="rustytree")
    xr.testing.assert_identical(ref, rst)
    assert str(rst["category"].sel(category="medium").item()) == "medium"


def test_eager_string_scalar_no_lazy_read(
    tmp_path: Path, track_raw_indexing_calls: list[tuple]
) -> None:
    """A 0-D `object`-dtype scalar (mirrors `sweep_mode`) is eager after the
    walk and never round-trips through the lazy backend. Its decoded value
    matches ``engine="zarr"`` exactly via `assert_identical` — the actual
    materialised dtype for a 0-D vlen-string scalar turns out to be `<U…`
    for *both* backends (xarray's CF-decode resolves 0-D string scalars
    differently than 1-D arrays, which do keep `StringDType` — see
    `test_vlen_open_datatree`'s `labels` check), so parity with
    `engine="zarr"`, not a literal dtype claim, is the correctness bar."""
    path = tmp_path / "scalar.zarr"
    root = zarr.create_group(store=str(path), zarr_format=3)
    root.create_array("mode", shape=(), dtype="string")[()] = SCALAR_VALUE

    by_name = vars_by_name(open_datatree(str(path)))
    assert "data" in by_name["mode"]

    dt = xr.open_datatree(str(path), engine="rustytree")
    assert str(dt["mode"].item()) == SCALAR_VALUE
    assert track_raw_indexing_calls == [], (
        f"eager-fetched scalar triggered a lazy read: {track_raw_indexing_calls}"
    )

    ref = xr.open_datatree(path, engine="zarr", consolidated=False)
    xr.testing.assert_identical(ref, dt)


def test_object_dtype_over_count_cap_stays_lazy(tmp_path: Path) -> None:
    """An `object`-dtype self-named coord above `EAGER_FETCH_MAX_OBJECT_ELEMENTS`
    (4096) must stay lazy — coverage of the count cap, mirroring
    `test_eager_fetch.py::test_oversized_coord_stays_lazy` for the numeric cap."""
    path = tmp_path / "big_object.zarr"
    root = zarr.create_group(store=str(path), zarr_format=3)
    n = 5000
    arr = root.create_array("big", shape=(n,), dtype="string", dimension_names=("big",))
    arr[:] = [str(i) for i in range(n)]

    by_name = vars_by_name(open_datatree(str(path)))
    assert "data" not in by_name["big"], "object var above the count cap should stay lazy"

    ref = xr.open_datatree(path, engine="zarr", consolidated=False)
    rst = xr.open_datatree(path, engine="rustytree")
    xr.testing.assert_identical(ref, rst)


def test_object_dtype_over_byte_cap_stays_lazy(tmp_path: Path) -> None:
    """An `object`-dtype self-named coord under the *count* cap but whose
    total payload exceeds `EAGER_FETCH_MAX_OBJECT_BYTES` (8 MiB) must also
    stay lazy — proves the byte cap gates independently of the count cap."""
    path = tmp_path / "big_bytes.zarr"
    root = zarr.create_group(store=str(path), zarr_format=3)
    # 4 elements, well under the 4096-element cap, but ~9 MiB combined.
    big_strings = ["x" * (3 * 1024 * 1024) for _ in range(3)]
    arr = root.create_array("blobs", shape=(3,), dtype="string", dimension_names=("blobs",))
    arr[:] = big_strings

    by_name = vars_by_name(open_datatree(str(path)))
    assert "data" not in by_name["blobs"], "object var above the byte cap should stay lazy"

    ref = xr.open_datatree(path, engine="zarr", consolidated=False)
    rst = xr.open_datatree(path, engine="rustytree")
    xr.testing.assert_identical(ref, rst)


def test_eager_predicate_skips_nd_object_dtype_var(string_2d_zarr_store: Path) -> None:
    """A small (12-element) N-D, non-self-named `object`-dtype data variable
    must stay lazy — the object-dtype eager branch is deliberately narrower
    than the numeric one (scalars + self-named 1-D coords only) so an
    ordinary small string data variable doesn't lose its dask laziness just
    because it fits under the element-count cap. Mirrors
    `test_eager_fetch.py::test_eager_predicate_skips_ordinary_data_var` for
    the numeric case; without this, a future loosening of the predicate
    (e.g. dropping the self-named/1-D gate) would regress silently, since
    `test_identical_to_zarr_engine`'s `assert_identical` would still pass
    either way."""
    by_name = vars_by_name(open_datatree(str(string_2d_zarr_store)))
    assert "data" not in by_name["grid"], "N-D object-dtype data var should stay lazy"


def test_multiple_eager_vars_one_group(tmp_path: Path) -> None:
    """Mixed numeric + string eager vars in one group — the real raw2zarr
    FM301 shape (self-named numeric coords alongside string scalars like
    `sweep_mode`). Catches an index mixup in `eager_phase`'s `(ni, vi)`
    assign-back-into-`nodes` loop that a single-eager-var-per-group test
    can't."""
    path = tmp_path / "mixed.zarr"
    root = zarr.create_group(store=str(path), zarr_format=3)
    root.create_array("x", shape=(4,), dtype="float64", dimension_names=("x",))[:] = np.arange(4.0)
    cat = root.create_array("category", shape=(3,), dtype="string", dimension_names=("category",))
    cat[:] = ["low", "medium", "high"]
    root.create_array("sweep_mode", shape=(), dtype="string")[()] = SCALAR_VALUE

    by_name = vars_by_name(open_datatree(str(path)))
    assert "data" in by_name["x"]
    assert "data" in by_name["category"]
    assert "data" in by_name["sweep_mode"]

    ref = xr.open_datatree(path, engine="zarr", consolidated=False)
    rst = xr.open_datatree(path, engine="rustytree")
    xr.testing.assert_identical(ref, rst)


@pytest.mark.parametrize("n, expect_eager", [(4096, True), (4097, False)])
def test_object_dtype_count_cap_boundary(tmp_path: Path, n: int, expect_eager: bool) -> None:
    """Exact boundary of `EAGER_FETCH_MAX_OBJECT_ELEMENTS` (4096): at the cap
    stays eager, one past it degrades to lazy."""
    path = tmp_path / f"boundary_{n}.zarr"
    root = zarr.create_group(store=str(path), zarr_format=3)
    arr = root.create_array("v", shape=(n,), dtype="string", dimension_names=("v",))
    arr[:] = [str(i) for i in range(n)]

    by_name = vars_by_name(open_datatree(str(path)))
    assert ("data" in by_name["v"]) is expect_eager


@pytest.mark.parametrize(
    "n_bytes, expect_eager",
    [(8 * 1024 * 1024, True), (8 * 1024 * 1024 + 1, False)],
)
def test_object_dtype_byte_cap_boundary(tmp_path: Path, n_bytes: int, expect_eager: bool) -> None:
    """Exact boundary of `EAGER_FETCH_MAX_OBJECT_BYTES` (8 MiB), well under
    the element-count cap (a single element), isolating the byte check."""
    path = tmp_path / f"byte_boundary_{n_bytes}.zarr"
    root = zarr.create_group(store=str(path), zarr_format=3)
    arr = root.create_array("v", shape=(1,), dtype="string", dimension_names=("v",))
    arr[:] = ["x" * n_bytes]

    by_name = vars_by_name(open_datatree(str(path)))
    assert ("data" in by_name["v"]) is expect_eager


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_bytes_dtype_stays_lazy(tmp_path: Path) -> None:
    """`variable_length_bytes` (also declared `object`, same as `string`) must
    not be eagerly fetched: `should_eager_fetch`'s object branch doesn't
    distinguish `String`/`Bytes` by dtype name alone, but `fetch_all_elements`
    only implements the `String` retrieval — `Bytes` falls through to the
    unsupported-dtype arm and degrades to lazy, exercised here through the
    new predicate rather than the old numeric-only one."""
    path = tmp_path / "bytes.zarr"
    root = zarr.create_group(store=str(path), zarr_format=3)
    arr = root.create_array("b", shape=(3,), dtype="variable_length_bytes", dimension_names=("b",))
    arr[:] = [b"aa", b"bb", b"cc"]

    by_name = vars_by_name(open_datatree(str(path)))
    assert "data" not in by_name["b"], "Bytes-dtype var should stay lazy"


def test_vlen_icechunk_eager_predicate(vlen_icechunk_repo: Path) -> None:
    """The icechunk snapshot walker runs the same `eager_phase` as the
    vanilla walker — confirm the object-dtype branch actually fires through
    it, not just through the vanilla-store code path every other test in
    this module exercises."""
    by_name = vars_by_name(open_datatree(str(vlen_icechunk_repo)))
    assert "data" in by_name["sweep_mode"], "0-D scalar should be eager via the icechunk walker"
    assert "data" not in by_name["labels"], "non-self-named 1-D var should stay lazy"


def test_string_empty_selection(vlen_zarr_store: Path) -> None:
    """An empty selection returns a 0-length array (the `is_empty` path builds an
    empty Vec) rather than reading or mis-slicing."""
    flat = _handle(vlen_zarr_store, "labels").read_subset([(1, 1)])
    assert len(flat) == 0

    ref = xr.open_datatree(vlen_zarr_store, engine="zarr", consolidated=False)
    rst = xr.open_datatree(vlen_zarr_store, engine="rustytree")
    xr.testing.assert_identical(
        ref["labels"].isel(n=slice(1, 1)), rst["labels"].isel(n=slice(1, 1))
    )


def test_utf32_handle_dtypes(utf32_zarr_store: Path) -> None:
    """utf32: declared == read == `<U{n}` (no divergence, unlike vlen `string`)."""
    handle = _handle(utf32_zarr_store, "sweep_mode")
    assert handle.dtype == "<U20"  # "azimuth_surveillance" is 20 code points
    assert handle.read_dtype == "<U20"


def test_utf32_icechunk(utf32_icechunk_repo: Path) -> None:
    """The icechunk snapshot walker parses a `fixed_length_utf32` zarr.json."""
    dt = xr.open_datatree(utf32_icechunk_repo, engine="rustytree")
    assert dt["sweep_mode"].variable.dtype.kind == "U"
    assert str(dt["sweep_mode"].item()) == SCALAR_VALUE
    assert [str(x) for x in dt["modes"].values] == LABELS


def test_utf32_lone_surrogate_dropped(tmp_path: Path) -> None:
    """KNOWN LIMITATION (documented divergence, not parity): `fixed_length_utf32`
    decodes through Rust `char`, which cannot represent lone surrogate code
    points (U+D800–U+DFFF). zarr preserves the raw bytes; rustytree drops the
    surrogate. This is malformed Unicode and vanishingly rare (valid content
    round-trips exactly — see `test_identical_to_zarr_engine`); pinned so the
    divergence stays visible rather than silent."""
    path = tmp_path / "surr.zarr"
    ds = xr.Dataset({"s": ("n", np.array(["\ud800bad"], dtype="<U5"))})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds.to_zarr(path, zarr_format=3, consolidated=False, mode="w")

    rst = xr.open_datatree(path, engine="rustytree")
    # The lone surrogate is dropped; the rest of the string survives.
    assert str(rst["s"].values[0]) == "bad"
