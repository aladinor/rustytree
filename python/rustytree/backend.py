"""xarray backend entrypoint for rustytree.

Registered via ``[project.entry-points."xarray.backends"]`` in
pyproject.toml so that ``xr.open_datatree(url, engine="rustytree")``
resolves here. The Rust core (``rustytree._rustytree``) does the
hierarchy walk and produces a ``dict[str, NodeData]`` keyed by absolute
group path; this entrypoint marshals that into a real ``xr.DataTree``
by:

  1. Wrapping each var's `ZarrsArrayHandle` as a `RustyBackendArray`,
     then `LazilyIndexedArray` so xarray treats it like any other lazy
     backend array.
  2. Building one `xr.Variable` per array (chunk reads still deferred).
  3. Running `xr.conventions.decode_cf_variables(...)` on each group's
     variable map so `mask_and_scale` / `decode_times` / `decode_coords`
     behave the same as `engine="zarr"`.
  4. Splitting decoded variables into data_vars and coord_vars (mirroring
     ``StoreBackendEntrypoint.open_dataset``: a variable is a coord if
     CF flagged it, OR if its name matches its only dim).
  5. Assembling a ``dict[str, xr.Dataset]`` and handing it to
     ``xarray.backends.common.datatree_from_dict_with_io_cleanup``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
from xarray import conventions
from xarray.backends.common import (
    BackendEntrypoint,
    datatree_from_dict_with_io_cleanup,
)
from xarray.coding import times as _xtimes
from xarray.core import indexing
from xarray.core.coordinates import Coordinates
from xarray.core.dataset import Dataset
from xarray.core.datatree import DataTree
from xarray.core.variable import Variable

from rustytree._array import RustyBackendArray

ROOT = "/"


@contextlib.contextmanager
def _metadata_only_datetime_dtype() -> Iterator[None]:
    """Skip xarray's per-time-variable first/last chunk peek.

    `xarray.coding.times._decode_cf_datetime_dtype` peeks the first and
    last element of every CF time variable to infer the result dtype
    (xarray issue #11303 / PR #11304). On cold-cache S3 with hundreds
    of time vars across a multi-group icechunk repo, that's the
    dominant ~17 s tail in `xr.open_datatree(engine="rustytree")`.

    The peek is purely so the function can call
    `decode_cf_datetime(example, units, calendar, ...)` and return
    `result.dtype`. We can substitute synthesised values (two zeros of
    the variable's int dtype, which decode to the reference date)
    instead of peeking real data — same dtype out, no chunk read.

    Falls back to the original (peeking) implementation if the
    synthesised path raises (e.g. malformed units), so error reporting
    stays accurate.

    Defensive: if `_decode_cf_datetime_dtype` is missing (xarray
    refactored or moved it — most likely once PR #11304 lands and
    flips the function name or class structure), the context manager
    becomes a no-op rather than raising at the call site. The parity
    tests against `engine="zarr"` will surface any correctness
    regression; the only cost of a stale patch is the original cold-
    cache slowdown returning until we ship a new release.

    Removable once xarray PR #11304 lands upstream and rustytree's
    xarray floor moves past it. Tracked in issue (TBD).
    """
    original = getattr(_xtimes, "_decode_cf_datetime_dtype", None)
    if original is None:
        # xarray no longer exposes this private helper — assume
        # upstream replaced the peek path and skip patching.
        yield
        return

    def patched(
        data: Any,
        units: str,
        calendar: str | None = None,
        use_cftime: bool | None = None,
        time_unit: Any = "ns",
    ) -> np.dtype:
        try:
            data_dtype = getattr(data, "dtype", np.dtype("int64"))
            example_value = np.array([0, 0], dtype=data_dtype)
            result = _xtimes.decode_cf_datetime(
                example_value, units, calendar, use_cftime, time_unit
            )
            return getattr(result, "dtype", np.dtype("object"))
        except Exception:
            return original(data, units, calendar, use_cftime, time_unit)

    _xtimes._decode_cf_datetime_dtype = patched  # type: ignore[assignment]
    try:
        yield
    finally:
        _xtimes._decode_cf_datetime_dtype = original  # type: ignore[assignment]


def _build_rust_kwargs(
    *,
    group: str | None,
    branch: str | None,
    storage_options: dict[str, Any] | None,
    max_concurrency: int | None,
) -> dict[str, Any]:
    """Forward only the kwargs the user actually set, so the Rust side
    sees its own defaults (e.g. `max_concurrency=32`) rather than a
    sea of `None`s."""
    kwargs: dict[str, Any] = {}
    if group is not None:
        kwargs["group"] = group
    if branch is not None:
        kwargs["branch"] = branch
    if storage_options is not None:
        kwargs["storage_options"] = storage_options
    if max_concurrency is not None:
        kwargs["max_concurrency"] = max_concurrency
    return kwargs


def _to_rust_source(filename_or_obj: Any) -> Any:
    """Translate the user's `filename_or_obj` into the shape the Rust
    `open_datatree` expects.

    Accepts:
      - `icechunk.Session` -> serialise its inner `_session` (a PySession)
        to msgpack bytes via `as_bytes()`. The bytes round-trip back
        through `icechunk::session::Session::from_bytes` on the Rust
        side; both crates link the same `icechunk` version so the format
        matches. Snapshot of the session state at call time, which is
        exactly what a read-side metadata walk needs.
      - `icechunk.IcechunkStore` (what `session.store` returns) ->
        reach in through `store._store.session.as_bytes()`. This uses
        the documented-but-underscored attribute path; the parity tests
        catch any breakage when icechunk-python refactors.
      - `str` / `pathlib.Path` -> return `str(...)` unchanged. Treated
        as a URL or local path on the Rust side.

    The cross-extension `bytes` handoff exists because PyO3 type
    extraction can't reach across cdylib boundaries: rustytree's
    `_rustytree.so` and icechunk-python's `_icechunk_python.so` have
    independent `type_object` instances for `PySession` even though
    both link the same Rust crate. Bytes round-trip is the agreed
    workaround until icechunk exposes a stable C-API or PyCapsule.
    """
    # Deferred import: only pulls in icechunk at the boundary, not at
    # plugin discovery, so users who never use icechunk don't pay for
    # it. Optional dependency from rustytree's POV.
    try:
        import icechunk
    except ImportError:
        icechunk = None  # type: ignore[assignment]

    if icechunk is not None:
        # An icechunk Session has `._session` (a PySession with as_bytes).
        # `PySession.as_bytes()` already returns `bytes`, so no wrap.
        if isinstance(filename_or_obj, icechunk.Session):
            return filename_or_obj._session.as_bytes()
        # An IcechunkStore (i.e. `session.store`) wraps a PyStore
        # whose `.session` is the same PySession.
        if isinstance(filename_or_obj, icechunk.IcechunkStore):
            return filename_or_obj._store.session.as_bytes()

    # Anything else: assume str/Path-like.
    return str(filename_or_obj)


def _reroot(groups: dict[str, Dataset], root: str) -> dict[str, Dataset]:
    """Strip `root` from every absolute path, anchoring the result at "/"
    to match `xr.open_datatree(group=...)`'s subtree contract."""
    root = root.rstrip("/")
    prefix = root + "/"
    out: dict[str, Dataset] = {}
    for path, ds in groups.items():
        if path == root:
            out[ROOT] = ds
        elif path.startswith(prefix):
            out[ROOT + path[len(prefix):]] = ds
    return out


def _node_to_dataset(
    node: dict,
    *,
    mask_and_scale: bool,
    decode_times: bool,
    concat_characters: bool,
    decode_coords: bool | str,
    drop_variables: str | Iterable[str] | None,
    use_cftime: bool | None,
    decode_timedelta: bool | None,
) -> Dataset:
    """Convert a single ``NodeData`` dict into an ``xr.Dataset``.

    Mirrors `StoreBackendEntrypoint.open_dataset`'s shape: lazy
    `RustyBackendArray`s wrapped in `LazilyIndexedArray`, then CF
    decoding, then split into data_vars and coord_vars. The result has
    no `_close` callback — chunk reads flow through the Rust runtime,
    which lives for the lifetime of the process.
    """
    raw_vars: dict[str, Variable] = {}
    for var in node["vars"]:
        name = var["name"]
        dims = tuple(var["dims"])
        # Rust-side Phase C eagerly fetches "decoder-trigger" vars (1-D
        # self-named dim coords + CF time-likes) in parallel; when it
        # does, the marshaller emits a `"data"` numpy array on the var
        # dict and we use that directly. Other vars stay lazy.
        if "data" in var:
            data: Any = var["data"]
        else:
            data = indexing.LazilyIndexedArray(RustyBackendArray(var["handle"]))
        # Surface the on-disk chunk shape on `encoding` so xarray can
        # honour `chunks={}` (preferred chunks) and round-trip
        # `to_zarr`. Without these xarray's chunking pass collapses to
        # a single chunk per dim and `chunks={}` produces dask arrays
        # with the wrong shape.
        chunks_tuple = tuple(var["handle"].chunks)
        encoding = {
            "chunks": chunks_tuple,
            "preferred_chunks": dict(zip(dims, chunks_tuple)),
        }
        raw_vars[name] = Variable(
            dims=dims,
            data=data,
            attrs=dict(var["attrs"]),
            encoding=encoding,
        )

    with _metadata_only_datetime_dtype():
        decoded_vars, decoded_attrs, coord_names = conventions.decode_cf_variables(
            raw_vars,
            dict(node["attrs"]),
            mask_and_scale=mask_and_scale,
            decode_times=decode_times,
            concat_characters=concat_characters,
            decode_coords=decode_coords,
            drop_variables=drop_variables,
            use_cftime=use_cftime,
            decode_timedelta=decode_timedelta,
        )

    data_vars: dict[str, Variable] = {}
    coord_vars: dict[str, Variable] = {}
    for name, variable in decoded_vars.items():
        # CF-flagged OR self-named 1D dimension coordinate.
        if name in coord_names or variable.dims == (name,):
            coord_vars[name] = variable
        else:
            data_vars[name] = variable

    coords = Coordinates(coord_vars, indexes={})
    return Dataset(data_vars, coords=coords, attrs=decoded_attrs)


class RustytreeBackendEntrypoint(BackendEntrypoint):
    """xarray backend that opens Zarr DataTrees via the rustytree Rust core."""

    description = "Open Zarr DataTrees concurrently using the rustytree Rust backend"
    url = "https://github.com/aladinor/rustytree"

    supports_groups = True

    # No `chunks=` kwarg: the rustytree path doesn't return dask arrays
    # from `open_*`; xarray adds dask wrappers itself if asked.
    open_dataset_parameters: tuple[str, ...] = (
        "filename_or_obj",
        "drop_variables",
        "group",
        "branch",
        "storage_options",
        "max_concurrency",
        "mask_and_scale",
        "decode_times",
        "concat_characters",
        "decode_coords",
        "use_cftime",
        "decode_timedelta",
    )

    def open_datatree(
        self,
        filename_or_obj: Any,
        *,
        drop_variables: str | Iterable[str] | None = None,
        group: str | None = None,
        branch: str | None = None,
        storage_options: dict[str, Any] | None = None,
        max_concurrency: int | None = None,
        mask_and_scale: bool = True,
        decode_times: bool = True,
        concat_characters: bool = True,
        decode_coords: bool | str = True,
        use_cftime: bool | None = None,
        decode_timedelta: bool | None = None,
    ) -> DataTree:
        # Lazy-imported so plugin discovery (which only needs the entrypoint
        # class object) doesn't pay the cdylib load cost.
        from rustytree._rustytree import open_datatree as _rust_open

        # `_to_rust_source` returns `bytes` for icechunk Session/Store
        # inputs (cross-extension serialise round-trip) and `str` for
        # path/URL inputs. The Rust side dispatches on type.
        source = _to_rust_source(filename_or_obj)
        # `storage_options` is meaningful only for vanilla S3 URLs; for
        # icechunk-via-bytes the user already encoded credentials into
        # the session before serialising. Drop it on the bytes path so
        # we don't pretend to honour something we can't.
        if isinstance(source, bytes):
            storage_options_arg = None
        else:
            storage_options_arg = storage_options

        tree = _rust_open(
            source,
            **_build_rust_kwargs(
                group=group,
                branch=branch,
                storage_options=storage_options_arg,
                max_concurrency=max_concurrency,
            ),
        )

        groups: dict[str, Dataset] = {
            path: _node_to_dataset(
                node,
                mask_and_scale=mask_and_scale,
                decode_times=decode_times,
                concat_characters=concat_characters,
                decode_coords=decode_coords,
                drop_variables=drop_variables,
                use_cftime=use_cftime,
                decode_timedelta=decode_timedelta,
            )
            for path, node in tree.items()
        }
        if group and group != ROOT:
            groups = _reroot(groups, group)
        return datatree_from_dict_with_io_cleanup(groups)

    def open_dataset(
        self,
        filename_or_obj: Any,
        *,
        drop_variables: str | Iterable[str] | None = None,
        group: str | None = None,
        branch: str | None = None,
        storage_options: dict[str, Any] | None = None,
        max_concurrency: int | None = None,
        mask_and_scale: bool = True,
        decode_times: bool = True,
        concat_characters: bool = True,
        decode_coords: bool | str = True,
        use_cftime: bool | None = None,
        decode_timedelta: bool | None = None,
    ) -> Dataset:
        # Delegates to `open_datatree` and pulls out the requested
        # node's Dataset. Today this means the rust walk recurses below
        # `group` and we drop the descendants — wasteful for deep trees.
        # A `recursive=False` knob on the Rust open is queued as a
        # follow-up (see plan): worth it once anyone does
        # `xr.open_dataset(s3_url, group="/deep/leaf")` in anger.
        tree = self.open_datatree(
            filename_or_obj,
            drop_variables=drop_variables,
            group=group,
            branch=branch,
            storage_options=storage_options,
            max_concurrency=max_concurrency,
            mask_and_scale=mask_and_scale,
            decode_times=decode_times,
            concat_characters=concat_characters,
            decode_coords=decode_coords,
            use_cftime=use_cftime,
            decode_timedelta=decode_timedelta,
        )
        # `open_datatree` re-roots when `group` is non-trivial, so the
        # requested group is always at "/" of the returned DataTree.
        return tree.dataset

    @classmethod
    def guess_can_open(cls, filename_or_obj: Any) -> bool:
        # rustytree only kicks in when the user explicitly passes
        # `engine="rustytree"` — auto-detection would steal paths the
        # user expected to land on `engine="zarr"`.
        return False
