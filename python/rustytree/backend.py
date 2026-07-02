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
import re
from collections.abc import Iterable, Iterator
from pathlib import PurePosixPath
from typing import Any

import numpy as np
from xarray.backends.common import (
    AbstractDataStore,
    BackendEntrypoint,
    datatree_from_dict_with_io_cleanup,
)
from xarray.backends.store import StoreBackendEntrypoint
from xarray.backends.zarr import FillValueCoder
from xarray.coding import times as _xtimes
from xarray.core import indexing
from xarray import merge
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
    recursive: bool | None = None,
    glob: str | None = None,
) -> dict[str, Any]:
    """Forward only the kwargs the user actually set, so the Rust side
    sees its own defaults (e.g. `max_concurrency=32`, `recursive=true`)
    rather than a sea of `None`s."""
    kwargs: dict[str, Any] = {}
    if group is not None:
        kwargs["group"] = group
    if branch is not None:
        kwargs["branch"] = branch
    if storage_options is not None:
        kwargs["storage_options"] = storage_options
    if max_concurrency is not None:
        kwargs["max_concurrency"] = max_concurrency
    if recursive is not None:
        kwargs["recursive"] = recursive
    if glob is not None:
        kwargs["glob"] = glob
    return kwargs


# Mirror xarray PR #11302's `_is_glob_pattern`: any of `*`, `?`, `[`
# triggers glob handling. Plain literal paths take the non-recursive
# fast-path in `open_dataset`. Glob support in `open_dataset` raises
# until Phase 8 — `open_dataset` returns a single Dataset, so a
# multi-match glob has no defined target.
_GLOB_CHARS = re.compile(r"[*?\[]")


def _check_zarr_v3_only(zarr_format: int | None, consolidated: bool | None) -> None:
    """Reject xarray kwargs that would imply Zarr v2 mode.

    rustytree currently only supports Zarr v3 (which icechunk uses
    natively). The two kwargs xarray's stock ``engine="zarr"`` accepts
    that imply v2 mode are:

      - ``zarr_format=2``: explicit Zarr v2 store. Reject.
      - ``consolidated=True``: read a ``.zmetadata`` consolidated index.
        That index format is the v2 consolidated-metadata convention
        (and the way it's commonly used today). icechunk's snapshot
        plays the same role for icechunk repos, and v3 stores walk the
        hierarchy directly via ``zarr.json``, so we reject this kwarg
        and direct the user at ``engine="zarr"`` for v2 stores.

    ``zarr_format=3`` / ``None`` and ``consolidated=False`` / ``None``
    pass through silently — those are the v3 path.
    """
    # `zarr_format` is checked before `consolidated`: when both v2-implying
    # kwargs are passed together, the user sees the more specific
    # "Zarr v3 only" message first.
    if zarr_format is not None and zarr_format != 3:
        raise NotImplementedError(
            f"rustytree currently supports Zarr v3 only; got "
            f"zarr_format={zarr_format!r}. Use `engine='zarr'` for "
            f"Zarr v{zarr_format} stores."
        )
    if consolidated is True:
        raise NotImplementedError(
            "rustytree does not support consolidated metadata "
            "(`consolidated=True`); that's the Zarr v2 convention and "
            "rustytree currently supports Zarr v3 only. icechunk's "
            "snapshot plays the same role for icechunk repos. Use "
            "`engine='zarr'` if you need consolidated v2 metadata, or "
            "pass `consolidated=False` (or omit the kwarg) for Zarr v3."
        )


def _normalize_literal_group(group: str | None, is_glob: bool) -> str | None:
    """Normalise a literal group path to absolute, canonical form.
    Globs are left alone: ``PurePosixPath.match`` treats relative
    patterns as suffix matches, absolute ones as full-path matches —
    meaningfully different.

    Rules:
      - ``None`` → ``None`` (no group selection).
      - Empty string → ``"/"`` (treat as root, matching xarray's
        ``open_*`` convention).
      - Otherwise: prepend ``/`` if absent, then canonicalise via
        ``PurePosixPath`` to collapse ``//`` runs and strip trailing
        slashes. icechunk's path validator and our Rust walk both
        emit canonical paths; without normalisation, a user-supplied
        ``/foo//bar`` or ``/foo/`` would miss the lookup.
    """
    if is_glob or group is None:
        return group
    if group == "":
        return ROOT
    if not group.startswith("/"):
        group = "/" + group
    return str(PurePosixPath(group))


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

    `bytes` input is passed through unchanged so callers that already
    serialised once (e.g. the ancestor-merge loop in `open_datatree`)
    can re-enter `open_dataset` without re-encoding the same session
    snapshot per ancestor.
    """
    if isinstance(filename_or_obj, bytes):
        return filename_or_obj

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


def _is_python_credentials_fetcher_error(exc: ValueError) -> bool:
    """True if `exc` is icechunk failing to deserialize a Python credentials
    fetcher whose typetag the Rust core doesn't register.

    The Rust core ships a shim (`src/py_credentials.rs`) that re-registers
    icechunk-python's `PythonCredentialsFetcher`, so this normally doesn't fire.
    It's a safety net for *version drift*: if a future icechunk-python renames or
    reshapes that fetcher, `Session::from_bytes` again raises
    ``unknown variant `...`, there are no variants`` and we want to explain the
    mismatch instead of leaking the raw msgpack error. Matched two ways so a
    rename still trips it.
    """
    msg = str(exc)
    return "PythonCredentialsFetcher" in msg or (
        "unknown variant" in msg and "there are no variants" in msg
    )


def _rust_open_or_explain(rust_open: Any, source: Any, **kwargs: Any) -> Any:
    """Call the Rust `open_datatree`, translating the credentials-fetcher
    deserialize failure into an actionable error.

    Only icechunk-via-bytes inputs can hit this; path/URL sources can't carry a
    Python credentials fetcher, so they re-raise unchanged.
    """
    try:
        return rust_open(source, **kwargs)
    except ValueError as exc:
        if isinstance(source, bytes) and _is_python_credentials_fetcher_error(exc):
            raise ValueError(
                "rustytree could not deserialize this icechunk session: it carries "
                "a credentials fetcher the Rust core does not recognize. This usually "
                "means the installed icechunk-python is newer than the icechunk "
                "version rustytree was built against and the credentials format "
                "drifted. Align icechunk / icechunk-python versions, or rebuild the "
                "session with anonymous or static credentials."
            ) from exc
        raise


def _filter_by_glob(tree: dict[str, Any], pattern: str) -> dict[str, Any]:
    """Filter ``{path: NodeData}`` by glob, mirroring xarray PR #11302.

    Uses ``PurePosixPath.match`` and auto-includes every ancestor of a
    matched path so ``DataTree.from_dict`` sees a well-formed hierarchy.
    """
    matched = {p for p in tree if PurePosixPath(p).match(pattern)}
    keep: set[str] = set(matched)
    for p in matched:
        for ancestor in PurePosixPath(p).parents:
            keep.add(str(ancestor))
    return {path: node for path, node in tree.items() if path in keep}


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


class _RustyDataStore(AbstractDataStore):
    """Adapt a Rust ``NodeData`` dict to xarray's ``AbstractDataStore`` so
    ``StoreBackendEntrypoint.open_dataset`` can build the per-group Dataset."""

    __slots__ = ("_node",)

    def __init__(self, node: dict) -> None:
        self._node = node

    def get_variables(self) -> dict[str, Variable]:
        out: dict[str, Variable] = {}
        for var in self._node["vars"]:
            dims = tuple(var["dims"])
            # Phase C may have pre-fetched the variable's full contents
            # (1-D self-named dim coords + CF time-likes); other vars
            # stay lazy through `RustyBackendArray`.
            data: Any = (
                var["data"]
                if "data" in var
                else indexing.LazilyIndexedArray(RustyBackendArray(var["handle"]))
            )
            handle = var["handle"]
            chunks = tuple(handle.chunks)
            attrs = dict(var["attrs"])
            # Mirror xarray's zarr backend: decode a base64 str/bytes `_FillValue`
            # (raw zarr wire form, as icechunk/virtual stores emit) to a numeric
            # sentinel so CF masking sees one fill value, not a str `_FillValue`
            # beside a numeric `missing_value`. Numeric fills, and list/tuple wire
            # forms `FillValueCoder` can't decode (e.g. complex), pass through.
            fv = attrs.get("_FillValue")
            if isinstance(fv, (str, bytes)):
                attrs["_FillValue"] = FillValueCoder.decode(fv, handle.dtype)
            out[var["name"]] = Variable(
                dims=dims,
                data=data,
                attrs=attrs,
                encoding={"chunks": chunks, "preferred_chunks": dict(zip(dims, chunks))},
            )
        return out

    def get_attrs(self) -> dict[str, Any]:
        return dict(self._node["attrs"])


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
    """Convert a single ``NodeData`` dict into an ``xr.Dataset`` by
    delegating to xarray's ``StoreBackendEntrypoint`` over a
    ``_RustyDataStore`` shim — inherits CF decode, data/coord promotion,
    and ``encoding``/``set_close`` wiring instead of reimplementing them."""
    with _metadata_only_datetime_dtype():
        return StoreBackendEntrypoint().open_dataset(
            _RustyDataStore(node),
            mask_and_scale=mask_and_scale,
            decode_times=decode_times,
            concat_characters=concat_characters,
            decode_coords=decode_coords,
            drop_variables=drop_variables,
            use_cftime=use_cftime,
            decode_timedelta=decode_timedelta,
        )


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
        # xarray's `engine="zarr"` accepts these; we accept them for
        # call-site compatibility but reject the v2-implying values
        # (see `_check_zarr_v3_only`).
        "zarr_format",
        "consolidated",
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
        zarr_format: int | None = None,
        consolidated: bool | None = None,
        include_ancestor_coords: bool = True,
    ) -> DataTree:
        _check_zarr_v3_only(zarr_format, consolidated)
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

        # Glob `group=` (xarray PR #11302 semantics): the Rust walk
        # accepts the pattern as a conservative prefix predicate to
        # prune subtrees that can't match (Phase 8 / part 2). Python's
        # `_filter_by_glob` is the source of truth — it runs after
        # the walk and drops any over-walked nodes the Rust prune
        # was conservative about.
        is_glob = group is not None and bool(_GLOB_CHARS.search(group))
        group = _normalize_literal_group(group, is_glob)

        tree = _rust_open_or_explain(
            _rust_open,
            source,
            **_build_rust_kwargs(
                group=None if is_glob else group,
                branch=branch,
                storage_options=storage_options_arg,
                max_concurrency=max_concurrency,
                glob=group if is_glob else None,
            ),
        )

        if is_glob:
            tree = _filter_by_glob(tree, group)
        elif group and group != ROOT and group not in tree:
            # Missing-literal-path: silent-empty is misleading (matches
            # the user-visible behaviour of `open_dataset`, which already
            # raises). Globs are exempt — empty match is valid for them
            # per xarray PR #11302's semantics.
            raise KeyError(
                f"rustytree.open_datatree: group {group!r} not found in store"
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
        # Glob results stay rooted at "/" — matched paths may span
        # ancestors with no common non-root prefix; only literal
        # subtree paths get rerooted and the ancestor merge.
        is_literal_subtree = bool(group) and group != ROOT and not is_glob
        if is_literal_subtree:
            groups = _reroot(groups, group)
        # Promote ancestor groups onto the new root so a literal subtree
        # open matches `dt_full[group].to_dataset(inherit="all_coords")`.
        # `compat="override"`: new root wins over ancestors; closer
        # ancestors win over farther (`PurePosixPath.parents` is
        # closest-first).
        if include_ancestor_coords and is_literal_subtree:
            # `source` is reused so each ancestor open skips the
            # icechunk session re-encode in `_to_rust_source`.
            ancestor_dses = [
                self.open_dataset(
                    source,
                    group=str(ancestor),
                    drop_variables=drop_variables,
                    branch=branch,
                    storage_options=storage_options,
                    max_concurrency=max_concurrency,
                    mask_and_scale=mask_and_scale,
                    decode_times=decode_times,
                    concat_characters=concat_characters,
                    decode_coords=decode_coords,
                    use_cftime=use_cftime,
                    decode_timedelta=decode_timedelta,
                    zarr_format=zarr_format,
                    consolidated=consolidated,
                )
                for ancestor in PurePosixPath(group).parents
            ]
            groups[ROOT] = merge(
                [groups[ROOT], *ancestor_dses], compat="override"
            )
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
        zarr_format: int | None = None,
        consolidated: bool | None = None,
    ) -> Dataset:
        _check_zarr_v3_only(zarr_format, consolidated)
        # `open_dataset` returns one Dataset, so for literal-path opens
        # (`group=None`/`"/"` or any non-glob path) we ask the Rust
        # walk to skip recursion past `group` — the descendants would
        # be discarded anyway. On `s3://nexrad-arco/KLOT` that drops a
        # full-tree open (107 nodes) to a single-group open. Glob
        # patterns are rejected here: `open_dataset` returns one
        # Dataset, so a multi-match glob has no defined target. Use
        # `open_datatree(group="*/sweep_0")` (Phase 8) instead.
        from rustytree._rustytree import open_datatree as _rust_open

        if group is not None and _GLOB_CHARS.search(group):
            raise NotImplementedError(
                f"rustytree.open_dataset: glob `group=` patterns ({group!r}) "
                "are not supported because `open_dataset` returns a single "
                "Dataset. Use `xr.open_datatree(group=...)` instead."
            )
        group = _normalize_literal_group(group, is_glob=False)

        source = _to_rust_source(filename_or_obj)
        storage_options_arg = None if isinstance(source, bytes) else storage_options

        tree = _rust_open_or_explain(
            _rust_open,
            source,
            **_build_rust_kwargs(
                group=group,
                branch=branch,
                storage_options=storage_options_arg,
                max_concurrency=max_concurrency,
                recursive=False,
            ),
        )

        # With `recursive=False` the Rust dict has exactly one entry,
        # keyed by the absolute path of the requested group.
        target_path = group if (group and group != ROOT) else ROOT
        if target_path not in tree:
            raise KeyError(
                f"rustytree.open_dataset: group {target_path!r} not found in store"
            )
        return _node_to_dataset(
            tree[target_path],
            mask_and_scale=mask_and_scale,
            decode_times=decode_times,
            concat_characters=concat_characters,
            decode_coords=decode_coords,
            drop_variables=drop_variables,
            use_cftime=use_cftime,
            decode_timedelta=decode_timedelta,
        )

    @classmethod
    def guess_can_open(cls, filename_or_obj: Any) -> bool:
        # rustytree only kicks in when the user explicitly passes
        # `engine="rustytree"` — auto-detection would steal paths the
        # user expected to land on `engine="zarr"`.
        return False
