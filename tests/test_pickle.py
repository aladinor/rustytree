"""Picklability of rustytree array handles for ``dask.distributed`` (issue #44).

A ``ZarrsArrayHandle`` from an icechunk-session store must survive a pickle
round-trip and reopen with identical data, so the dask task graph can be sent to
distributed workers. This mirrors icechunk's own approach: the reopen state is
the session's ``as_bytes()`` msgpack, reopened via ``Session::from_bytes`` — no
rustytree-side credential handling. Vanilla ``s3://`` / local stores are not yet
picklable and must raise a clear error rather than the opaque default.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import icechunk
import numpy as np
import pytest
import xarray as xr

# These tests open with `chunks={}`, which routes through xarray's dask chunk
# manager — so they need dask. dask is optional for rustytree (a runtime dep of
# neither rustytree nor icechunk; only a dev/test extra), so skip cleanly rather
# than erroring when it's absent, mirroring how icechunk keeps dask opt-in.
pytest.importorskip("dask")


def _readonly_store(repo_path: Path):
    """Reopen a committed local icechunk repo and return its readonly store."""
    storage = icechunk.local_filesystem_storage(str(repo_path))
    repo = icechunk.Repository.open(storage)
    return repo.readonly_session("main").store


def test_raw_handle_pickles_and_reads_identically(tiny_icechunk_repo: Path) -> None:
    tree = xr.open_datatree(_readonly_store(tiny_icechunk_repo), engine="rustytree", chunks={})
    temp = tree["temp"]
    expected = temp.values

    revived = pickle.loads(pickle.dumps(temp))
    np.testing.assert_array_equal(revived.values, expected)


def test_datatree_variable_pickles(tiny_icechunk_repo: Path) -> None:
    """The lazy dask-backed variable (which holds the handle in its graph) must
    pickle and recompute correctly."""
    tree = xr.open_datatree(_readonly_store(tiny_icechunk_repo), engine="rustytree", chunks={})
    mask = tree["mask"]
    blob = pickle.dumps(mask)
    revived = pickle.loads(blob)
    np.testing.assert_array_equal(revived.values, mask.values)


def test_nested_group_arrays_pickle_and_read_identically(
    multilevel_icechunk_repo: Path,
) -> None:
    """Reopen uses the array's *absolute* store path (``array.path()``). Exercise
    nested groups — a bug that stored a group-relative name or mangled the
    leading slash would make ``Array::async_open`` fail on the worker for these
    but pass for root-level arrays."""
    tree = xr.open_datatree(
        _readonly_store(multilevel_icechunk_repo), engine="rustytree", chunks={}
    )

    # depth-1 array with real (arange) values → catches value corruption too
    temp = tree["volume_a"]["temp"]
    revived_temp = pickle.loads(pickle.dumps(temp))
    np.testing.assert_array_equal(revived_temp.values, np.arange(4, dtype="float64"))

    # depth-2 array → the key path-resolution check (raises on a path bug)
    dbz = tree["volume_a/sweep_0"]["dbz"]
    revived_dbz = pickle.loads(pickle.dumps(dbz))
    assert revived_dbz.shape == (8, 16)
    assert revived_dbz.dtype == np.float32
    np.testing.assert_array_equal(revived_dbz.values, dbz.values)


def test_revived_handle_repickles(tiny_icechunk_repo: Path) -> None:
    """A revived handle must stay picklable (dask task-stealing / scatter /
    memory-spill re-serialize an already-deserialized graph). Guards against
    ``_reopen_array_handle`` dropping the spec on the revived handle."""
    tree = xr.open_datatree(_readonly_store(tiny_icechunk_repo), engine="rustytree", chunks={})
    temp = tree["temp"]
    expected = temp.values

    hop1 = pickle.loads(pickle.dumps(temp))
    hop2 = pickle.loads(pickle.dumps(hop1))  # re-pickle the revived handle
    np.testing.assert_array_equal(hop2.values, expected)


def test_corrupt_pickle_state_raises_valueerror() -> None:
    """The public reconstructor must surface a corrupt/truncated task-graph
    payload as a clean ``ValueError``, not a Rust panic that aborts the worker."""
    from rustytree._rustytree import _reopen_array_handle

    with pytest.raises(ValueError, match="corrupt"):
        _reopen_array_handle(b"\xff\xff not msgpack \x00")


def test_vanilla_store_handle_is_not_picklable(tiny_zarr_store: Path) -> None:
    """A handle from a non-icechunk (vanilla) store carries no reopen spec and
    must raise a clear, actionable error at pickle time — not a panic, and not
    the opaque default ``cannot pickle 'ZarrsArrayHandle'``. The message must keep
    its actionable guidance (open via an icechunk Session, or use the threaded
    scheduler)."""
    tree = xr.open_datatree(str(tiny_zarr_store), engine="rustytree", chunks={})
    temp = tree["temp"]
    with pytest.raises(ValueError) as exc:
        pickle.dumps(temp)
    msg = str(exc.value)
    assert "not picklable" in msg
    assert "icechunk Session" in msg
    assert "scheduler" in msg


@pytest.mark.distributed
def test_distributed_compute_matches_threaded(tiny_icechunk_repo: Path) -> None:
    """The #44 repro: compute a reduction under a real distributed cluster and
    check it equals the threaded-scheduler result."""
    distributed = pytest.importorskip("distributed")

    store = _readonly_store(tiny_icechunk_repo)
    tree = xr.open_datatree(store, engine="rustytree", chunks={})
    threaded = float(tree["temp"].sum())

    with (
        distributed.LocalCluster(
            processes=True,
            n_workers=2,
            threads_per_worker=1,
            dashboard_address=None,
        ) as cluster,
        distributed.Client(cluster),
    ):
        got = float(tree["temp"].sum().compute())

    assert got == threaded
