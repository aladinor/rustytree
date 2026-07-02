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


def test_vanilla_store_handle_is_not_picklable(tiny_zarr_store: Path) -> None:
    """A handle from a non-icechunk (vanilla) store carries no reopen spec and
    must raise a clear, actionable error at pickle time — not a panic, and not
    the opaque default ``cannot pickle 'ZarrsArrayHandle'``."""
    tree = xr.open_datatree(str(tiny_zarr_store), engine="rustytree", chunks={})
    temp = tree["temp"]
    with pytest.raises(Exception, match="not picklable"):
        pickle.dumps(temp)


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
