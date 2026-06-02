"""Walk an icechunk repository (recursive, dict-keyed-by-path).

Two input paths exercised here:
  - **Local-fs path string**: rustytree's `_rustytree.open_datatree`
    detects the on-disk icechunk layout and opens it via
    `Repository::open` + `readonly_session` itself. Convenience for
    zero-credential local dev; preserved through the Phase 7 redesign.
  - **`bytes` from `PySession.as_bytes()`**: the cross-extension
    handoff. Users construct an icechunk `Session` in icechunk-python
    (with whatever branch / credentials / cache they want) and pass
    the serialised session bytes; rustytree rehydrates via
    `icechunk::session::Session::from_bytes`. Primary path for
    icechunk-on-S3 after Phase 7 dropped the URL dispatch.
"""

from __future__ import annotations

import os
from pathlib import Path

import icechunk
import pytest

from rustytree._rustytree import open_datatree


def test_open_root_returns_tree_with_root_node(tiny_icechunk_repo: Path) -> None:
    tree = open_datatree(str(tiny_icechunk_repo))
    assert isinstance(tree, dict)
    assert "/" in tree
    root = tree["/"]
    assert set(root) == {"path", "attrs", "vars"}
    assert root["path"] == "/"


def test_root_attrs_round_trip(tiny_icechunk_repo: Path) -> None:
    tree = open_datatree(str(tiny_icechunk_repo))
    assert tree["/"]["attrs"] == {"title": "tiny"}


def test_vars_listed(tiny_icechunk_repo: Path) -> None:
    tree = open_datatree(str(tiny_icechunk_repo))
    assert {var["name"] for var in tree["/"]["vars"]} == {"temp", "mask"}


def test_var_metadata_matches_vanilla(tiny_icechunk_repo: Path) -> None:
    tree = open_datatree(str(tiny_icechunk_repo))
    by_name = {var["name"]: var for var in tree["/"]["vars"]}

    temp = by_name["temp"]
    assert temp["dims"] == ["lat", "lon"]
    assert temp["shape"] == [4, 3]
    assert temp["attrs"] == {"units": "K"}

    mask = by_name["mask"]
    assert mask["dims"] == ["lat", "lon"]
    assert mask["shape"] == [4, 3]


def test_explicit_main_branch_is_default(tiny_icechunk_repo: Path) -> None:
    # Compare metadata only; var dicts now carry a ZarrsArrayHandle that uses
    # identity comparison so two equal opens would fall out of equality.
    def metadata_only(tree: dict) -> dict:
        return {
            path: {
                "path": node["path"],
                "attrs": node["attrs"],
                "vars": [{k: v for k, v in var.items() if k != "handle"} for var in node["vars"]],
            }
            for path, node in tree.items()
        }

    a = open_datatree(str(tiny_icechunk_repo))
    b = open_datatree(str(tiny_icechunk_repo), branch="main")
    assert metadata_only(a) == metadata_only(b)


def test_unknown_branch_raises(tiny_icechunk_repo: Path) -> None:
    with pytest.raises(RuntimeError):
        open_datatree(str(tiny_icechunk_repo), branch="does-not-exist")


KTWX_PATH = Path("/home/alfonso-ladino/python/raw2zarr/zarr/KTWX")


@pytest.mark.skipif(
    not KTWX_PATH.exists() or os.environ.get("RUSTYTREE_SKIP_KTWX") == "1",
    reason="KTWX repo not present (set RUSTYTREE_SKIP_KTWX=1 to skip explicitly)",
)
def test_open_ktwx_recursive_walk() -> None:
    """Smoke test against the user's actual radar icechunk repo.

    Only runs when the repo is present locally. Asserts the recursive
    walk produces a tree whose root node is well-formed; group + variable
    names aren't pinned because the schema may evolve.
    """
    tree = open_datatree(str(KTWX_PATH))
    assert isinstance(tree, dict)
    assert "/" in tree
    root = tree["/"]
    assert root["path"] == "/"
    assert isinstance(root["attrs"], dict)
    assert isinstance(root["vars"], list)
    # Recursive walk should surface at least the root for any tree.
    assert len(tree) >= 1


def test_open_via_session_bytes(tiny_icechunk_repo: Path) -> None:
    """Cross-extension handoff: open the same local fixture via msgpack
    session bytes (the path remote/cloud users will take). Confirms
    `_rustytree.open_datatree(bytes)` reconstructs an equivalent tree
    to the path-based open."""
    storage = icechunk.local_filesystem_storage(str(tiny_icechunk_repo))
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session("main")
    session_bytes = bytes(session._session.as_bytes())

    tree_via_bytes = open_datatree(session_bytes)
    tree_via_path = open_datatree(str(tiny_icechunk_repo))

    # Same set of paths and same per-var metadata (handles compare by
    # identity, so we strip them for the comparison).
    def metadata_only(tree: dict) -> dict:
        return {
            path: {
                "path": node["path"],
                "attrs": node["attrs"],
                "vars": [
                    {k: v for k, v in var.items() if k not in ("handle", "data")}
                    for var in node["vars"]
                ],
            }
            for path, node in tree.items()
        }

    assert metadata_only(tree_via_bytes) == metadata_only(tree_via_path)


def test_garbage_session_bytes_rejected() -> None:
    """Misshapen bytes surface as a clear error rather than panicking
    across the FFI boundary."""
    with pytest.raises((ValueError, RuntimeError), match="Session"):
        open_datatree(b"\x00\x01not msgpack\x02\x03")


@pytest.mark.skipif(
    os.environ.get("RUSTYTREE_S3_SMOKE") != "1",
    reason="Network-gated S3 smoke test (set RUSTYTREE_S3_SMOKE=1 to run)",
)
def test_open_nexrad_arco_klot_anon_s3() -> None:
    """Open the public anonymous icechunk repo on AWS via session bytes.

    User flow:
      1. Construct icechunk `Session` (with credentials, branch, etc.).
      2. Serialise via `session._session.as_bytes()`.
      3. Pass bytes to `_rustytree.open_datatree`; rustytree
         rehydrates via `Session::from_bytes` and walks.

    Network-gated because it hits real AWS; opt-in via
    `RUSTYTREE_S3_SMOKE=1`.
    """
    storage = icechunk.s3_storage(
        bucket="nexrad-arco",
        prefix="KLOT",
        region="us-east-1",
        anonymous=True,
    )
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session("main")
    session_bytes = bytes(session._session.as_bytes())

    tree = open_datatree(session_bytes)
    assert "/" in tree
    root = tree["/"]
    assert root["path"] == "/"
    assert isinstance(root["attrs"], dict)
    assert isinstance(root["vars"], list)


@pytest.mark.skipif(
    os.environ.get("RUSTYTREE_ARRAYLAKE_SMOKE") != "1",
    reason=(
        "arraylake-gated smoke test for issue #40 (set "
        "RUSTYTREE_ARRAYLAKE_SMOKE=1 and be logged into arraylake to run)"
    ),
)
def test_open_arraylake_goes16_refreshable_credentials() -> None:
    """Regression for issue #40: open an arraylake/Earthmover icechunk session.

    arraylake builds the session with a *refreshable* S3 credentials fetcher
    serialized under the typetag `PythonCredentialsFetcher`. Before the
    `src/py_credentials.rs` shim, `Session::from_bytes` rejected it with
    "unknown variant `PythonCredentialsFetcher`, there are no variants". This
    mirrors the notebook in `raw2zarr/notebooks/goes-16-vzarr.ipynb`.

    arraylake-gated (needs the `arraylake` package + a login); opt-in via
    `RUSTYTREE_ARRAYLAKE_SMOKE=1`.
    """
    import xarray as xr

    arraylake = pytest.importorskip("arraylake")

    client = arraylake.Client()
    repo = client.get_repo("earthmover-public/goes-16")
    session = repo.readonly_session("main")

    # This is the exact call that raised in the notebook.
    dt = xr.open_datatree(
        session.store,
        engine="rustytree",
        zarr_format=3,
        consolidated=False,
    )
    assert "/" in dt.groups or dt.children or dt.dataset is not None
