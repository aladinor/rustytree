"""Walk a single group inside a local icechunk repository.

These tests exercise the icechunk dispatch path end-to-end: rustytree
detects the repository layout, opens it via `Repository::open` +
`readonly_session`, wraps the session as a zarrs-compatible store, and
runs the same walk code that powers the vanilla Zarr v3 path.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rustytree._rustytree import open_datatree


def test_open_root_returns_node_dict(tiny_icechunk_repo: Path) -> None:
    node = open_datatree(str(tiny_icechunk_repo))
    assert isinstance(node, dict)
    assert set(node) == {"path", "attrs", "vars"}
    assert node["path"] == "/"


def test_root_attrs_round_trip(tiny_icechunk_repo: Path) -> None:
    node = open_datatree(str(tiny_icechunk_repo))
    assert node["attrs"] == {"title": "tiny"}


def test_vars_listed(tiny_icechunk_repo: Path) -> None:
    node = open_datatree(str(tiny_icechunk_repo))
    assert {var["name"] for var in node["vars"]} == {"temp", "mask"}


def test_var_metadata_matches_vanilla(tiny_icechunk_repo: Path) -> None:
    node = open_datatree(str(tiny_icechunk_repo))
    by_name = {var["name"]: var for var in node["vars"]}

    temp = by_name["temp"]
    assert temp["dims"] == ["lat", "lon"]
    assert temp["shape"] == [4, 3]
    assert temp["attrs"] == {"units": "K"}

    mask = by_name["mask"]
    assert mask["dims"] == ["lat", "lon"]
    assert mask["shape"] == [4, 3]


def test_explicit_main_branch_is_default(tiny_icechunk_repo: Path) -> None:
    a = open_datatree(str(tiny_icechunk_repo))
    b = open_datatree(str(tiny_icechunk_repo), branch="main")
    assert a == b


def test_unknown_branch_raises(tiny_icechunk_repo: Path) -> None:
    with pytest.raises(RuntimeError):
        open_datatree(str(tiny_icechunk_repo), branch="does-not-exist")


KTWX_PATH = Path("/home/alfonso-ladino/python/raw2zarr/zarr/KTWX")


@pytest.mark.skipif(
    not KTWX_PATH.exists() or os.environ.get("RUSTYTREE_SKIP_KTWX") == "1",
    reason="KTWX repo not present (set RUSTYTREE_SKIP_KTWX=1 to skip explicitly)",
)
def test_open_ktwx_root() -> None:
    """Smoke test against the user's actual radar icechunk repo.

    Only runs when the repo is present locally. Asserts the engine opens
    the root and produces a structurally valid `NodeData`. Group + variable
    names aren't pinned because the schema may evolve; only the shape of
    the response is.
    """
    node = open_datatree(str(KTWX_PATH))
    assert isinstance(node, dict)
    assert set(node) == {"path", "attrs", "vars"}
    assert node["path"] == "/"
    assert isinstance(node["attrs"], dict)
    assert isinstance(node["vars"], list)


@pytest.mark.skipif(
    os.environ.get("RUSTYTREE_S3_SMOKE") != "1",
    reason="Network-gated S3 smoke test (set RUSTYTREE_S3_SMOKE=1 to run)",
)
def test_open_nexrad_arco_klot_anon_s3() -> None:
    """Open the public anonymous icechunk repo on AWS.

    Exercises the `s3://`-icechunk dispatch end-to-end:
      1. URL parsing produces `StoreSpec::S3 {bucket, prefix}`.
      2. `s3_is_icechunk` HEADs `KLOT/repo` → True.
      3. `open_s3_icechunk` calls icechunk's `new_s3_storage` +
         `Repository::open` + `readonly_session("main")` against AWS.
      4. The walk reads the root group's `zarr.json` from the icechunk
         manifest and returns a structurally valid `NodeData`.

    Network-gated because it hits real AWS; opt-in via
    `RUSTYTREE_S3_SMOKE=1`.
    """
    node = open_datatree(
        "s3://nexrad-arco/KLOT",
        storage_options={"region": "us-east-1", "anon": True},
    )
    assert node["path"] == "/"
    assert isinstance(node["attrs"], dict)
    assert isinstance(node["vars"], list)
