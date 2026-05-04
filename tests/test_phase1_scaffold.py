"""Phase 1 scaffold contract: entry point resolves, stubs raise NotImplementedError."""

from __future__ import annotations

import pytest
import xarray as xr

import rustytree
from rustytree.backend import RustytreeBackendEntrypoint


def test_package_imports_with_version() -> None:
    assert rustytree.__version__ == "0.1.0"


def test_rust_extension_importable() -> None:
    from rustytree._rustytree import open_datatree

    assert callable(open_datatree)


def test_rust_open_datatree_stub_raises() -> None:
    from rustytree._rustytree import open_datatree

    with pytest.raises(NotImplementedError, match="Phase 2"):
        open_datatree("file:///tmp/does-not-matter")


def test_xarray_engine_registered() -> None:
    engines = xr.backends.list_engines()
    assert "rustytree" in engines
    assert isinstance(engines["rustytree"], RustytreeBackendEntrypoint)


def test_supports_groups_advertised() -> None:
    assert RustytreeBackendEntrypoint.supports_groups is True


def test_guess_can_open_returns_false() -> None:
    ep = RustytreeBackendEntrypoint()
    assert ep.guess_can_open("anything.zarr") is False


@pytest.mark.parametrize("method", ["open_dataset", "open_datatree", "open_groups_as_dict"])
def test_python_stubs_raise_not_implemented(method: str) -> None:
    ep = RustytreeBackendEntrypoint()
    with pytest.raises(NotImplementedError):
        getattr(ep, method)("s3://fake/path")
