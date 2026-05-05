"""Plugin-discovery contract.

Phase 1 originally locked the scaffold's stub behaviour
(NotImplementedError). Phase 5 wired `open_datatree` and `open_dataset`
end-to-end, so the stub assertion was retired; what remains is the
plugin-discovery surface (entrypoint resolves, supports_groups True,
guess_can_open False).
"""

from __future__ import annotations

import xarray as xr

import rustytree
from rustytree.backend import RustytreeBackendEntrypoint


def test_package_imports_with_version() -> None:
    assert rustytree.__version__ == "0.1.0"


def test_rust_extension_importable() -> None:
    from rustytree._rustytree import open_datatree

    assert callable(open_datatree)


def test_xarray_engine_registered() -> None:
    engines = xr.backends.list_engines()
    assert "rustytree" in engines
    assert isinstance(engines["rustytree"], RustytreeBackendEntrypoint)


def test_supports_groups_advertised() -> None:
    assert RustytreeBackendEntrypoint.supports_groups is True


def test_guess_can_open_returns_false() -> None:
    ep = RustytreeBackendEntrypoint()
    assert ep.guess_can_open("anything.zarr") is False
