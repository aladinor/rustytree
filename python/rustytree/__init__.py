"""rustytree — Rust-backed xarray DataTree backend for fast Zarr access from object storage.

The Rust extension (``rustytree._rustytree``) is loaded lazily by
``RustytreeBackendEntrypoint`` so that ``import rustytree`` (and xarray's
plugin discovery) doesn't pay the cdylib load cost upfront.
"""

from rustytree.backend import RustytreeBackendEntrypoint

__all__ = ["RustytreeBackendEntrypoint"]
__version__ = "0.2.1"
