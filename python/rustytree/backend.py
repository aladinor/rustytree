"""xarray backend entrypoint for rustytree.

Registered via ``[project.entry-points."xarray.backends"]`` in pyproject.toml so
that ``xr.open_datatree(url, engine="rustytree")`` resolves here. The actual
hierarchy walk + Zarr decoding lives in the Rust extension
``rustytree._rustytree`` and lands in Phase 2+.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from xarray.backends.common import BackendEntrypoint


class RustytreeBackendEntrypoint(BackendEntrypoint):
    """xarray backend that opens Zarr DataTrees via the rustytree Rust core."""

    description = "Open Zarr DataTrees concurrently using the rustytree Rust backend"
    url = "https://github.com/aladinor/rustytree"

    supports_groups = True

    def open_dataset(
        self,
        filename_or_obj: Any,
        *,
        drop_variables: str | Iterable[str] | None = None,
    ):
        raise NotImplementedError(
            "rustytree.open_dataset lands in Phase 5 — use xr.open_datatree(engine='rustytree')"
            " once the multi-node walk is wired."
        )

    def open_datatree(
        self,
        filename_or_obj: Any,
        *,
        drop_variables: str | Iterable[str] | None = None,
    ):
        raise NotImplementedError(
            "rustytree.open_datatree lands in Phase 4 — Phase 1 only wires plugin discovery."
        )

    def open_groups_as_dict(
        self,
        filename_or_obj: Any,
        *,
        drop_variables: str | Iterable[str] | None = None,
    ):
        raise NotImplementedError("rustytree.open_groups_as_dict lands in Phase 4.")

    @classmethod
    def guess_can_open(cls, filename_or_obj: Any) -> bool:
        # Phase 2+ will sniff for zarr.json / .zgroup at the URL root.
        return False
