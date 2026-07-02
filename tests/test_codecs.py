"""Numcodecs-namespace codec decoding (`numcodecs.zlib` + `numcodecs.shuffle`).

zarr-python can write Zarr v3 arrays whose codec pipeline uses the
non-standard `numcodecs.*` namespace (it warns they "may not be supported by
other zarr implementations"). The public ``earthmover-public/goes-16`` arraylake
dataset does exactly this: ``bytes -> numcodecs.shuffle -> numcodecs.zlib``.
icechunk reads such data by delegating chunk decode to zarr-python's numcodecs
shim; rustytree decodes via the Rust ``zarrs`` crate, whose ``zlib`` codec is
documented byte-compatible with zarr-python's.

These tests pin that compatibility with a local store (no network), so the
``zlib`` cargo feature can't silently regress. ``numcodecs.shuffle`` is always
compiled by ``zarrs``; ``numcodecs.zlib`` requires the feature enabled in
``Cargo.toml``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr

from rustytree._rustytree import open_datatree

# Skip cleanly if this zarr-python build lacks the numcodecs.zarr3 shim used to
# *write* the fixture (the Rust decode side is what we're really testing).
_numcodecs = pytest.importorskip("zarr.codecs.numcodecs")


@pytest.fixture
def numcodecs_zlib_store(tmp_path: Path) -> tuple[Path, np.ndarray]:
    """Vanilla v3 store mirroring the GOES-16 codec pipeline.

    One ``int16`` array compressed with ``numcodecs.shuffle`` (elementsize 2)
    then ``numcodecs.zlib`` (level 1), spanning multiple chunks so decode runs
    on more than one block. Returns the store path and the written values.
    """
    path = tmp_path / "numcodecs.zarr"
    root = zarr.create_group(store=str(path), zarr_format=3)

    values = np.arange(-20, 20, dtype="int16").reshape(8, 5)
    field = root.create_array(
        "field",
        shape=values.shape,
        dtype="int16",
        chunks=(2, 5),
        compressors=[
            _numcodecs.Shuffle(elementsize=2),
            _numcodecs.Zlib(level=1),
        ],
        dimension_names=("y", "x"),
    )
    field[:] = values
    return path, values


def test_walk_constructs_numcodecs_zlib_array(
    numcodecs_zlib_store: tuple[Path, np.ndarray],
) -> None:
    """The walk must construct the array (codec chain resolves) — this is the
    step that raised "codec numcodecs.zlib is not supported" before the fix."""
    path, _ = numcodecs_zlib_store
    tree = open_datatree(str(path))
    names = {var["name"] for var in tree["/"]["vars"]}
    assert "field" in names


def test_numcodecs_zlib_decodes_to_written_values(
    numcodecs_zlib_store: tuple[Path, np.ndarray],
) -> None:
    """Full decode round-trip: rustytree must reproduce the written values
    bit-for-bit through the shuffle + zlib pipeline."""
    path, values = numcodecs_zlib_store
    # mask_and_scale=False: compare the raw decoded integers directly.
    dt = xr.open_datatree(
        str(path), engine="rustytree", mask_and_scale=False, decode_times=False
    )
    decoded = dt["field"].values
    np.testing.assert_array_equal(decoded, values)
