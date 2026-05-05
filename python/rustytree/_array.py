"""xarray ``BackendArray`` adapter over a Rust ``ZarrsArrayHandle``.

The walk hands back one ``ZarrsArrayHandle`` per array in each group's
``vars`` list. Wrapping it as ``RustyBackendArray`` lets xarray treat the
array like any other lazy ``BackendArray`` — chunk reads happen on
``__getitem__`` rather than at open time, and xarray's
``LazilyIndexedArray`` wrapper handles the index-translation gymnastics
for us.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from xarray.backends.common import BackendArray
from xarray.core import indexing


class RustyBackendArray(BackendArray):
    """Lazy view over a Rust-side `ZarrsArrayHandle`.

    The handle is opened once during the walk (so shape and dtype are
    cached cheaply); this wrapper only fetches chunk bytes when xarray
    indexes into it. We advertise ``IndexingSupport.BASIC`` because
    `ZarrsArrayHandle.read_subset` accepts hyperrectangular slabs only;
    fancy indexing (boolean masks, integer arrays) is handled by
    xarray's `explicit_indexing_adapter`, which translates fancy indices
    into a sequence of basic reads.
    """

    __slots__ = ("_handle", "shape", "dtype")

    def __init__(self, handle: Any) -> None:
        self._handle = handle
        self.shape = tuple(handle.shape)
        self.dtype = np.dtype(handle.dtype)

    def __getitem__(self, key: indexing.ExplicitIndexer) -> np.ndarray:
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.BASIC,
            self._raw_indexing_method,
        )

    def _raw_indexing_method(self, key: tuple[Any, ...]) -> np.ndarray:
        # `key` is a tuple, one entry per dimension, each one of:
        #   - int                    (collapse this axis)
        #   - slice(start, stop, 1)  (xarray uses unit step at this level)
        # Translate every entry into an inclusive (start, stop) range and
        # remember which axes the int entries collapse so we can squeeze
        # them back out after the read.
        ranges: list[tuple[int, int]] = []
        squeeze_axes: list[int] = []
        for axis, (entry, dim_size) in enumerate(zip(key, self.shape, strict=True)):
            if isinstance(entry, slice):
                start, stop, step = entry.indices(dim_size)
                if step != 1:
                    msg = (
                        f"RustyBackendArray: stepped slicing not supported (axis {axis} "
                        f"has step {step}); xarray's explicit_indexing_adapter is "
                        "expected to keep us at BASIC support"
                    )
                    raise NotImplementedError(msg)
                ranges.append((start, stop))
            else:
                # Integer index — clamp into a single-element slab and
                # mark the axis for squeezing.
                idx = int(entry)
                if idx < 0:
                    idx += dim_size
                ranges.append((idx, idx + 1))
                squeeze_axes.append(axis)

        flat = self._handle.read_subset(ranges)
        out_shape = tuple(stop - start for start, stop in ranges)
        out = flat.reshape(out_shape)
        if squeeze_axes:
            out = out.squeeze(axis=tuple(squeeze_axes))
        return out
