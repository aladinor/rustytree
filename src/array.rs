//! `ZarrsArrayHandle`: `PyO3` wrapper around an opened `zarrs::Array` that
//! defers chunk reads until Python asks for them.
//!
//! The walk eagerly opens every array (one `Array::async_open` per array,
//! which reads `<path>/zarr.json` from the store). That gives us shape,
//! dtype, dims, and attributes without having to fetch any chunks. The
//! resulting `Array` is then handed out wrapped in a `ZarrsArrayHandle`
//! so xarray can call back through `read_subset` whenever it needs data.
//!
//! `read_subset` runs `runtime.block_on(array.async_retrieve_array_subset::<Vec<T>>(...))`
//! with the GIL released (`Python::detach`) so concurrent loads from a
//! Python thread pool overlap on the network rather than serialising
//! through the GIL.

use std::borrow::Cow;
use std::sync::Arc;

use numpy::PyArray1;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyIndexError, PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyString, PyTuple};
use tokio::runtime::Handle;
use zarrs::array::{Array, ArraySubset, DataType, data_type};
use zarrs::plugin::ExtensionName;
use zarrs_storage::AsyncReadableListableStorageTraits;

use crate::dtype_dispatch::for_each_supported_dtype;
use crate::store::ReopenSpec;

/// A handle to an already-opened `zarrs::Array` that can read array
/// subsets back to `NumPy`.
///
/// Constructed by the walk; consumed by `RustyBackendArray` on the
/// Python side. The `Array` is shared via `Arc` so multiple handles can
/// alias the same underlying array without re-opening it.
#[pyclass(module = "rustytree._rustytree", name = "ZarrsArrayHandle")]
pub(crate) struct ZarrsArrayHandle {
    array: Arc<Array<dyn AsyncReadableListableStorageTraits>>,
    runtime: Handle,
    /// How to reopen this array's store on a fresh process, so the handle can
    /// survive a pickle round-trip onto a `dask.distributed` worker (issue #44).
    /// `None` for store types not yet picklable (vanilla `s3://` / local) —
    /// `__reduce__` then raises a clear error rather than a panic. `Arc` so all
    /// handles in one tree share one copy of the (potentially large) spec. The
    /// array's own path (`array.path()`) supplies the other half of the pickle
    /// state, so we don't store it separately.
    spec: Option<Arc<ReopenSpec>>,
}

impl ZarrsArrayHandle {
    pub(crate) fn new(
        array: Arc<Array<dyn AsyncReadableListableStorageTraits>>,
        runtime: Handle,
        spec: Option<Arc<ReopenSpec>>,
    ) -> Self {
        Self {
            array,
            runtime,
            spec,
        }
    }

    /// Read a string-like dtype (`string` / `fixed_length_utf32`) as a 1-D numpy
    /// `object` array of `str`, or `Ok(None)` when the dtype isn't string-like
    /// (the numeric macro handles those). Both flavours reduce to a
    /// `Vec<String>` from zarrs; the Python adapter casts the `object` array to
    /// the declared `StringDType` / `<U…` dtype. Split out of `read_subset` so
    /// that method stays within the line budget.
    fn read_string_subset<'py>(
        &self,
        py: Python<'py>,
        subset: &ArraySubset,
        is_empty: bool,
        aligned_shape: &[u64],
        offsets: &[u64],
        request_shape: &[u64],
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        let dtype = self.array.data_type();
        let is_string = dtype.is::<data_type::StringDataType>();
        let is_utf32 = dtype.is::<data_type::FixedLengthUTF32DataType>();
        if !is_string && !is_utf32 {
            return Ok(None);
        }
        let strings: Vec<String> = if is_empty {
            Vec::new()
        } else {
            let array = self.array.clone();
            let runtime = self.runtime.clone();
            py.detach(|| -> PyResult<Vec<String>> {
                let read_err = |err| PyValueError::new_err(format!("zarrs read failed: {err}"));
                if is_string {
                    runtime
                        .block_on(array.async_retrieve_array_subset::<Vec<String>>(subset))
                        .map_err(read_err)
                } else {
                    // zarrs decodes `fixed_length_utf32` into one `Vec<char>`
                    // per element (trailing U+0000 padding trimmed). Known
                    // limitation: Rust `char` can't hold lone surrogate code
                    // points (U+D800–U+DFFF), so zarrs drops them — unlike numpy
                    // `<U`, which keeps the raw bytes. That diverges from
                    // `engine="zarr"` only for malformed Unicode; valid content
                    // round-trips exactly (see `test_utf32_lone_surrogate_dropped`).
                    let rows: Vec<Vec<char>> = runtime
                        .block_on(array.async_retrieve_array_subset::<Vec<Vec<char>>>(subset))
                        .map_err(read_err)?;
                    Ok(rows
                        .into_iter()
                        .map(|chars| chars.into_iter().collect())
                        .collect())
                }
            })?
        };
        Ok(Some(strings_to_object_pyarray(
            py,
            strings,
            aligned_shape,
            offsets,
            request_shape,
        )?))
    }
}

#[pymethods]
impl ZarrsArrayHandle {
    /// The array's shape as a tuple of `int`.
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.array.shape())
    }

    /// The array's chunk shape as a tuple of `int` (one per dimension).
    /// Used by the Python entrypoint to populate
    /// `Variable.encoding["chunks"]` and `["preferred_chunks"]`, which
    /// xarray reads when the user passes `chunks={}` to
    /// `xr.open_datatree` so dask gets the on-disk chunk shape rather
    /// than a single big chunk per array.
    #[getter]
    fn chunks<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let chunk_shape = self
            .array
            .chunk_shape(&vec![0; self.array.dimensionality()])
            .map_err(|err| {
                PyValueError::new_err(format!("zarrs: chunk_shape lookup failed: {err}"))
            })?;
        let dims: Vec<u64> = chunk_shape.iter().map(|n| n.get()).collect();
        PyTuple::new(py, dims)
    }

    /// The array's dtype as a `NumPy` dtype string (e.g. `"float64"`,
    /// `"int8"`). Naming is broader than reading: a dtype can be named
    /// here and still be refused by `read_subset` until it is added to
    /// `for_each_supported_dtype!`. Dtypes we can neither read nor name
    /// safely yield a string `numpy.dtype()` rejects — see
    /// [`zarrs_dtype_to_numpy_str`] for why that is deliberate.
    #[getter]
    fn dtype(&self) -> String {
        zarrs_dtype_to_numpy_str(self.array.data_type())
    }

    /// The `NumPy` dtype the read materialises, which `RustyBackendArray` casts
    /// the returned buffer to. Differs from [`Self::dtype`] only for vlen
    /// `string` (declared `object`, read as numpy-2 `StringDType`), mirroring
    /// `engine="zarr"`. See [`zarrs_dtype_read_numpy_str`].
    #[getter]
    fn read_dtype(&self) -> String {
        zarrs_dtype_read_numpy_str(self.array.data_type())
    }

    /// Pickle support (issue #44): make the handle survive a `dask.distributed`
    /// task-graph serialization by reducing it to `(reconstructor, (state,))`,
    /// where `state` is the msgpack of `(reopen_spec, array_path)`. The worker
    /// calls [`_reopen_array_handle`] to rebuild the store and re-open the array.
    ///
    /// Only handles that carry a reopen spec (icechunk-session stores) are
    /// picklable; others raise a clear error rather than the opaque default
    /// `cannot pickle 'ZarrsArrayHandle'`. No credential handling lives here —
    /// the spec is icechunk's own serialized session (see [`ReopenSpec`]).
    fn __reduce__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyBytes>,))> {
        let Some(spec) = &self.spec else {
            return Err(PyValueError::new_err(
                "rustytree: this array is not picklable — pickling (e.g. for \
                 dask.distributed) is currently supported only for stores opened via an \
                 icechunk Session, not vanilla s3:// / local Zarr stores. Either open the \
                 store through an icechunk Session, or compute with the threaded scheduler \
                 (`dask.config.set(scheduler=\"threads\")`).",
            ));
        };
        let state =
            rmp_serde::to_vec(&(spec.as_ref(), self.array.path().as_str())).map_err(|err| {
                PyValueError::new_err(format!(
                    "rustytree: failed to serialise array handle: {err}"
                ))
            })?;
        let reconstructor = py
            .import("rustytree._rustytree")?
            .getattr("_reopen_array_handle")?;
        Ok((reconstructor, (PyBytes::new(py, &state),)))
    }

    /// Read a hyperrectangular slab of the array.
    ///
    /// `ranges` is a list of already-normalised half-open `(start, stop)`
    /// tuples, one per dimension, satisfying
    /// `0 <= start <= stop <= shape[i]`. Translating Python slice
    /// semantics — reversed, negative or out-of-range endpoints — is the
    /// caller's job; `RustyBackendArray._raw_indexing_method` is where
    /// that happens. A reversed range is rejected here rather than
    /// silently read as empty, so a malformed range from any other caller
    /// still surfaces.
    ///
    /// Returns a 1-D `NumPy` array of length `prod(stop_i - start_i)`;
    /// the Python adapter (`RustyBackendArray`) reshapes to the
    /// requested shape.
    ///
    /// Releases the GIL while the chunk read is in flight so concurrent
    /// loads from a `concurrent.futures` thread pool actually overlap on
    /// the network instead of serialising through Python.
    // PyO3's argument extraction needs an owned `Vec`; clippy then complains
    // it isn't consumed (we only iterate). The Vec is what Python hands over,
    // so the by-value signature is forced by the FFI boundary — `expect` so
    // we'd notice if PyO3 ever grew slice support and the silencer became
    // unnecessary.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 argument extraction requires owned types; cannot take &[(u64, u64)]"
    )]
    fn read_subset<'py>(
        &self,
        py: Python<'py>,
        ranges: Vec<(u64, u64)>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if ranges.len() != self.array.dimensionality() {
            return Err(PyIndexError::new_err(format!(
                "expected {} ranges, got {}",
                self.array.dimensionality(),
                ranges.len()
            )));
        }
        let shape = self.array.shape();
        for (i, (start, stop)) in ranges.iter().enumerate() {
            if start > stop {
                return Err(PyValueError::new_err(format!(
                    "range[{i}]: start {start} > stop {stop}"
                )));
            }
            if *stop > shape[i] {
                return Err(PyIndexError::new_err(format!(
                    "range[{i}] stop {stop} exceeds dim size {}",
                    shape[i]
                )));
            }
        }

        // An empty selection (any axis with `start == stop`, e.g.
        // `isel(time=slice(1, 1))`) has nothing to fetch. Skipping the
        // read matters: chunk alignment below would expand it to a whole
        // chunk and pull that off the network — a full round-trip on S3 —
        // only to slice everything back out again.
        let is_empty = ranges.iter().any(|(start, stop)| start == stop);

        // Align the requested ranges to chunk-grid boundaries so each
        // chunk read goes through zarrs's `async_retrieve_chunk_opt`
        // fast path (which falls back to the array's `fill_value` when
        // a chunk is missing from storage). The slow path
        // `async_retrieve_chunk_subset_opt` does NOT do that fallback —
        // it asks the storage directly via `AsyncStoragePartialDecoder`
        // and propagates the icechunk `ChunkNotFound` error. By
        // expanding the read to whole chunks we bypass that bug; we
        // then slice the result down to what was actually asked for.
        // Cost: over-fetches when the request is much smaller than a
        // chunk, but that's usually a CF-decode peek and the over-fetch
        // is a single chunk worth of data.
        // Upstream: https://github.com/LDeakin/zarrs (chunk subset
        // partial-decode bypasses fill_value fallback).
        let chunk_shape_nz = self
            .array
            .chunk_shape(&vec![0; self.array.dimensionality()])
            .map_err(|err| {
                PyValueError::new_err(format!("zarrs: chunk_shape lookup failed: {err}"))
            })?;
        let chunk_shape: Vec<u64> = chunk_shape_nz.iter().map(|n| n.get()).collect();

        let mut aligned_ranges: Vec<(u64, u64)> = Vec::with_capacity(ranges.len());
        let mut request_offsets_in_aligned: Vec<u64> = Vec::with_capacity(ranges.len());
        let mut request_shape: Vec<u64> = Vec::with_capacity(ranges.len());
        let mut aligned_shape: Vec<u64> = Vec::with_capacity(ranges.len());
        for (i, (start, stop)) in ranges.iter().enumerate() {
            let cs = chunk_shape[i].max(1);
            let aligned_start = (start / cs) * cs;
            let aligned_stop_unbounded = stop.div_ceil(cs) * cs;
            let aligned_stop = aligned_stop_unbounded.min(shape[i]);
            aligned_ranges.push((aligned_start, aligned_stop));
            request_offsets_in_aligned.push(start - aligned_start);
            request_shape.push(stop - start);
            aligned_shape.push(aligned_stop - aligned_start);
        }
        let aligned_std_ranges: Vec<std::ops::Range<u64>> =
            aligned_ranges.iter().map(|(s, e)| *s..*e).collect();
        let subset = ArraySubset::new_with_ranges(&aligned_std_ranges);

        // Dispatch on dtype via the shared macro: each branch decodes
        // into the matching primitive type and hands a 1-D `NumPy`
        // array back to Python. The Python adapter reshapes; doing it
        // here would force every dtype to materialise an ndarray crate
        // type, which costs an extra dependency for no benefit.
        // String-like dtypes (`string`, `fixed_length_utf32`) are handled ahead
        // of the numeric `for_each_supported_dtype!` macro (which is shared with
        // the eager-fetch path and stays numeric-only).
        if let Some(result) = self.read_string_subset(
            py,
            &subset,
            is_empty,
            &aligned_shape,
            &request_offsets_in_aligned,
            &request_shape,
        )? {
            return Ok(result);
        }

        let array = self.array.clone();
        let runtime = self.runtime.clone();

        for_each_supported_dtype!(self.array.data_type(), T => {
            // The empty case skips the read but still flows through this
            // one dispatch, so the returned array carries the same dtype
            // a non-empty read would, from a single code path.
            let elements: Vec<T> = if is_empty {
                Vec::new()
            } else {
                py.detach(|| -> PyResult<Vec<T>> {
                    runtime
                        .block_on(array.async_retrieve_array_subset::<Vec<T>>(&subset))
                        .map_err(|err| PyValueError::new_err(format!("zarrs read failed: {err}")))
                })?
            };
            let sliced = slice_nd(
                elements,
                &aligned_shape,
                &request_offsets_in_aligned,
                &request_shape,
            );
            PyArray1::from_vec(py, sliced).into_bound_py_any(py)
        }, other => {
            // The store's own spelling — see `zarrs_dtype_zarr_name`.
            let name = zarrs_dtype_zarr_name(other);
            Err(PyNotImplementedError::new_err(format!(
                "rustytree: dtype {name} is not yet supported by ZarrsArrayHandle.read_subset; \
                 supported today: bool, int{{8,16,32,64}}, uint{{8,16,32,64}}, float{{32,64}}"
            )))
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "ZarrsArrayHandle(shape={:?}, dtype={})",
            self.array.shape(),
            zarrs_dtype_to_numpy_str(self.array.data_type())
        )
    }
}

/// Slice an N-dimensional row-major buffer down to a hyperrectangle.
///
/// `elements` is a flat row-major buffer of shape `aligned_shape`; we
/// extract a contiguous-in-the-trailing-axes sub-rectangle starting at
/// `offsets` with shape `out_shape`. Used by `read_subset` to slice a
/// chunk-aligned read down to the actually-requested ranges.
// `T: Clone` (not `Copy`) so the same slicer serves the numeric arms and the
// `String` buffers produced by the string-like arms. Zero-cost for numeric `T`:
// the identity fast-path still moves `elements`, and cloning a `Copy` value is
// a bit copy.
fn slice_nd<T: Clone>(
    elements: Vec<T>,
    aligned_shape: &[u64],
    offsets: &[u64],
    out_shape: &[u64],
) -> Vec<T> {
    debug_assert_eq!(aligned_shape.len(), offsets.len());
    debug_assert_eq!(aligned_shape.len(), out_shape.len());
    // Fast path: the aligned read already matches the request (common
    // when the request is itself chunk-aligned, e.g. full-array reads).
    let identity = aligned_shape == out_shape && offsets.iter().all(|o| *o == 0);
    if identity {
        return elements;
    }
    // Use usize internally — we're indexing into a `Vec<T>` so values
    // fit usize by construction (zarrs allocated this Vec, so it cannot
    // be larger than the address space).
    let aligned_shape: Vec<usize> = aligned_shape
        .iter()
        .map(|n| usize::try_from(*n).expect("aligned_shape fits usize"))
        .collect();
    let offsets: Vec<usize> = offsets
        .iter()
        .map(|n| usize::try_from(*n).expect("offset fits usize"))
        .collect();
    let out_shape: Vec<usize> = out_shape
        .iter()
        .map(|n| usize::try_from(*n).expect("out_shape fits usize"))
        .collect();
    let total_out: usize = out_shape.iter().product();
    let mut out: Vec<T> = Vec::with_capacity(total_out);
    // Row-major strides for the aligned (source) buffer.
    let n = aligned_shape.len();
    let mut src_strides = vec![1_usize; n];
    for i in (0..n.saturating_sub(1)).rev() {
        src_strides[i] = src_strides[i + 1] * aligned_shape[i + 1];
    }
    // Walk the output in row-major order, computing the source index for
    // each destination element. Count-driven rather than exiting on a
    // carry rollover, so an empty request is zero iterations and the loop
    // is total by construction — that was #65.
    let mut idx = vec![0_usize; n];
    for _ in 0..total_out {
        let mut src = 0_usize;
        for i in 0..n {
            src += (offsets[i] + idx[i]) * src_strides[i];
        }
        out.push(elements[src].clone());
        // Increment the multidim index, last axis first, carrying into
        // the next axis up when one wraps.
        for axis in (0..n).rev() {
            idx[axis] += 1;
            if idx[axis] < out_shape[axis] {
                break;
            }
            idx[axis] = 0;
        }
    }
    out
}

/// Slice a flat `Vec<String>` down to the requested hyperrectangle and hand it
/// back as a 1-D numpy `object` array of `PyString`s.
///
/// Both string-like dtypes (`string`, `fixed_length_utf32`) funnel through
/// here: the `numpy` crate can only build `object` arrays, so the Python
/// adapter (`RustyBackendArray`) reshapes and casts the result to the declared
/// `StringDType` / `<U…` dtype.
fn strings_to_object_pyarray<'py>(
    py: Python<'py>,
    strings: Vec<String>,
    aligned_shape: &[u64],
    offsets: &[u64],
    request_shape: &[u64],
) -> PyResult<Bound<'py, PyAny>> {
    let sliced = slice_nd(strings, aligned_shape, offsets, request_shape);
    // `from_slice` (not `from_vec`): for object dtype it clones each ref into a
    // numpy-owned buffer, so numpy and the local `objs` Vec own their
    // references independently — no double-decref hazard if the array is later
    // mutated. One incref/element, negligible at coord/scalar sizes.
    let objs: Vec<Py<PyAny>> = sliced
        .into_iter()
        .map(|s| PyString::new(py, &s).into_any().unbind())
        .collect();
    PyArray1::from_slice(py, &objs).into_bound_py_any(py)
}

/// Reconstruct a `ZarrsArrayHandle` from its pickled state (see
/// [`ZarrsArrayHandle::__reduce__`]). Called on a `dask.distributed` worker to
/// revive a handle: rebuild the store from the reopen spec (icechunk
/// `Session::from_bytes`, via `build_store_from_spec`) and re-open the single
/// array by path. This mirrors the initial open — reusing icechunk's own serde,
/// with no rustytree-side credential handling.
#[pyfunction]
pub(crate) fn _reopen_array_handle(py: Python<'_>, state: &[u8]) -> PyResult<ZarrsArrayHandle> {
    let (spec, path): (ReopenSpec, String) = rmp_serde::from_slice(state).map_err(|err| {
        PyValueError::new_err(format!(
            "rustytree: corrupt ZarrsArrayHandle pickle state: {err}"
        ))
    })?;

    let runtime = crate::runtime::handle();
    // Release the GIL while rebuilding the store + reading `<path>/zarr.json`
    // over the network, matching `read_subset`.
    let array = py.detach(|| -> crate::error::Result<_> {
        runtime.block_on(async {
            let store = crate::store::build_store_from_spec(&spec)?.into_store();
            let array = Array::async_open(store, &path).await.map_err(|err| {
                crate::error::RustytreeError::Other(format!(
                    "rustytree: failed to reopen array {path:?} on worker: {err}"
                ))
            })?;
            Ok(Arc::new(array))
        })
    })?;

    Ok(ZarrsArrayHandle::new(
        array,
        runtime.handle().clone(),
        Some(Arc::new(spec)),
    ))
}

/// Translate `zarrs::array::DataType` to the canonical `NumPy` dtype
/// string.
///
/// The Zarr V3 name *is* the `NumPy` name for every dtype we support
/// (`float64`, `int8`, `complex128`, …), so this is a lookup rather than
/// a table. `maps_every_named_dtype_to_its_numpy_name` pins the sixteen
/// names we actually care about, so an upstream alias rename on any of
/// those fails a test; zarrs registers ~45 dtypes in total and the rest
/// pass through unchecked. rustytree is V3-only, so the V3 spelling is
/// the one the metadata actually used.
///
/// Deliberately does *not* use `Display`: zarrs 0.23 renders both Zarr
/// spellings when they differ, so `float64` comes out as
/// `"float64 / <f8"` — which `numpy.dtype()` rejects.
///
/// Two behaviours worth knowing:
///
/// - `bytes` and variable-length `string` are both declared `object`.
///   `np.dtype("bytes")` is `|S0` (fixed-width, zero itemsize), so passing the
///   V3 name through would mislabel a vlen binary array. `string` is declared
///   `object` *even though the read materialises numpy-2 `StringDType`*
///   ([`zarrs_dtype_read_numpy_str`]): that mirrors `engine="zarr"`, whose
///   variable is `object` while its values are `StringDType`. Declaring
///   `object` up-front is load-bearing — xarray's CF-decode flattens a
///   `StringDType`-declared variable's *values* to `object`, so declaring
///   `object` is the only way to keep the values as `StringDType`.
/// - `fixed_length_utf32` (numpy `<U…`) → `"<U{n}"`, fixed-width unicode with
///   `n` = the element capacity in code points — matching what `engine="zarr"`
///   materialises for that dtype (declared and read alike).
/// - Names we can't read and can't safely rename (`numpy.datetime64`, `r32`, …)
///   pass through and make `np.dtype()` raise `TypeError` at open. That loud
///   failure is intended.
pub(crate) fn zarrs_dtype_to_numpy_str(dtype: &DataType) -> String {
    if dtype.is::<data_type::BytesDataType>() || dtype.is::<data_type::StringDataType>() {
        return "object".to_owned();
    }
    if let Some(utf32) = dtype.downcast_ref::<data_type::FixedLengthUTF32DataType>() {
        return format!("<U{}", utf32.capacity_code_points().get());
    }
    zarrs_dtype_zarr_name(dtype)
}

/// The `NumPy` dtype the *read* materialises — what `RustyBackendArray` casts the
/// returned buffer to. Identical to [`zarrs_dtype_to_numpy_str`] except for
/// variable-length `string`, which reads back as numpy-2 `StringDType` (`"T"`)
/// while being *declared* `object`. This declared/read split is exactly how
/// `engine="zarr"` surfaces vlen strings on numpy≥2 (numpy collapses a 0-D
/// `StringDType` to `<U` by itself, matching zarr's scalar behaviour). This is
/// why the package floor is `numpy>=2.0`: `StringDType` did not exist before.
pub(crate) fn zarrs_dtype_read_numpy_str(dtype: &DataType) -> String {
    if dtype.is::<data_type::StringDataType>() {
        return "T".to_owned();
    }
    zarrs_dtype_to_numpy_str(dtype)
}

/// The dtype's Zarr V3 name — the spelling that appears in the store's
/// own `zarr.json`.
///
/// Use this for *diagnostics*, not for handing a dtype to `NumPy`.
/// [`zarrs_dtype_to_numpy_str`] deliberately renames `bytes` to `object`,
/// which is right for `np.dtype()` but wrong in an error message: a user
/// whose metadata says `"variable_length_bytes"` should not be told that
/// "dtype object is not supported" — that name appears nowhere in their
/// store.
pub(crate) fn zarrs_dtype_zarr_name(dtype: &DataType) -> String {
    dtype
        .name_v3()
        .map_or_else(|| dtype.to_string(), Cow::into_owned)
}

#[cfg(test)]
mod tests {
    use super::{slice_nd, zarrs_dtype_read_numpy_str, zarrs_dtype_to_numpy_str};
    use crate::dtype_dispatch::for_each_supported_dtype;
    use zarrs::array::{DataType, data_type};

    /// Regression tests for #65: the walk used to push an element before
    /// it could test the bounds.
    ///
    /// `read_subset` now skips the read for an empty selection, so these
    /// arrangements are the ones it *would* build absent that skip. They
    /// are kept deliberately: `slice_nd` is a `pub(crate)` helper whose
    /// totality should not depend on one caller's fast path, and the
    /// failure mode was an out-of-bounds panic.
    mod empty_selection {
        use super::slice_nd;

        /// `read_subset([(1, 1)])` on shape 12 / chunks 4.
        #[test]
        fn at_unaligned_offset_yields_no_elements() {
            assert_eq!(
                slice_nd(vec![0.0, 1.0, 2.0, 3.0], &[4], &[1], &[0]),
                Vec::<f64>::new()
            );
        }

        /// `read_subset([(13, 13)])` on shape 13 / chunks 4 — the
        /// arrangement that panicked rather than returning a wrong length.
        #[test]
        fn at_end_of_clamped_ragged_chunk_yields_no_elements() {
            assert_eq!(slice_nd(vec![12.0], &[1], &[1], &[0]), Vec::<f64>::new());
        }

        /// Correct even before the fix, via the identity fast path.
        #[test]
        fn at_chunk_aligned_offset_yields_no_elements() {
            assert_eq!(
                slice_nd(Vec::<f64>::new(), &[0], &[0], &[0]),
                Vec::<f64>::new()
            );
        }

        #[test]
        fn with_one_empty_axis_of_two_yields_no_elements() {
            assert_eq!(
                slice_nd(vec![0.0, 1.0, 2.0, 3.0], &[2, 2], &[0, 1], &[2, 0]),
                Vec::<f64>::new()
            );
        }

        #[test]
        fn with_every_axis_empty_yields_no_elements() {
            assert_eq!(
                slice_nd(vec![0.0, 1.0, 2.0, 3.0], &[2, 2], &[1, 1], &[0, 0]),
                Vec::<f64>::new()
            );
        }
    }

    /// A 0-D scalar has an *empty* `out_shape` — no axes — which is not
    /// the same as a zero-extent axis, and must still yield one element.
    /// It reaches the walk with `total_out == 1`, so it no longer depends
    /// on the identity fast path intercepting it first.
    #[test]
    fn scalar_read_still_yields_one_element() {
        assert_eq!(slice_nd(vec![7.0_f64], &[], &[], &[]), vec![7.0]);
    }

    /// Stride arithmetic beyond 2-D. The suite otherwise only reads 3-D
    /// arrays whole, which takes the identity fast path and never
    /// exercises the per-axis strides.
    #[test]
    fn slices_a_subrectangle_out_of_a_3d_buffer() {
        let buf: Vec<i32> = (0..24).collect();
        assert_eq!(
            slice_nd(buf, &[2, 3, 4], &[1, 1, 1], &[1, 2, 2]),
            vec![17, 18, 21, 22]
        );
    }

    /// Non-empty slicing must be unaffected by the rewrite.
    #[test]
    fn slices_a_subrectangle_out_of_a_2d_buffer() {
        // 3x4 row-major buffer, take rows 1..3 x cols 1..3.
        let buf: Vec<i32> = (0..12).collect();
        assert_eq!(slice_nd(buf, &[3, 4], &[1, 1], &[2, 2]), vec![5, 6, 9, 10]);
    }

    /// zarrs 0.22's `DataType` was an enum, so the compiler checked the
    /// dtype dispatch for us. In 0.23 it is a newtype over
    /// `Arc<dyn DataTypeTraits>`, and we now lean on zarrs's own V3
    /// aliases for the names — so this table is what guarantees an
    /// upstream alias rename shows up as a failing test rather than as a
    /// dtype string numpy can't parse. Needs no store and no Python, so
    /// it also covers the dtypes our pytest fixtures never create.
    #[test]
    fn maps_every_named_dtype_to_its_numpy_name() {
        for (dtype, expected) in [
            (data_type::bool(), "bool"),
            (data_type::int8(), "int8"),
            (data_type::int16(), "int16"),
            (data_type::int32(), "int32"),
            (data_type::int64(), "int64"),
            (data_type::uint8(), "uint8"),
            (data_type::uint16(), "uint16"),
            (data_type::uint32(), "uint32"),
            (data_type::uint64(), "uint64"),
            (data_type::float32(), "float32"),
            (data_type::float64(), "float64"),
            (data_type::complex64(), "complex64"),
            (data_type::complex128(), "complex128"),
            // `np.dtype("bytes")` is `|S0`, so the V3 name must not pass
            // through; both `bytes` and vlen `string` are *declared* `object`
            // (`string`'s values still read back as `StringDType` — see
            // `zarrs_dtype_read_numpy_str`).
            (data_type::bytes(), "object"),
            (data_type::string(), "object"),
            // Readable-name-only fallthrough that numpy does accept.
            (data_type::float16(), "float16"),
        ] {
            assert_eq!(zarrs_dtype_to_numpy_str(&dtype), expected);
        }
        // vlen `string` is *declared* `object` but *read* as numpy-2
        // `StringDType` ("T"); every other dtype's read name equals its
        // declared name.
        assert_eq!(zarrs_dtype_read_numpy_str(&data_type::string()), "T");
        assert_eq!(zarrs_dtype_read_numpy_str(&data_type::float64()), "float64");
    }

    /// The macro's marker→primitive pairings, pinned the same way. A
    /// swapped arm here would silently reinterpret the chunk bytes at a
    /// wrong element width, so it matters more than the name table.
    #[test]
    fn dispatch_macro_binds_the_matching_primitive_type() {
        fn primitive_for(dtype: &DataType) -> &'static str {
            for_each_supported_dtype!(dtype, T => {
                std::any::type_name::<T>()
            }, _other => {
                "unsupported"
            })
        }

        for (dtype, expected) in [
            (data_type::bool(), "bool"),
            (data_type::int8(), "i8"),
            (data_type::int16(), "i16"),
            (data_type::int32(), "i32"),
            (data_type::int64(), "i64"),
            (data_type::uint8(), "u8"),
            (data_type::uint16(), "u16"),
            (data_type::uint32(), "u32"),
            (data_type::uint64(), "u64"),
            (data_type::float32(), "f32"),
            (data_type::float64(), "f64"),
        ] {
            assert_eq!(primitive_for(&dtype), expected);
        }

        // complex64 is nameable but not readable — it must reach the
        // fallback arm, not silently pick a primitive.
        assert_eq!(primitive_for(&data_type::complex64()), "unsupported");
    }

    /// Pins the upstream behaviour that forced this mapper to exist. If
    /// a future zarrs makes `Display` numpy-safe, this fails and the
    /// workaround can go.
    #[test]
    fn zarrs_display_is_not_a_numpy_dtype_string() {
        assert_eq!(data_type::float64().to_string(), "float64 / <f8");
    }
}
