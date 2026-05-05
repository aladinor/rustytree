//! `ZarrsArrayHandle`: `PyO3` wrapper around an opened `zarrs::Array` that
//! defers chunk reads until Python asks for them.
//!
//! The walk eagerly opens every array (one `Array::async_open` per array,
//! which reads `<path>/zarr.json` from the store). That gives us shape,
//! dtype, dims, and attributes without having to fetch any chunks. The
//! resulting `Array` is then handed out wrapped in a `ZarrsArrayHandle`
//! so xarray can call back through `read_subset` whenever it needs data.
//!
//! `read_subset` runs `runtime.block_on(array.async_retrieve_array_subset_elements::<T>(...))`
//! with the GIL released (`Python::detach`) so concurrent loads from a
//! Python thread pool overlap on the network rather than serialising
//! through the GIL.

use std::sync::Arc;

use numpy::PyArray1;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyIndexError, PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use tokio::runtime::Handle;
use zarrs::array::{Array, DataType};
use zarrs::array_subset::ArraySubset;
use zarrs_storage::AsyncReadableListableStorageTraits;

use crate::dtype_dispatch::for_each_supported_dtype;

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
}

impl ZarrsArrayHandle {
    pub(crate) fn new(
        array: Arc<Array<dyn AsyncReadableListableStorageTraits>>,
        runtime: Handle,
    ) -> Self {
        Self { array, runtime }
    }
}

#[pymethods]
impl ZarrsArrayHandle {
    /// The array's shape as a tuple of `int`.
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.array.shape())
    }

    /// The array's dtype as a `NumPy` dtype string (e.g. `"float64"`,
    /// `"int8"`). Falls back to `format!("{:?}")` for less common
    /// dtypes, which `numpy.dtype(...)` will raise on — those need
    /// explicit support added here when we encounter them.
    #[getter]
    fn dtype(&self) -> String {
        zarrs_dtype_to_numpy_str(self.array.data_type())
    }

    /// Read a hyperrectangular slab of the array.
    ///
    /// `ranges` is a list of `(start, stop)` tuples — one per
    /// dimension, exclusive stop, matching Python `slice` semantics.
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
        let dtype = self.array.data_type().clone();
        let array = self.array.clone();
        let runtime = self.runtime.clone();

        for_each_supported_dtype!(dtype, T => {
            let elements: Vec<T> = py.detach(|| -> PyResult<Vec<T>> {
                runtime
                    .block_on(array.async_retrieve_array_subset_elements::<T>(&subset))
                    .map_err(|err| PyValueError::new_err(format!("zarrs read failed: {err}")))
            })?;
            let sliced = slice_nd(
                elements,
                &aligned_shape,
                &request_offsets_in_aligned,
                &request_shape,
            );
            PyArray1::from_vec(py, sliced).into_bound_py_any(py)
        }, other => {
            Err(PyNotImplementedError::new_err(format!(
                "rustytree: dtype {other:?} is not yet supported by ZarrsArrayHandle.read_subset; \
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
fn slice_nd<T: Copy>(
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
    // Walk the output shape in row-major order, computing the source
    // index for each destination element.
    let mut idx = vec![0_usize; n];
    loop {
        let mut src = 0_usize;
        for i in 0..n {
            src += (offsets[i] + idx[i]) * src_strides[i];
        }
        out.push(elements[src]);
        // Increment the multidim index in row-major (last-axis-first)
        // order. Loop exits when the carry rolls past axis 0.
        let mut axis = n;
        loop {
            if axis == 0 {
                return out;
            }
            axis -= 1;
            idx[axis] += 1;
            if idx[axis] < out_shape[axis] {
                break;
            }
            idx[axis] = 0;
        }
    }
}

/// Translate `zarrs::array::DataType` to the canonical `NumPy` dtype
/// string. Anything unsupported returns `format!("{:?}")` so callers
/// see a clear error when they try to `numpy.dtype(...)` it.
pub(crate) fn zarrs_dtype_to_numpy_str(dtype: &DataType) -> String {
    match dtype {
        DataType::Bool => "bool".into(),
        DataType::Int8 => "int8".into(),
        DataType::Int16 => "int16".into(),
        DataType::Int32 => "int32".into(),
        DataType::Int64 => "int64".into(),
        DataType::UInt8 => "uint8".into(),
        DataType::UInt16 => "uint16".into(),
        DataType::UInt32 => "uint32".into(),
        DataType::UInt64 => "uint64".into(),
        DataType::Float32 => "float32".into(),
        DataType::Float64 => "float64".into(),
        DataType::Complex64 => "complex64".into(),
        DataType::Complex128 => "complex128".into(),
        other => format!("{other:?}"),
    }
}
