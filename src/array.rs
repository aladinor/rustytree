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
        let std_ranges: Vec<std::ops::Range<u64>> = ranges.iter().map(|(s, e)| *s..*e).collect();
        let subset = ArraySubset::new_with_ranges(&std_ranges);

        // Dispatch on dtype: each branch decodes into the matching
        // primitive type and hands a 1-D `NumPy` array back to Python.
        // The Python adapter reshapes; doing it here would force every
        // dtype to materialise an ndarray crate type, which costs an
        // extra dependency for no benefit.
        let dtype = self.array.data_type().clone();
        let array = self.array.clone();
        let runtime = self.runtime.clone();

        macro_rules! read_as {
            ($ty:ty) => {{
                let elements: Vec<$ty> = py.detach(|| -> PyResult<Vec<$ty>> {
                    runtime
                        .block_on(array.async_retrieve_array_subset_elements::<$ty>(&subset))
                        .map_err(|err| PyValueError::new_err(format!("zarrs read failed: {err}")))
                })?;
                PyArray1::from_vec(py, elements).into_bound_py_any(py)
            }};
        }

        match dtype {
            DataType::Bool => read_as!(bool),
            DataType::Int8 => read_as!(i8),
            DataType::Int16 => read_as!(i16),
            DataType::Int32 => read_as!(i32),
            DataType::Int64 => read_as!(i64),
            DataType::UInt8 => read_as!(u8),
            DataType::UInt16 => read_as!(u16),
            DataType::UInt32 => read_as!(u32),
            DataType::UInt64 => read_as!(u64),
            DataType::Float32 => read_as!(f32),
            DataType::Float64 => read_as!(f64),
            other => Err(PyNotImplementedError::new_err(format!(
                "rustytree: dtype {other:?} is not yet supported by ZarrsArrayHandle.read_subset; \
                 supported today: bool, int{{8,16,32,64}}, uint{{8,16,32,64}}, float{{32,64}}"
            ))),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ZarrsArrayHandle(shape={:?}, dtype={})",
            self.array.shape(),
            zarrs_dtype_to_numpy_str(self.array.data_type())
        )
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
