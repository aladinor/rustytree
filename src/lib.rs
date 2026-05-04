//! rustytree — Rust-backed xarray `DataTree` backend (Phase 1 scaffold).
//!
//! Phase 1 only proves that `xr.open_datatree(engine="rustytree")` resolves
//! through the entry point and returns a controlled `NotImplementedError`.
//! The async walk + zarrs integration land in Phase 2.

use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;

/// Phase 2 entry point. Returns `NotImplementedError` until the async walk lands.
#[pyfunction]
fn open_datatree(url: &str) -> PyResult<PyObject> {
    let _ = url;
    Err(PyNotImplementedError::new_err(
        "rustytree.open_datatree lands in Phase 2",
    ))
}

#[pymodule]
fn _rustytree(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(open_datatree, m)?)?;
    Ok(())
}
