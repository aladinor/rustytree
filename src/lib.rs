//! rustytree — Rust-backed xarray `DataTree` backend (scaffold + async runtime).
//!
//! `xr.open_datatree(engine="rustytree")` resolves through this module's
//! `open_datatree` `PyO3` function. The hierarchy walk + zarrs / icechunk
//! integration are not yet implemented; the function currently raises
//! `NotImplementedError`.

use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;

mod error;
mod runtime;

/// Returns `NotImplementedError` until the async hierarchy walk is wired up.
#[pyfunction]
fn open_datatree(url: &str) -> PyResult<Py<PyAny>> {
    let _ = url;
    Err(PyNotImplementedError::new_err(
        "rustytree.open_datatree: async hierarchy walk not yet implemented",
    ))
}

#[pymodule]
fn _rustytree(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(open_datatree, m)?)?;
    Ok(())
}
