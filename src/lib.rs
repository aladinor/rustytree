//! rustytree — Rust-backed xarray `DataTree` backend.
//!
//! `xr.open_datatree(engine="rustytree")` resolves through this module's
//! `open_datatree` `PyO3` function. Today only local-filesystem vanilla
//! Zarr v3 stores are supported, and only a single group's metadata is
//! returned. Recursive multi-node walks, icechunk dispatch, and lazy chunk
//! reads land in follow-up PRs.

use std::collections::BTreeMap;
use std::path::PathBuf;

use pyo3::Bound;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde_json::Value as JsonValue;

mod error;
mod icechunk_store;
mod node;
mod runtime;
mod store;
mod walk;

use crate::error::Result;
use crate::node::{NodeData, VarMeta};

/// Open a Zarr v3 store at `path` and return a metadata snapshot of the
/// group at `group` (default `"/"`).
///
/// `path` may point at either a vanilla Zarr v3 directory or an icechunk
/// repository on local filesystem; rustytree auto-detects the layout.
/// `branch` selects the icechunk branch (default `"main"`); it's silently
/// ignored on the vanilla path.
///
/// The returned dict has:
///
/// ```text
/// {
///     "path":   <group path>,
///     "attrs":  <dict of group attrs>,
///     "vars":   [
///         {"name", "dims", "dtype", "shape", "attrs"},
///         ...
///     ],
/// }
/// ```
///
/// Remote stores (`s3://`, `gs://`, ...) are not supported in this build;
/// passing one yields `NotImplementedError`.
#[pyfunction]
#[pyo3(signature = (path, *, group = None, branch = None))]
fn open_datatree<'py>(
    py: Python<'py>,
    path: &str,
    group: Option<&str>,
    branch: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let path = parse_local_path(path)?;
    let group_path = group.unwrap_or("/");
    let branch = branch.map(str::to_string);

    let node = py.detach(move || -> Result<NodeData> {
        runtime::handle().block_on(async {
            let store = store::build_local_store(&path, branch.as_deref()).await?;
            walk::open_single(&store, group_path).await
        })
    })?;

    node_to_pydict(py, &node)
}

/// Parse the `path` argument. Today: only local-filesystem paths or
/// `file://` URLs. Remote URL schemes are rejected with
/// `NotImplementedError` (the follow-up PR adds remote `object_store`
/// dispatch).
fn parse_local_path(input: &str) -> PyResult<PathBuf> {
    if let Some(rest) = input.strip_prefix("file://") {
        return Ok(PathBuf::from(rest));
    }
    if input.contains("://") {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
            "rustytree.open_datatree: only local-filesystem paths are supported in this build; got {input}"
        )));
    }
    Ok(PathBuf::from(input))
}

/// Marshal a `NodeData` into a Python dict using xarray-friendly key names.
fn node_to_pydict<'py>(py: Python<'py>, node: &NodeData) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("path", &node.path)?;
    dict.set_item("attrs", attrs_to_pydict(py, &node.attrs)?)?;

    let vars = PyList::empty(py);
    for var in &node.vars {
        vars.append(var_to_pydict(py, var)?)?;
    }
    dict.set_item("vars", vars)?;
    Ok(dict)
}

fn var_to_pydict<'py>(py: Python<'py>, var: &VarMeta) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("name", &var.name)?;
    dict.set_item("dims", &var.dims)?;
    dict.set_item("dtype", &var.dtype)?;
    dict.set_item("shape", &var.shape)?;
    dict.set_item("attrs", attrs_to_pydict(py, &var.attrs)?)?;
    Ok(dict)
}

fn attrs_to_pydict<'py>(
    py: Python<'py>,
    attrs: &BTreeMap<String, JsonValue>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (k, v) in attrs {
        dict.set_item(k, json_to_py(py, v)?)?;
    }
    Ok(dict)
}

/// Convert a `serde_json::Value` to a Python object. Numbers that don't fit
/// `i64` fall back to `f64`; arbitrary-precision integers aren't preserved
/// (none of Zarr v3's metadata uses them today).
fn json_to_py<'py>(py: Python<'py>, value: &JsonValue) -> PyResult<Bound<'py, PyAny>> {
    match value {
        JsonValue::Null => Ok(py.None().into_bound(py)),
        JsonValue::Bool(b) => b.into_bound_py_any(py),
        JsonValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into_bound_py_any(py)
            } else if let Some(u) = n.as_u64() {
                u.into_bound_py_any(py)
            } else if let Some(f) = n.as_f64() {
                f.into_bound_py_any(py)
            } else {
                Err(PyValueError::new_err(format!(
                    "unrepresentable JSON number: {n}"
                )))
            }
        }
        JsonValue::String(s) => s.into_bound_py_any(py),
        JsonValue::Array(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(json_to_py(py, item)?)?;
            }
            Ok(list.into_any())
        }
        JsonValue::Object(map) => {
            let dict = PyDict::new(py);
            for (k, v) in map {
                dict.set_item(k, json_to_py(py, v)?)?;
            }
            Ok(dict.into_any())
        }
    }
}

#[pymodule]
fn _rustytree(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(open_datatree, m)?)?;
    Ok(())
}
