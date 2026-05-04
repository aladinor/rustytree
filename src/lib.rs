//! rustytree — Rust-backed xarray `DataTree` backend.
//!
//! `xr.open_datatree(engine="rustytree")` resolves through this module's
//! `open_datatree` `PyO3` function. Supported inputs today: local-filesystem
//! paths (icechunk + vanilla Zarr v3) and `s3://` URLs (vanilla Zarr v3
//! only — icechunk-on-S3 lands later). Only a single group's metadata is
//! returned; recursive multi-node walks and lazy chunk reads ship in
//! follow-up PRs.

use std::collections::{BTreeMap, HashMap};

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
mod url;
mod walk;

use crate::error::Result;
use crate::node::{NodeData, VarMeta};
use crate::url::StoreSpec;

/// Open a Zarr v3 store at `path` and return a metadata snapshot of the
/// group at `group` (default `"/"`).
///
/// `path` may be a local-filesystem path, a `file://` URL, or an `s3://`
/// URL. Local paths auto-detect between icechunk repositories and vanilla
/// Zarr v3 directories. `branch` selects the icechunk branch (default
/// `"main"`); silently ignored on non-icechunk paths.
///
/// `storage_options` accepts the standard fsspec/xarray-style keys for the
/// chosen scheme. For `s3://`: `region`, `endpoint`, `access_key_id`,
/// `secret_access_key`, `session_token`, `allow_http`, `skip_signature`
/// (alias `anon`). Unknown keys are rejected so typos surface immediately.
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
#[pyfunction]
#[pyo3(signature = (path, *, group = None, branch = None, storage_options = None))]
fn open_datatree<'py>(
    py: Python<'py>,
    path: &str,
    group: Option<&str>,
    branch: Option<&str>,
    storage_options: Option<&Bound<'_, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    let spec = url::parse_store_spec(path)?;
    let options = storage_options
        .map(parse_storage_options)
        .transpose()?
        .unwrap_or_default();
    let group_path = group.unwrap_or("/");
    let branch = branch.map(str::to_string);

    let node = py.detach(move || -> Result<NodeData> {
        runtime::handle().block_on(async {
            let store = match &spec {
                StoreSpec::Local(p) => store::build_local_store(p, branch.as_deref()).await?,
                StoreSpec::S3 { bucket, prefix } => {
                    store::build_vanilla_s3(bucket, prefix, &options)?
                }
            };
            walk::open_single(&store, group_path).await
        })
    })?;

    node_to_pydict(py, &node)
}

/// Convert a Python dict of `storage_options` into an owned
/// `HashMap<String, String>`.
///
/// Values are passed through Python's `str()` so the user can write
/// `{"anon": True}` or `{"timeout": 30}` instead of having to stringify
/// themselves — matches the fsspec / xarray convention. The downstream
/// per-key parsers (e.g. `apply_s3_option`) recognise `"True"` / `"False"`
/// alongside `"true"` / `"false"` so the round-trip is lossless.
fn parse_storage_options(dict: &Bound<'_, PyDict>) -> PyResult<HashMap<String, String>> {
    let mut out = HashMap::with_capacity(dict.len());
    for (key, value) in dict.iter() {
        let key: String = key.extract()?;
        let value: String = value.str()?.extract()?;
        out.insert(key, value);
    }
    Ok(out)
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
