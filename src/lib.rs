//! rustytree — Rust-backed xarray `DataTree` backend.
//!
//! `xr.open_datatree(engine="rustytree")` resolves through this module's
//! `open_datatree` `PyO3` function. Supported inputs today: local-filesystem
//! paths (icechunk + vanilla Zarr v3) and `s3://` URLs (icechunk + vanilla,
//! auto-detected via a HEAD probe on `<prefix>/repo`). The walk is
//! recursive and parallel; each variable carries a `ZarrsArrayHandle`
//! that defers chunk reads until xarray asks for them.

use std::collections::{BTreeMap, HashMap};

use numpy::PyArray1;
use pyo3::Bound;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};
use serde_json::Value as JsonValue;

mod array;
mod dtype_dispatch;
mod error;
mod icechunk_store;
mod node;
mod runtime;
mod store;
mod url;
mod walk;

use crate::array::ZarrsArrayHandle;
use crate::error::Result;
use crate::node::{EagerElements, NodeData, VarMeta};
use crate::url::StoreSpec;

/// Open a Zarr v3 store and return a metadata snapshot of every group
/// in the tree rooted at `group` (default `"/"`).
///
/// `source` is one of:
///   - **`bytes`**: msgpack-serialised `icechunk::session::Session`,
///     produced by `icechunk-python`'s `PySession.as_bytes()`. The
///     primary cross-extension handoff for icechunk: users construct
///     the Session in icechunk-python with whatever branch / creds /
///     cache config they want, then hand the bytes to us.
///   - **`str`**: a local-filesystem path, `file://` URL, or `s3://`
///     URL pointing at a **vanilla Zarr v3** store, OR a local-fs path
///     pointing at an icechunk repository (auto-detected via the
///     `<root>/repo` + `<root>/snapshots/` heuristic). Remote icechunk
///     URL dispatch (the old `s3://...icechunk_repo` path) is no
///     longer supported — pass session bytes instead.
///
/// `branch` selects the icechunk branch on local-fs icechunk paths
/// (default `"main"`); silently ignored on every other input.
///
/// `storage_options` accepts the fsspec/xarray-style keys for `s3://`
/// vanilla Zarr v3 inputs (`region`, `endpoint`, `access_key_id`,
/// `secret_access_key`, `session_token`, `allow_http`,
/// `skip_signature` / alias `anon`). Unknown keys are rejected.
/// Ignored for `bytes` inputs (icechunk owns its own credentials via
/// the user's session).
///
/// `max_concurrency` caps the number of concurrent group-discovery I/O
/// operations (default 32). Per-array fan-out within each group is
/// independent of this cap.
///
/// `recursive` (default `true`) controls whether the walk descends past
/// `group`. With `recursive=false` the result contains exactly one
/// node — the requested group itself, with its arrays. The Python
/// `open_dataset` path auto-selects `recursive=false` for literal-group
/// opens so a single-Dataset request doesn't pay for siblings it
/// discards. Glob `group=` patterns require `recursive=true`.
///
/// The returned dict is keyed by absolute group path:
///
/// ```text
/// {
///     "/":          {"path", "attrs", "vars": [...]},
///     "/group_a":   {"path", "attrs", "vars": [...]},
///     "/group_a/x": {"path", "attrs", "vars": [...]},
///     ...
/// }
/// ```
#[pyfunction]
#[pyo3(signature = (source, *, group = None, branch = None, storage_options = None, max_concurrency = None, recursive = None))]
fn open_datatree<'py>(
    py: Python<'py>,
    source: &Bound<'_, PyAny>,
    group: Option<&str>,
    branch: Option<&str>,
    storage_options: Option<&Bound<'_, PyDict>>,
    max_concurrency: Option<usize>,
    recursive: Option<bool>,
) -> PyResult<Bound<'py, PyDict>> {
    let group_path = group.unwrap_or("/").to_string();
    let max_concurrency_val = max_concurrency;
    let recursive_val = recursive.unwrap_or(true);

    // Branch on input type: `bytes` = icechunk session handoff;
    // `str`/`Path` = local path or vanilla Zarr v3 URL.
    if let Ok(bytes) = source.cast::<PyBytes>() {
        let session_bytes = bytes.as_bytes().to_vec();
        let nodes = py.detach(move || -> Result<Vec<NodeData>> {
            runtime::handle().block_on(async {
                let bundle = icechunk_store::bundle_from_session_bytes(&session_bytes)?;
                walk::walk_recursive(
                    store::WalkSource::Icechunk(bundle),
                    &group_path,
                    max_concurrency_val,
                    recursive_val,
                )
                .await
            })
        })?;
        return nodes_to_pydict(py, &nodes);
    }

    // Fall through: string-shaped input (URL or path).
    let path: String = source.extract().map_err(|_| {
        PyValueError::new_err(
            "rustytree.open_datatree: `source` must be either `bytes` (icechunk session) or \
             `str`/`Path` (URL or local path); got an object of an unsupported type",
        )
    })?;
    let spec = url::parse_store_spec(&path)?;
    let options = storage_options
        .map(parse_storage_options)
        .transpose()?
        .unwrap_or_default();
    let branch = branch.map(str::to_string);

    let nodes = py.detach(move || -> Result<Vec<NodeData>> {
        runtime::handle().block_on(async {
            let walk_source = match &spec {
                StoreSpec::Local(p) => store::build_local_store(p, branch.as_deref()).await?,
                StoreSpec::S3 { bucket, prefix } => {
                    // Vanilla Zarr v3 only. icechunk-on-S3 used to live
                    // here as an auto-detected URL dispatch; Phase 7
                    // dropped it — users construct the icechunk Session
                    // themselves and pass the session bytes instead.
                    store::WalkSource::Vanilla(store::build_vanilla_s3(bucket, prefix, &options)?)
                }
            };
            walk::walk_recursive(walk_source, &group_path, max_concurrency_val, recursive_val)
                .await
        })
    })?;

    nodes_to_pydict(py, &nodes)
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

/// Marshal a list of `NodeData` into a Python dict keyed by absolute path.
fn nodes_to_pydict<'py>(py: Python<'py>, nodes: &[NodeData]) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for node in nodes {
        dict.set_item(&node.path, node_to_pydict(py, node)?)?;
    }
    Ok(dict)
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
    let handle = ZarrsArrayHandle::new(var.array.clone(), runtime::handle().handle().clone());
    dict.set_item("handle", Py::new(py, handle)?)?;
    if let Some(eager) = &var.eager {
        let arr = eager_to_pyarray(py, eager, &var.shape)?;
        dict.set_item("data", arr)?;
    }
    Ok(dict)
}

/// Marshal an `EagerElements` into a Python numpy ndarray reshaped to
/// the variable's full shape. Called from `var_to_pydict` when the
/// walk's Phase C pre-fetched the array's contents.
fn eager_to_pyarray<'py>(
    py: Python<'py>,
    eager: &EagerElements,
    shape: &[u64],
) -> PyResult<Bound<'py, PyAny>> {
    // 1-D `PyArray1::from_slice` materialises the data; numpy's
    // `reshape` then gives us the natural N-D view that xarray expects
    // when constructing a `Variable` with this as `data`.
    let flat = match eager {
        EagerElements::Bool(v) => PyArray1::from_slice(py, v).into_any(),
        EagerElements::I8(v) => PyArray1::from_slice(py, v).into_any(),
        EagerElements::I16(v) => PyArray1::from_slice(py, v).into_any(),
        EagerElements::I32(v) => PyArray1::from_slice(py, v).into_any(),
        EagerElements::I64(v) => PyArray1::from_slice(py, v).into_any(),
        EagerElements::U8(v) => PyArray1::from_slice(py, v).into_any(),
        EagerElements::U16(v) => PyArray1::from_slice(py, v).into_any(),
        EagerElements::U32(v) => PyArray1::from_slice(py, v).into_any(),
        EagerElements::U64(v) => PyArray1::from_slice(py, v).into_any(),
        EagerElements::F32(v) => PyArray1::from_slice(py, v).into_any(),
        EagerElements::F64(v) => PyArray1::from_slice(py, v).into_any(),
    };
    let shape_tuple = PyTuple::new(py, shape)?;
    flat.call_method1("reshape", (shape_tuple,))
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
    m.add_class::<ZarrsArrayHandle>()?;
    Ok(())
}
