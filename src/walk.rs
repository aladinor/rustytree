//! Async metadata walk for a single Zarr v3 group.
//!
//! Opens the group at `path`, lists its child arrays, opens each, and
//! returns a `NodeData` snapshot — no chunk data is fetched.
//!
//! Recursive multi-node walks land in a follow-up PR (this one is single-
//! group only).

use std::collections::BTreeMap;

use futures::future::try_join_all;
use zarrs::array::Array;
use zarrs::group::Group;
use zarrs_storage::AsyncReadableListableStorage;

use crate::error::{Result, RustytreeError};
use crate::node::{NodeData, VarMeta};

/// Open a single group and capture its metadata + that of its child arrays.
pub(crate) async fn open_single(
    store: &AsyncReadableListableStorage,
    path: &str,
) -> Result<NodeData> {
    let group = Group::async_open(store.clone(), path)
        .await
        .map_err(|err| RustytreeError::Other(format!("failed to open group {path}: {err}")))?;

    let array_paths = group
        .async_child_array_paths()
        .await
        .map_err(|err| RustytreeError::Other(format!("failed to list arrays in {path}: {err}")))?;

    // Opening each array's metadata is independent of the others — fan out
    // with `try_join_all` so the eventual recursive walk doesn't have to
    // retrofit parallelism into this loop. Bounded by `array_paths.len()`,
    // so no semaphore needed at this level.
    let vars = try_join_all(
        array_paths
            .into_iter()
            .map(|array_path| open_array_meta(store, array_path.to_string())),
    )
    .await?;

    Ok(NodeData {
        path: path.to_string(),
        attrs: clone_attrs(group.attributes()),
        vars,
    })
}

/// Clone a `serde_json::Map` of attributes into a `BTreeMap`.
///
/// We use `BTreeMap` (not `HashMap`) so callers — and the Python dict the
/// `lib.rs` marshalling produces — see attrs in deterministic alphabetical
/// order rather than `HashMap`'s randomised iteration order.
fn clone_attrs(
    map: &serde_json::Map<String, serde_json::Value>,
) -> BTreeMap<String, serde_json::Value> {
    map.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
}

/// Open one array and project its metadata into a `VarMeta`.
async fn open_array_meta(
    store: &AsyncReadableListableStorage,
    array_path: String,
) -> Result<VarMeta> {
    let array = Array::async_open(store.clone(), &array_path)
        .await
        .map_err(|err| {
            RustytreeError::Other(format!("failed to open array {array_path}: {err}"))
        })?;

    let name = array_path
        .rsplit_once('/')
        .map_or_else(|| array_path.clone(), |(_, last)| last.to_string());

    let dims = array.dimension_names().as_ref().map_or_else(
        || {
            (0..array.shape().len())
                .map(|i| format!("dim_{i}"))
                .collect()
        },
        |names| {
            // `DimensionName` is `Option<String>` — replace `None` with a
            // synthetic `dim_<i>` placeholder so xarray always has a name.
            names
                .iter()
                .enumerate()
                .map(|(i, n)| n.clone().unwrap_or_else(|| format!("dim_{i}")))
                .collect()
        },
    );

    Ok(VarMeta {
        name,
        dims,
        dtype: format!("{}", array.data_type()),
        shape: array.shape().to_vec(),
        attrs: clone_attrs(array.attributes()),
    })
}
