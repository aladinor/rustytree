//! Async metadata walk for Zarr v3 groups.
//!
//! Two entry points:
//!   - [`open_single`] reads metadata for one group only.
//!   - [`walk_recursive`] discovers descendant groups starting from a root,
//!     then opens every group's metadata in parallel.
//!
//! No chunk data is fetched at this stage; the lazy chunk-read path lands
//! alongside the `BackendArray` adapter in a later PR.

use std::collections::BTreeMap;
use std::sync::Arc;

use futures::future::try_join_all;
use tokio::sync::Semaphore;
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
    // with `try_join_all`. Bounded by `array_paths.len()`, so no semaphore
    // needed at this level (the recursive walk gates the total in-flight
    // work via its own semaphore).
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

/// Recursively walk every descendant group starting at `root_path` and
/// return one [`NodeData`] per group, root first.
///
/// Discovery and metadata-fetch are split into two phases:
///   1. Walk the group tree from `root_path` to discover every descendant
///      group path. The walk is async and parallel; concurrent
///      `Group::async_open` calls are gated by `max_concurrency`.
///   2. Fan out [`open_single`] across every discovered path via
///      `try_join_all`. Each per-group call internally fans out array
///      opens, so the total parallelism is approximately
///      `min(num_groups * max_arrays_per_group, OS network limit)`,
///      with `max_concurrency` capping the per-call concurrency at the
///      group level.
///
/// `max_concurrency` defaults to **32** when `None` — empirically a good
/// balance for object-storage backends that pipeline ~16 in-flight
/// requests per HTTP client. Override with `max_concurrency=` from the
/// Python boundary.
pub(crate) async fn walk_recursive(
    store: AsyncReadableListableStorage,
    root_path: &str,
    max_concurrency: Option<usize>,
) -> Result<Vec<NodeData>> {
    let semaphore = Arc::new(Semaphore::new(max_concurrency.unwrap_or(32).max(1)));
    let mut paths = Vec::new();
    discover_paths(&store, root_path.to_string(), &semaphore, &mut paths).await?;

    // All groups' metadata in parallel. Each `open_single` itself fans
    // out per-array work via `try_join_all`, so this gives true two-level
    // concurrency.
    let nodes = try_join_all(paths.iter().map(|p| open_single(&store, p))).await?;
    Ok(nodes)
}

/// Depth-first descent that records every group path beneath `path`
/// (inclusive). Each level's `Group::async_open` + child-listing happens
/// concurrently with siblings; the semaphore caps total in-flight work.
///
/// The result is in pre-order traversal order so that `from_dict`-style
/// consumers see parents before children — matters for xarray's
/// `DataTree.from_dict` which auto-creates intermediate nodes but works
/// best with a topologically sorted input.
///
/// The semaphore permit is released **before** recursing into children.
/// Holding a permit across the recursive `try_join_all` would deadlock at
/// any `max_concurrency` smaller than the tree depth — children fight the
/// parent for permits the parent has no intent to drop until children
/// return. The acquire/release window covers exactly the I/O that needs
/// bounding (`Group::async_open` + `async_child_group_paths`).
fn discover_paths<'a>(
    store: &'a AsyncReadableListableStorage,
    path: String,
    semaphore: &'a Arc<Semaphore>,
    out: &'a mut Vec<String>,
) -> futures::future::BoxFuture<'a, Result<()>> {
    Box::pin(async move {
        let child_paths = {
            let _permit = semaphore
                .acquire()
                .await
                .map_err(|_| RustytreeError::Other("walk semaphore closed unexpectedly".into()))?;

            let group = Group::async_open(store.clone(), &path)
                .await
                .map_err(|err| {
                    RustytreeError::Other(format!("failed to open group {path} during walk: {err}"))
                })?;

            group.async_child_group_paths().await.map_err(|err| {
                RustytreeError::Other(format!("failed to list child groups in {path}: {err}"))
            })?
        };
        // permit dropped here; safe to recurse.

        out.push(path);

        let child_strs: Vec<String> = child_paths.into_iter().map(|p| p.to_string()).collect();
        let mut sub_results: Vec<Vec<String>> = try_join_all(child_strs.into_iter().map(|c| {
            let store = store.clone();
            let sem = Arc::clone(semaphore);
            async move {
                let mut sub = Vec::new();
                discover_paths(&store, c, &sem, &mut sub).await?;
                Ok::<_, RustytreeError>(sub)
            }
        }))
        .await?;
        for sub in sub_results.drain(..) {
            out.extend(sub);
        }
        Ok(())
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
