//! Async metadata walk for Zarr v3 groups.
//!
//! Three entry points:
//!   - [`open_single`] reads metadata for one group only.
//!   - [`walk_recursive`] is the public dispatcher. For icechunk inputs
//!     it routes to [`walk_icechunk_session_snapshot`] (the in-memory
//!     snapshot fast-path); for vanilla Zarr v3 it routes to the
//!     generic [`walk_via_zarrs`] which does per-node `Group::async_open`
//!     + `Array::async_open` round-trips.
//!   - Phase C eager-fetch runs after either path produces `NodeData`.
//!
//! No chunk data is fetched at the metadata stage; the lazy chunk-read
//! path lands alongside the `BackendArray` adapter.

use std::collections::BTreeMap;
use std::sync::Arc;

use futures::future::try_join_all;
use icechunk::format::Path as IcePath;
use icechunk::format::snapshot::NodeData as IceNodeData;
use icechunk::session::Session;
use tokio::sync::{RwLock, Semaphore};
use zarrs::array::{Array, ArrayMetadata};
use zarrs::array_subset::ArraySubset;
use zarrs::group::{Group, GroupMetadata};
use zarrs_storage::{AsyncReadableListableStorage, AsyncReadableListableStorageTraits};

use crate::dtype_dispatch::for_each_supported_dtype;
use crate::error::{Result, RustytreeError};
use crate::node::{EagerElements, NodeData, VarMeta};
use crate::store::WalkSource;

/// Cap on element count for eager pre-fetching. Coords / time-likes
/// larger than this stay lazy — at 1 M elements an int64 coord is 8 MB,
/// already a meaningful cold-cache cost we don't want to pay
/// unconditionally during open. Tunable later if profiles say so.
const EAGER_FETCH_MAX_ELEMENTS: u64 = 1 << 20;

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
///
/// `recursive` (default `true`) controls whether we walk past
/// `root_path`. With `recursive=false` we return exactly one
/// [`NodeData`] — the requested group plus its arrays.
pub(crate) async fn walk_recursive(
    source: WalkSource,
    root_path: &str,
    max_concurrency: Option<usize>,
    recursive: bool,
) -> Result<Vec<NodeData>> {
    let semaphore = Arc::new(Semaphore::new(max_concurrency.unwrap_or(32).max(1)));
    let mut nodes = match source {
        WalkSource::Icechunk(bundle) => {
            // Snapshot fast-path: read group + array metadata directly
            // from icechunk's already-fetched in-memory snapshot tree
            // instead of issuing per-node `Group::async_open` /
            // `Array::async_open` round-trips through the session.
            walk_icechunk_session_snapshot(&bundle.session, &bundle.store, root_path, recursive)
                .await?
        }
        WalkSource::Vanilla(store) => {
            walk_via_zarrs(&store, root_path, &semaphore, recursive).await?
        }
    };

    // Phase C: eager-fetch values for vars that xarray's per-node CF
    // decoders + Index construction will read anyway (1-D self-named
    // dim coords + CF time-likes). Doing it here in parallel turns
    // hundreds of serialised lazy-backend reads into one bounded
    // `try_join_all` against the same tokio runtime. See
    // `should_eager_fetch` for the predicate.
    eager_phase(&semaphore, &mut nodes).await?;

    Ok(nodes)
}

/// Generic zarrs-based walker for vanilla Zarr v3 stores. Discovers
/// every descendant group via async `list_dir` + `Group::async_open`,
/// then fans out per-group metadata fetches via `open_single`. With
/// `recursive=false` the discovery step is skipped: only `root_path`
/// itself is opened.
async fn walk_via_zarrs(
    store: &AsyncReadableListableStorage,
    root_path: &str,
    semaphore: &Arc<Semaphore>,
    recursive: bool,
) -> Result<Vec<NodeData>> {
    let paths: Vec<String> = if recursive {
        let mut paths = Vec::new();
        discover_paths(store, root_path.to_string(), semaphore, &mut paths).await?;
        paths
    } else {
        vec![root_path.to_string()]
    };

    // All groups' metadata in parallel. Each `open_single` itself fans
    // out per-array work via `try_join_all`, so this gives true two-level
    // concurrency.
    try_join_all(paths.iter().map(|p| open_single(store, p))).await
}

/// Snapshot-fast-path walker for icechunk sessions.
///
/// `Session::list_nodes(parent)` enumerates every descendant group +
/// array inline from the already-fetched snapshot manifest. Each
/// `NodeSnapshot` carries `user_data: Bytes` — the on-disk `zarr.json`
/// payload — which we deserialise into zarrs's `ArrayMetadata` /
/// `GroupMetadata` types and then construct an `Arc<Array>` /
/// `Group` from via `new_with_metadata` (synchronous; no I/O).
///
/// This bypasses zarrs's `Group::async_open` + `Array::async_open`
/// round-trips through the session for every node. On
/// `s3://nexrad-arco/KLOT` (107 groups + 1300 arrays) the generic
/// walker pays ~1.6 s wall on this work; the snapshot walker reduces
/// it to <100 ms wall (the snapshot is fully in-memory after
/// `Repository::open` user-side).
async fn walk_icechunk_session_snapshot(
    session: &Arc<RwLock<Session>>,
    store: &AsyncReadableListableStorage,
    root_path: &str,
    recursive: bool,
) -> Result<Vec<NodeData>> {
    let parent = IcePath::new(root_path).map_err(|err| {
        RustytreeError::InvalidInput(format!("icechunk: invalid root path {root_path:?}: {err}"))
    })?;

    // Hold the read lock just long enough to gather the node list. The
    // iterator borrows from the session, so we collect into an owned
    // Vec before releasing the guard.
    let session_guard = session.read().await;
    let nodes_iter = session_guard.list_nodes(&parent).await.map_err(|err| {
        RustytreeError::Other(format!("icechunk: list_nodes({root_path}) failed: {err}"))
    })?;
    let snapshots: Vec<icechunk::format::snapshot::NodeSnapshot> = nodes_iter
        .collect::<icechunk::session::SessionResult<Vec<_>>>()
        .map_err(|err| {
            RustytreeError::Other(format!("icechunk: snapshot enumeration failed: {err}"))
        })?;
    drop(session_guard);

    // Two passes: build a map of group-path -> NodeData (with empty
    // vars list), then attach each Array node as a VarMeta on its
    // parent group. The second pass needs the first's keys, so we
    // can't fuse them.
    //
    // When `recursive` is false we keep only the root group itself
    // and arrays whose immediate parent is `root_path`. icechunk's
    // `list_nodes` always returns the full subtree (it's already in
    // memory), so we filter at materialisation time — the win is
    // skipping the CPU-bound `Array::new_with_metadata` for arrays
    // we'd otherwise drop.
    let root_normalized = normalize_path(root_path);
    let mut groups: BTreeMap<String, NodeData> = BTreeMap::new();
    let mut arrays: Vec<icechunk::format::snapshot::NodeSnapshot> = Vec::new();
    for snap in snapshots {
        match &snap.node_data {
            IceNodeData::Group => {
                let path = ice_path_to_string(&snap.path);
                if !recursive && path != root_normalized {
                    continue;
                }
                let attrs = parse_group_attrs(&snap.user_data, &path)?;
                groups.insert(
                    path.clone(),
                    NodeData {
                        path,
                        attrs,
                        vars: Vec::new(),
                    },
                );
            }
            IceNodeData::Array { .. } => {
                if !recursive {
                    let array_path = ice_path_to_string(&snap.path);
                    if parent_of(&array_path) != root_normalized {
                        continue;
                    }
                }
                arrays.push(snap);
            }
        }
    }

    // Build VarMetas in parallel, batched by parent group.
    // `Array::new_with_metadata` is synchronous CPU work
    // (codec/chunk-grid construction); for 1300+ arrays in a
    // multi-VCP radar tree it's the dominant cost of the fast-path.
    // Spawning one blocking task per array drowned in scheduling
    // overhead (μs of work per task vs μs of spawn cost) — batching
    // by group amortises the spawn while still letting up to N_cores
    // groups build concurrently.
    let mut by_parent: BTreeMap<String, Vec<icechunk::format::snapshot::NodeSnapshot>> =
        BTreeMap::new();
    for snap in arrays {
        let array_path = ice_path_to_string(&snap.path);
        by_parent
            .entry(parent_of(&array_path))
            .or_default()
            .push(snap);
    }

    let group_jobs: Vec<(String, Vec<VarMeta>)> =
        try_join_all(by_parent.into_iter().map(|(parent_path, snaps)| {
            let store = store.clone();
            tokio::task::spawn_blocking(move || -> Result<(String, Vec<VarMeta>)> {
                let mut vars = Vec::with_capacity(snaps.len());
                for snap in snaps {
                    let array_path = ice_path_to_string(&snap.path);
                    vars.push(build_var_meta_from_snapshot(
                        &snap,
                        &array_path,
                        store.clone(),
                    )?);
                }
                Ok((parent_path, vars))
            })
        }))
        .await
        .map_err(|err| RustytreeError::Other(format!("icechunk: array build join failed: {err}")))?
        .into_iter()
        .collect::<Result<Vec<_>>>()?;

    for (parent_path, vars) in group_jobs {
        if let Some(node) = groups.get_mut(&parent_path) {
            node.vars = vars;
        } else if !vars.is_empty() {
            return Err(RustytreeError::Other(format!(
                "icechunk: arrays under {parent_path} have no parent group"
            )));
        }
    }

    // Pre-order traversal: BTreeMap sorts by path, which gives us
    // root-first ordering compatible with `xr.DataTree.from_dict`'s
    // expectation that parents land before children.
    Ok(groups.into_values().collect())
}

/// Convert icechunk's `Path` to a `/`-rooted string. icechunk's
/// `Display` on `Path` yields the same shape (`"/foo/bar"`).
fn ice_path_to_string(p: &IcePath) -> String {
    p.to_string()
}

/// Strip a trailing slash from a non-root path so two equivalent shapes
/// (`/foo` and `/foo/`) compare equal. Empty input is treated as root.
fn normalize_path(path: &str) -> String {
    if path.is_empty() {
        return "/".to_string();
    }
    if path.len() > 1 && path.ends_with('/') {
        path.trim_end_matches('/').to_string()
    } else {
        path.to_string()
    }
}

/// Return the parent directory of an absolute Zarr path. The root's
/// parent is the root itself.
fn parent_of(path: &str) -> String {
    match path.rsplit_once('/') {
        Some(("", _)) | None => "/".to_string(),
        Some((parent, _)) => parent.to_string(),
    }
}

/// Parse a group's `user_data` bytes (the `zarr.json` payload) into a
/// `GroupMetadata` and project the user attrs into a `BTreeMap`.
fn parse_group_attrs(user_data: &[u8], path: &str) -> Result<BTreeMap<String, serde_json::Value>> {
    if user_data.is_empty() {
        // Root or implicit groups have no user-data; treat as empty
        // attrs rather than failing.
        return Ok(BTreeMap::new());
    }
    let metadata: GroupMetadata = serde_json::from_slice(user_data).map_err(|err| {
        RustytreeError::Other(format!(
            "icechunk: failed to parse group zarr.json at {path}: {err}"
        ))
    })?;
    let attrs_map = match &metadata {
        GroupMetadata::V3(v3) => &v3.attributes,
        GroupMetadata::V2(v2) => &v2.attributes,
    };
    Ok(attrs_map
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect())
}

/// Build a `VarMeta` from a snapshot's array node — both the on-disk
/// `zarr.json` (parsed via zarrs's `ArrayMetadata`) and the
/// pre-projected snapshot fields (`shape`, `dimension_names`).
///
/// The resulting `Arc<Array>` is constructed via
/// `Array::new_with_metadata` which is synchronous and does no I/O.
fn build_var_meta_from_snapshot(
    snap: &icechunk::format::snapshot::NodeSnapshot,
    array_path: &str,
    store: AsyncReadableListableStorage,
) -> Result<VarMeta> {
    let metadata: ArrayMetadata = serde_json::from_slice(&snap.user_data).map_err(|err| {
        RustytreeError::Other(format!(
            "icechunk: failed to parse array zarr.json at {array_path}: {err}"
        ))
    })?;

    let array = Array::new_with_metadata(store, array_path, metadata).map_err(|err| {
        RustytreeError::Other(format!(
            "icechunk: failed to construct Array from snapshot for {array_path}: {err}"
        ))
    })?;

    let name = array_path
        .rsplit_once('/')
        .map_or_else(|| array_path.to_string(), |(_, last)| last.to_string());

    let dims = array.dimension_names().as_ref().map_or_else(
        || {
            (0..array.shape().len())
                .map(|i| format!("dim_{i}"))
                .collect()
        },
        |names| {
            names
                .iter()
                .enumerate()
                .map(|(i, n)| n.clone().unwrap_or_else(|| format!("dim_{i}")))
                .collect()
        },
    );

    let dtype = format!("{}", array.data_type());
    let shape = array.shape().to_vec();
    let attrs = clone_attrs(array.attributes());

    Ok(VarMeta {
        name,
        dims,
        dtype,
        shape,
        attrs,
        array: Arc::new(array),
        eager: None,
    })
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

/// Open one array and project its metadata into a `VarMeta`. The opened
/// array is kept alive in an `Arc` so `ZarrsArrayHandle` can read chunks
/// later without re-opening.
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

    let dtype = format!("{}", array.data_type());
    let shape = array.shape().to_vec();
    let attrs = clone_attrs(array.attributes());

    Ok(VarMeta {
        name,
        dims,
        dtype,
        shape,
        attrs,
        array: Arc::new(array),
        eager: None,
    })
}

/// Decide whether a var's values should be pre-fetched during the walk
/// rather than deferred to xarray's lazy backend.
///
/// Triggers:
///   - **1-D self-named dim coord** (`var.dims == [var.name]`): xarray
///     promotes these into `Index` objects; index construction reads
///     the values, which would otherwise hit our lazy backend
///     once-per-coord-per-node.
///   - **CF time-like** (`attrs["units"]` is a string containing
///     `" since "`): xarray's `_decode_cf_datetime_dtype` peeks the
///     first and last element of every such variable to infer dtype.
///     That's 2 RTTs/var × N vars × N nodes serialised through our
///     lazy backend — the dominant cost on cold-cache S3.
///
/// Skip if total elements > `EAGER_FETCH_MAX_ELEMENTS` (1 M) — guards
/// against accidentally pulling a multi-GB array that just happens to
/// match the heuristic.
fn should_eager_fetch(var: &VarMeta) -> bool {
    // `shape` is empty for 0-D scalars; `iter().product()` gives 1.
    let n_elements: u64 = if var.shape.is_empty() {
        1
    } else {
        var.shape.iter().product()
    };
    if n_elements > EAGER_FETCH_MAX_ELEMENTS {
        return false;
    }
    // Self-named 1-D dim coord: xarray's `_maybe_create_default_indexes`
    // post-pass reads these on every node to construct pandas Index
    // objects. Pre-fetching here turns N×serial into one bounded
    // `try_join_all` against the same tokio runtime. CF time-likes
    // are NOT pre-fetched anymore — the metadata-only patch in
    // `backend.py` handles them without reading any chunks.
    var.dims.len() == 1 && var.dims[0] == var.name
}

/// Read the entire array's elements eagerly. Used by Phase C to
/// materialise small coord/time arrays so xarray's CF decoders see
/// resident numpy data instead of triggering chunk reads through our
/// lazy backend. Bounded by the same semaphore the walk uses, so this
/// can't stampede the underlying `object_store` client.
async fn fetch_all_elements(
    array: &Arc<Array<dyn AsyncReadableListableStorageTraits>>,
) -> Result<EagerElements> {
    let subset = ArraySubset::new_with_shape(array.shape().to_vec());
    let dtype = array.data_type().clone();
    for_each_supported_dtype!(dtype, T => {
        let elements: Vec<T> = array
            .async_retrieve_array_subset_elements::<T>(&subset)
            .await
            .map_err(|err| {
                RustytreeError::Other(format!("eager fetch failed: {err}"))
            })?;
        Ok(eager_from_vec(elements))
    }, other => {
        // Unsupported dtype — skip eagerly; the var stays lazy. The
        // caller treats this `Err` as "leave eager=None and continue".
        Err(RustytreeError::Other(format!(
            "eager fetch: dtype {other:?} not yet supported"
        )))
    })
}

/// One eager-fetch task: source coords + the array to read.
type EagerTask = (
    usize,
    usize,
    Arc<Array<dyn AsyncReadableListableStorageTraits>>,
);

/// Run the eager fetch over every eligible (node, var) pair, in
/// parallel, and assign the results back into `nodes` in place.
async fn eager_phase(semaphore: &Arc<Semaphore>, nodes: &mut [NodeData]) -> Result<()> {
    // Build the work list while still holding `&nodes` immutably; we
    // assign results via (node_idx, var_idx) so the borrow doesn't
    // need to outlive the await points.
    let mut work: Vec<EagerTask> = Vec::new();
    for (ni, node) in nodes.iter().enumerate() {
        for (vi, var) in node.vars.iter().enumerate() {
            if should_eager_fetch(var) {
                work.push((ni, vi, var.array.clone()));
            }
        }
    }
    if work.is_empty() {
        return Ok(());
    }

    // Each task acquires a permit so total in-flight reads (across
    // discover_paths, open_single, and Phase C) stay bounded by
    // `max_concurrency`. The permit is held for the full read.
    let results = try_join_all(work.into_iter().map(|(ni, vi, array)| {
        let sem = Arc::clone(semaphore);
        async move {
            let _permit = sem
                .acquire()
                .await
                .map_err(|_| RustytreeError::Other("eager phase: semaphore closed".into()))?;
            // Best-effort: an unsupported-dtype error here just means
            // the var stays lazy. Leak everything else as a real
            // error so the open call fails cleanly.
            match fetch_all_elements(&array).await {
                Ok(elements) => Ok::<_, RustytreeError>((ni, vi, Some(elements))),
                Err(_) => Ok::<_, RustytreeError>((ni, vi, None)),
            }
        }
    }))
    .await?;

    for (ni, vi, eager) in results {
        if let Some(eager) = eager {
            nodes[ni].vars[vi].eager = Some(eager);
        }
    }
    Ok(())
}

/// Wrap a typed `Vec<T>` in the matching `EagerElements` variant. The
/// dispatch is by Rust type, so the caller sees `EagerElements` without
/// having to inspect the original `DataType` again.
fn eager_from_vec<T: EagerElementType>(v: Vec<T>) -> EagerElements {
    T::wrap(v)
}

trait EagerElementType: Sized {
    fn wrap(v: Vec<Self>) -> EagerElements;
}

macro_rules! impl_eager_element {
    ($t:ty, $variant:ident) => {
        impl EagerElementType for $t {
            fn wrap(v: Vec<Self>) -> EagerElements {
                EagerElements::$variant(v)
            }
        }
    };
}

impl_eager_element!(bool, Bool);
impl_eager_element!(i8, I8);
impl_eager_element!(i16, I16);
impl_eager_element!(i32, I32);
impl_eager_element!(i64, I64);
impl_eager_element!(u8, U8);
impl_eager_element!(u16, U16);
impl_eager_element!(u32, U32);
impl_eager_element!(u64, U64);
impl_eager_element!(f32, F32);
impl_eager_element!(f64, F64);

#[cfg(test)]
mod tests {
    use super::{normalize_path, parent_of};

    #[test]
    fn normalize_path_handles_root_and_trailing_slash() {
        assert_eq!(normalize_path("/"), "/");
        assert_eq!(normalize_path("/foo"), "/foo");
        assert_eq!(normalize_path("/foo/"), "/foo");
        assert_eq!(normalize_path("/foo/bar"), "/foo/bar");
        assert_eq!(normalize_path("/foo/bar/"), "/foo/bar");
        assert_eq!(normalize_path(""), "/");
    }

    #[test]
    fn parent_of_root_and_descendants() {
        assert_eq!(parent_of("/"), "/");
        assert_eq!(parent_of("/foo"), "/");
        assert_eq!(parent_of("/foo/bar"), "/foo");
        assert_eq!(parent_of("/a/b/c"), "/a/b");
    }
}
