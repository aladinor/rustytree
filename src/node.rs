//! Per-group metadata snapshot.
//!
//! `NodeData` is what the async walk produces for each Zarr group: enough
//! information for the Python side to assemble an `xr.Dataset`. `VarMeta`
//! carries one entry per array (data variable or coordinate) inside the
//! group, plus a live `Arc<zarrs::Array>` so the lazy chunk-read path
//! (`ZarrsArrayHandle`) can call back through `async_retrieve_array_subset`
//! without re-opening the array.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::Value as JsonValue;
use zarrs::array::Array;
use zarrs_storage::AsyncReadableListableStorageTraits;

/// Metadata snapshot for a single array within a group, plus a live
/// `Arc<Array>` so subsequent reads don't re-open the array.
pub(crate) struct VarMeta {
    /// Array name as it appears under the parent group (no leading slash).
    pub name: String,
    /// Dimension names from the array's `dimension_names` (synthesised as
    /// `dim_0`, `dim_1`, ... when the array doesn't declare them).
    pub dims: Vec<String>,
    /// Numpy-style dtype string (e.g. `"<f8"`, `"<i4"`).
    pub dtype: String,
    /// Array shape in elements.
    pub shape: Vec<u64>,
    /// User-attribute map from `zarr.json` `attributes`.
    pub attrs: BTreeMap<String, JsonValue>,
    /// Live array handle for lazy chunk reads. Wrapped in `Arc` so the
    /// `ZarrsArrayHandle` produced for Python can keep the underlying
    /// array alive independently of the `NodeData` lifetime.
    pub array: Arc<Array<dyn AsyncReadableListableStorageTraits>>,
}

/// Metadata snapshot for a single group.
pub(crate) struct NodeData {
    /// Group path (e.g. `"/"`, `"/sweep_0"`).
    pub path: String,
    /// User-attribute map from the group's `zarr.json`.
    pub attrs: BTreeMap<String, JsonValue>,
    /// Arrays found directly under this group.
    pub vars: Vec<VarMeta>,
}
