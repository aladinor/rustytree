//! Per-group metadata snapshot.
//!
//! `NodeData` is what the async walk produces for each Zarr group: enough
//! information for the Python side to assemble an `xr.Dataset`. `VarMeta`
//! carries one entry per array (data variable or coordinate) inside the
//! group.
//!
//! The walk is metadata-only. Live `zarrs::Array` handles for the lazy
//! chunk-read path are added by the follow-up PR that wires the
//! `BackendArray` adapter — keeping them out here keeps `NodeData` cheap
//! to clone and easy to marshal across the FFI boundary.

use std::collections::BTreeMap;

use serde_json::Value as JsonValue;

/// Metadata snapshot for a single array within a group.
#[derive(Debug, Clone)]
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
}

/// Metadata snapshot for a single group.
#[derive(Debug, Clone)]
pub(crate) struct NodeData {
    /// Group path (e.g. `"/"`, `"/sweep_0"`).
    pub path: String,
    /// User-attribute map from the group's `zarr.json`.
    pub attrs: BTreeMap<String, JsonValue>,
    /// Arrays found directly under this group. Child groups are handled by
    /// the recursive walk in the multi-node PR.
    pub vars: Vec<VarMeta>,
}
