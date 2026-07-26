//! Per-group metadata snapshot.
//!
//! `NodeData` is what the async walk produces for each Zarr group: enough
//! information for the Python side to assemble an `xr.Dataset`. `VarMeta`
//! carries one entry per array (data variable or coordinate) inside the
//! group, plus a live `Arc<zarrs::Array>` so the lazy chunk-read path
//! (`ZarrsArrayHandle`) can call back through `async_retrieve_array_subset`
//! without re-opening the array.
//!
//! `VarMeta::eager` carries already-materialised element values for vars
//! that the walk pre-fetched in parallel during Phase C (1-D self-named
//! dim coords + CF time-like vars). The Python entrypoint uses this to
//! build `xr.Variable`s with resident numpy data, bypassing xarray's
//! per-node serial decoder calls back through our lazy backend.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::Value as JsonValue;
use zarrs::array::Array;
use zarrs_storage::AsyncReadableListableStorageTraits;

/// Element-typed eager-fetched array contents. One variant per dtype that
/// `ZarrsArrayHandle::read_subset` already supports, plus `Str` for vlen
/// `string` (handled outside `for_each_supported_dtype!`, same split as the
/// lazy `read_subset` path).
pub(crate) enum EagerElements {
    Bool(Vec<bool>),
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    Str(Vec<String>),
}

/// Metadata snapshot for a single array within a group, plus a live
/// `Arc<Array>` so subsequent reads don't re-open the array.
pub(crate) struct VarMeta {
    /// Array name as it appears under the parent group (no leading slash).
    pub name: String,
    /// Dimension names from the array's `dimension_names` (synthesised as
    /// `dim_0`, `dim_1`, ... when the array doesn't declare them).
    pub dims: Vec<String>,
    /// Canonical `NumPy` dtype name (e.g. `"float64"`, `"int32"`), as
    /// produced by `array::zarrs_dtype_to_numpy_str` — not the Zarr V2
    /// spelling (`"<f8"`) and not zarrs's `Display`, which renders both
    /// spellings at once and is not parseable by `numpy.dtype()`.
    pub dtype: String,
    /// Array shape in elements.
    pub shape: Vec<u64>,
    /// User-attribute map from `zarr.json` `attributes`.
    pub attrs: BTreeMap<String, JsonValue>,
    /// Live array handle for lazy chunk reads. Wrapped in `Arc` so the
    /// `ZarrsArrayHandle` produced for Python can keep the underlying
    /// array alive independently of the `NodeData` lifetime.
    pub array: Arc<Array<dyn AsyncReadableListableStorageTraits>>,
    /// Eagerly-fetched element values, populated by `walk::eager_phase`
    /// for variables that match the eager-fetch predicate. `None` means
    /// the Python side will build a lazy `RustyBackendArray` instead.
    pub eager: Option<EagerElements>,
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
