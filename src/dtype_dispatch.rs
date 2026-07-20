//! Shared dtype-dispatch macro.
//!
//! `read_subset` (in `array.rs`) and the Phase C eager fanout (in
//! `walk.rs`) both need to map a `zarrs::array::DataType` onto a
//! concrete primitive type to call
//! `async_retrieve_array_subset::<Vec<T>>(...)`. Without sharing, the
//! 11-arm dispatch drifts between the two sites; with this macro the
//! dtype→primitive-type mapping lives in one place. Naming a dtype for
//! Python is a separate concern handled by
//! [`crate::array::zarrs_dtype_to_numpy_str`], which can name more
//! dtypes than we can read.
//!
//! zarrs 0.23 replaced the `DataType` enum with a newtype over
//! `Arc<dyn DataTypeTraits>`, so this is an `is::<Marker>()` chain
//! rather than a `match`.
//!
//! Usage:
//!
//! ```ignore
//! // takes a `&DataType` — e.g. straight from `array.data_type()`
//! for_each_supported_dtype!(array.data_type(), T => {
//!     // body has `T` bound to the matching primitive type
//!     let v: Vec<T> = ...;
//! }, other => {
//!     // fallback for unsupported dtypes; `other` is &DataType
//!     return Err(...);
//! });
//! ```
//!
//! The `body` block is type-substituted per arm, so callers can use `T`
//! as both a type and a turbofish.

/// Run `body` with `T` bound to the primitive type for each supported
/// `zarrs::array::DataType`, falling back to `fallback` for anything we
/// don't yet handle (complex floats, low-precision floats, strings, raw
/// bits).
macro_rules! for_each_supported_dtype {
    ($dt:expr, $T:ident => $body:block, $other:ident => $fallback:block) => {{
        // Bound once (not re-evaluated per arm) and annotated so a
        // wrong-typed `$dt` fails here rather than deep inside a branch.
        let dt: &::zarrs::array::DataType = $dt;
        if dt.is::<::zarrs::array::data_type::BoolDataType>() {
            type $T = bool;
            $body
        } else if dt.is::<::zarrs::array::data_type::Int8DataType>() {
            type $T = i8;
            $body
        } else if dt.is::<::zarrs::array::data_type::Int16DataType>() {
            type $T = i16;
            $body
        } else if dt.is::<::zarrs::array::data_type::Int32DataType>() {
            type $T = i32;
            $body
        } else if dt.is::<::zarrs::array::data_type::Int64DataType>() {
            type $T = i64;
            $body
        } else if dt.is::<::zarrs::array::data_type::UInt8DataType>() {
            type $T = u8;
            $body
        } else if dt.is::<::zarrs::array::data_type::UInt16DataType>() {
            type $T = u16;
            $body
        } else if dt.is::<::zarrs::array::data_type::UInt32DataType>() {
            type $T = u32;
            $body
        } else if dt.is::<::zarrs::array::data_type::UInt64DataType>() {
            type $T = u64;
            $body
        } else if dt.is::<::zarrs::array::data_type::Float32DataType>() {
            type $T = f32;
            $body
        } else if dt.is::<::zarrs::array::data_type::Float64DataType>() {
            type $T = f64;
            $body
        } else {
            let $other = dt;
            $fallback
        }
    }};
}

pub(crate) use for_each_supported_dtype;
