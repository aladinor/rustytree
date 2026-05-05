//! Shared dtype-dispatch macro.
//!
//! `read_subset` (in `array.rs`) and the Phase C eager fanout (in
//! `walk.rs`) both need to map a `zarrs::array::DataType` variant onto
//! a concrete primitive type to call
//! `async_retrieve_array_subset_elements::<T>(...)`. Without sharing,
//! the 11-arm match drifts between the two sites; with this macro the
//! supported-dtype list lives in one place.
//!
//! Usage:
//!
//! ```ignore
//! for_each_supported_dtype!(dt, T => {
//!     // body has `T` bound to the matching primitive type
//!     let v: Vec<T> = ...;
//! }, other => {
//!     // fallback for unsupported dtypes; `other` is &DataType
//!     return Err(...);
//! });
//! ```
//!
//! The macro expands to a `match` on `dt`. The `body` block is
//! type-substituted per arm, so callers can use `T` as both a type and
//! a turbofish.

/// Run `body` with `T` bound to the primitive type for each supported
/// `zarrs::array::DataType` variant, falling back to `fallback` for
/// anything we don't yet handle (complex floats, low-precision floats,
/// strings, raw bits).
macro_rules! for_each_supported_dtype {
    ($dt:expr, $T:ident => $body:block, $other:ident => $fallback:block) => {
        match $dt {
            ::zarrs::array::DataType::Bool => {
                type $T = bool;
                $body
            }
            ::zarrs::array::DataType::Int8 => {
                type $T = i8;
                $body
            }
            ::zarrs::array::DataType::Int16 => {
                type $T = i16;
                $body
            }
            ::zarrs::array::DataType::Int32 => {
                type $T = i32;
                $body
            }
            ::zarrs::array::DataType::Int64 => {
                type $T = i64;
                $body
            }
            ::zarrs::array::DataType::UInt8 => {
                type $T = u8;
                $body
            }
            ::zarrs::array::DataType::UInt16 => {
                type $T = u16;
                $body
            }
            ::zarrs::array::DataType::UInt32 => {
                type $T = u32;
                $body
            }
            ::zarrs::array::DataType::UInt64 => {
                type $T = u64;
                $body
            }
            ::zarrs::array::DataType::Float32 => {
                type $T = f32;
                $body
            }
            ::zarrs::array::DataType::Float64 => {
                type $T = f64;
                $body
            }
            $other => $fallback,
        }
    };
}

pub(crate) use for_each_supported_dtype;
