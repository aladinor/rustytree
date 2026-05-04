//! Error type for the rustytree Rust core.
//!
//! `RustytreeError` aggregates the failure modes the (in-progress) hierarchy
//! walk and chunk-read code paths can hit, and converts cleanly to a `PyErr`
//! at the FFI boundary so Python sees a meaningful exception type.
//!
//! The `From<RustytreeError> for PyErr` impl maps:
//!   - I/O failures   -> `OSError`
//!   - missing keys   -> `KeyError`
//!   - bad input      -> `ValueError`
//!   - everything else (catch-all upstream errors) -> `RuntimeError`
//!
//! Concrete `From<UpstreamError>` impls (zarrs, icechunk, `object_store`) land
//! alongside the modules that introduce those dependencies.

use pyo3::exceptions::{PyKeyError, PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use thiserror::Error;

/// Result alias used throughout the Rust crate.
#[expect(dead_code, reason = "wired up by the upcoming async hierarchy walk PR")]
pub(crate) type Result<T> = std::result::Result<T, RustytreeError>;

/// Top-level error type for rustytree's Rust core.
#[derive(Debug, Error)]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "variants get callers in the upcoming async hierarchy walk PR"
    )
)]
pub(crate) enum RustytreeError {
    /// Underlying I/O failure (filesystem, socket, etc.).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// A key (e.g. group path or chunk coordinate) was not found in the store.
    #[error("not found: {0}")]
    NotFound(String),

    /// Caller-supplied input was rejected (malformed URL, invalid kwargs, etc.).
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// Transitional catch-all for upstream errors that don't yet have a typed
    /// variant. Replace each call site with a typed variant (`Zarrs`,
    /// `Icechunk`, `ObjectStore`, ...) as the corresponding crates are added,
    /// then remove this variant.
    #[error("{0}")]
    Other(String),
}

impl From<RustytreeError> for PyErr {
    fn from(err: RustytreeError) -> Self {
        let msg = err.to_string();
        match err {
            RustytreeError::Io(_) => PyOSError::new_err(msg),
            RustytreeError::NotFound(_) => PyKeyError::new_err(msg),
            RustytreeError::InvalidInput(_) => PyValueError::new_err(msg),
            RustytreeError::Other(_) => PyRuntimeError::new_err(msg),
        }
    }
}

// Cross-FFI behaviour (the `From<RustytreeError> for PyErr` impl mapping each
// variant to the right Python exception type) is verified by pytest in
// `tests/`. Standalone Rust unit tests would need libpython linked at runtime,
// which conflicts with the cdylib's `extension-module` feature.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_io_error() {
        let err = RustytreeError::Io(std::io::Error::other("boom"));
        assert_eq!(err.to_string(), "I/O error: boom");
    }

    #[test]
    fn display_not_found() {
        let err = RustytreeError::NotFound("/data/missing".into());
        assert_eq!(err.to_string(), "not found: /data/missing");
    }

    #[test]
    fn display_invalid_input() {
        let err = RustytreeError::InvalidInput("bad url".into());
        assert_eq!(err.to_string(), "invalid input: bad url");
    }

    #[test]
    fn display_other() {
        let err = RustytreeError::Other("upstream failure".into());
        assert_eq!(err.to_string(), "upstream failure");
    }
}
