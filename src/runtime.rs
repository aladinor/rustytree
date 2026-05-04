//! Process-global Tokio runtime.
//!
//! rustytree owns one multi-threaded runtime for the lifetime of the process.
//! It's reused for the (future) async hierarchy walk and for lazy chunk reads
//! triggered by indexing into a `BackendArray`. The `PyO3` entry points cross
//! the FFI boundary once per call, releasing the GIL via
//! `Python::allow_threads(...)` before invoking `runtime.block_on(...)`.
//!
//! Initialised on first access via `OnceLock`; never torn down.

use std::sync::OnceLock;
use tokio::runtime::{Builder, Runtime};

static RUNTIME: OnceLock<Runtime> = OnceLock::new();

/// Returns a reference to the process-global Tokio runtime, initialising it
/// on first call.
///
/// # Panics
/// Panics if the runtime cannot be constructed (e.g. the OS refuses to spawn
/// the worker thread pool). This is unrecoverable; the only sensible response
/// is to surface it as a panic and let Python translate it into `PanicException`.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "wired up by the upcoming async hierarchy walk PR")
)]
pub(crate) fn handle() -> &'static Runtime {
    RUNTIME.get_or_init(|| {
        Builder::new_multi_thread()
            .enable_all()
            .thread_name("rustytree-tokio")
            .build()
            .expect("rustytree: failed to start Tokio runtime")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handle_is_idempotent() {
        let a = handle();
        let b = handle();
        assert!(
            std::ptr::eq(a, b),
            "OnceLock should hand back the same runtime"
        );
    }

    #[test]
    fn handle_runs_a_future() {
        let value = handle().block_on(async { 42_u32 });
        assert_eq!(value, 42);
    }
}
