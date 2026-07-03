//! Process-global Tokio runtime.
//!
//! rustytree owns one multi-threaded runtime for the lifetime of the process.
//! It's reused for the (future) async hierarchy walk and for lazy chunk reads
//! triggered by indexing into a `BackendArray`. The `PyO3` entry points cross
//! the FFI boundary once per call, releasing the GIL via
//! `Python::allow_threads(...)` before invoking `runtime.block_on(...)`.
//!
//! Initialised on first access via `OnceLock`; never torn down.
//!
//! **Not fork-safe.** A tokio runtime's worker/IO-driver threads do not survive
//! `fork()`, so a process that inherited an already-built runtime would hang on
//! `block_on`. `dask.distributed` uses the `spawn` start-method by default (a
//! fresh interpreter per worker), which sidesteps this entirely and is the
//! supported path for issue #44. Fork-based multiprocessing (e.g.
//! `multiprocessing` / `ProcessPoolExecutor` with the Linux-default `fork`
//! start-method) after a store has been opened is unsupported. Making the
//! runtime survive fork (rebuild-on-PID-change + route `read_subset` through
//! `handle()` instead of a cached `Handle`) is tracked as a follow-up.

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
