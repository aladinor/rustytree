//! Build a zarrs `AsyncReadableListableStorage` from a path or URL.
//!
//! Today only local-filesystem inputs are supported; remote `object_store`
//! backends (S3 / GCS / Azure / HTTP) and the icechunk dispatch (which
//! produces an `AsyncIcechunkStore` instead) land in the follow-up PR.

use std::path::Path;
use std::sync::Arc;

use zarrs_object_store::AsyncObjectStore;
use zarrs_object_store::object_store::local::LocalFileSystem;
use zarrs_storage::AsyncReadableListableStorage;

use crate::error::{Result, RustytreeError};

/// Open a local-filesystem-backed store rooted at `path`.
///
/// `path` must exist and be a directory; on either failure the caller gets
/// `RustytreeError::Io` mapped to `OSError` at the FFI boundary.
pub(crate) fn build_local_store(path: &Path) -> Result<AsyncReadableListableStorage> {
    let local = LocalFileSystem::new_with_prefix(path).map_err(|err| {
        RustytreeError::Io(std::io::Error::other(format!(
            "failed to open local filesystem at {}: {err}",
            path.display()
        )))
    })?;
    Ok(Arc::new(AsyncObjectStore::new(local)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_local_store_succeeds_for_existing_dir() {
        // The cargo manifest dir always exists while tests are running.
        let path = Path::new(env!("CARGO_MANIFEST_DIR"));
        assert!(build_local_store(path).is_ok());
    }

    #[test]
    fn build_local_store_fails_for_missing_dir() {
        let path = Path::new("/this/path/does/not/exist/rustytree-test");
        // `Ok` is `Arc<dyn ...>` (not `Debug`), so reach for the err side
        // explicitly instead of `expect_err`.
        match build_local_store(path) {
            Err(RustytreeError::Io(_)) => {}
            Err(other) => panic!("expected Io variant, got {other:?}"),
            Ok(_) => panic!("expected Err for non-existent dir"),
        }
    }
}
