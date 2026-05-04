//! Build a zarrs `AsyncReadableListableStorage` from a path.
//!
//! Auto-detects between an icechunk repository and a vanilla Zarr v3
//! directory: the path is checked for icechunk's on-disk layout
//! (`<root>/refs/` + `<root>/snapshots/`) and routed to either
//! `icechunk_store::open_local_icechunk` or a plain
//! `zarrs_object_store::AsyncObjectStore` over `LocalFileSystem`.
//!
//! Remote `object_store` backends (S3 / GCS / Azure / HTTP) land in a
//! follow-up PR.

use std::path::Path;
use std::sync::Arc;

use zarrs_object_store::AsyncObjectStore;
use zarrs_object_store::object_store::local::LocalFileSystem;
use zarrs_storage::AsyncReadableListableStorage;

use crate::error::{Result, RustytreeError};
use crate::icechunk_store::{looks_like_icechunk_repo, open_local_icechunk};

/// Build a zarrs storage handle for a local-filesystem path.
///
/// Detects icechunk repositories automatically and opens them at the given
/// `branch` (defaulting to `"main"` when `None`); other directories are
/// opened as vanilla Zarr v3 stores. The `branch` parameter is silently
/// ignored on the vanilla path — branches are an icechunk concept.
pub(crate) async fn build_local_store(
    path: &Path,
    branch: Option<&str>,
) -> Result<AsyncReadableListableStorage> {
    if looks_like_icechunk_repo(path) {
        return open_local_icechunk(path, branch.unwrap_or("main")).await;
    }
    build_vanilla_local(path)
}

/// Open a directory as a vanilla Zarr v3 store via `LocalFileSystem`.
fn build_vanilla_local(path: &Path) -> Result<AsyncReadableListableStorage> {
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
    fn vanilla_local_succeeds_for_existing_dir() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"));
        assert!(build_vanilla_local(path).is_ok());
    }

    #[test]
    fn vanilla_local_fails_for_missing_dir() {
        let path = Path::new("/this/path/does/not/exist/rustytree-test");
        match build_vanilla_local(path) {
            Err(RustytreeError::Io(_)) => {}
            Err(other) => panic!("expected Io variant, got {other:?}"),
            Ok(_) => panic!("expected Err for non-existent dir"),
        }
    }
}
