//! icechunk-backed Zarr v3 store.
//!
//! Builds an `AsyncReadableListableStorage` from an icechunk repository on
//! local filesystem so the rest of the walk code stays polymorphic over
//! icechunk and vanilla object_store-backed stores.
//!
//! Remote icechunk storage (S3 / GCS / Azure / HTTP via icechunk's storage
//! factories) and Python-side `IcechunkStore`/`Session` unwrap land in
//! follow-up PRs.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use icechunk::Repository;
use icechunk::repository::VersionInfo;
use icechunk::storage::new_local_filesystem_storage;
use zarrs_icechunk::AsyncIcechunkStore;
use zarrs_storage::AsyncReadableListableStorage;

use crate::error::{Result, RustytreeError};

/// Heuristic detector for an on-disk icechunk repository.
///
/// icechunk's local-filesystem layout (as of `icechunk = "2"`) places a
/// `repo` manifest file at the root and a `snapshots/` directory holding
/// immutable snapshot files. Their joint presence is a strong signal —
/// vanilla Zarr v3 stores do not produce either. Branch/tag refs live
/// inside the `repo` file, not in a `refs/` directory.
pub(crate) fn looks_like_icechunk_repo(path: &Path) -> bool {
    path.join("repo").is_file() && path.join("snapshots").is_dir()
}

/// Open an icechunk repository on local filesystem at `path` and produce a
/// zarrs-compatible store rooted at the given branch.
///
/// `branch` defaults to `"main"` when callers don't specify one.
pub(crate) async fn open_local_icechunk(
    path: &Path,
    branch: &str,
) -> Result<AsyncReadableListableStorage> {
    let storage = new_local_filesystem_storage(path).await.map_err(|err| {
        RustytreeError::Other(format!(
            "icechunk: failed to open local-filesystem storage at {}: {err}",
            path.display()
        ))
    })?;

    let repo = Repository::open(None, storage, HashMap::new())
        .await
        .map_err(|err| {
            RustytreeError::Other(format!(
                "icechunk: failed to open repository at {}: {err}",
                path.display()
            ))
        })?;

    let version = VersionInfo::BranchTipRef(branch.to_string());
    let session = repo.readonly_session(&version).await.map_err(|err| {
        RustytreeError::Other(format!(
            "icechunk: failed to open readonly session on branch {branch} at {}: {err}",
            path.display()
        ))
    })?;

    Ok(Arc::new(AsyncIcechunkStore::new(session)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_icechunk_layout() {
        let tmp = std::env::temp_dir().join("rustytree-icechunk-detect-test");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("snapshots")).expect("create snapshots/");
        std::fs::write(tmp.join("repo"), b"placeholder").expect("create repo file");
        assert!(looks_like_icechunk_repo(&tmp));
        std::fs::remove_dir_all(&tmp).expect("cleanup");
    }

    #[test]
    fn rejects_non_icechunk_dirs() {
        let tmp = std::env::temp_dir().join("rustytree-icechunk-reject-test");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).expect("create tmp dir");
        assert!(!looks_like_icechunk_repo(&tmp));
        // snapshots/ alone is not enough.
        std::fs::create_dir(tmp.join("snapshots")).expect("create snapshots/ only");
        assert!(!looks_like_icechunk_repo(&tmp));
        // repo as a directory (not a file) is not enough.
        std::fs::create_dir(tmp.join("repo")).expect("create repo/ as dir");
        assert!(!looks_like_icechunk_repo(&tmp));
        std::fs::remove_dir_all(&tmp).expect("cleanup");
    }

    #[test]
    fn rejects_missing_path() {
        let path = Path::new("/this/path/does/not/exist/rustytree");
        assert!(!looks_like_icechunk_repo(path));
    }
}
