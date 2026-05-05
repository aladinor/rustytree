//! icechunk-backed Zarr v3 store.
//!
//! Two construction paths:
//!   - [`open_local_icechunk`]: open an icechunk repository from a
//!     local-filesystem path. Convenience for zero-credential local dev.
//!   - [`store_from_session_bytes`]: rehydrate a `Session` from msgpack
//!     bytes produced by `icechunk-python`'s `PySession.as_bytes()`. The
//!     primary path: users construct the icechunk Session themselves
//!     (with whatever branch / credentials / cache config they want),
//!     then hand `session.store` to xarray; we cross the FFI boundary by
//!     unwrapping the Python `IcechunkStore` to bytes.
//!
//! The remote-S3 icechunk URL dispatch that lived here through PR #15
//! has been removed (Phase 7 redesign): users construct an
//! `icechunk.Repository.open(s3_storage_factory)` themselves rather
//! than passing a `s3://` URL to rustytree.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use icechunk::Repository;
use icechunk::repository::VersionInfo;
use icechunk::session::Session;
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

/// Rehydrate an `icechunk::session::Session` from the msgpack bytes
/// produced by `icechunk-python`'s `PySession.as_bytes()` and wrap it
/// as a zarrs-compatible store.
///
/// This is the cross-extension handoff: `PyO3` type extraction can't reach
/// across cdylibs, but the bytes serialisation is `pub` icechunk API
/// (using `rmp_serde`). We deserialise with `Session::from_bytes` —
/// which is the inverse of what `PySession::as_bytes` calls — so the
/// reconstructed session is byte-identical to the user's. Read-only:
/// any uncommitted writes the user made on the live session are
/// discarded by `as_bytes` (they're not part of the serialised state),
/// which is exactly what we want for a read-side metadata walk.
///
/// Caveats:
///   - The bytes format is icechunk-internal (rmp_serde-encoded
///     `Session`). Format stability is bounded by icechunk's semver;
///     we pin `icechunk = "2"` and CI must keep `icechunk` and
///     `icechunk-python` versions matched.
pub(crate) fn store_from_session_bytes(bytes: &[u8]) -> Result<AsyncReadableListableStorage> {
    let session = Session::from_bytes(bytes).map_err(|err| {
        RustytreeError::Other(format!(
            "icechunk: failed to reconstruct Session from bytes ({} bytes): {err}",
            bytes.len()
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

    #[test]
    fn store_from_session_bytes_rejects_garbage() {
        let result = store_from_session_bytes(b"not msgpack");
        assert!(matches!(result, Err(RustytreeError::Other(_))));
    }
}
