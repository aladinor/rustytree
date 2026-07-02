//! Build a zarrs `AsyncReadableListableStorage` from a path or URL.
//!
//! Local-filesystem inputs auto-detect between icechunk repositories
//! (`<root>/repo` file + `<root>/snapshots/` dir → routed through
//! `icechunk_store::open_local_icechunk`) and vanilla Zarr v3 directories
//! (routed through `zarrs_object_store::AsyncObjectStore` +
//! `LocalFileSystem`).
//!
//! `s3://` URLs go through `AmazonS3Builder` for **vanilla Zarr v3 only**.
//! icechunk-on-S3 used to live here as URL dispatch; Phase 7 dropped that
//! path — users construct the icechunk Session via `icechunk-python` and
//! pass `session.store` to `xr.open_datatree`. Other remote schemes
//! (`gs://`, `az://`, `http(s)://`) are not yet supported.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use zarrs_object_store::AsyncObjectStore;
use zarrs_object_store::object_store::aws::AmazonS3Builder;
use zarrs_object_store::object_store::local::LocalFileSystem;
use zarrs_storage::AsyncReadableListableStorage;

use crate::error::{Result, RustytreeError};
use crate::icechunk_store::{
    IcechunkBundle, bundle_from_session_bytes, looks_like_icechunk_repo, open_local_icechunk,
};

/// A serializable recipe for reopening a store on a fresh process — the state
/// a [`crate::array::ZarrsArrayHandle`] carries so it can survive a pickle
/// round-trip and reopen on a `dask.distributed` worker (see issue #44).
///
/// v1 covers only the icechunk-session path, and does so as a **pure mirror of
/// icechunk's own serialization**: we retain the msgpack bytes produced by
/// `icechunk-python`'s `PySession.as_bytes()` and reopen via
/// `Session::from_bytes` (through [`bundle_from_session_bytes`]). rustytree adds
/// **no** credential handling of its own — icechunk's typed credential enum
/// rides along inside the bytes exactly as icechunk stores it, so `from_env` /
/// `anonymous` sessions carry no secret into the task graph. Vanilla `s3://` /
/// local stores have no icechunk `Session` to reuse and are a deliberate
/// follow-up (a non-icechunk credential mechanism would not be a literal
/// mirror). Handles built from those inputs carry no spec and refuse to pickle.
#[derive(Clone, Serialize, Deserialize)]
pub(crate) enum ReopenSpec {
    /// The msgpack bytes from `PySession.as_bytes()`. Kept as a plain `Vec<u8>`
    /// so `serde` needs no `rc` feature; handles wrap the spec in an `Arc` to
    /// share these (potentially large) bytes across a tree's arrays in memory.
    // Follow-up: `#[serde(with = "serde_bytes")]` would encode this as a msgpack
    // `bin` instead of an int array, shrinking the pickled state. Deferred with
    // the other wire-size work (per-array duplication) — see issue #44.
    IcechunkSession { bytes: Vec<u8> },
}

/// Build a [`WalkSource`] from a [`ReopenSpec`]. The single entry point shared
/// by the initial open and the per-worker reopen, so both reconstruct the store
/// exactly the same way. Sync today because the only variant
/// (`bundle_from_session_bytes` → `Session::from_bytes`) is sync; add `async`
/// back when a variant needs it (e.g. the vanilla-store follow-up).
pub(crate) fn build_store_from_spec(spec: &ReopenSpec) -> Result<WalkSource> {
    match spec {
        ReopenSpec::IcechunkSession { bytes } => {
            Ok(WalkSource::Icechunk(bundle_from_session_bytes(bytes)?))
        }
    }
}

/// What kind of store the walk should consume.
///
/// `Icechunk` carries both the live `Session` (used by the snapshot
/// fast-path metadata walker) and the matching zarrs store (used for
/// lazy chunk reads). `Vanilla` is the generic-zarrs path for
/// non-icechunk Zarr v3 stores; only the store handle is needed.
pub(crate) enum WalkSource {
    Icechunk(IcechunkBundle),
    Vanilla(AsyncReadableListableStorage),
}

impl WalkSource {
    /// The zarrs store backing this source. Used when reopening a single array
    /// by path (the pickle-revive path) where the session's metadata walker is
    /// not needed — only lazy chunk reads.
    pub(crate) fn into_store(self) -> AsyncReadableListableStorage {
        match self {
            WalkSource::Icechunk(bundle) => bundle.store,
            WalkSource::Vanilla(store) => store,
        }
    }
}

/// Build a [`WalkSource`] for a local-filesystem path.
///
/// Detects icechunk repositories automatically and opens them at the given
/// `branch` (defaulting to `"main"` when `None`); other directories are
/// opened as vanilla Zarr v3 stores. The `branch` parameter is silently
/// ignored on the vanilla path — branches are an icechunk concept.
pub(crate) async fn build_local_store(path: &Path, branch: Option<&str>) -> Result<WalkSource> {
    if looks_like_icechunk_repo(path) {
        let bundle = open_local_icechunk(path, branch.unwrap_or("main")).await?;
        return Ok(WalkSource::Icechunk(bundle));
    }
    Ok(WalkSource::Vanilla(build_vanilla_local(path)?))
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

/// Build a vanilla Zarr v3 store rooted at an S3 bucket + prefix.
///
/// `options` accepts the standard fsspec/xarray-style S3 keys: `region`,
/// `endpoint`, `access_key_id`, `secret_access_key`, `session_token`,
/// `allow_http` (bool string), `skip_signature` (bool string; `anon` is an
/// alias). Unknown keys are rejected so typos surface immediately rather
/// than silently disabling auth.
///
/// `AmazonS3Builder::from_env()` is the base, so `AWS_REGION` /
/// `AWS_ACCESS_KEY_ID` / etc. are picked up automatically; anything in
/// `options` overrides.
pub(crate) fn build_vanilla_s3(
    bucket: &str,
    prefix: &str,
    options: &HashMap<String, String>,
) -> Result<AsyncReadableListableStorage> {
    // Use `with_url` to lock both bucket and prefix in one call.
    let url = if prefix.is_empty() {
        format!("s3://{bucket}")
    } else {
        format!("s3://{bucket}/{prefix}")
    };

    let mut builder = AmazonS3Builder::from_env().with_url(&url);
    for (key, value) in options {
        builder = apply_s3_option(builder, key, value)?;
    }

    let store = builder.build().map_err(|err| {
        RustytreeError::Other(format!("failed to build S3 store for {url}: {err}"))
    })?;
    Ok(Arc::new(AsyncObjectStore::new(store)))
}

/// Apply one fsspec/xarray-style S3 option to the builder.
fn apply_s3_option(builder: AmazonS3Builder, key: &str, value: &str) -> Result<AmazonS3Builder> {
    let bool_opt = || -> Result<bool> {
        match value {
            "true" | "True" | "1" | "yes" => Ok(true),
            "false" | "False" | "0" | "no" => Ok(false),
            other => Err(RustytreeError::InvalidInput(format!(
                "s3 storage option `{key}` expects a boolean, got {other:?}"
            ))),
        }
    };
    match key {
        "region" => Ok(builder.with_region(value)),
        "endpoint" => Ok(builder.with_endpoint(value)),
        "access_key_id" => Ok(builder.with_access_key_id(value)),
        "secret_access_key" => Ok(builder.with_secret_access_key(value)),
        "session_token" => Ok(builder.with_token(value)),
        "allow_http" => Ok(builder.with_allow_http(bool_opt()?)),
        // `anon` is the fsspec spelling; map it to skip_signature.
        "skip_signature" | "anon" => Ok(builder.with_skip_signature(bool_opt()?)),
        other => Err(RustytreeError::InvalidInput(format!(
            "unknown s3 storage option: `{other}`"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reopen_spec_round_trips_through_rmp_serde() {
        // The pickle state serialised in `ZarrsArrayHandle::__reduce__` is
        // `(ReopenSpec, path)`; pin that the ReopenSpec half survives the same
        // msgpack codec (a silent format break would corrupt every worker's
        // reopen, which the Python E2E test wouldn't localise to here).
        let spec = ReopenSpec::IcechunkSession {
            bytes: vec![1, 2, 3, 4, 250, 128, 0],
        };
        let encoded = rmp_serde::to_vec(&spec).expect("serialize ReopenSpec");
        let ReopenSpec::IcechunkSession { bytes } =
            rmp_serde::from_slice(&encoded).expect("deserialize ReopenSpec");
        assert_eq!(bytes, vec![1, 2, 3, 4, 250, 128, 0]);
    }

    #[test]
    fn build_store_from_spec_rejects_garbage_session_bytes() {
        // The reopen seam must surface bad session bytes as the icechunk-session
        // error variant (→ Python ValueError), not panic on the worker.
        let spec = ReopenSpec::IcechunkSession {
            bytes: b"not a real icechunk session".to_vec(),
        };
        match build_store_from_spec(&spec) {
            Err(RustytreeError::IcechunkSession(_)) => {}
            Err(other) => panic!("expected IcechunkSession error, got {other:?}"),
            Ok(_) => panic!("expected error for garbage session bytes"),
        }
    }

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
