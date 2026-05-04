//! Build a zarrs `AsyncReadableListableStorage` from a path or URL.
//!
//! Local-filesystem inputs auto-detect between icechunk repositories
//! (`<root>/repo` file + `<root>/snapshots/` dir → routed through
//! `icechunk_store::open_local_icechunk`) and vanilla Zarr v3 directories
//! (routed through `zarrs_object_store::AsyncObjectStore` +
//! `LocalFileSystem`).
//!
//! `s3://` URLs go through `AmazonS3Builder` (vanilla Zarr v3 only —
//! icechunk-on-S3 dispatch lands in a follow-up PR). Other remote schemes
//! (`gs://`, `az://`, `http(s)://`) are not yet supported.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use zarrs_object_store::AsyncObjectStore;
use zarrs_object_store::object_store::ObjectStoreExt;
use zarrs_object_store::object_store::aws::AmazonS3Builder;
use zarrs_object_store::object_store::local::LocalFileSystem;
use zarrs_object_store::object_store::path::Path as ObjectStorePath;
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

/// Probe an S3 prefix to see if it points at an icechunk repository.
///
/// icechunk's on-prefix layout has a top-level `repo` manifest object;
/// vanilla Zarr v3 stores have a top-level `zarr.json` instead. A single
/// HEAD on `<prefix>/repo` distinguishes them. The probe respects
/// `storage_options` so anonymous public buckets work without credentials.
///
/// Returns `Ok(true)` for icechunk, `Ok(false)` for vanilla. Network or
/// auth failures (anything other than `NotFound`) propagate so the caller
/// sees the real error instead of being silently routed to the wrong
/// backend.
pub(crate) async fn s3_is_icechunk(
    bucket: &str,
    prefix: &str,
    options: &HashMap<String, String>,
) -> Result<bool> {
    let mut builder = AmazonS3Builder::from_env().with_bucket_name(bucket);
    for (key, value) in options {
        builder = apply_s3_option(builder, key, value)?;
    }
    let client = builder.build().map_err(|err| {
        RustytreeError::Other(format!(
            "failed to build S3 client for s3://{bucket} (icechunk probe): {err}"
        ))
    })?;

    let key = if prefix.is_empty() {
        "repo".to_string()
    } else {
        format!("{prefix}/repo")
    };
    let path = ObjectStorePath::from(key);
    match client.head(&path).await {
        Ok(_) => Ok(true),
        Err(zarrs_object_store::object_store::Error::NotFound { .. }) => Ok(false),
        Err(err) => Err(RustytreeError::Other(format!(
            "icechunk probe failed for s3://{bucket}/{prefix}: {err}"
        ))),
    }
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
