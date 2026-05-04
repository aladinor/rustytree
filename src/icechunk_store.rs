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
use icechunk::storage::{
    S3Credentials, S3Options, S3StaticCredentials, new_local_filesystem_storage, new_s3_storage,
};
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

/// Open an icechunk repository on S3 and produce a zarrs-compatible store
/// rooted at the given branch.
///
/// `options` accepts the same fsspec/xarray-style keys we recognise on the
/// vanilla S3 path (see `store::apply_s3_option`); they're translated into
/// icechunk's `S3Options` + `S3Credentials` shape here. `anon=True`
/// (alias `skip_signature=True`) maps to `S3Credentials::Anonymous`;
/// `access_key_id` + `secret_access_key` map to a `Static` credential
/// triple. Otherwise icechunk picks up creds from the AWS env / IAM role.
pub(crate) async fn open_s3_icechunk(
    bucket: &str,
    prefix: &str,
    branch: &str,
    options: &HashMap<String, String>,
) -> Result<AsyncReadableListableStorage> {
    let (config, credentials) = parse_s3_options(options)?;
    let prefix = if prefix.is_empty() {
        None
    } else {
        Some(prefix.to_string())
    };

    let storage =
        new_s3_storage(config, bucket.to_string(), prefix.clone(), credentials).map_err(|err| {
            RustytreeError::Other(format!(
                "icechunk: failed to build S3 storage for s3://{bucket}/{}: {err}",
                prefix.as_deref().unwrap_or("")
            ))
        })?;

    let repo = Repository::open(None, storage, HashMap::new())
        .await
        .map_err(|err| {
            RustytreeError::Other(format!(
                "icechunk: failed to open S3 repository at s3://{bucket}/{}: {err}",
                prefix.as_deref().unwrap_or("")
            ))
        })?;

    let version = VersionInfo::BranchTipRef(branch.to_string());
    let session = repo.readonly_session(&version).await.map_err(|err| {
        RustytreeError::Other(format!(
            "icechunk: failed to open readonly session on branch {branch} \
             at s3://{bucket}/{}: {err}",
            prefix.as_deref().unwrap_or("")
        ))
    })?;

    Ok(Arc::new(AsyncIcechunkStore::new(session)))
}

/// Translate fsspec/xarray-style options into icechunk's `S3Options` +
/// `S3Credentials`. Recognised keys mirror the vanilla S3 path; unknown
/// keys are rejected so typos surface as `ValueError`.
fn parse_s3_options(
    options: &HashMap<String, String>,
) -> Result<(S3Options, Option<S3Credentials>)> {
    let mut config = S3Options {
        region: None,
        endpoint_url: None,
        anonymous: false,
        allow_http: false,
        force_path_style: false,
        network_stream_timeout_seconds: None,
        requester_pays: false,
    };

    let mut access_key_id: Option<String> = None;
    let mut secret_access_key: Option<String> = None;
    let mut session_token: Option<String> = None;
    let mut anonymous = false;

    let bool_opt = |key: &str, value: &str| -> Result<bool> {
        match value {
            "true" | "True" | "1" | "yes" => Ok(true),
            "false" | "False" | "0" | "no" => Ok(false),
            other => Err(RustytreeError::InvalidInput(format!(
                "s3 storage option `{key}` expects a boolean, got {other:?}"
            ))),
        }
    };

    for (key, value) in options {
        match key.as_str() {
            "region" => config.region = Some(value.clone()),
            "endpoint" => config.endpoint_url = Some(value.clone()),
            "allow_http" => config.allow_http = bool_opt(key, value)?,
            "skip_signature" | "anon" => anonymous = bool_opt(key, value)?,
            "access_key_id" => access_key_id = Some(value.clone()),
            "secret_access_key" => secret_access_key = Some(value.clone()),
            "session_token" => session_token = Some(value.clone()),
            other => {
                return Err(RustytreeError::InvalidInput(format!(
                    "unknown s3 storage option: `{other}`"
                )));
            }
        }
    }

    config.anonymous = anonymous;

    let credentials = match (anonymous, access_key_id, secret_access_key) {
        (true, _, _) => Some(S3Credentials::Anonymous),
        (false, Some(access_key_id), Some(secret_access_key)) => {
            Some(S3Credentials::Static(S3StaticCredentials {
                access_key_id,
                secret_access_key,
                session_token,
                expires_after: None,
            }))
        }
        // No anon, no static creds → let icechunk default to FromEnv.
        (false, None, None) => None,
        // Partial static credentials are a clear caller mistake.
        (false, Some(_), None) | (false, None, Some(_)) => {
            return Err(RustytreeError::InvalidInput(
                "s3 storage_options: access_key_id and secret_access_key must be set together"
                    .into(),
            ));
        }
    };

    Ok((config, credentials))
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

    fn opts(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn parse_s3_options_empty_yields_default_config_and_from_env_creds() {
        let (config, creds) = parse_s3_options(&HashMap::new()).expect("empty options ok");
        assert_eq!(config.region, None);
        assert_eq!(config.endpoint_url, None);
        assert!(!config.anonymous);
        assert!(!config.allow_http);
        // None means "fall back to FromEnv inside icechunk".
        assert!(creds.is_none());
    }

    #[test]
    fn parse_s3_options_anon_yields_anonymous_credentials() {
        let (config, creds) = parse_s3_options(&opts(&[("anon", "true")])).expect("ok");
        assert!(config.anonymous);
        assert!(matches!(creds, Some(S3Credentials::Anonymous)));
    }

    #[test]
    fn parse_s3_options_skip_signature_is_alias_for_anon() {
        let (_, creds) = parse_s3_options(&opts(&[("skip_signature", "True")])).expect("ok");
        assert!(matches!(creds, Some(S3Credentials::Anonymous)));
    }

    #[test]
    fn parse_s3_options_static_credentials() {
        let (_, creds) = parse_s3_options(&opts(&[
            ("access_key_id", "AKIA..."),
            ("secret_access_key", "shhh"),
            ("session_token", "tok"),
        ]))
        .expect("ok");
        match creds {
            Some(S3Credentials::Static(c)) => {
                assert_eq!(c.access_key_id, "AKIA...");
                assert_eq!(c.secret_access_key, "shhh");
                assert_eq!(c.session_token.as_deref(), Some("tok"));
            }
            other => panic!("expected Static, got {other:?}"),
        }
    }

    #[test]
    fn parse_s3_options_partial_static_creds_rejected() {
        let err = parse_s3_options(&opts(&[("access_key_id", "AKIA...")])).expect_err("partial");
        assert!(matches!(err, RustytreeError::InvalidInput(_)), "{err:?}");
    }

    #[test]
    fn parse_s3_options_region_and_endpoint_set() {
        let (config, _) = parse_s3_options(&opts(&[
            ("region", "us-east-1"),
            ("endpoint", "https://minio.local:9000"),
            ("allow_http", "true"),
        ]))
        .expect("ok");
        assert_eq!(config.region.as_deref(), Some("us-east-1"));
        assert_eq!(
            config.endpoint_url.as_deref(),
            Some("https://minio.local:9000")
        );
        assert!(config.allow_http);
    }

    #[test]
    fn parse_s3_options_unknown_key_rejected() {
        let err = parse_s3_options(&opts(&[("regin", "us-east-1")])).expect_err("typo");
        assert!(matches!(err, RustytreeError::InvalidInput(_)), "{err:?}");
    }

    #[test]
    fn parse_s3_options_invalid_bool_rejected() {
        let err = parse_s3_options(&opts(&[("anon", "maybe")])).expect_err("bad bool");
        assert!(matches!(err, RustytreeError::InvalidInput(_)), "{err:?}");
    }
}
