//! Parser for the `path` argument to `open_datatree`.
//!
//! Maps a string into a `StoreSpec` describing where to read from. Today:
//! local-filesystem paths (`file://...` or bare) and S3 URLs (`s3://...`).
//! GCS / Azure / HTTP and icechunk-on-remote-storage land in follow-up PRs.

use std::path::PathBuf;

use crate::error::{Result, RustytreeError};

/// Where to read a Zarr v3 store from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum StoreSpec {
    /// Local filesystem (icechunk vs vanilla auto-detected at the path).
    Local(PathBuf),
    /// S3 bucket + prefix (vanilla Zarr v3 only in this build —
    /// icechunk-on-S3 lands later).
    S3 {
        /// S3 bucket name.
        bucket: String,
        /// Object-key prefix (no leading slash, no trailing slash).
        prefix: String,
    },
}

/// Parse the `path` argument into a `StoreSpec`.
///
/// Supported inputs:
/// - bare paths: `/abs/path`, `relative/path`
/// - `file://` URLs: `file:///abs/path`
/// - `s3://` URLs: `s3://bucket`, `s3://bucket/prefix/store.zarr`
///
/// Unsupported schemes are rejected with a clear error message; future PRs
/// add `gs://`, `az://` / `abfs://`, and `http(s)://`.
pub(crate) fn parse_store_spec(input: &str) -> Result<StoreSpec> {
    if let Some(rest) = input.strip_prefix("file://") {
        return Ok(StoreSpec::Local(PathBuf::from(rest)));
    }
    if let Some(rest) = input.strip_prefix("s3://") {
        let (bucket, prefix) = rest.split_once('/').unwrap_or((rest, ""));
        if bucket.is_empty() {
            return Err(RustytreeError::InvalidInput(format!(
                "s3 URL is missing a bucket name: {input}"
            )));
        }
        return Ok(StoreSpec::S3 {
            bucket: bucket.to_string(),
            prefix: prefix.trim_matches('/').to_string(),
        });
    }
    if input.contains("://") {
        let scheme = input.split_once("://").map_or("", |(s, _)| s);
        return Err(RustytreeError::InvalidInput(format!(
            "unsupported URL scheme `{scheme}://` in {input}; \
             rustytree currently supports `file://` and `s3://` only"
        )));
    }
    Ok(StoreSpec::Local(PathBuf::from(input)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unwrap_local(spec: StoreSpec) -> PathBuf {
        match spec {
            StoreSpec::Local(p) => p,
            spec @ StoreSpec::S3 { .. } => panic!("expected Local, got {spec:?}"),
        }
    }

    fn unwrap_s3(spec: StoreSpec) -> (String, String) {
        match spec {
            StoreSpec::S3 { bucket, prefix } => (bucket, prefix),
            spec @ StoreSpec::Local(_) => panic!("expected S3, got {spec:?}"),
        }
    }

    #[test]
    fn bare_path_is_local() {
        let spec = parse_store_spec("/data/store.zarr").expect("parse ok");
        assert_eq!(unwrap_local(spec), PathBuf::from("/data/store.zarr"));
    }

    #[test]
    fn relative_path_is_local() {
        let spec = parse_store_spec("relative/store").expect("parse ok");
        assert_eq!(unwrap_local(spec), PathBuf::from("relative/store"));
    }

    #[test]
    fn file_url_is_local() {
        let spec = parse_store_spec("file:///data/store.zarr").expect("parse ok");
        assert_eq!(unwrap_local(spec), PathBuf::from("/data/store.zarr"));
    }

    #[test]
    fn s3_url_with_prefix() {
        let spec = parse_store_spec("s3://my-bucket/path/to/store.zarr").expect("parse ok");
        let (bucket, prefix) = unwrap_s3(spec);
        assert_eq!(bucket, "my-bucket");
        assert_eq!(prefix, "path/to/store.zarr");
    }

    #[test]
    fn s3_url_no_prefix() {
        let spec = parse_store_spec("s3://my-bucket").expect("parse ok");
        let (bucket, prefix) = unwrap_s3(spec);
        assert_eq!(bucket, "my-bucket");
        assert_eq!(prefix, "");
    }

    #[test]
    fn s3_url_trailing_slash_normalised() {
        let spec = parse_store_spec("s3://my-bucket/").expect("parse ok");
        let (bucket, prefix) = unwrap_s3(spec);
        assert_eq!(bucket, "my-bucket");
        assert_eq!(prefix, "");
    }

    #[test]
    fn s3_url_with_leading_and_trailing_slashes_in_prefix() {
        let spec = parse_store_spec("s3://b//p/").expect("parse ok");
        let (bucket, prefix) = unwrap_s3(spec);
        assert_eq!(bucket, "b");
        assert_eq!(prefix, "p");
    }

    #[test]
    fn s3_url_without_bucket_rejected() {
        let err = parse_store_spec("s3:///orphan-prefix").expect_err("missing bucket");
        assert!(
            matches!(err, RustytreeError::InvalidInput(_)),
            "got {err:?}"
        );
    }

    #[test]
    fn unsupported_scheme_rejected() {
        let err = parse_store_spec("gs://bucket/store.zarr").expect_err("gs not supported yet");
        let msg = err.to_string();
        assert!(msg.contains("gs"), "{msg}");
        assert!(msg.contains("supports"), "{msg}");
    }
}
