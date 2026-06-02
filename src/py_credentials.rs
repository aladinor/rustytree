//! Cross-extension credential shim for icechunk sessions built by
//! `icechunk-python` (and tools layered on it, e.g. arraylake / Earthmover).
//!
//! ## Why this module exists
//!
//! When a user constructs an icechunk `Session` with *refreshable* object-store
//! credentials, icechunk stores them as
//! `S3Credentials::Refreshable(Arc<dyn S3CredentialsFetcher>)` (and the GCS /
//! Azure analogues). `S3CredentialsFetcher` is a `#[typetag::serde]` trait, so
//! the serialized session carries a tag naming the concrete fetcher type.
//! `icechunk-python` registers its fetcher under the type name
//! **`PythonCredentialsFetcher`** — but only inside *its own* `_icechunk_python`
//! cdylib. rustytree's `_rustytree` cdylib links the vanilla `icechunk` crate,
//! whose `S3CredentialsFetcher` typetag registry has **no** impls registered, so
//! `Session::from_bytes` fails with:
//!
//! ```text
//! unknown variant `PythonCredentialsFetcher`, there are no variants
//! ```
//!
//! `xr.open_zarr(session.store)` never hits this because it talks to the *live*
//! in-process `IcechunkStore` and never round-trips through bytes;
//! `icechunk-python`'s own pickling round-trips inside its own cdylib where the
//! fetcher *is* registered. rustytree is the only consumer that deserializes the
//! bytes in a *foreign* cdylib — hence the gap.
//!
//! ## What this module does
//!
//! It re-registers, inside rustytree's cdylib, a fetcher named
//! `PythonCredentialsFetcher` for each of the three credential traits, with the
//! same serialized shape `icechunk-python` produces
//! (`{ pickled_function: bytes, initial: Option<Creds> }`). Once registered,
//! `Session::from_bytes` resolves the tag and deserialization succeeds.
//!
//! `get()` mirrors `icechunk-python`'s own behaviour
//! (`icechunk-python/src/config.rs`): return the scattered `initial` static
//! credentials while they are still fresh; otherwise acquire the GIL and re-run
//! the embedded pickled Python credential function to refresh them — exactly
//! what `icechunk-python` does, so long walks and
//! `scatter_initial_credentials=False` sessions both keep working.
//!
//! ## Fragility
//!
//! This couples rustytree to `icechunk-python`'s internal serialized shape (the
//! `PythonCredentialsFetcher` type name and its `pickled_function` / `initial`
//! fields). That coupling sits inside the byte-format coupling the project
//! already accepts by pinning `icechunk = "=2.0.5"` and requiring a matching
//! `icechunk-python`. If a future `icechunk-python` renames the type, the tag no
//! longer resolves and the friendly error in `python/rustytree/backend.py`
//! explains the version mismatch. See
//! <https://github.com/aladinor/rustytree/issues/40>.
//!
//! The module has no direct callers: the `#[typetag::serde]` impls register
//! themselves into `inventory` at link time, and icechunk's `Session`
//! deserializer finds them there. `mod py_credentials;` in `lib.rs` is what pulls
//! the registrations into the cdylib.

use async_trait::async_trait;
use chrono::{DateTime, TimeDelta, Utc};
use icechunk::storage::{
    AzureCredentialsFetcher, AzureRefreshableCredential, GcsBearerCredential,
    GcsCredentialsFetcher, S3CredentialsFetcher, S3StaticCredentials,
};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::{Deserialize, Serialize};

/// Re-fetch credentials this many seconds *before* their stated expiry. Matches
/// the lower bound of `icechunk-python`'s 120-180s refresh window; we use a
/// fixed value (not a random one) because rustytree's read-only walk doesn't
/// have a credential-stampede problem worth jittering for.
const REFRESH_BUFFER_SECS: i64 = 180;

/// Wire-compatible mirror of `icechunk-python`'s internal
/// `PythonCredentialsFetcher<C>` (see module docs). The field names and order
/// must match what `icechunk-python` serializes: `pickled_function` then
/// `initial`.
///
/// `C` is the per-backend static credential type the embedded Python callable
/// returns: `S3StaticCredentials`, `GcsBearerCredential`, or
/// `AzureRefreshableCredential`.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct PythonCredentialsFetcher<C> {
    /// `pickle.dumps`-encoded Python callable returning fresh credentials. Run
    /// under the GIL on the refresh path; opaque to Rust otherwise.
    pickled_function: Vec<u8>,
    /// Credentials scattered at construction time
    /// (`scatter_initial_credentials=True`, arraylake's default). `None` when
    /// the user opted out, in which case every `get()` must run the callable.
    initial: Option<C>,
}

/// `true` when `expires_after` is absent or far enough in the future that the
/// cached credentials are still safe to hand out.
fn is_fresh(expires_after: Option<DateTime<Utc>>) -> bool {
    match expires_after {
        None => true,
        Some(expiry) => expiry > Utc::now() + TimeDelta::seconds(REFRESH_BUFFER_SECS),
    }
}

/// Unpickle the stored Python callable and invoke it with no arguments,
/// returning the Python credentials object it produces. Runs under the GIL.
fn call_pickled<'py>(py: Python<'py>, pickled: &[u8]) -> PyResult<Bound<'py, PyAny>> {
    let pickle = PyModule::import(py, "pickle")?;
    let callable = pickle
        .getattr("loads")?
        .call1((PyBytes::new(py, pickled),))?;
    callable.call0()
}

/// Build the actionable error string for a failed refresh. We can't run the
/// pickled callable in this process (missing imports, no network, etc.), so tell
/// the user how to sidestep it.
fn refresh_error(provider: &str, err: &PyErr) -> String {
    format!(
        "rustytree could not run the Python credential callback embedded in this \
         icechunk session to refresh {provider} credentials ({err}). This session \
         was likely created with delegated/refreshable credentials (e.g. \
         arraylake/Earthmover). Either rebuild it with anonymous or static \
         credentials, keep `scatter_initial_credentials=True` so fresh static \
         credentials travel with the session, or make the credential function \
         importable in this process."
    )
}

/// Read a Python `icechunk.S3StaticCredentials`-shaped object's attributes into
/// the Rust `S3StaticCredentials`. We duck-type rather than depend on
/// `icechunk-python`'s private `PyS3StaticCredentials` (it lives in another
/// cdylib and isn't reachable from here).
fn s3_from_pyobj(obj: &Bound<'_, PyAny>) -> PyResult<S3StaticCredentials> {
    Ok(S3StaticCredentials {
        access_key_id: obj.getattr("access_key_id")?.extract()?,
        secret_access_key: obj.getattr("secret_access_key")?.extract()?,
        session_token: obj.getattr("session_token")?.extract()?,
        expires_after: obj.getattr("expires_after")?.extract()?,
    })
}

/// Read a Python `icechunk.GcsBearerCredential`-shaped object into the Rust type.
fn gcs_from_pyobj(obj: &Bound<'_, PyAny>) -> PyResult<GcsBearerCredential> {
    Ok(GcsBearerCredential {
        bearer: obj.getattr("bearer")?.extract()?,
        expires_after: obj.getattr("expires_after")?.extract()?,
    })
}

/// Read a Python `icechunk.AzureRefreshableCredential`-shaped object into the
/// Rust enum. The variant is discriminated by which value-bearing attribute is
/// present (`key` / `token` / `bearer`); every variant also carries
/// `expires_after`.
fn azure_from_pyobj(obj: &Bound<'_, PyAny>) -> PyResult<AzureRefreshableCredential> {
    let expires_after = obj.getattr("expires_after")?.extract()?;
    if let Ok(key) = obj.getattr("key") {
        return Ok(AzureRefreshableCredential::AccessKey {
            key: key.extract()?,
            expires_after,
        });
    }
    if let Ok(token) = obj.getattr("token") {
        return Ok(AzureRefreshableCredential::SASToken {
            token: token.extract()?,
            expires_after,
        });
    }
    if let Ok(bearer) = obj.getattr("bearer") {
        return Ok(AzureRefreshableCredential::BearerToken {
            bearer: bearer.extract()?,
            expires_after,
        });
    }
    Err(pyo3::exceptions::PyValueError::new_err(
        "icechunk Azure credential object had none of `key`, `token`, or `bearer`",
    ))
}

#[async_trait]
#[typetag::serde]
impl S3CredentialsFetcher for PythonCredentialsFetcher<S3StaticCredentials> {
    async fn get(&self) -> Result<S3StaticCredentials, String> {
        if let Some(creds) = self.initial.as_ref()
            && is_fresh(creds.expires_after)
        {
            return Ok(creds.clone());
        }
        Python::attach(|py| s3_from_pyobj(&call_pickled(py, &self.pickled_function)?))
            .map_err(|err| refresh_error("S3", &err))
    }
}

#[async_trait]
#[typetag::serde]
impl GcsCredentialsFetcher for PythonCredentialsFetcher<GcsBearerCredential> {
    async fn get(&self) -> Result<GcsBearerCredential, String> {
        if let Some(creds) = self.initial.as_ref()
            && is_fresh(creds.expires_after)
        {
            return Ok(creds.clone());
        }
        Python::attach(|py| gcs_from_pyobj(&call_pickled(py, &self.pickled_function)?))
            .map_err(|err| refresh_error("GCS", &err))
    }
}

#[async_trait]
#[typetag::serde]
impl AzureCredentialsFetcher for PythonCredentialsFetcher<AzureRefreshableCredential> {
    async fn get(&self) -> Result<AzureRefreshableCredential, String> {
        if let Some(creds) = self.initial.as_ref()
            && is_fresh(creds.expires_after())
        {
            return Ok(creds.clone());
        }
        Python::attach(|py| azure_from_pyobj(&call_pickled(py, &self.pickled_function)?))
            .map_err(|err| refresh_error("Azure", &err))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The GIL-bound refresh path (`call_pickled`, `*_from_pyobj`,
    // `Python::attach`) needs a live interpreter, which conflicts with the
    // cdylib's `extension-module` feature in `cargo test`; it's covered by the
    // pytest integration tests instead. Here we cover the pure-Rust surface:
    // freshness logic and the typetag round-trip that makes deserialization work.

    #[test]
    fn fresh_when_no_expiry() {
        assert!(is_fresh(None));
    }

    #[test]
    fn stale_when_expired() {
        let past = Utc::now() - TimeDelta::hours(1);
        assert!(!is_fresh(Some(past)));
    }

    #[test]
    fn stale_inside_refresh_buffer() {
        // Expires in 60s, but the buffer is 180s — treat as stale so we refresh
        // before the credentials actually lapse mid-walk.
        let soon = Utc::now() + TimeDelta::seconds(60);
        assert!(!is_fresh(Some(soon)));
    }

    #[test]
    fn fresh_well_beyond_buffer() {
        let later = Utc::now() + TimeDelta::hours(1);
        assert!(is_fresh(Some(later)));
    }

    #[test]
    fn deserializes_python_credentials_fetcher_tag() {
        // Prove the typetag registration resolves the same tag
        // `icechunk-python` emits. Serialize an `S3Credentials::Refreshable`
        // wrapping our shim, then deserialize it back through the icechunk enum:
        // this is exactly the step that previously failed with
        // "unknown variant `PythonCredentialsFetcher`".
        use icechunk::storage::S3Credentials;
        use std::sync::Arc;

        let fetcher = PythonCredentialsFetcher::<S3StaticCredentials> {
            pickled_function: b"not-a-real-pickle".to_vec(),
            initial: Some(S3StaticCredentials {
                access_key_id: "AKIA".to_string(),
                secret_access_key: "secret".to_string(),
                session_token: None,
                expires_after: None,
            }),
        };
        let creds = S3Credentials::Refreshable(Arc::new(fetcher));

        let bytes = rmp_serde::to_vec(&creds).expect("serialize Refreshable creds");
        let decoded: S3Credentials =
            rmp_serde::from_slice(&bytes).expect("deserialize must resolve the typetag");

        match decoded {
            S3Credentials::Refreshable(_) => {}
            other => panic!("expected Refreshable, got {other:?}"),
        }
    }
}
