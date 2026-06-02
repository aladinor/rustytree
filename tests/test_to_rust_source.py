"""Unit tests for ``_to_rust_source``.

`backend.py::_to_rust_source` is the single dispatch site that decides
whether the user gave us an icechunk Session/IcechunkStore (in which
case we serialise to msgpack bytes for the Rust side) or a path/URL
(passed straight through as a string). Each input type is exercised
here so a future regression in the dispatch — e.g. an icechunk
attribute rename, or accidentally swallowing a non-string non-icechunk
input — surfaces in unit tests rather than in the integration paths.
"""

from __future__ import annotations

from pathlib import Path

import icechunk
import pytest

from rustytree.backend import (
    _is_python_credentials_fetcher_error,
    _rust_open_or_explain,
    _to_rust_source,
)


def test_session_input_returns_bytes(tiny_icechunk_repo: Path) -> None:
    storage = icechunk.local_filesystem_storage(str(tiny_icechunk_repo))
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session("main")

    out = _to_rust_source(session)
    assert isinstance(out, bytes)
    assert len(out) > 0


def test_icechunk_store_input_returns_bytes(tiny_icechunk_repo: Path) -> None:
    storage = icechunk.local_filesystem_storage(str(tiny_icechunk_repo))
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session("main")

    out = _to_rust_source(session.store)
    assert isinstance(out, bytes)
    assert len(out) > 0


def test_session_and_store_produce_equivalent_bytes(tiny_icechunk_repo: Path) -> None:
    """Both the `Session` and the `IcechunkStore` should serialise to the
    same underlying PySession bytes — they wrap the same inner state.
    """
    storage = icechunk.local_filesystem_storage(str(tiny_icechunk_repo))
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session("main")

    via_session = _to_rust_source(session)
    via_store = _to_rust_source(session.store)
    assert via_session == via_store


def test_str_input_returns_str() -> None:
    out = _to_rust_source("/some/local/path")
    assert isinstance(out, str)
    assert out == "/some/local/path"


def test_pathlib_path_returns_str(tmp_path: Path) -> None:
    out = _to_rust_source(tmp_path)
    assert isinstance(out, str)
    assert out == str(tmp_path)


def test_unknown_object_falls_back_to_str() -> None:
    """Non-icechunk, non-string objects get coerced via `str(...)`. This
    matches xarray's permissive `filename_or_obj` contract — the actual
    Rust dispatch will fail at the type check there with a clear error
    if the resulting string isn't a valid URL/path."""

    class Custom:
        def __str__(self) -> str:
            return "custom://opaque"

    out = _to_rust_source(Custom())
    assert isinstance(out, str)
    assert out == "custom://opaque"


def test_bytes_input_passes_through_identity() -> None:
    """`bytes` input short-circuits the icechunk encode path. This is
    relied on by `open_datatree`'s ancestor-merge loop (passes the
    already-serialised `source` so each ancestor open skips a fresh
    `session.as_bytes()` call). Identity-equality (`is`) pins that no
    copy or transform happens — re-encoding bytes would defeat the
    point of the short-circuit.
    """
    payload = b"\x00\x01\x02pre-serialised-session-bytes\xff"
    out = _to_rust_source(payload)
    assert out is payload


def test_works_when_icechunk_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the user doesn't have icechunk installed, `_to_rust_source`
    must still handle the str/Path branch — the icechunk import is
    deferred precisely to allow this. We simulate the missing import
    by unloading the module before calling."""
    import sys

    # Drop icechunk from sys.modules so the deferred import inside
    # `_to_rust_source` raises ImportError.
    monkeypatch.setitem(sys.modules, "icechunk", None)

    out = _to_rust_source("/path/with/no/icechunk")
    assert out == "/path/with/no/icechunk"


# --- credentials-fetcher drift detection + friendly error -------------------
#
# The Rust core normally deserializes arraylake/Earthmover sessions via the
# `src/py_credentials.rs` shim. These cover the Python-side safety net that
# explains a *version drift* failure (an icechunk-python that reshapes the
# fetcher), without needing S3/network.


def test_detects_unknown_variant_drift_message() -> None:
    exc = ValueError(
        "icechunk session: unknown error: unknown variant "
        "`PythonCredentialsFetcher`, there are no variants"
    )
    assert _is_python_credentials_fetcher_error(exc)


def test_detects_renamed_fetcher_via_unknown_variant() -> None:
    # Even if icechunk-python renames the fetcher, the "unknown variant ...
    # there are no variants" shape still trips the detector.
    exc = ValueError(
        "icechunk session: unknown error: unknown variant "
        "`SomeRenamedFetcher`, there are no variants"
    )
    assert _is_python_credentials_fetcher_error(exc)


def test_rejects_unrelated_value_error() -> None:
    exc = ValueError("icechunk session: invalid msgpack: corrupt blob")
    assert not _is_python_credentials_fetcher_error(exc)


def test_rust_open_or_explain_translates_credentials_error() -> None:
    def boom(_source: object, **_kwargs: object) -> object:
        raise ValueError(
            "icechunk session: unknown error: unknown variant "
            "`PythonCredentialsFetcher`, there are no variants"
        )

    with pytest.raises(ValueError, match="could not deserialize this icechunk session"):
        _rust_open_or_explain(boom, b"session-bytes")

    # The original error is chained for debuggability.
    with pytest.raises(ValueError) as excinfo:
        _rust_open_or_explain(boom, b"session-bytes")
    assert "issues/40" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ValueError)
    assert "PythonCredentialsFetcher" in str(excinfo.value.__cause__)


def test_rust_open_or_explain_passes_through_other_errors() -> None:
    def boom(_source: object, **_kwargs: object) -> object:
        raise ValueError("icechunk session: corrupt blob")

    with pytest.raises(ValueError, match="corrupt blob"):
        _rust_open_or_explain(boom, b"session-bytes")


def test_rust_open_or_explain_ignores_credentials_error_for_str_source() -> None:
    # A path/URL source can't carry a Python credentials fetcher; even a
    # matching message must re-raise unchanged rather than mislead.
    def boom(_source: object, **_kwargs: object) -> object:
        raise ValueError("unknown variant `X`, there are no variants")

    with pytest.raises(ValueError, match="unknown variant"):
        _rust_open_or_explain(boom, "/some/path")


def test_rust_open_or_explain_returns_result_on_success() -> None:
    sentinel = {"/": "node"}

    def ok(source: object, **kwargs: object) -> object:
        assert source == b"session-bytes"
        assert kwargs == {"group": "/g"}
        return sentinel

    assert _rust_open_or_explain(ok, b"session-bytes", group="/g") is sentinel
