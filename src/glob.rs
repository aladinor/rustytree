//! Conservative glob prefix predicate for the discovery walk.
//!
//! Compiled from the user's `group=` glob pattern. Used by
//! `walk_recursive` to prune subtrees that *cannot* match the
//! pattern — saving `Group::async_open` + `async_child_group_paths`
//! per pruned subtree on vanilla Zarr v3, and `Array::new_with_metadata`
//! + Phase C eager-fetch per skipped descendant on icechunk.
//!
//! The Python `_filter_by_glob` post-filter
//! (`pathlib.PurePosixPath.match`) is the source of truth for which
//! paths land in the result. This Rust predicate is purely an
//! optimisation: a false positive (we walk a path Python would
//! reject) is wasted work; a false negative (we skip a path Python
//! would accept) loses data. Bias toward false positives; when
//! unsure, walk it.
//!
//! Supported wildcard: `*` (zero or more non-slash chars). Patterns
//! using `?` or `[` fall back to "no prune" — the Python filter
//! still applies them correctly.

/// Conservative prefix predicate.
#[derive(Clone)]
pub(crate) struct GlobPredicate {
    /// Per-segment patterns when the pattern is absolute and uses
    /// only `*` wildcards. `None` means "can't safely prune anything"
    /// (relative pattern, or contains `?` / `[`).
    segments: Option<Vec<String>>,
}

impl GlobPredicate {
    /// Parse a glob pattern into a prefix predicate.
    pub(crate) fn parse(pattern: &str) -> Self {
        // Relative patterns match any suffix per `PurePosixPath.match`,
        // so we can't prune anything based on prefix.
        if !pattern.starts_with('/') {
            return Self { segments: None };
        }
        // Bail on `?` / `[`: we don't fully implement fnmatch, and the
        // Python filter handles them correctly anyway.
        if pattern.contains('?') || pattern.contains('[') {
            return Self { segments: None };
        }
        // Filter empty segments produced by `//` runs. `PurePosixPath`
        // coalesces `//` into a single separator (`Path("/foo/bar")`
        // matches `Path("/foo//bar")`), so we have to too — otherwise
        // the empty segment becomes a literal "" that nothing can
        // match, producing a false negative (data loss).
        let segments: Vec<String> = pattern
            .trim_start_matches('/')
            .trim_end_matches('/')
            .split('/')
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect();
        Self {
            segments: Some(segments),
        }
    }

    /// Can the walk discover a match at or below `path`?
    ///
    /// `path` is an absolute path like `/foo/bar`. Returns `true`
    /// unless we can prove no descendant of `path` could match the
    /// pattern. Conservative: when the predicate can't reason about
    /// the pattern (relative or unsupported wildcard), always returns
    /// `true` so the walk proceeds.
    pub(crate) fn could_descendant_match(&self, path: &str) -> bool {
        let Some(segs) = &self.segments else {
            return true;
        };
        let path_segs: Vec<&str> = path
            .trim_start_matches('/')
            .split('/')
            .filter(|s| !s.is_empty())
            .collect();
        // Absolute patterns require exact segment-count equality.
        // A path deeper than the pattern can't have a descendant
        // match (descending only adds more segments).
        if path_segs.len() > segs.len() {
            return false;
        }
        // Each path segment must match the corresponding pattern
        // segment. Mismatch on any prefix segment proves no
        // descendant can match.
        for (path_seg, pat_seg) in path_segs.iter().zip(segs.iter()) {
            if !star_match(pat_seg, path_seg) {
                return false;
            }
        }
        true
    }
}

/// Match a single name against a single pattern segment that may
/// contain zero or more `*` wildcards. Each `*` matches zero or more
/// non-slash characters.
fn star_match(pattern: &str, name: &str) -> bool {
    let parts: Vec<&str> = pattern.split('*').collect();
    // No `*`: exact compare.
    if parts.len() == 1 {
        return parts[0] == name;
    }
    // The first piece must prefix-match the name.
    if !name.starts_with(parts[0]) {
        return false;
    }
    // The last piece must suffix-match.
    let last = parts[parts.len() - 1];
    if !name.ends_with(last) {
        return false;
    }
    // The prefix and suffix together must not exceed the name length.
    if parts[0].len() + last.len() > name.len() {
        return false;
    }
    // Middle pieces must appear in order between the prefix and the
    // suffix. Walk a cursor through the name; each piece must be
    // findable forward of the cursor.
    let mut cursor = parts[0].len();
    let stop = name.len() - last.len();
    for piece in &parts[1..parts.len() - 1] {
        if cursor > stop {
            return false;
        }
        match name[cursor..stop].find(piece) {
            Some(idx) => cursor += idx + piece.len(),
            None => return false,
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::{GlobPredicate, star_match};

    #[test]
    fn star_match_literal() {
        assert!(star_match("sweep_0", "sweep_0"));
        assert!(!star_match("sweep_0", "sweep_1"));
    }

    #[test]
    fn star_match_wildcard() {
        assert!(star_match("*", "anything"));
        assert!(star_match("*", ""));
        assert!(star_match("VCP-*", "VCP-12"));
        assert!(!star_match("VCP-*", "VOL-12"));
        assert!(star_match("*sweep*", "my_sweep_0"));
    }

    #[test]
    fn star_match_empty_pattern() {
        assert!(star_match("", ""));
        assert!(!star_match("", "foo"));
    }

    #[test]
    fn predicate_skips_relative_pattern() {
        let p = GlobPredicate::parse("*/sweep_0");
        assert!(p.could_descendant_match("/anything"));
    }

    #[test]
    fn predicate_skips_unsupported_wildcards() {
        for pat in ["/[vV]olume/*", "/?CP-12/*"] {
            let p = GlobPredicate::parse(pat);
            assert!(p.could_descendant_match("/whatever"));
        }
    }

    #[test]
    fn predicate_descendant_match() {
        let p = GlobPredicate::parse("/*/sweep_0");
        assert!(p.could_descendant_match("/"));
        assert!(p.could_descendant_match("/VCP-12"));
        assert!(p.could_descendant_match("/VCP-12/sweep_0"));
        // Too deep; can't have a match below.
        assert!(!p.could_descendant_match("/VCP-12/sweep_0/extra"));
    }

    #[test]
    fn predicate_exact_depth_match() {
        // At the pattern's exact depth, `could_descendant_match`
        // doubles as a precise match check.
        let p = GlobPredicate::parse("/*/sweep_0");
        assert!(p.could_descendant_match("/VCP-12/sweep_0"));
        assert!(!p.could_descendant_match("/VCP-12/sweep_1"));
    }

    #[test]
    fn predicate_handles_trailing_slash() {
        // `PurePosixPath` strips trailing slash; we should too.
        let p = GlobPredicate::parse("/*/sweep_0/");
        assert!(p.could_descendant_match("/VCP-12/sweep_0"));
    }

    #[test]
    fn predicate_root_descendant_check() {
        let p = GlobPredicate::parse("/foo/bar");
        assert!(p.could_descendant_match("/"));
        assert!(p.could_descendant_match("/foo"));
        assert!(!p.could_descendant_match("/baz"));
    }

    #[test]
    fn predicate_collapses_double_slash() {
        // `PurePosixPath("/foo/bar").match("/foo//bar")` is `True`
        // because PurePosixPath coalesces `//`. We must collapse the
        // pattern's empty segments too — otherwise the empty literal
        // segment would never match and we'd produce a Rust false
        // negative (silent data loss). See `parse`'s filter.
        let p = GlobPredicate::parse("/foo//bar");
        assert!(p.could_descendant_match("/foo/bar"));
        assert!(p.could_descendant_match("/foo"));
    }
}
