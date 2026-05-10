# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-05-10

### Fixed
- `CacheStats::disk_bytes` now reports the actual file size in bytes
  rather than always returning 0. The `Cache` stashes its `Path` at open
  time so `stats()` can `metadata()` it accurately.

### Added
- `Cache::path()` accessor for callers that need the underlying database
  file path (backups, manual `fsync`, etc.).

## [0.1.0] - 2026-05-09

### Added

- Initial public release.
- Rust core crate `embedcache-core` with a redb-backed embedded KV store.
- Content-addressed cache keys via blake3 over `(text, model_name)`.
- Optional TTL (per-cache) with explicit `purge_expired()` and an optional
  `purge_to_size(bytes)` for size-bounded eviction.
- Python package `embedcache` with PyO3-backed native module
  (`embedcache._native`).
- Convenience `get_or_compute(text, model, fn)` and bulk
  `get_or_compute_many(texts, model, batch_fn)` that only invoke the user
  function on cache misses.
- abi3-py310 wheel: one wheel for CPython 3.10 through 3.13.

[Unreleased]: https://github.com/MukundaKatta/embedcache/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/MukundaKatta/embedcache/releases/tag/v0.1.0
