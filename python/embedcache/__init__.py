"""Content-addressed local embedding cache.

The native Rust core (``embedcache._native``) handles redb storage and
blake3 hashing. This module re-exports its public types and provides the
``get_or_compute`` and ``get_or_compute_many`` convenience methods that
make caching a one-line drop-in around any embedding function.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from importlib import metadata
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from embedcache._native import (
    EmbedCache as _NativeCache,
)
from embedcache._native import (
    EmbedCacheError,
    key_for,
)


def _read_version() -> str:
    try:
        return metadata.version("embedcache")
    except metadata.PackageNotFoundError:
        return "0.0.0"


__version__: Final[str] = _read_version()

__all__ = [
    "EmbedCache",
    "EmbedCacheError",
    "__version__",
    "key_for",
]


class EmbedCache:
    """Disk-backed content-addressed embedding cache.

    Wraps the Rust-backed `_native.EmbedCache`. The added value over the
    raw native object is the `get_or_compute` / `get_or_compute_many`
    helpers that integrate with whatever embedding function you already
    have.
    """

    def __init__(self, path: str | Path, *, ttl_seconds: int | None = None) -> None:
        self._inner = _NativeCache(str(Path(path)), ttl_seconds=ttl_seconds)

    def get(self, text: str, model: str) -> NDArray[np.float32] | None:
        """Return the cached vector for `(text, model)` or `None`."""
        return self._inner.get(text, model)

    def put(self, text: str, model: str, vector: NDArray[np.float32]) -> None:
        """Insert or overwrite a vector for `(text, model)`."""
        if vector.dtype != np.float32:
            vector = vector.astype(np.float32, copy=False)
        if vector.ndim != 1:
            raise ValueError(f"vector must be 1-D, got shape {vector.shape}")
        self._inner.put(text, model, vector)

    def remove(self, text: str, model: str) -> bool:
        """Remove a single entry; return ``True`` if it existed."""
        return bool(self._inner.remove(text, model))

    def clear(self) -> int:
        """Remove all entries; return the count removed."""
        return int(self._inner.clear())

    def purge_expired(self) -> int:
        """Remove every expired entry; return the count removed."""
        return int(self._inner.purge_expired())

    def purge_to_size(self, max_bytes: int) -> int:
        """Evict oldest entries until total bytes <= `max_bytes`."""
        return int(self._inner.purge_to_size(max_bytes))

    def stats(self) -> dict[str, int]:
        """Return entries / value_bytes / disk_bytes."""
        return dict(self._inner.stats())

    def contains(self, text: str, model: str) -> bool:
        """`True` if `(text, model)` has a non-expired entry."""
        return bool(self._inner.contains(text, model))

    def __len__(self) -> int:
        return len(self._inner)

    def __contains__(self, key: tuple[str, str]) -> bool:
        text, model = key
        return self.contains(text, model)

    def __repr__(self) -> str:
        return repr(self._inner)

    def get_or_compute(
        self,
        text: str,
        model: str,
        compute: Callable[[str], NDArray[np.float32]],
    ) -> NDArray[np.float32]:
        """Return the cached vector or call `compute(text)` and cache it.

        `compute` is invoked exactly once for a cache miss. The returned
        array is stored as ``np.float32`` regardless of dtype and must be
        1-D.
        """
        cached = self.get(text, model)
        if cached is not None:
            return cached
        vec = compute(text)
        self.put(text, model, vec)
        return vec

    def get_or_compute_many(
        self,
        texts: Sequence[str],
        model: str,
        compute_batch: Callable[[list[str]], list[NDArray[np.float32]]],
    ) -> list[NDArray[np.float32]]:
        """Return one vector per input text. Calls `compute_batch` only on misses.

        `compute_batch` receives the deduplicated list of missing texts and
        must return a list of the same length, in the same order. The
        returned list aligns 1:1 with `texts`.
        """
        results: list[NDArray[np.float32] | None] = [None] * len(texts)
        seen_misses: dict[str, list[int]] = {}

        for i, t in enumerate(texts):
            cached = self.get(t, model)
            if cached is not None:
                results[i] = cached
            else:
                seen_misses.setdefault(t, []).append(i)

        if seen_misses:
            unique_misses = list(seen_misses.keys())
            new_vecs = compute_batch(unique_misses)
            if len(new_vecs) != len(unique_misses):
                raise ValueError(
                    f"compute_batch returned {len(new_vecs)} vectors for "
                    f"{len(unique_misses)} inputs"
                )
            for text, vec in zip(unique_misses, new_vecs, strict=True):
                self.put(text, model, vec)
                for i in seen_misses[text]:
                    results[i] = vec

        out: list[NDArray[np.float32]] = []
        for i, r in enumerate(results):
            if r is None:  # pragma: no cover - logic above guarantees populated
                raise RuntimeError(f"index {i} unexpectedly missing")
            out.append(r)
        return out
