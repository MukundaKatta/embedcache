"""End-to-end tests for the Python facade."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from embedcache import EmbedCache, EmbedCacheError, key_for


def test_version_present() -> None:
    import embedcache

    assert isinstance(embedcache.__version__, str)
    assert embedcache.__version__ != ""


def test_put_get_round_trip(cache: EmbedCache, vec3: np.ndarray) -> None:
    cache.put("hello", "m", vec3)
    got = cache.get("hello", "m")
    assert got is not None
    assert got.dtype == np.float32
    assert np.array_equal(got, vec3)


def test_get_returns_none_on_miss(cache: EmbedCache) -> None:
    assert cache.get("nope", "m") is None


def test_put_overwrites(cache: EmbedCache) -> None:
    cache.put("k", "m", np.array([1.0], dtype=np.float32))
    cache.put("k", "m", np.array([2.0, 3.0], dtype=np.float32))
    assert np.array_equal(cache.get("k", "m"), np.array([2.0, 3.0], dtype=np.float32))


def test_remove(cache: EmbedCache, vec3: np.ndarray) -> None:
    cache.put("k", "m", vec3)
    assert cache.remove("k", "m") is True
    assert cache.remove("k", "m") is False
    assert cache.get("k", "m") is None


def test_clear_removes_all(cache: EmbedCache) -> None:
    for i in range(5):
        cache.put(f"k{i}", "m", np.array([float(i)], dtype=np.float32))
    assert len(cache) == 5
    removed = cache.clear()
    assert removed == 5
    assert len(cache) == 0


def test_contains_and_in_operator(cache: EmbedCache, vec3: np.ndarray) -> None:
    cache.put("hello", "m", vec3)
    assert cache.contains("hello", "m")
    assert ("hello", "m") in cache
    assert ("hello", "other_model") not in cache


def test_stats_shape(cache: EmbedCache, vec3: np.ndarray) -> None:
    cache.put("k", "m", vec3)
    s = cache.stats()
    assert set(s.keys()) == {"entries", "value_bytes", "disk_bytes"}
    assert s["entries"] == 1
    assert s["value_bytes"] > 0


def test_repr(cache: EmbedCache, vec3: np.ndarray) -> None:
    cache.put("k", "m", vec3)
    assert "EmbedCache" in repr(cache)


def test_get_or_compute_calls_once(cache: EmbedCache) -> None:
    calls: list[str] = []

    def compute(text: str) -> np.ndarray:
        calls.append(text)
        return np.array([1.0, 2.0], dtype=np.float32)

    cache.get_or_compute("hello", "m", compute)
    cache.get_or_compute("hello", "m", compute)
    cache.get_or_compute("hello", "m", compute)
    assert calls == ["hello"]


def test_get_or_compute_many_only_calls_for_misses(cache: EmbedCache) -> None:
    cache.put("a", "m", np.array([10.0], dtype=np.float32))
    captured: list[list[str]] = []

    def batch(missing: list[str]) -> list[np.ndarray]:
        captured.append(list(missing))
        return [np.array([float(i)], dtype=np.float32) for i, _ in enumerate(missing)]

    out = cache.get_or_compute_many(["a", "b", "c"], "m", batch)
    assert len(out) == 3
    # `a` was cached; only b, c hit batch.
    assert captured == [["b", "c"]]
    # Returned cached vector for `a`.
    assert out[0][0] == 10.0


def test_get_or_compute_many_dedupes_within_batch(cache: EmbedCache) -> None:
    captured: list[list[str]] = []

    def batch(missing: list[str]) -> list[np.ndarray]:
        captured.append(list(missing))
        return [np.array([float(i)], dtype=np.float32) for i, _ in enumerate(missing)]

    out = cache.get_or_compute_many(["x", "x", "y", "x", "y"], "m", batch)
    # Unique misses sent to batch: ["x", "y"] in first-seen order.
    assert captured == [["x", "y"]]
    # All x's resolve to same vec (the first computed for x).
    assert np.array_equal(out[0], out[1])
    assert np.array_equal(out[0], out[3])
    assert np.array_equal(out[2], out[4])


def test_get_or_compute_many_validates_batch_length(cache: EmbedCache) -> None:
    def bad_batch(missing: list[str]) -> list[np.ndarray]:
        return [np.array([1.0], dtype=np.float32)]  # too few

    with pytest.raises(ValueError, match="returned 1 vectors for 2 inputs"):
        cache.get_or_compute_many(["a", "b"], "m", bad_batch)


def test_put_rejects_2d_arrays(cache: EmbedCache) -> None:
    with pytest.raises(ValueError, match="must be 1-D"):
        cache.put("k", "m", np.zeros((3, 4), dtype=np.float32))


def test_put_coerces_dtype(cache: EmbedCache) -> None:
    cache.put("k", "m", np.array([1.0, 2.0], dtype=np.float64))
    got = cache.get("k", "m")
    assert got is not None
    assert got.dtype == np.float32


def test_ttl_expires(cache_with_ttl: EmbedCache, vec3: np.ndarray) -> None:
    cache_with_ttl.put("k", "m", vec3)
    assert cache_with_ttl.get("k", "m") is not None
    # ttl_seconds=1 means the entry is dead after >=1 elapsed second; sleep
    # generously over a second-boundary to remove flake risk on slow CI.
    time.sleep(2.1)
    assert cache_with_ttl.get("k", "m") is None


def test_ttl_zero_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        EmbedCache(tmp_path / "c.redb", ttl_seconds=0)


def test_purge_to_size(cache: EmbedCache) -> None:
    for i in range(20):
        cache.put(f"k{i}", "m", np.array([float(i)], dtype=np.float32))
    before = cache.stats()
    target = before["value_bytes"] // 4
    removed = cache.purge_to_size(target)
    after = cache.stats()
    assert removed > 0
    assert after["value_bytes"] <= target


def test_key_for_is_32_bytes() -> None:
    k = key_for("m", "hello")
    assert isinstance(k, bytes)
    assert len(k) == 32


def test_key_for_collision_resistance() -> None:
    # Without separator, ("a", "bc") and ("ab", "c") would collide.
    assert key_for("a", "bc") != key_for("ab", "c")


def test_native_error_class_exposed() -> None:
    assert issubclass(EmbedCacheError, Exception)
