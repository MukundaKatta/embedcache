"""Shared fixtures for embedcache tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from embedcache import EmbedCache


@pytest.fixture
def cache(tmp_path: Path) -> Iterator[EmbedCache]:
    yield EmbedCache(tmp_path / "c.redb")


@pytest.fixture
def cache_with_ttl(tmp_path: Path) -> Iterator[EmbedCache]:
    yield EmbedCache(tmp_path / "c.redb", ttl_seconds=1)


@pytest.fixture
def vec3() -> np.ndarray:
    return np.array([0.1, 0.2, 0.3], dtype=np.float32)
