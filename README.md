# embedcache

Content-addressed local embedding cache. Skip duplicate embedding API calls.
Rust core, Python frontend.

## The problem

You re-embed the same documents over and over. Some are identical, some differ
by a trailing newline, all of them cost real money (and time) to embed at the
provider. The fix is a content-addressed cache keyed on the exact bytes you
would have sent: same input + same model → cached vector, otherwise compute.

`embedcache` is that cache, fast enough that the lookup overhead is below the
network round-trip you would have paid otherwise.

## Install

```bash
pip install embedcache
```

## 30-second quickstart

```python
import numpy as np
from embedcache import EmbedCache

cache = EmbedCache("./.embedcache.redb", ttl_seconds=86400 * 30)

def embed(text: str) -> np.ndarray:
    # your real call: openai.embeddings.create(...), bedrock, cohere, etc.
    return np.zeros(384, dtype=np.float32)

vec = cache.get_or_compute("hello world", "text-embedding-3-small", embed)
```

For bulk ingestion, `get_or_compute_many` calls your batch function only on
the misses:

```python
texts = ["a", "b", "c", "d"]

def embed_batch(missing: list[str]) -> list[np.ndarray]:
    return [np.zeros(384, dtype=np.float32) for _ in missing]

vectors = cache.get_or_compute_many(texts, "text-embedding-3-small", embed_batch)
```

## Why it is fast

- **Hashing.** blake3 keys (~5x faster than SHA-256 on the 1–10 KB strings
  most prompts are).
- **Storage.** redb is an embedded ACID KV store with a single-file format,
  log-structured, no Python in the hot path.
- **GIL.** PyO3 releases the GIL on every `get`/`put`, so a Python thread
  pool calling the cache from a batch loop does not serialize on the cache.

## API

```python
class EmbedCache:
    def __init__(
        self,
        path: str | Path,
        *,
        ttl_seconds: int | None = None,
    ) -> None: ...

    def get(self, text: str, model: str) -> NDArray[np.float32] | None: ...
    def put(self, text: str, model: str, vector: NDArray[np.float32]) -> None: ...

    def get_or_compute(
        self,
        text: str,
        model: str,
        compute: Callable[[str], NDArray[np.float32]],
    ) -> NDArray[np.float32]: ...

    def get_or_compute_many(
        self,
        texts: Sequence[str],
        model: str,
        compute_batch: Callable[[list[str]], list[NDArray[np.float32]]],
    ) -> list[NDArray[np.float32]]: ...

    def purge_expired(self) -> int: ...
    def purge_to_size(self, max_bytes: int) -> int: ...
    def clear(self) -> None: ...
    def stats(self) -> dict[str, int]: ...
    def __len__(self) -> int: ...
```

## License

Dual-licensed under MIT or Apache-2.0 at your option.
