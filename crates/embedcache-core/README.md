# embedcache-core

Pure-Rust core for [embedcache](https://github.com/MukundaKatta/embedcache):
a content-addressed local embedding cache.

Backed by [redb](https://crates.io/crates/redb). Keys are blake3 hashes over
`(text, model_name)`. Values are length-prefixed `f32` little-endian vectors
plus an inserted-at timestamp so an optional TTL can be enforced.

## Quick example

```rust
use embedcache_core::Cache;

let dir = tempfile::tempdir()?;
let cache = Cache::open(dir.path().join("cache.redb"))?;
cache.put("hello world", "text-embedding-3-small", &vec![0.1, 0.2, 0.3])?;
let vec = cache.get("hello world", "text-embedding-3-small")?;
assert_eq!(vec.unwrap(), vec![0.1, 0.2, 0.3]);
# Ok::<(), embedcache_core::CacheError>(())
```

## License

Dual-licensed under MIT or Apache-2.0.
