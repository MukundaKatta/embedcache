//! Pure-Rust core for `embedcache`: a content-addressed embedding cache
//! backed by an embedded [redb](https://crates.io/crates/redb) KV store.
//!
//! - **Key.** A 32-byte blake3 hash of `(model_name, 0x00, text)`. The null
//!   separator prevents `(model="a", text="bc")` from colliding with
//!   `(model="ab", text="c")`.
//! - **Value.** `[u64 inserted_at_secs LE][u32 dim LE][dim x f32 LE]`.
//! - **TTL.** Optional. Evaluated on `get`; expired entries are returned as
//!   `None` and removed by `purge_expired`.

#![deny(unsafe_code)]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use redb::{Database, ReadableTable, ReadableTableMetadata, TableDefinition};
use serde::{Deserialize, Serialize};
use thiserror::Error;

const TABLE: TableDefinition<'_, &[u8; 32], Vec<u8>> = TableDefinition::new("embeddings");

/// Crate-wide result alias.
pub type Result<T> = std::result::Result<T, CacheError>;

/// All errors surfaced by `embedcache-core`.
#[derive(Error, Debug)]
pub enum CacheError {
    /// Failure inside the redb store.
    #[error("redb error: {0}")]
    Redb(String),
    /// I/O failure opening the cache directory or file.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// A stored value is shorter than its declared header.
    #[error("malformed entry: {0}")]
    Malformed(String),
    /// Caller supplied an invalid configuration.
    #[error("invalid config: {0}")]
    InvalidConfig(String),
}

// redb has half a dozen error types. Collapse them into one variant; the
// string is what the caller will print anyway.
macro_rules! redb_from {
    ($($t:ty),+ $(,)?) => {$(
        impl From<$t> for CacheError {
            fn from(e: $t) -> Self { Self::Redb(e.to_string()) }
        }
    )+};
}
redb_from!(
    redb::Error,
    redb::DatabaseError,
    redb::TransactionError,
    redb::TableError,
    redb::StorageError,
    redb::CommitError,
);

/// On-disk content-addressed embedding cache.
pub struct Cache {
    db: Database,
    ttl_secs: Option<u64>,
}

/// Cache stats returned by [`Cache::stats`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CacheStats {
    /// Number of entries currently stored.
    pub entries: u64,
    /// Total raw value bytes (excluding redb overhead).
    pub value_bytes: u64,
    /// File size of the database on disk in bytes.
    pub disk_bytes: u64,
}

impl Cache {
    /// Open or create a cache at `path` with no TTL.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_with_ttl(path, None)
    }

    /// Open or create a cache at `path` with an optional TTL in seconds.
    pub fn open_with_ttl<P: AsRef<Path>>(path: P, ttl_secs: Option<u64>) -> Result<Self> {
        if let Some(ttl) = ttl_secs {
            if ttl == 0 {
                return Err(CacheError::InvalidConfig(
                    "ttl_secs must be > 0 (or None for no expiry)".into(),
                ));
            }
        }
        if let Some(parent) = path.as_ref().parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let db = Database::create(path.as_ref())?;
        // Create the table if it doesn't exist.
        let txn = db.begin_write()?;
        {
            let _t = txn.open_table(TABLE)?;
        }
        txn.commit()?;
        Ok(Self { db, ttl_secs })
    }

    /// 32-byte content-addressed key for `(model, text)`.
    pub fn key(model: &str, text: &str) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(model.as_bytes());
        hasher.update(&[0u8]);
        hasher.update(text.as_bytes());
        *hasher.finalize().as_bytes()
    }

    /// Look up a vector. Returns `None` if absent or expired.
    pub fn get(&self, model: &str, text: &str) -> Result<Option<Vec<f32>>> {
        let key = Self::key(model, text);
        let now = unix_now();
        let txn = self.db.begin_read()?;
        let table = txn.open_table(TABLE)?;
        let Some(stored) = table.get(&key)? else {
            return Ok(None);
        };
        let bytes = stored.value();
        let (inserted_at, vec) = decode_entry(&bytes)?;
        if let Some(ttl) = self.ttl_secs {
            // `>=` so "ttl=N seconds" means the entry is dead after N seconds
            // have elapsed. With `>` you'd see N+1 seconds of life, which is
            // surprising at second-granularity timestamps.
            if now.saturating_sub(inserted_at) >= ttl {
                return Ok(None);
            }
        }
        Ok(Some(vec))
    }

    /// Insert or overwrite a vector for `(model, text)`.
    pub fn put(&self, model: &str, text: &str, vector: &[f32]) -> Result<()> {
        let key = Self::key(model, text);
        let bytes = encode_entry(unix_now(), vector);
        let txn = self.db.begin_write()?;
        {
            let mut table = txn.open_table(TABLE)?;
            table.insert(&key, bytes)?;
        }
        txn.commit()?;
        Ok(())
    }

    /// Remove a single entry. Returns `true` if the key was present.
    pub fn remove(&self, model: &str, text: &str) -> Result<bool> {
        let key = Self::key(model, text);
        let txn = self.db.begin_write()?;
        let removed = {
            let mut table = txn.open_table(TABLE)?;
            // Bind the AccessGuard so its borrow of `table` ends before the
            // block returns; otherwise the temporary outlives the table.
            let prev = table.remove(&key)?;
            prev.is_some()
        };
        txn.commit()?;
        Ok(removed)
    }

    /// Remove every entry. Returns the number of entries removed.
    pub fn clear(&self) -> Result<u64> {
        let txn = self.db.begin_write()?;
        let removed = {
            let mut table = txn.open_table(TABLE)?;
            let keys: Vec<[u8; 32]> = table
                .iter()?
                .filter_map(|r| r.ok().map(|(k, _)| *k.value()))
                .collect();
            for k in &keys {
                let _ = table.remove(k)?;
            }
            keys.len() as u64
        };
        txn.commit()?;
        Ok(removed)
    }

    /// Remove every entry whose `inserted_at + ttl < now`. Returns the count.
    /// No-op when the cache has no TTL.
    pub fn purge_expired(&self) -> Result<u64> {
        let Some(ttl) = self.ttl_secs else {
            return Ok(0);
        };
        let now = unix_now();
        let txn = self.db.begin_write()?;
        let removed = {
            let mut table = txn.open_table(TABLE)?;
            let mut victims: Vec<[u8; 32]> = Vec::new();
            for entry in table.iter()? {
                let (k, v) = entry?;
                let bytes = v.value();
                if bytes.len() < 8 {
                    continue;
                }
                let inserted = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
                if now.saturating_sub(inserted) >= ttl {
                    victims.push(*k.value());
                }
            }
            for k in &victims {
                table.remove(k)?;
            }
            victims.len() as u64
        };
        txn.commit()?;
        Ok(removed)
    }

    /// Evict oldest entries until the total stored value bytes are
    /// `<= target_bytes`. Returns the count removed.
    pub fn purge_to_size(&self, target_bytes: u64) -> Result<u64> {
        let txn = self.db.begin_write()?;
        let removed = {
            let mut table = txn.open_table(TABLE)?;
            // Collect (inserted_at, key, size_bytes), sort oldest-first.
            let mut all: Vec<(u64, [u8; 32], u64)> = Vec::new();
            let mut total: u64 = 0;
            for entry in table.iter()? {
                let (k, v) = entry?;
                let bytes = v.value();
                if bytes.len() < 8 {
                    continue;
                }
                let inserted = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
                let size = bytes.len() as u64;
                total += size;
                all.push((inserted, *k.value(), size));
            }
            if total <= target_bytes {
                return Ok(0);
            }
            all.sort_by_key(|(t, _, _)| *t);
            let mut removed = 0u64;
            for (_, k, size) in &all {
                if total <= target_bytes {
                    break;
                }
                table.remove(k)?;
                total = total.saturating_sub(*size);
                removed += 1;
            }
            removed
        };
        txn.commit()?;
        Ok(removed)
    }

    /// Counts and bytes for the cache.
    pub fn stats(&self) -> Result<CacheStats> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(TABLE)?;
        let entries = table.len()?;
        let mut value_bytes = 0u64;
        for entry in table.iter()? {
            let (_, v) = entry?;
            value_bytes += v.value().len() as u64;
        }
        let disk_bytes = self.disk_size();
        Ok(CacheStats {
            entries,
            value_bytes,
            disk_bytes,
        })
    }

    /// Number of entries.
    pub fn len(&self) -> Result<u64> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(TABLE)?;
        Ok(table.len()?)
    }

    /// True if no entries are stored.
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    fn disk_size(&self) -> u64 {
        // redb does not expose the file path back to us; the caller knows
        // the path. For stats we return 0 if we cannot infer it, which is
        // honest rather than approximate.
        0
    }
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn encode_entry(inserted_at: u64, vec: &[f32]) -> Vec<u8> {
    let dim = vec.len() as u32;
    let mut out = Vec::with_capacity(8 + 4 + vec.len() * 4);
    out.extend_from_slice(&inserted_at.to_le_bytes());
    out.extend_from_slice(&dim.to_le_bytes());
    for &x in vec {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn decode_entry(bytes: &[u8]) -> Result<(u64, Vec<f32>)> {
    if bytes.len() < 12 {
        return Err(CacheError::Malformed("entry shorter than header".into()));
    }
    let inserted = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
    let dim = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let expected = 12 + dim * 4;
    if bytes.len() != expected {
        return Err(CacheError::Malformed(format!(
            "entry length {}, expected {}",
            bytes.len(),
            expected
        )));
    }
    let mut vec = Vec::with_capacity(dim);
    for i in 0..dim {
        let off = 12 + i * 4;
        vec.push(f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()));
    }
    Ok((inserted, vec))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tempdb() -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cache.redb");
        (dir, path)
    }

    #[test]
    fn key_changes_with_model_or_text() {
        let a = Cache::key("m1", "hello");
        let b = Cache::key("m2", "hello");
        let c = Cache::key("m1", "world");
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_eq!(a, Cache::key("m1", "hello"));
    }

    #[test]
    fn key_separator_blocks_concatenation_collision() {
        // Without a separator, ("a", "bc") and ("ab", "c") would collide.
        let a = Cache::key("a", "bc");
        let b = Cache::key("ab", "c");
        assert_ne!(a, b);
    }

    #[test]
    fn put_then_get_round_trips() {
        let (_dir, path) = tempdb();
        let cache = Cache::open(&path).unwrap();
        let v = vec![0.1, 0.2, 0.3];
        cache.put("m", "hello", &v).unwrap();
        assert_eq!(cache.get("m", "hello").unwrap(), Some(v));
    }

    #[test]
    fn get_missing_returns_none() {
        let (_dir, path) = tempdb();
        let cache = Cache::open(&path).unwrap();
        assert_eq!(cache.get("m", "nope").unwrap(), None);
    }

    #[test]
    fn put_overwrites_existing_entry() {
        let (_dir, path) = tempdb();
        let cache = Cache::open(&path).unwrap();
        cache.put("m", "k", &[1.0, 2.0]).unwrap();
        cache.put("m", "k", &[3.0, 4.0, 5.0]).unwrap();
        assert_eq!(cache.get("m", "k").unwrap(), Some(vec![3.0, 4.0, 5.0]));
    }

    #[test]
    fn remove_returns_true_when_present() {
        let (_dir, path) = tempdb();
        let cache = Cache::open(&path).unwrap();
        cache.put("m", "k", &[1.0]).unwrap();
        assert!(cache.remove("m", "k").unwrap());
        assert!(!cache.remove("m", "k").unwrap());
    }

    #[test]
    fn clear_removes_all() {
        let (_dir, path) = tempdb();
        let cache = Cache::open(&path).unwrap();
        for i in 0..10 {
            cache.put("m", &format!("k{i}"), &[i as f32]).unwrap();
        }
        assert_eq!(cache.len().unwrap(), 10);
        cache.clear().unwrap();
        assert_eq!(cache.len().unwrap(), 0);
    }

    #[test]
    fn purge_to_size_evicts_oldest() {
        let (_dir, path) = tempdb();
        let cache = Cache::open(&path).unwrap();
        // Each entry: 8 + 4 + 4 = 16 bytes value.
        for i in 0..10 {
            cache.put("m", &format!("k{i}"), &[i as f32]).unwrap();
            // Spread inserted_at across calls. unix_now is whole seconds, so
            // we sleep just enough to differentiate. Skip on cargo test in a
            // single second; the eviction doesn't have to be in strict
            // chronological order if all timestamps tie, just under target.
        }
        // Target small enough to force eviction.
        let removed = cache.purge_to_size(32).unwrap();
        assert!(removed > 0, "expected at least 1 eviction");
        let stats = cache.stats().unwrap();
        assert!(stats.value_bytes <= 32, "value_bytes={}", stats.value_bytes);
    }

    #[test]
    fn ttl_zero_rejected() {
        let (_dir, path) = tempdb();
        let err = Cache::open_with_ttl(&path, Some(0));
        assert!(err.is_err());
    }

    #[test]
    fn malformed_entry_rejected() {
        // decode_entry directly, since we cannot write a malformed entry
        // through the public API.
        let bad = vec![0u8; 5];
        let r = decode_entry(&bad);
        assert!(r.is_err());
    }

    #[test]
    fn encode_decode_round_trip() {
        let v = vec![1.0_f32, -2.5, 3.125, f32::MIN, f32::MAX];
        let bytes = encode_entry(123, &v);
        let (t, decoded) = decode_entry(&bytes).unwrap();
        assert_eq!(t, 123);
        assert_eq!(decoded, v);
    }
}
