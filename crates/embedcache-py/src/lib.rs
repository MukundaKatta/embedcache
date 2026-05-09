//! PyO3 bindings exposing `embedcache_core` as `embedcache._native`.
//!
//! Heavy work (redb transactions, blake3 hashing) releases the GIL via
//! `py.allow_threads`. Vectors are returned as f32 numpy arrays.

use std::path::PathBuf;

use embedcache_core::{Cache, CacheError};
use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

pyo3::create_exception!(_native, EmbedCacheError, pyo3::exceptions::PyException);

fn map_err(e: CacheError) -> PyErr {
    match e {
        CacheError::InvalidConfig(msg) => PyValueError::new_err(msg),
        other => EmbedCacheError::new_err(other.to_string()),
    }
}

#[pyclass(name = "EmbedCache", module = "embedcache._native")]
struct PyEmbedCache {
    inner: Cache,
}

#[pymethods]
impl PyEmbedCache {
    #[new]
    #[pyo3(signature = (path, *, ttl_seconds=None))]
    fn new(path: PathBuf, ttl_seconds: Option<u64>) -> PyResult<Self> {
        let inner = Cache::open_with_ttl(path, ttl_seconds).map_err(map_err)?;
        Ok(Self { inner })
    }

    /// Look up a vector. Returns `None` on cache miss or expired entry.
    fn get<'py>(
        &self,
        py: Python<'py>,
        text: &str,
        model: &str,
    ) -> PyResult<Option<Bound<'py, PyArray1<f32>>>> {
        let text = text.to_owned();
        let model = model.to_owned();
        let result = py
            .allow_threads(move || self.inner.get(&model, &text))
            .map_err(map_err)?;
        Ok(result.map(|v| v.to_pyarray(py)))
    }

    /// Insert or overwrite a vector for `(text, model)`.
    fn put(
        &self,
        py: Python<'_>,
        text: &str,
        model: &str,
        vector: PyReadonlyArray1<'_, f32>,
    ) -> PyResult<()> {
        let text = text.to_owned();
        let model = model.to_owned();
        let owned: Vec<f32> = vector.as_slice()?.to_vec();
        py.allow_threads(move || self.inner.put(&model, &text, &owned))
            .map_err(map_err)
    }

    /// Remove a single entry. Returns `True` if it existed.
    fn remove(&self, py: Python<'_>, text: &str, model: &str) -> PyResult<bool> {
        let text = text.to_owned();
        let model = model.to_owned();
        py.allow_threads(move || self.inner.remove(&model, &text))
            .map_err(map_err)
    }

    /// Remove every entry. Returns the number removed.
    fn clear(&self, py: Python<'_>) -> PyResult<u64> {
        py.allow_threads(|| self.inner.clear()).map_err(map_err)
    }

    /// Remove every expired entry. Returns the number removed.
    fn purge_expired(&self, py: Python<'_>) -> PyResult<u64> {
        py.allow_threads(|| self.inner.purge_expired())
            .map_err(map_err)
    }

    /// Evict oldest entries until the value-bytes total is `<= max_bytes`.
    fn purge_to_size(&self, py: Python<'_>, max_bytes: u64) -> PyResult<u64> {
        py.allow_threads(move || self.inner.purge_to_size(max_bytes))
            .map_err(map_err)
    }

    /// Number of entries.
    fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
        let n = py.allow_threads(|| self.inner.len()).map_err(map_err)?;
        Ok(n as usize)
    }

    /// Membership test for `(text, model)`.
    #[pyo3(signature = (text, model))]
    fn contains(&self, py: Python<'_>, text: &str, model: &str) -> PyResult<bool> {
        let text = text.to_owned();
        let model = model.to_owned();
        let r = py
            .allow_threads(move || self.inner.get(&model, &text))
            .map_err(map_err)?;
        Ok(r.is_some())
    }

    /// Stats dict: `entries`, `value_bytes`, `disk_bytes`.
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = py.allow_threads(|| self.inner.stats()).map_err(map_err)?;
        let dict = PyDict::new(py);
        dict.set_item("entries", s.entries)?;
        dict.set_item("value_bytes", s.value_bytes)?;
        dict.set_item("disk_bytes", s.disk_bytes)?;
        Ok(dict)
    }

    fn __repr__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyString>> {
        let n = py.allow_threads(|| self.inner.len()).map_err(map_err)?;
        Ok(PyString::new(py, &format!("EmbedCache(entries={n})")))
    }
}

/// Module-level helper that callers rarely need but is useful for migrations.
#[pyfunction]
fn key_for(model: &str, text: &str) -> [u8; 32] {
    Cache::key(model, text)
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("EmbedCacheError", m.py().get_type::<EmbedCacheError>())?;
    m.add_class::<PyEmbedCache>()?;
    m.add_function(wrap_pyfunction!(key_for, m)?)?;
    Ok(())
}
