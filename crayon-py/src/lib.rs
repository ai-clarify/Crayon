//! Python client for Crayon.
//!
//! Wraps `crayon::client::ClusterClient` behind a private tokio runtime so
//! Python callers get plain blocking methods. Object and task ids travel as
//! hex strings; payloads as `bytes`. Task arguments are pythonic: a `bytes`
//! value is an inline argument, a `str` is an object id reference.
//!
//!     import crayon
//!     c = crayon.Client("127.0.0.1:7777")
//!     oid = c.put(b"payload")                       # shared-memory arena put
//!     task, out = c.submit("llm", "rollout", 1, [json.dumps(cfg).encode()])
//!     result = c.get(out, timeout_ms=120_000)
//!     c.release(oid)

use std::time::Duration;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes};

use ::crayon::{
    client::ClusterClient,
    ids::{ObjectId, TaskId},
    operation::{Codec, Operation, OperationDescriptor, OperationKey, TaskArg},
    resources::ResourceSet,
};

fn runtime_err(error: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

fn parse_object_id(id: &str) -> PyResult<ObjectId> {
    id.parse().map_err(runtime_err)
}

fn parse_codec(codec: &str) -> PyResult<Codec> {
    match codec {
        "json" => Ok(Codec::JsonV1),
        "bincode" => Ok(Codec::BincodeV1),
        "raw" => Ok(Codec::RawBytes),
        "pickle" => Ok(Codec::PythonPickleV1),
        other => Err(runtime_err(format!("unknown codec {other:?}"))),
    }
}

/// bytes -> inline argument, str -> object-id reference.
fn parse_args(args: &Bound<'_, PyAny>, codec: &Codec) -> PyResult<Vec<TaskArg>> {
    let mut out = Vec::new();
    for item in args.try_iter()? {
        let item = item?;
        if let Ok(bytes) = item.extract::<Vec<u8>>() {
            out.push(TaskArg::Inline {
                codec: codec.clone(),
                bytes,
            });
        } else if let Ok(id) = item.extract::<String>() {
            out.push(TaskArg::Object(parse_object_id(&id)?));
        } else {
            return Err(runtime_err("task args must be bytes (inline) or str (object id)"));
        }
    }
    Ok(out)
}

fn descriptor(namespace: &str, name: &str, version: u32, codec: &Codec) -> OperationDescriptor {
    OperationDescriptor {
        key: OperationKey::new(namespace, name, version),
        input_codec: codec.clone(),
        output_codec: codec.clone(),
        max_inline_arg_bytes: 64 * 1024,
    }
}

#[pyclass]
struct Client {
    runtime: tokio::runtime::Runtime,
    inner: ClusterClient,
}

#[pymethods]
impl Client {
    /// Connects and discovers the coordinator epoch (and, same-host, maps the
    /// shared-memory arena so puts and gets bypass the socket).
    #[new]
    fn new(address: String) -> PyResult<Self> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .map_err(runtime_err)?;
        let mut inner = ClusterClient::connect(address);
        runtime.block_on(inner.connect_epoch()).map_err(runtime_err)?;
        Ok(Self { runtime, inner })
    }

    /// Stores raw bytes; returns the hex object id. Same-host, the payload is
    /// written straight into the shared-memory arena (no size limit from RPC
    /// framing — gigabytes are fine).
    fn put(&self, py: Python<'_>, data: &[u8]) -> PyResult<String> {
        py.allow_threads(|| self.runtime.block_on(self.inner.put_bytes(data)))
            .map(|id| id.to_string())
            .map_err(runtime_err)
    }

    /// Fetches an object's bytes, waiting up to `timeout_ms` for a pending
    /// task output.
    #[pyo3(signature = (id, timeout_ms = 0))]
    fn get<'py>(&self, py: Python<'py>, id: &str, timeout_ms: u64) -> PyResult<Bound<'py, PyBytes>> {
        let id = parse_object_id(id)?;
        let (_codec, bytes) = py
            .allow_threads(|| {
                self.runtime
                    .block_on(self.inner.get_bytes_within(id, Duration::from_millis(timeout_ms)))
            })
            .map_err(runtime_err)?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// Fetches many objects in one blocking batch; one result per id, in order.
    #[pyo3(signature = (ids, timeout_ms = 0))]
    fn get_many<'py>(
        &self,
        py: Python<'py>,
        ids: Vec<String>,
        timeout_ms: u64,
    ) -> PyResult<Vec<Bound<'py, PyBytes>>> {
        let ids = ids
            .iter()
            .map(|id| parse_object_id(id))
            .collect::<PyResult<Vec<_>>>()?;
        let results = py
            .allow_threads(|| {
                self.runtime
                    .block_on(self.inner.get_bytes_many(&ids, Duration::from_millis(timeout_ms)))
            })
            .map_err(runtime_err)?;
        results
            .into_iter()
            .map(|result| {
                result
                    .map(|(_codec, bytes)| PyBytes::new(py, &bytes))
                    .map_err(runtime_err)
            })
            .collect()
    }

    /// `ray.wait`: blocks until at least `min_ready` of `ids` resolve or
    /// `timeout_ms` elapses, then returns `(index, bytes)` for every ready slot
    /// (index into `ids`). Lets a driver drain finished rollouts and re-`wait` on
    /// the stragglers instead of blocking on the slowest. Order follows `ids`.
    #[pyo3(signature = (ids, min_ready, timeout_ms = 0))]
    fn wait<'py>(
        &self,
        py: Python<'py>,
        ids: Vec<String>,
        min_ready: usize,
        timeout_ms: u64,
    ) -> PyResult<Vec<(usize, Bound<'py, PyBytes>)>> {
        let ids = ids
            .iter()
            .map(|id| parse_object_id(id))
            .collect::<PyResult<Vec<_>>>()?;
        let results = py
            .allow_threads(|| {
                self.runtime.block_on(self.inner.wait_bytes_many(
                    &ids,
                    min_ready,
                    Duration::from_millis(timeout_ms),
                ))
            })
            .map_err(runtime_err)?;
        Ok(results
            .into_iter()
            .enumerate()
            .filter_map(|(index, result)| {
                result.ok().map(|(_codec, bytes)| (index, PyBytes::new(py, &bytes)))
            })
            .collect())
    }

    /// Submits one task; returns `(task_id, output_id)` hex strings.
    #[pyo3(signature = (namespace, name, version, args, cpu = 1.0, max_attempts = 3, codec = "json"))]
    #[allow(clippy::too_many_arguments)]
    fn submit(
        &self,
        py: Python<'_>,
        namespace: &str,
        name: &str,
        version: u32,
        args: &Bound<'_, PyAny>,
        cpu: f64,
        max_attempts: u32,
        codec: &str,
    ) -> PyResult<(String, String)> {
        let codec = parse_codec(codec)?;
        let operation: Operation<serde_json::Value, serde_json::Value> =
            Operation::new(descriptor(namespace, name, version, &codec)).map_err(runtime_err)?;
        let args = parse_args(args, &codec)?;
        let resources = ResourceSet::cpu_gpu(cpu, 0.0).map_err(runtime_err)?;
        let handle = py
            .allow_threads(|| {
                self.runtime
                    .block_on(self.inner.submit(&operation, args, resources, max_attempts))
            })
            .map_err(runtime_err)?;
        Ok((handle.task_id.to_string(), handle.output.id.to_string()))
    }

    /// Submits many tasks of one operation in a single RPC; returns
    /// `(task_id, output_id)` per spec, in order.
    #[pyo3(signature = (namespace, name, version, batch, cpu = 1.0, max_attempts = 3, codec = "json"))]
    #[allow(clippy::too_many_arguments)]
    fn submit_batch(
        &self,
        py: Python<'_>,
        namespace: &str,
        name: &str,
        version: u32,
        batch: &Bound<'_, PyAny>,
        cpu: f64,
        max_attempts: u32,
        codec: &str,
    ) -> PyResult<Vec<(String, String)>> {
        let codec = parse_codec(codec)?;
        let operation: Operation<serde_json::Value, serde_json::Value> =
            Operation::new(descriptor(namespace, name, version, &codec)).map_err(runtime_err)?;
        let mut specs = Vec::new();
        for args in batch.try_iter()? {
            specs.push(parse_args(&args?, &codec)?);
        }
        let resources = ResourceSet::cpu_gpu(cpu, 0.0).map_err(runtime_err)?;
        let handles = py
            .allow_threads(|| {
                self.runtime.block_on(self.inner.submit_batch(
                    &operation,
                    specs,
                    resources,
                    max_attempts,
                ))
            })
            .map_err(runtime_err)?;
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .map(|h| (h.task_id.to_string(), h.output.id.to_string()))
                    .map_err(runtime_err)
            })
            .collect()
    }

    /// Releases an object so the cluster reclaims its storage.
    fn release(&self, py: Python<'_>, id: &str) -> PyResult<()> {
        let id = parse_object_id(id)?;
        py.allow_threads(|| self.runtime.block_on(self.inner.release(id)))
            .map_err(runtime_err)
    }

    fn cancel(&self, py: Python<'_>, task_id: &str) -> PyResult<()> {
        let task_id: TaskId = task_id.parse().map_err(runtime_err)?;
        py.allow_threads(|| self.runtime.block_on(self.inner.cancel(task_id)))
            .map_err(runtime_err)
    }

    /// Task status as a JSON string.
    fn status(&self, py: Python<'_>, task_id: &str) -> PyResult<String> {
        let task_id: TaskId = task_id.parse().map_err(runtime_err)?;
        let view = py
            .allow_threads(|| self.runtime.block_on(self.inner.status(task_id)))
            .map_err(runtime_err)?;
        serde_json::to_string(&view).map_err(runtime_err)
    }

    /// Registered workers as a JSON string.
    fn workers(&self, py: Python<'_>) -> PyResult<String> {
        let workers = py
            .allow_threads(|| self.runtime.block_on(self.inner.workers()))
            .map_err(runtime_err)?;
        serde_json::to_string(&workers).map_err(runtime_err)
    }
}

#[pymodule]
fn crayon(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Client>()?;
    Ok(())
}
