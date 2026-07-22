//! PyO3 Python bindings for the Crayon distributed runtime.
//!
//! Exposes a Python API mirroring Ray's core: `Ray`, `ObjectRef`, `Resources`,
//! and actor handles. Python objects are serialized via `pickle` and stored in
//! Crayon's object store as raw bytes.

use std::sync::Arc;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};

use crayon_rs::common::{ObjectID, TaskID};
use crayon_rs::resources::Resources as RustResources;
use crayon_rs::Ray as RustRay;

// ---------------------------------------------------------------------------
// ObjectRef
// ---------------------------------------------------------------------------

/// A reference to an object stored in Crayon's object store.
#[pyclass]
#[derive(Clone)]
struct ObjectRef {
    inner: crayon_rs::common::ObjectRef<Vec<u8>>,
}

#[pymethods]
impl ObjectRef {
    /// The object's unique identifier (hex string).
    #[getter]
    fn id(&self) -> String {
        format!("{}", self.inner.id)
    }

    #[getter]
    fn task_id(&self) -> Option<String> {
        self.inner.task_id().map(|id| id.to_string())
    }

    fn __repr__(&self) -> String {
        format!("ObjectRef({})", self.id())
    }
}

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------

/// Resource requirements for a task or actor.
#[pyclass]
#[derive(Clone, Copy)]
struct Resources {
    #[pyo3(get, set)]
    cpu: f64,
    #[pyo3(get, set)]
    gpu: f64,
}

#[pymethods]
impl Resources {
    #[new]
    #[pyo3(signature = (cpu=1.0, gpu=0.0))]
    fn new(cpu: f64, gpu: f64) -> Self {
        Resources { cpu, gpu }
    }

    fn __repr__(&self) -> String {
        format!("Resources(cpu={}, gpu={})", self.cpu, self.gpu)
    }
}

impl From<Resources> for RustResources {
    fn from(r: Resources) -> Self {
        RustResources::new(r.cpu, r.gpu)
    }
}

// ---------------------------------------------------------------------------
// ActorHandle
// ---------------------------------------------------------------------------

/// A handle to a stateful actor created via `Ray.create_actor`.
#[pyclass]
struct ActorHandle {
    inner: crayon_rs::actor::ActorHandle<Vec<u8>>,
    runtime: Arc<tokio::runtime::Runtime>,
}

#[pymethods]
impl ActorHandle {
    /// Call a method on the actor's state object.
    ///
    /// `method` is the name of the method to invoke on the state object;
    /// `args` are passed positionally. Returns an `ObjectRef` for the result.
    #[pyo3(signature = (method, *args))]
    fn call(&self, method: &str, args: Bound<'_, PyTuple>) -> PyResult<ObjectRef> {
        let method = method.to_string();
        let args_vec: Vec<PyObject> = args.iter().map(|a| a.unbind()).collect();

        let inner = self.inner.clone();
        let fut = async move {
            inner
                .call_result(
                    move |state: &mut Vec<u8>| -> Result<Vec<u8>, crayon_rs::common::CrayonError> {
                        Python::with_gil(|py| {
                            let state_obj = pickle_loads(py, state).map_err(|error| {
                                crayon_rs::common::CrayonError::TaskFailed(error.to_string())
                            })?;
                            let state_bound = state_obj.bind(py);
                            let method_obj =
                                state_bound.getattr(method.as_str()).map_err(|error| {
                                    crayon_rs::common::CrayonError::TaskFailed(error.to_string())
                                })?;
                            let args_tuple = PyTuple::new_bound(py, &args_vec);
                            let result = method_obj.call1(&args_tuple).map_err(|error| {
                                crayon_rs::common::CrayonError::TaskFailed(error.to_string())
                            })?;
                            *state = pickle_dumps(py, state_bound).map_err(|error| {
                                crayon_rs::common::CrayonError::Serialize(error.to_string())
                            })?;
                            pickle_dumps(py, &result).map_err(|error| {
                                crayon_rs::common::CrayonError::Serialize(error.to_string())
                            })
                        })
                    },
                )
                .await
        };

        let obj_ref = Python::with_gil(|py| {
            py.allow_threads(|| {
                self.runtime
                    .block_on(fut)
                    .map_err(|e| PyRuntimeError::new_err(format!("actor call failed: {:?}", e)))
            })
        })?;

        Ok(ObjectRef { inner: obj_ref })
    }

    /// Kill the actor. New calls will fail.
    fn kill(&self) {
        self.inner.kill();
    }

    fn __repr__(&self) -> String {
        format!("ActorHandle(id={})", self.inner.id())
    }
}

// ---------------------------------------------------------------------------
// Ray
// ---------------------------------------------------------------------------

/// The Crayon runtime handle.
#[pyclass]
struct Ray {
    inner: RustRay,
    runtime: Arc<tokio::runtime::Runtime>,
}

#[pymethods]
impl Ray {
    /// Initialize the runtime with `num_workers` worker threads.
    ///
    /// Memory management (LRU eviction + disk spilling) is enabled by default
    /// with a limit of 50% of available RAM. Pass `max_memory_bytes` and/or
    /// `spill_dir` to customize — this is what keeps Crayon from OOM-killing
    /// like Ray's plasma store on large RL workloads.
    #[new]
    #[pyo3(signature = (num_workers, max_memory_bytes=None, spill_dir=None))]
    fn new(
        num_workers: usize,
        max_memory_bytes: Option<usize>,
        spill_dir: Option<String>,
    ) -> PyResult<Self> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("failed to build tokio runtime: {e}")))?;
        let runtime = Arc::new(runtime);

        let inner = match (max_memory_bytes, spill_dir) {
            (None, None) => runtime.block_on(async { RustRay::init(num_workers) }),
            (mem, dir) => {
                let max_mem = mem.unwrap_or_else(|| {
                    // Reuse Rust's default detection when only spill_dir is given.
                    2 * 1024 * 1024 * 1024
                });
                let spill = dir.map(std::path::PathBuf::from).unwrap_or_else(|| {
                    std::env::temp_dir().join(format!("crayon-spill-{}", std::process::id()))
                });
                runtime.block_on(async { RustRay::init_with_memory(num_workers, max_mem, spill) })
            }
        };

        Ok(Ray { inner, runtime })
    }

    /// Store a Python object in the object store. Returns an `ObjectRef`.
    fn put(&self, value: &Bound<'_, PyAny>) -> PyResult<ObjectRef> {
        let py = value.py();
        let bytes = pickle_dumps(py, value)?;
        let obj_ref = self.inner.put_bytes(bytes);
        Ok(ObjectRef { inner: obj_ref })
    }

    /// Store a Python object in-process by reference (no serialization).
    ///
    /// The object is stored in the object store's local slot — shared with
    /// workers via `Arc`, zero copy. Only valid within a single process
    /// (Crayon's workers are in-process threads). This is the fast path for
    /// single-node RL: pass model weights, large tensors, etc. without
    /// pickling. Ray cannot do this because its workers are separate processes.
    fn put_local(&self, value: &Bound<'_, PyAny>) -> PyResult<ObjectRef> {
        let id = ObjectID::new();
        let obj = value.clone().unbind();
        self.inner.store().put_local(id, obj);
        let obj_ref = self.inner.store().reserve_ref::<Vec<u8>>(id);
        Ok(ObjectRef { inner: obj_ref })
    }

    /// Fetch an object by reference. Blocks until the object is available.
    fn get(&self, obj_ref: &ObjectRef) -> PyResult<PyObject> {
        // Fast path: in-process local object — zero copy, no pickle.
        if let Ok(local) = self.inner.store().get_local::<PyObject>(obj_ref.inner.id) {
            return Python::with_gil(|py| Ok(local.as_ref().clone_ref(py)));
        }
        let inner = self.inner.clone();
        let obj_ref = obj_ref.inner.clone();
        let bytes = Python::with_gil(|py| {
            py.allow_threads(|| {
                self.runtime
                    .block_on(async move { inner.get_bytes(&obj_ref).await })
                    .map(|bytes| bytes.to_vec())
            })
        })
        .map_err(|e| PyRuntimeError::new_err(format!("get failed: {:?}", e)))?;
        Python::with_gil(|py| pickle_loads(py, &bytes))
    }

    /// Fetch many objects concurrently. Returns a list of results.
    fn get_batch(&self, obj_refs: &Bound<'_, PyList>) -> PyResult<PyObject> {
        let store = self.inner.store();
        let py = obj_refs.py();

        // Pre-allocate results; local refs fill in-place, remote refs fetched later.
        let mut results: Vec<Option<PyObject>> = Vec::with_capacity(obj_refs.len());
        let mut remote_refs: Vec<crayon_rs::common::ObjectRef<Vec<u8>>> = Vec::new();
        let mut remote_positions: Vec<usize> = Vec::new();

        for (i, r) in obj_refs.iter().enumerate() {
            let r: ObjectRef = r.extract().unwrap();
            if let Ok(local) = store.get_local::<PyObject>(r.inner.id) {
                results.push(Some(local.as_ref().clone_ref(py)));
            } else {
                results.push(None);
                remote_refs.push(r.inner);
                remote_positions.push(i);
            }
        }

        // Fetch remote refs concurrently through the object store.
        if !remote_refs.is_empty() {
            let inner = self.inner.clone();
            let remote_results = py.allow_threads(|| {
                self.runtime.block_on(async move {
                    futures::future::join_all(
                        remote_refs
                            .iter()
                            .map(|reference| inner.get_bytes(reference)),
                    )
                    .await
                })
            });
            for (pos, res) in remote_positions.into_iter().zip(remote_results) {
                match res {
                    Ok(bytes) => results[pos] = Some(pickle_loads(py, &bytes)?),
                    Err(e) => {
                        results[pos] =
                            Some(PyRuntimeError::new_err(format!("{:?}", e)).to_object(py));
                    }
                }
            }
        }

        let list = PyList::empty_bound(py);
        for obj in results.into_iter().flatten() {
            list.append(obj)?;
        }
        Ok(list.unbind().into_any())
    }

    /// Run a Python callable as a remote task.
    ///
    /// `func` is called with the given positional `args`. Any `ObjectRef`
    /// arguments are automatically fetched (resolved) before `func` runs,
    /// mirroring Ray's automatic dependency resolution. Returns an `ObjectRef`
    /// for the result.
    #[pyo3(signature = (func, *args))]
    fn spawn(&self, func: &Bound<'_, PyAny>, args: Bound<'_, PyTuple>) -> PyResult<ObjectRef> {
        self.spawn_inner(func, args, None)
    }

    /// Like `spawn` but with explicit resource requirements.
    #[pyo3(signature = (func, resources, *args))]
    fn spawn_with_resources(
        &self,
        func: &Bound<'_, PyAny>,
        resources: Resources,
        args: Bound<'_, PyTuple>,
    ) -> PyResult<ObjectRef> {
        self.spawn_inner(func, args, Some(resources))
    }

    /// Shared implementation for `spawn` and `spawn_with_resources`.
    #[pyo3(signature = (func, args, resources=None))]
    fn spawn_inner(
        &self,
        func: &Bound<'_, PyAny>,
        args: Bound<'_, PyTuple>,
        resources: Option<Resources>,
    ) -> PyResult<ObjectRef> {
        let func_obj = Arc::new(func.clone().unbind());
        let store = self.inner.store();
        let mut ref_args = Vec::new();
        let mut plain_args = Vec::new();
        let mut is_ref = Vec::new();
        for arg in args.iter() {
            if let Ok(reference) = arg.extract::<ObjectRef>() {
                if let Ok(local) = store.get_local::<PyObject>(reference.inner.id) {
                    plain_args.push(local.as_ref().clone_ref(args.py()));
                    is_ref.push(false);
                } else {
                    ref_args.push(crayon_rs::args::RawBytesArg(reference.inner));
                    is_ref.push(true);
                }
            } else {
                plain_args.push(arg.unbind());
                is_ref.push(false);
            }
        }
        let plain_args = Arc::new(plain_args);
        let resources = resources
            .map(RustResources::from)
            .unwrap_or_else(RustResources::default_task);
        let obj_ref = self
            .inner
            .spawn_bytes_with_args(ref_args, resources, move |resolved| {
                run_task(
                    func_obj.clone(),
                    resolved,
                    plain_args.clone(),
                    is_ref.clone(),
                )
                .map_err(|error| crayon_rs::common::CrayonError::TaskFailed(error))
            });
        Ok(ObjectRef { inner: obj_ref })
    }

    /// Create a stateful actor with the given name and initial state.
    fn create_actor(&self, name: &str, state: &Bound<'_, PyAny>) -> PyResult<ActorHandle> {
        let py = state.py();
        let state_bytes = pickle_dumps(py, state)?;
        // Enter the runtime context so tokio::spawn inside spawn_actor works.
        let _guard = self.runtime.enter();
        let handle = self.inner.create_actor(name, state_bytes);
        Ok(ActorHandle {
            inner: handle,
            runtime: self.runtime.clone(),
        })
    }

    /// Snapshot the current system status as a Python dict.
    fn status(&self) -> PyResult<PyObject> {
        let status = self.inner.status();
        Python::with_gil(|py| {
            let dict = PyDict::new_bound(py);
            dict.set_item("objects", status.objects)?;
            dict.set_item("tasks_total", status.tasks_total)?;
            dict.set_item("tasks_finished", status.tasks_finished)?;
            dict.set_item("tasks_failed", status.tasks_failed)?;
            dict.set_item("tasks_cancelled", status.tasks_cancelled)?;
            dict.set_item("tasks_pending", status.tasks_pending)?;
            dict.set_item("tasks_running", status.tasks_running)?;
            dict.set_item("worker_utilization", status.worker_utilization)?;
            dict.set_item("memory_used_bytes", status.memory_used_bytes)?;
            dict.set_item("memory_limit_bytes", status.memory_limit_bytes)?;
            dict.set_item("spill_failures", status.spill_failures)?;

            let actors_list = PyList::empty_bound(py);
            for a in &status.actors {
                let entry = (
                    a.id.clone(),
                    a.name.clone(),
                    a.state.clone(),
                    a.pending,
                    a.completed,
                );
                actors_list.append(entry)?;
            }
            dict.set_item("actors", actors_list)?;

            let workers_list = PyList::empty_bound(py);
            for w in &status.workers {
                let entry = (w.id, w.busy, w.tasks_done);
                workers_list.append(entry)?;
            }
            dict.set_item("workers", workers_list)?;

            Ok(dict.unbind().into_any())
        })
    }

    /// Cancel a task by its id (hex string). Returns `False` if unknown.
    fn cancel(&self, task_id: &str) -> PyResult<bool> {
        let id = parse_task_id(task_id)?;
        Ok(self.inner.cancel(id))
    }

    fn __repr__(&self) -> String {
        "Ray(...)".to_string()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Serialize a Python object to bytes via `pickle.dumps`.
fn pickle_dumps(py: Python, obj: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    let pickle = py.import_bound("pickle")?;
    let bytes_obj = pickle.call_method1("dumps", (obj,))?;
    let bytes = bytes_obj.downcast::<PyBytes>()?;
    Ok(bytes.as_bytes().to_vec())
}

/// Deserialize bytes back to a Python object via `pickle.loads`.
fn pickle_loads(py: Python, bytes: &[u8]) -> PyResult<PyObject> {
    let pickle = py.import_bound("pickle")?;
    let py_bytes = PyBytes::new_bound(py, bytes);
    let obj = pickle.call_method1("loads", (py_bytes,))?;
    Ok(obj.unbind())
}

/// Execute a Python task function with the given resolved and plain arguments.
///
/// `resolved` contains the pickled bytes of ObjectRef args (in order), and
/// `plain_args` contains directly-passed Python objects (in order). `is_ref`
/// has one entry per original argument, indicating whether that slot is filled
/// from `resolved` (true) or `plain_args` (false). The two cursors walk
/// `resolved`/`plain_args` in lockstep with `is_ref` to reconstruct the full
/// argument list in its original order.
fn run_task(
    func_obj: Arc<PyObject>,
    resolved: Vec<Vec<u8>>,
    plain_args: Arc<Vec<PyObject>>,
    is_ref: Vec<bool>,
) -> Result<Vec<u8>, String> {
    Python::with_gil(|py| {
        let func = func_obj.bind(py);
        let mut resolved_iter = resolved.into_iter();
        let mut plain_iter = plain_args.iter();
        let mut call_args = Vec::with_capacity(is_ref.len());
        for is_reference in is_ref {
            if is_reference {
                let bytes = resolved_iter
                    .next()
                    .ok_or_else(|| "resolved argument missing".to_string())?;
                call_args.push(pickle_loads(py, &bytes).map_err(|error| error.to_string())?);
            } else {
                let value = plain_iter
                    .next()
                    .ok_or_else(|| "plain argument missing".to_string())?;
                call_args.push(value.clone_ref(py));
            }
        }
        let args_tuple = PyTuple::new_bound(py, &call_args);
        let result = func.call1(&args_tuple).map_err(|error| error.to_string())?;
        pickle_dumps(py, &result).map_err(|error| error.to_string())
    })
}

/// Parse a hex string into a `TaskID`.
fn parse_task_id(s: &str) -> PyResult<TaskID> {
    let bytes = hex::decode(s)
        .map_err(|e| PyRuntimeError::new_err(format!("invalid task id '{s}': {e}")))?;
    if bytes.len() != 16 {
        return Err(PyRuntimeError::new_err(format!(
            "task id must be 32 hex chars, got {}",
            bytes.len() * 2
        )));
    }
    let mut arr = [0u8; 16];
    arr.copy_from_slice(&bytes);
    let id: TaskID = bincode::deserialize(&arr)
        .map_err(|e| PyRuntimeError::new_err(format!("failed to parse task id: {e}")))?;
    Ok(id)
}

// ---------------------------------------------------------------------------
// VersionedStore — RL artifact versioning (checkpoints, rollouts, metrics)
// ---------------------------------------------------------------------------

/// Versioned artifact store for RL training.
///
/// Keeps a bounded history of named artifacts (policy weights, rollout batches,
/// metrics) so you can roll back, reproduce, and compare across training steps.
/// Old versions are evicted automatically — part of Crayon's OOM prevention.
#[pyclass]
struct VersionedStore {
    inner: crayon_rs::versioning::VersionedStore,
    runtime: Arc<tokio::runtime::Runtime>,
}

#[pymethods]
impl VersionedStore {
    /// Create a versioned store. `keep_last` bounds versions per name (0 = unlimited).
    #[new]
    fn new(ray: &Ray, keep_last: usize) -> PyResult<Self> {
        Ok(VersionedStore {
            inner: ray.inner.versioned_store(keep_last),
            runtime: ray.runtime.clone(),
        })
    }

    /// Store a new version of `name`. If `version` is None, auto-increments.
    /// Returns the version number used.
    #[pyo3(signature = (name, value, version=None))]
    fn put(&self, name: &str, value: &Bound<'_, PyAny>, version: Option<u64>) -> PyResult<u64> {
        let py = value.py();
        let bytes = pickle_dumps(py, value)?;
        let v = version.unwrap_or_else(|| self.inner.latest_version(name) + 1);
        let meta = self.inner.put_bytes_with_version(name, bytes, v);
        Ok(meta.version)
    }

    /// Get the latest version of `name`.
    fn get(&self, name: &str) -> PyResult<PyObject> {
        let inner = self.inner.clone();
        let name_owned = name.to_string();
        let bytes = Python::with_gil(|py| {
            py.allow_threads(|| {
                self.runtime.block_on(async move {
                    inner
                        .get_bytes(&name_owned)
                        .await
                        .map(|bytes| bytes.to_vec())
                })
            })
        })
        .ok_or_else(|| PyRuntimeError::new_err(format!("no version found for '{name}'")))?;
        Python::with_gil(|py| pickle_loads(py, &bytes))
    }

    /// Get a specific version of `name`.
    fn get_at(&self, name: &str, version: u64) -> PyResult<PyObject> {
        let inner = self.inner.clone();
        let name_owned = name.to_string();
        let bytes = Python::with_gil(|py| {
            py.allow_threads(|| {
                self.runtime.block_on(async move {
                    inner
                        .get_bytes_at(&name_owned, version)
                        .await
                        .map(|bytes| bytes.to_vec())
                })
            })
        })
        .ok_or_else(|| {
            PyRuntimeError::new_err(format!("version {version} of '{name}' not found"))
        })?;
        Python::with_gil(|py| pickle_loads(py, &bytes))
    }

    /// List all versions of `name` as (version, created_at) tuples.
    fn history(&self, name: &str) -> PyResult<PyObject> {
        let history = self.inner.history(name);
        Python::with_gil(|py| {
            let list = PyList::empty_bound(py);
            for m in history {
                list.append((m.version, m.created_at))?;
            }
            Ok(list.unbind().into_any())
        })
    }

    /// Latest version number for `name`, or 0 if none.
    fn latest_version(&self, name: &str) -> u64 {
        self.inner.latest_version(name)
    }

    /// Delete all versions of `name`.
    fn remove(&self, name: &str) {
        self.inner.remove(name);
    }
}

/// Module definition.
#[pymodule]
fn crayon(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Ray>()?;
    m.add_class::<ObjectRef>()?;
    m.add_class::<Resources>()?;
    m.add_class::<ActorHandle>()?;
    m.add_class::<VersionedStore>()?;
    Ok(())
}
