//! GPU device detection and assignment.
//!
//! Crayon's resource model counts GPUs, but to actually use one a task needs
//! to know *which* device it was scheduled onto. This module detects available
//! GPUs and provides a thread-local [`current_device`] that the scheduler sets
//! before running a GPU task.
//!
//! Tasks read the device id and pass it to their ML framework:
//! ```ignore
//! let device = crayon::device::current_device().unwrap_or(0);
//! let model = Model::load(&candle_core::Device::Cuda(device))?;
//! ```

use std::cell::Cell;

thread_local! {
    static CURRENT_DEVICE: Cell<Option<usize>> = Cell::new(None);
}

/// The GPU device id assigned to the current worker thread, if any.
pub fn current_device() -> Option<usize> {
    CURRENT_DEVICE.with(|d| d.get())
}

/// Set the GPU device id for the current thread. Called by the scheduler
/// before running a GPU task.
pub(crate) fn set_current_device(device: Option<usize>) {
    CURRENT_DEVICE.with(|d| d.set(device));
}

/// Detect the number of CUDA GPUs on this node.
///
/// Order of checks:
/// 1. `CUDA_VISIBLE_DEVICES` env var (respects container/device limits)
/// 2. `/proc/driver/nvidia/gpus/` directory (Linux, no subprocess)
/// 3. `nvidia-smi -L` (fallback, requires nvidia-smi)
/// 4. 0 if none found
pub fn detect_gpu_count() -> usize {
    // 1. CUDA_VISIBLE_DEVICES
    if let Ok(visible) = std::env::var("CUDA_VISIBLE_DEVICES") {
        if !visible.is_empty() && visible != "-1" {
            return visible.split(',').filter(|s| !s.trim().is_empty()).count();
        }
        return 0;
    }

    // 2. /proc/driver/nvidia/gpus/ (Linux)
    if let Ok(entries) = std::fs::read_dir("/proc/driver/nvidia/gpus/") {
        let count = entries.filter(|e| e.is_ok()).count();
        if count > 0 {
            return count;
        }
    }

    // 3. nvidia-smi -L
    if let Ok(out) = std::process::Command::new("nvidia-smi")
        .arg("-L")
        .output()
    {
        if out.status.success() {
            let s = String::from_utf8_lossy(&out.stdout);
            let count = s.lines().filter(|l| l.starts_with("GPU ")).count();
            if count > 0 {
                return count;
            }
        }
    }

    0
}
