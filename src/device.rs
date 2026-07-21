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
use std::sync::OnceLock;

thread_local! {
    static CURRENT_DEVICE: Cell<Option<usize>> = const { Cell::new(None) };
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

/// Detect the number of CUDA GPUs on this node. Cached after first call —
/// GPU count never changes during a process lifetime, and the `nvidia-smi`
/// fallback can take 100ms+.
pub fn detect_gpu_count() -> usize {
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(detect_gpu_count_uncached)
}

fn detect_gpu_count_uncached() -> usize {
    // 1. CUDA_VISIBLE_DEVICES (respects container/device limits)
    if let Ok(visible) = std::env::var("CUDA_VISIBLE_DEVICES") {
        return if visible.is_empty() || visible == "-1" {
            0
        } else {
            visible.split(',').filter(|s| !s.trim().is_empty()).count()
        };
    }

    // 2. & 3. Try each detector in order; return the first positive count.
    let detectors: [fn() -> Option<usize>; 2] = [
        || {
            std::fs::read_dir("/proc/driver/nvidia/gpus/")
                .ok()
                .map(|e| e.filter(|e| e.is_ok()).count())
                .filter(|&c| c > 0)
        },
        || {
            std::process::Command::new("nvidia-smi")
                .arg("-L")
                .output()
                .ok()
                .filter(|o| o.status.success())
                .map(|o| {
                    String::from_utf8_lossy(&o.stdout)
                        .lines()
                        .filter(|l| l.starts_with("GPU "))
                        .count()
                })
                .filter(|&c| c > 0)
        },
    ];
    detectors.iter().find_map(|f| f()).unwrap_or(0)
}
