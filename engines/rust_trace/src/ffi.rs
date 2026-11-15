//! C ABI surface consumed by Python (ctypes) and the Nim CLI (dynlib).
//!
//! All functions are `extern "C"`, panic-free (release profile uses
//! `panic = "abort"`), and touch only POD types. Strings are NUL-terminated
//! UTF-8; callers own the memory they pass in.

use std::os::raw::{c_char, c_int};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::reader::{self, Event, ScanStats, EV_RECORD_SIZE};
use crate::ring::{self, RingStats};

static LOCK: AtomicBool = AtomicBool::new(false);

struct Guard;
impl Guard {
    fn acquire() -> Guard {
        while LOCK
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            std::hint::spin_loop();
        }
        Guard
    }
}
impl Drop for Guard {
    fn drop(&mut self) {
        LOCK.store(false, Ordering::Release);
    }
}

/// ABI version; the Python bridge refuses to load on a mismatch.
#[no_mangle]
pub extern "C" fn trace_abi_version() -> c_int {
    5
}

/// Scan a trace file; fills `out`. Returns 0 on success, negative on error.
#[no_mangle]
pub extern "C" fn trace_scan(path: *const c_char, out: *mut ScanStats) -> c_int {
    if path.is_null() || out.is_null() {
        return -1;
    }
    let _g = Guard::acquire();
    let Ok(s) = unsafe { std::ffi::CStr::from_ptr(path) }.to_str() else {
        return -2;
    };
    match reader::scan_string(s) {
        Ok(st) => {
            unsafe { *out = st };
            0
        }
        Err(reader::TraceError::Io(_)) => -3,
        Err(reader::TraceError::BadMagic) => -4,
        Err(reader::TraceError::Truncated) => -5,
        Err(reader::TraceError::Empty) => -6,
        Err(reader::TraceError::BadFloats { .. }) => -7,
    }
}

/// Read the first `max` events of a trace into `out` (struct of arrays).
/// Returns the count written, or a negative error code.
#[no_mangle]
pub extern "C" fn trace_read(
    path: *const c_char,
    out: *mut Event,
    max: usize,
) -> isize {
    if path.is_null() || (out.is_null() && max > 0) {
        return -1;
    }
    let _g = Guard::acquire();
    let Ok(s) = unsafe { std::ffi::CStr::from_ptr(path) }.to_str() else {
        return -2;
    };
    let Ok(mapped) = reader::MappedBytes::open(s) else {
        return -3;
    };
    let Ok(iter) = reader::TraceIter::new(&mapped) else {
        return -4;
    };
    let mut n = 0usize;
    for ev in iter.take(max) {
        // SAFETY: caller promises capacity `max` at `out`.
        unsafe { std::ptr::write_unaligned(out.add(n), ev) };
        n += 1;
    }
    n as isize
}

/// Create the process-global shared ring (`capacity` is rounded up to a power
/// of two). Returns 0 on success.
#[no_mangle]
pub extern "C" fn trace_ring_new(capacity: u64) -> c_int {
    let _g = Guard::acquire();
    if ring::global_init(capacity) {
        0
    } else {
        -1
    }
}

/// Pointer + byte size of the mapped ring, for Python `mmap.mmap(fileno=-1)`.
#[no_mangle]
pub extern "C" fn trace_ring_ptr() -> *mut u8 {
    match ring::global() {
        Some(r) => r.descriptor().base,
        None => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn trace_ring_bytes() -> usize {
    ring::global_bytes()
}

/// Push one event into the ring. Returns 1 if the oldest event was dropped.
#[no_mangle]
pub extern "C" fn trace_ring_push(
    tick: u64,
    x: f32,
    y: f32,
    z: f32,
    conf: f32,
    flags: u8,
) -> c_int {
    match ring::global() {
        Some(r) => r.push(Event { tick, x, y, z, confidence: conf, flags }) as c_int,
        None => -1,
    }
}

/// Pop one event into `out`. Returns 1 on success, 0 when empty, <0 on error.
#[no_mangle]
pub extern "C" fn trace_ring_pop(out: *mut Event) -> c_int {
    if out.is_null() {
        return -1;
    }
    match ring::global() {
        Some(r) => match r.pop() {
            Some(ev) => {
                unsafe { std::ptr::write_unaligned(out, ev) };
                1
            }
            None => 0,
        },
        None => -2,
    }
}

#[no_mangle]
pub extern "C" fn trace_ring_stats(out: *mut RingStats) -> c_int {
    if out.is_null() {
        return -1;
    }
    match ring::global() {
        Some(r) => {
            unsafe { *out = r.stats() };
            0
        }
        None => -2,
    }
}

/// Stream a whole trace into the global ring (producer helper).
#[no_mangle]
pub extern "C" fn trace_ring_load_file(path: *const c_char) -> isize {
    if path.is_null() {
        return -1;
    }
    let Some(r) = ring::global() else {
        return -2;
    };
    let Ok(s) = unsafe { std::ffi::CStr::from_ptr(path) }.to_str() else {
        return -3;
    };
    let Ok(mapped) = reader::MappedBytes::open(s) else {
        return -4;
    };
    let Ok(iter) = reader::TraceIter::new(&mapped) else {
        return -5;
    };
    let mut n = 0isize;
    for ev in iter {
        r.push(ev);
        n += 1;
    }
    n
}

/// Copy `out.len()` events starting at event index `start` into `out`.
/// Returns the number written, or a negative error code.
#[no_mangle]
pub extern "C" fn trace_read_window(
    path: *const c_char,
    start: usize,
    out: *mut Event,
    max: usize,
) -> isize {
    if path.is_null() || (out.is_null() && max > 0) {
        return -1;
    }
    let _g = Guard::acquire();
    let Ok(s) = unsafe { std::ffi::CStr::from_ptr(path) }.to_str() else {
        return -2;
    };
    let mut tmp = vec![Event::default(); max];
    match reader::read_window(s, start, &mut tmp) {
        Ok(n) => {
            unsafe { std::ptr::copy_nonoverlapping(tmp.as_ptr(), out, n) };
            n as isize
        }
        Err(_) => -3,
    }
}

/// Reset head/tail of the global ring to zero (drops whatever is buffered).
/// Lets a host process reuse the singleton between runs without unloading.
#[no_mangle]
pub extern "C" fn trace_ring_reset() -> c_int {
    let Some(r) = ring::global() else {
        return -2;
    };
    let _g = Guard::acquire();
    r.reset_counters();
    0
}

/// Layout probe: returns the compiled `ScanStats` size and field offsets so
/// the Python bridge can verify its ctypes definition at load time instead of
/// silently misreading a shifted struct.
#[no_mangle]
pub extern "C" fn trace_layout_probe(out: *mut u32, len: usize) -> usize {
    let info: [u32; 10] = [
        std::mem::size_of::<ScanStats>() as u32,
        std::mem::offset_of!(ScanStats, min_x) as u32,
        std::mem::offset_of!(ScanStats, max_x) as u32,
        std::mem::offset_of!(ScanStats, min_y) as u32,
        std::mem::offset_of!(ScanStats, min_z) as u32,
        std::mem::offset_of!(ScanStats, elapsed_s) as u32,
        std::mem::size_of::<Event>() as u32,
        std::mem::size_of::<RingStats>() as u32,
        std::mem::offset_of!(RingStats, dropped) as u32,
        EV_RECORD_SIZE as u32,
    ];
    let n = info.len().min(len);
    if !out.is_null() && n > 0 {
        unsafe { std::ptr::copy_nonoverlapping(info.as_ptr(), out, n) };
    }
    n
}
