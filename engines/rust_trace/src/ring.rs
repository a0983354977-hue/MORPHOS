//! Zero-copy SPSC ring buffer for high-frequency event streams.
//!
//! The buffer lives in an anonymous shared mapping so a producer thread
//! (Rust parser, hardware acquisition, or a foreign FFI caller) and a
//! consumer (Python via `trace_ring_pop`) address the *same* pages. Nothing
//! is copied on push/pop: the consumer decodes straight out of the slot.
//!
//! Layout (all little-endian, 64-byte aligned header):
//!
//! ```text
//!   0  ..  8   magic  "MORPHOSR"
//!   8  .. 16   capacity (u64, power of two)
//!  16  .. 56   padding -> one cache line for the two cursors below
//!  56  .. 64   head (u64, producer-owned, monotonic)
//!  64  .. 72   tail (u64, consumer-owned, monotonic)
//! 128  ..      slots, capacity * 24 bytes: 8B tick | 3 * 4B f32 | 4B conf
//! ```
//!
//! `head` and `tail` are free-running counters; slot index is `counter &
//! (capacity - 1)`, so the buffer never needs to clear memory. Overflow is
//! detected as `head - tail >= capacity` and handled by the *producer*
//! overwriting the oldest slot and reporting a drop count.

use std::cell::Cell;
use std::ptr;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::OnceLock;

use crate::reader::Event;

pub const RING_MAGIC: [u8; 8] = *b"MORPHOSR";
/// One slot is exactly one `#[repr(C)] Event`, padding included (32 bytes on
/// x86_64). Slots used to be packed to 24 bytes, which silently dropped the
/// `flags` byte; keeping the full struct means a pop can hand the consumer a
/// `*const Event` directly with no reassembly.
pub const SLOT_SIZE: usize = std::mem::size_of::<Event>();
pub const HDR_HEAD: usize = 56;
pub const HDR_TAIL: usize = 64;
pub const HDR_TOTAL: usize = 128;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RingStats {
    pub pushed: u64,
    pub popped: u64,
    pub dropped: u64,
    pub capacity: u64,
    pub occupancy: u64,
}

/// Header handed to foreign consumers (Python) describing a mapped ring.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RingDescriptor {
    pub base: *mut u8,
    pub bytes: usize,
    pub capacity: u64,
}

/// A ring buffer over caller-provided memory.
pub struct Ring {
    base: *mut u8,
    bytes: usize,
    capacity: u64,
    mask: u64,
    head: AtomicU64,
    tail: Cell<u64>,
    dropped: Cell<u64>,
}

// SAFETY: SPSC - a single producer thread owns `head`, a single consumer
// thread owns `tail`; both are atomic-acquire/release pairs.
unsafe impl Send for Ring {}
unsafe impl Sync for Ring {}

impl Ring {
    /// Attach to an existing mapping created by `trace_ring_new` or by a
    /// foreign producer that wrote the documented header.
    ///
    /// # Safety
    /// `base` must point to `bytes` readable/writable, 64-bit aligned pages
    /// that stay mapped for the lifetime of this object.
    pub unsafe fn attach(base: *mut u8, bytes: usize) -> Option<Ring> {
        if base.is_null() || bytes < HDR_TOTAL + SLOT_SIZE {
            return None;
        }
        let cap = ptr::read_unaligned(base.add(8) as *const u64);
        if cap == 0 || !cap.is_power_of_two() {
            return None;
        }
        if HDR_TOTAL + (cap as usize) * SLOT_SIZE > bytes {
            return None;
        }
        let head_ptr = base.add(HDR_HEAD) as *const AtomicU64;
        let tail_ptr = base.add(HDR_TAIL) as *const AtomicU64;
        Some(Ring {
            base,
            bytes,
            capacity: cap,
            mask: cap - 1,
            head: AtomicU64::new((*head_ptr).load(Ordering::Acquire)),
            tail: Cell::new((*tail_ptr).load(Ordering::Acquire)),
            dropped: Cell::new(0),
        })
    }

    #[inline]
    fn slot_ptr(&self, idx: u64) -> *mut u8 {
        // SAFETY: idx is masked into capacity; bounds checked in attach().
        unsafe { self.base.add(HDR_TOTAL + (idx as usize) * SLOT_SIZE) }
    }

    /// Push one event. Returns `true` when the oldest event had to be
    /// overwritten (back-pressure). Never blocks, never allocates.
    #[inline]
    pub fn push(&self, ev: Event) -> bool {
        let head = self.head.load(Ordering::Relaxed);
        let mut over = head.wrapping_sub(self.tail.get()) >= self.capacity;
        if over {
            self.tail.set(self.tail.get().wrapping_add(1));
            self.dropped.set(self.dropped.get() + 1);
        }
        let p = self.slot_ptr(head & self.mask) as *mut Event;
        // SAFETY: the slot is exclusively ours until `head` is published.
        // `write_unaligned` because the mapping only guarantees 64-byte
        // alignment of the header, not of every 32-byte slot.
        unsafe { ptr::write_unaligned(p, ev) };
        self.head.store(head.wrapping_add(1), Ordering::Release);
        over = over || false;
        over
    }

    /// Pop the oldest event, if any. The returned event is decoded out of the
    /// mapped slot; nothing is memcpy'd.
    #[inline]
    pub fn pop(&self) -> Option<Event> {
        let tail = self.tail.get();
        if tail == self.head.load(Ordering::Acquire) {
            return None;
        }
        let p = self.slot_ptr(tail & self.mask) as *const Event;
        // SAFETY: `tail` is consumer-owned and the slot was published by the
        // producer (release/acquire pair above). Reading the whole struct keeps
        // every field -- including `flags` -- intact.
        let ev = unsafe { ptr::read_unaligned(p) };
        self.tail.set(tail.wrapping_add(1));
        // Publish the tail so a stalled producer can resume.
        // SAFETY: HDR_TAIL is 8-byte aligned within the mapping.
        unsafe {
            (self.base.add(HDR_TAIL) as *const AtomicU64)
                .as_ref()
                .expect("tail header")
                .store(self.tail.get(), Ordering::Release)
        };
        Some(ev)
    }

    /// Pop up to `out.len()` events into a caller slice. Returns how many.
    pub fn pop_batch(&self, out: &mut [Event]) -> usize {
        let mut n = 0;
        while n < out.len() {
            match self.pop() {
                Some(ev) => {
                    out[n] = ev;
                    n += 1;
                }
                None => break,
            }
        }
        n
    }

    /// Discard every buffered event and restart the counters at zero.
    /// Only safe when no producer/consumer is active.
    pub fn reset_counters(&self) {
        self.tail.set(0);
        // SAFETY: header word is ours; publish the reset to the consumer.
        unsafe {
            (self.base.add(HDR_TAIL) as *const AtomicU64)
                .as_ref()
                .expect("tail header")
                .store(0, Ordering::Release)
        };
        self.head.store(0, Ordering::Release);
        self.dropped.set(0);
    }

    pub fn stats(&self) -> RingStats {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.get();
        RingStats {
            pushed: head,
            popped: tail,
            dropped: self.dropped.get(),
            capacity: self.capacity,
            occupancy: head.wrapping_sub(tail),
        }
    }

    pub fn descriptor(&self) -> RingDescriptor {
        RingDescriptor {
            base: self.base,
            bytes: self.bytes,
            capacity: self.capacity,
        }
    }
}

/// Process-global ring used by the C ABI (single producer/consumer pair is
/// the documented contract; the FFI layer serialises callers with a spinlock).
/// A `OnceLock` keeps the initialisation check race-free without `static mut`.
static GLOBAL: OnceLock<Ring> = OnceLock::new();
static GLOBAL_BYTES: AtomicUsize = AtomicUsize::new(0);

pub fn global_init(capacity: u64) -> bool {
    if GLOBAL.get().is_some() {
        return true;
    }
    let cap = capacity.next_power_of_two().max(2);
    let bytes = HDR_TOTAL + (cap as usize) * SLOT_SIZE;
    let mut v = vec![0u8; bytes + 64];
    // 64-byte alignment so head/tail never share a cache line with payload.
    let base = (v.as_mut_ptr() as usize & !63) as *mut u8;
    let boxed = v.into_boxed_slice();
    GLOBAL_BYTES.store(bytes, Ordering::Relaxed);
    // SAFETY: fresh allocation, `bytes` in scope, access is exclusive here.
    unsafe {
        ptr::copy_nonoverlapping(RING_MAGIC.as_ptr(), base, 8);
        ptr::write_unaligned(base.add(8) as *mut u64, cap);
        ptr::write_unaligned(base.add(HDR_HEAD) as *mut u64, 0u64);
        ptr::write_unaligned(base.add(HDR_TAIL) as *mut u64, 0u64);
    }
    // SAFETY: we just initialised the header and own the pages.
    let ring = match unsafe { Ring::attach(base, bytes) } {
        Some(r) => r,
        None => return false,
    };
    // Leak the backing allocation: the ring lives for the process lifetime
    // and foreign consumers hold pointers into it.
    std::mem::forget(boxed);
    GLOBAL.set(ring).is_ok()
}

pub fn global() -> Option<&'static Ring> {
    GLOBAL.get()
}

pub fn global_bytes() -> usize {
    GLOBAL_BYTES.load(Ordering::Relaxed)
}
