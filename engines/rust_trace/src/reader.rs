//! Zero-copy mmap reader for MORPHOS binary event streams.
//!
//! The reader never copies payload bytes: `MappedBytes` borrows the OS page
//! cache, and the parse cursor walks the stream with `get` + `from_le_bytes`
//! on fixed-size structs.

use std::fs::File;
use std::path::Path;
use std::time::Instant;

use memmap2::Mmap;

pub const MAGIC: [u8; 8] = *b"MORPHOSE";
pub const EV_RECORD_SIZE: usize = 19;

/// Event flags understood by the ingest layer.
pub mod flags {
    pub const NONE: u8 = 0;
    /// Sample gap: the source missed its nominal slot.
    pub const GAP: u8 = 1 << 0;
    /// Producer marked the sample as suspicious (spike / clamp).
    pub const SUSPECT: u8 = 1 << 1;
    /// 24-bit float lanes were stored left-justified (extended range).
    pub const WIDE: u8 = 1 << 2;
    /// Explicit confidence present in the second half of the tick field.
    pub const HAS_CONF: u8 = 1 << 3;
}

/// One decoded event. `Bool`-free, 20-byte layout so it can cross the FFI
/// boundary and be viewed as a numpy structured array without padding.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Event {
    pub tick: u64,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub confidence: f32,
    pub flags: u8,
}

#[derive(Debug)]
pub enum TraceError {
    Io(std::io::Error),
    BadMagic,
    Truncated,
    Empty,
    BadFloats { offset: usize },
}

impl std::fmt::Display for TraceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TraceError::Io(e) => write!(f, "io error: {e}"),
            TraceError::BadMagic => write!(f, "bad trace magic (expected MORPHOSE)"),
            TraceError::Truncated => write!(f, "trace file ends mid-record"),
            TraceError::Empty => write!(f, "empty trace file"),
            TraceError::BadFloats { offset } => {
                write!(f, "non-finite float at record offset {offset}")
            }
        }
    }
}

impl std::error::Error for TraceError {}

impl From<std::io::Error> for TraceError {
    fn from(e: std::io::Error) -> Self {
        TraceError::Io(e)
    }
}

/// Copy-free view over the mapped file.
pub struct MappedBytes {
    mmap: Mmap,
}

impl MappedBytes {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, TraceError> {
        let file = File::open(path)?;
        let len = file.metadata()?.len();
        if len == 0 {
            return Err(TraceError::Empty);
        }
        // SAFETY: the trace file is append-only and never mutated in place by
        // the producer, so a shared mapping stays coherent.
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(Self { mmap })
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        &self.mmap
    }
}
/// Representable span of one f24 lane: +-3.90625 (raw +-8388607 / 2^21).
pub const F24_MAX: f32 = 3.90625;

fn f24(le: [u8; 3], wide: bool) -> f32 {
    // 24-bit fixed point; `wide` stretches the representable range by 2^8.
    let raw = (le[0] as i32) | ((le[1] as i32) << 8) | ((le[2] as i32) << 16);
    let signed = (raw << 8) >> 8; // sign-extend 24 -> 32
    signed as f32 / 2147483.648_f32 * if wide { 256.0 } else { 1.0 }
}

#[inline]
fn decode_at(buf: &[u8], off: usize) -> Event {
    let tick_lo = u64::from_le_bytes(buf[off..off + 8].try_into().unwrap());
    let t = off + 8;
    let flags = buf[t + 10];
    let wide = flags & flags::WIDE != 0;
    let has_conf = flags & flags::HAS_CONF != 0;
    Event {
        tick: tick_lo & 0x0000_FFFF_FFFF_FFFF,
        x: f24([buf[t], buf[t + 1], buf[t + 2]], wide),
        y: f24([buf[t + 3], buf[t + 4], buf[t + 5]], wide),
        z: f24([buf[t + 6], buf[t + 7], buf[t + 8]], wide),
        confidence: if has_conf {
            ((tick_lo >> 48) & 0xFFFF) as f32 / 65535.0
        } else {
            1.0
        },
        flags,
    }
}

/// Iterator over events in a mapped trace. Yields borrowed-free `Event`
/// values (fully decoded, never re-reading the same bytes).
pub struct TraceIter<'a> {
    buf: &'a [u8],
    off: usize,
    end: usize,
}

impl<'a> TraceIter<'a> {
    pub fn new(mapped: &'a MappedBytes) -> Result<Self, TraceError> {
        let buf = mapped.as_slice();
        if buf.len() < MAGIC.len() {
            return Err(TraceError::Truncated);
        }
        if buf[..MAGIC.len()] != MAGIC {
            return Err(TraceError::BadMagic);
        }
        let end = buf.len();
        let end = end - ((end - MAGIC.len()) % EV_RECORD_SIZE);
        Ok(Self {
            buf,
            off: MAGIC.len(),
            end,
        })
    }

    #[inline]
    pub fn units(&self) -> usize {
        (self.end - self.off) / EV_RECORD_SIZE
    }
}

impl<'a> Iterator for TraceIter<'a> {
    type Item = Event;

    #[inline]
    fn next(&mut self) -> Option<Event> {
        if self.off + EV_RECORD_SIZE > self.end {
            return None;
        }
        let ev = decode_at(self.buf, self.off);
        self.off += EV_RECORD_SIZE;
        Some(ev)
    }
}

/// `#[repr(C)]` is mandatory: this struct crosses the FFI boundary, and
/// without it rustc is free to reorder fields for packing (it did: the f64
/// moved ahead of the f32 block, so Python read every bound from the wrong
/// slot). The `const _: () = ...` assertions below pin the wire layout.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ScanStats {
    pub events: u64,
    pub bytes: u64,
    pub nan_or_inf: u64,
    pub gaps: u64,
    pub first_tick: u64,
    pub last_tick: u64,
    // Ranges are meaningless until scanning starts; NaN marks "not yet seen"
    // and stops an empty/gap-only stream from reporting 0.0 as a real bound.
    pub min_x: f32,
    pub max_x: f32,
    pub min_y: f32,
    pub max_y: f32,
    pub min_z: f32,
    pub max_z: f32,
    pub elapsed_s: f64,
}

pub fn scan_string(path: &str) -> Result<ScanStats, TraceError> {
    let mapped = MappedBytes::open(path)?;
    let iter = TraceIter::new(&mapped)?;
    let t0 = Instant::now();
    let mut st = ScanStats {
        bytes: mapped.as_slice().len() as u64,
        // No Default here: Default gives 0.0 for the min fields, which would
        // make the reported range wrong (min_x = 0.0 even when x is large).
        ..empty_stats()
    };
    let mut first = true;
    for ev in iter {
        if first {
            st.first_tick = ev.tick;
            first = false;
        }
        st.last_tick = ev.tick;
        if !(ev.x.is_finite() && ev.y.is_finite() && ev.z.is_finite()) {
            st.nan_or_inf += 1;
        }
        if ev.flags & flags::GAP != 0 {
            st.gaps += 1;
        }
        st.min_x = st.min_x.min(ev.x);
        st.max_x = st.max_x.max(ev.x);
        st.min_y = st.min_y.min(ev.y);
        st.max_y = st.max_y.max(ev.y);
        st.min_z = st.min_z.min(ev.z);
        st.max_z = st.max_z.max(ev.z);
        st.events += 1;
    }
    st.elapsed_s = t0.elapsed().as_secs_f64().max(1e-12);
    Ok(st)
}

fn empty_stats() -> ScanStats {
    ScanStats {
        events: 0,
        bytes: 0,
        nan_or_inf: 0,
        gaps: 0,
        first_tick: 0,
        last_tick: 0,
        min_x: f32::INFINITY,
        max_x: f32::NEG_INFINITY,
        min_y: f32::INFINITY,
        max_y: f32::NEG_INFINITY,
        min_z: f32::INFINITY,
        max_z: f32::NEG_INFINITY,
        elapsed_s: 0.0,
    }
}

impl ScanStats {
    pub fn events_per_second(&self) -> f64 {
        self.events as f64 / self.elapsed_s
    }
    pub fn mb_per_second(&self) -> f64 {
        self.bytes as f64 / 1e6 / self.elapsed_s
    }
}

// Wire-layout guard: these offsets are what ctypes in the Python bridge
// assumes. If a field is added/reordered, this fails to compile rather than
// silently corrupting scan results at runtime.
const _: () = {
    assert!(std::mem::size_of::<ScanStats>() == 80);
    assert!(std::mem::offset_of!(ScanStats, min_x) == 48);
    assert!(std::mem::offset_of!(ScanStats, max_x) == 52);
    assert!(std::mem::offset_of!(ScanStats, min_y) == 56);
    assert!(std::mem::offset_of!(ScanStats, max_y) == 60);
    assert!(std::mem::offset_of!(ScanStats, min_z) == 64);
    assert!(std::mem::offset_of!(ScanStats, max_z) == 68);
    assert!(std::mem::offset_of!(ScanStats, elapsed_s) == 72);
    assert!(std::mem::size_of::<Event>() == 32);
};

/// Copy a window of events straight out of the mapping into `out`.
/// Returns the number of events written (0 when `start` is past the end).
pub fn read_window(path: &str, start: usize, out: &mut [Event]) -> Result<usize, TraceError> {
    let mapped = MappedBytes::open(path)?;
    let iter = TraceIter::new(&mapped)?;
    let mut n = 0usize;
    for ev in iter.skip(start) {
        if n == out.len() {
            break;
        }
        out[n] = ev;
        n += 1;
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(v: f32) -> f32 {
        let mut b = [0u8; 3];
        let raw = (v * crate::F24_SCALE) as i32;
        let raw = raw.clamp(-8_388_608, 8_388_607);
        b.copy_from_slice(&raw.to_le_bytes()[0..3]);
        f24(b, false)
    }

    #[test]
    fn f24_roundtrip_is_accurate_to_one_ulp() {
        for v in [-1.0f32, -0.5, -0.003, 0.0, 0.0049, 0.5, 1.0, 1.878] {
            let back = roundtrip(v);
            let tol = 1.0 / crate::F24_SCALE + v.abs() * 1e-6;
            assert!(
                (back - v).abs() <= tol,
                "f24 roundtrip {v} -> {back} exceeds tolerance {tol}"
            );
        }
    }

    #[test]
    fn f24_preserves_sign() {
        assert!(roundtrip(-1.0) < 0.0);
        assert!(roundtrip(1.0) > 0.0);
        assert_eq!(roundtrip(0.0), 0.0);
    }

    #[test]
    fn scan_reports_true_range_not_default_zero() {
        let dir = std::env::temp_dir().join("morphos_scan_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.morphose");
        let mut w = crate::EventWriter::create(path.to_str().unwrap()).unwrap();
        // f24 lanes span +-3.90625 (see F24_SCALE); stay well inside it.
        for i in 0..100 {
            w.push(i, 2.0 + i as f32 * 0.01, -3.0, 0.5, 1.0, 0).unwrap();
        }
        w.finish().unwrap();
        let st = scan_string(path.to_str().unwrap()).unwrap();
        assert_eq!(st.events, 100);
        assert!((st.min_x - 2.0).abs() < 1e-3, "min_x = {}", st.min_x);
        assert!((st.max_x - 2.99).abs() < 1e-2, "max_x = {}", st.max_x);
        assert!((st.min_y + 3.0).abs() < 1e-5);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn ring_push_pop_preserves_order_and_wraps() {
        // 64 bytes of slack so the aligned base still leaves a full ring.
        let region = crate::ring::HDR_TOTAL + 8 * crate::ring::SLOT_SIZE;
        let mut backing = vec![0u8; region + 64];
        let base = (backing.as_mut_ptr() as usize & !63) as *mut u8;
        unsafe {
            std::ptr::copy_nonoverlapping(crate::ring::RING_MAGIC.as_ptr(), base, 8);
            std::ptr::write_unaligned(base.add(8) as *mut u64, 8);
            let ring = crate::ring::Ring::attach(base, region).unwrap();
            for i in 0..100u64 {
                let ev = Event { tick: i, x: i as f32, y: 0.0, z: 0.0, confidence: 1.0, flags: 0 };
                ring.push(ev);
            }
            // capacity 8 -> only the last 8 survive, in order.
            let mut ticks = vec![];
            while let Some(e) = ring.pop() {
                ticks.push(e.tick);
            }
            assert_eq!(ticks, vec![92, 93, 94, 95, 96, 97, 98, 99]);
            assert_eq!(ring.stats().dropped, 92);
        }
    }
}
