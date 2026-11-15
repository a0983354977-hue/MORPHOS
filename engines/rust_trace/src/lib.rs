//! MORPHOS high-performance binary event parser + zero-copy ring buffer.
//!
//! Three surfaces:
//!   * `reader` - mmap'd, copy-free decode of `.morphose` trace streams
//!   * `ring`   - SPSC ring buffer over shared memory (no allocation on hot path)
//!   * `ffi`    - C ABI for the Python core and the Nim CLI
//!
//! Trace wire format (little-endian, 8-byte magic + 19-byte records):
//!
//! ```text
//!  0 .. 8   magic "MORPHOSE"
//!  per record:
//!   0 .. 8   tick (48b) | confidence (16b fixed point, if HAS_CONF)
//!   8 .. 11  x  f24 fixed point
//!  11 .. 14  y  f24
//!  14 .. 17  z  f24
//!  17 .. 18  reserved
//!  18 .. 19  flags
//! ```

pub mod ffi;
pub mod reader;
pub mod ring;
pub mod writer;

pub use reader::{Event, MappedBytes, ScanStats, TraceError, TraceIter, EV_RECORD_SIZE, MAGIC};
pub use reader::flags as event_flags;
pub use ring::{Ring, RingDescriptor, RingStats, RING_MAGIC, SLOT_SIZE};
pub use writer::EventWriter;

/// Fixed-point scale shared by the 24-bit lane codec.
pub const F24_SCALE: f32 = 2147483.648;
