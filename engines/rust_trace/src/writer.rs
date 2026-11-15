//! Binary event stream writer (producer side), used by the CLI `mkstream`
//! command and the Python `RustTraceSource` simulator fallback.

use std::fs::File;
use std::io::{BufWriter, Write};

use crate::reader::{MAGIC, EV_RECORD_SIZE};

pub struct EventWriter {
    w: BufWriter<File>,
    written: u64,
}

/// Encode one lane. Values outside +-3.90625 saturate, and the caller is
/// expected to have normalised its inputs (the encoder z-scores to [-1,1]).
fn pack24(v: f32, buf: &mut [u8; 3]) {
    let clamped = if v.is_finite() { v } else { 0.0 };
    let raw = (clamped * 2147483.648_f32) as i32;
    let raw = raw.clamp(-8_388_608, 8_388_607);
    let b = raw.to_le_bytes();
    buf.copy_from_slice(&b[0..3]);
}

/// True when `v` fits an f24 lane without saturation.
#[inline]
pub fn fits(v: f32) -> bool {
    v.is_finite() && v.abs() <= crate::reader::F24_MAX
}

impl EventWriter {
    pub fn create(path: &str) -> std::io::Result<Self> {
        let f = File::create(path)?;
        let mut w = BufWriter::with_capacity(1 << 20, f);
        w.write_all(&MAGIC)?;
        Ok(Self { w, written: 0 })
    }

    #[inline]
    pub fn push(&mut self, tick: u64, x: f32, y: f32, z: f32, conf: f32, flags: u8) -> std::io::Result<()> {
        let mut rec = [0u8; EV_RECORD_SIZE];
        let conf_bits = ((conf.clamp(0.0, 1.0) * 65535.0) as u64) << 48;
        let tick_field = (tick & 0x0000_FFFF_FFFF_FFFF) | conf_bits;
        rec[0..8].copy_from_slice(&tick_field.to_le_bytes());
        pack24(x, (&mut rec[8..11]).try_into().unwrap());
        pack24(y, (&mut rec[11..14]).try_into().unwrap());
        pack24(z, (&mut rec[14..17]).try_into().unwrap());
        rec[18] = flags;
        self.w.write_all(&rec)?;
        self.written += 1;
        Ok(())
    }

    pub fn finish(mut self) -> std::io::Result<u64> {
        self.w.flush()?;
        Ok(self.written)
    }
}
