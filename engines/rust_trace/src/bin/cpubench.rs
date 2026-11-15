//! `morphos-cpubench` - steady-state CPU cost of the trace parser at a
//! realistic sensor rate, so the T5.2 acceptance figure (<2% of one core) is
//! measured rather than asserted.
//!
//! Model: one mmap is opened once (a real ingest daemon maps its stream file a
//! single time), then events are produced into the ring and drained in bulk at
//! the requested rate. Busy time is measured only around the parse + ring work,
//! never around the pacing sleep.

use std::time::{Duration, Instant};

use rust_trace::ring::global_init;
use rust_trace::{Event, EventWriter, MappedBytes, TraceIter};

fn main() {
    let mut args = std::env::args().skip(1);
    let hz: u64 = args.next().and_then(|v| v.parse().ok()).unwrap_or(1000);
    let secs: f64 = args.next().and_then(|v| v.parse().ok()).unwrap_or(5.0);

    let path = std::env::temp_dir().join(format!("morphos_cpubench_{hz}.morphose"));
    let p = path.to_str().expect("temp path").to_string();

    // Ten seconds of payload; the bench loops over it.
    let n = hz * 10;
    {
        let mut w = EventWriter::create(&p).expect("create");
        let mut x = 0.1f32;
        for i in 0..n {
            x = (x + 0.001).rem_euclid(2.0) - 1.0;
            w.push(i, x, 0.5, -0.25, 1.0, 0).expect("write");
        }
        w.finish().expect("flush");
    }

    assert!(global_init(1 << 15), "ring init");
    let ring = rust_trace::ring::global().expect("ring");

    // One mapping for the whole run, exactly like the ingest daemon.
    let mapped = MappedBytes::open(&p).expect("open");

    let produce_per_tick = (hz / 1000).max(1) as usize;
    let total = (hz as f64 * secs) as u64;
    let mut cursor = TraceIter::new(&mapped).expect("iter");
    let mut produced = 0u64;
    let mut consumed = 0u64;
    let mut busy = Duration::ZERO;
    let mut batch = [Event::default(); 256];
    let t_wall = Instant::now();
    let mut next = Instant::now();

    while produced < total {
        let t0 = Instant::now();
        // Producer work: pull `produce_per_tick` events from the live cursor.
        // A real ingest daemon holds one cursor and reads what is new; it never
        // re-walks the file, so neither does this benchmark.
        let mut pushed = 0;
        while pushed < produce_per_tick {
            match cursor.next() {
                Some(ev) => {
                    let mut e = ev;
                    e.tick = produced;
                    ring.push(e);
                    pushed += 1;
                    produced += 1;
                }
                None => {
                    cursor = TraceIter::new(&mapped).expect("iter");
                }
            }
        }
        // Consumer work: bulk drain.
        loop {
            let got = ring.pop_batch(&mut batch);
            consumed += got as u64;
            if got == 0 {
                break;
            }
        }
        busy += t0.elapsed();

        // Pace: one millisecond of wall clock between 1 kHz groups.
        next += Duration::from_millis(1);
        let now = Instant::now();
        if next > now {
            std::thread::sleep(next - now);
        }
    }

    let wall = t_wall.elapsed().as_secs_f64();
    let st = ring.stats();
    println!("sensor rate      {} Hz", hz);
    println!("wall duration    {:.3} s", wall);
    println!("events           {} produced / {} consumed (ring occupancy {})",
             produced, consumed, st.occupancy);
    println!("busy time        {:.4} s", busy.as_secs_f64());
    println!("cpu of one core  {:.3} %", 100.0 * busy.as_secs_f64() / wall);
    println!("parse throughput {:.2} Mev/s", produced as f64 / busy.as_secs_f64() / 1e6);
    std::fs::remove_file(&path).ok();
}
