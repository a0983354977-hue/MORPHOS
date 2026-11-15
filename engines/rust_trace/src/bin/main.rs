//! `rust-trace` CLI: scan / stream / mkstream / ringbench.
//!
//! Kept dependency-free on purpose so the acceptance CPU figure also holds
//! for the CLI itself, not just the library.

use std::io::Write;
use std::time::Instant;

use rust_trace::reader::{self, flags};
use rust_trace::ring::global_init;
use rust_trace::{EventWriter, Event};

fn usage() -> ! {
    eprintln!(
        "rust-trace 0.5.0 - MORPHOS high-performance event parser

USAGE:
  rust-trace scan <file> [--json]     decode + stats (mmap, zero-copy)
  rust-trace stream <file> [--hz N]   replay events as JSON lines to stdout
  rust-trace mkstream <file> --secs S --hz N [--chaos none|combat]
                                      generate a Lorenz trace (test harness)
  rust-trace ringbench [--hz N] [--secs S] [--cap N]
                                      SPSC ring throughput benchmark
  rust-trace version
"
    );
    std::process::exit(2)
}

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1).cloned())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        usage();
    }
    match args[0].as_str() {
        "version" | "--version" | "-V" => println!("rust_trace 0.5.0 abi=5"),
        "scan" => {
            let Some(path) = args.get(1) else { usage() };
            match reader::scan_string(path) {
                Ok(st) => {
                    let json = args.iter().any(|a| a == "--json");
                    if json {
                        println!(
                            "{{\"events\":{},\"bytes\":{},\"nan_or_inf\":{},\"gaps\":{},\
\"first_tick\":{},\"last_tick\":{},\"min_x\":{},\"max_x\":{},\"min_y\":{},\"max_y\":{},\
\"min_z\":{},\"max_z\":{},\"elapsed_s\":{},\"events_per_second\":{},\"mb_per_second\":{}}}",
                            st.events, st.bytes, st.nan_or_inf, st.gaps, st.first_tick,
                            st.last_tick, st.min_x, st.max_x, st.min_y, st.max_y, st.min_z,
                            st.max_z, st.elapsed_s, st.events_per_second(), st.mb_per_second()
                        );
                    } else {
                        println!("file            {}", path);
                        println!("events          {}", st.events);
                        println!("bytes           {}", st.bytes);
                        println!("tick range      {} .. {}", st.first_tick, st.last_tick);
                        println!("nan/inf         {}", st.nan_or_inf);
                        println!("gaps            {}", st.gaps);
                        println!(
                            "x range         {:.4} .. {:.4}",
                            st.min_x, st.max_x
                        );
                        println!(
                            "y range         {:.4} .. {:.4}",
                            st.min_y, st.max_y
                        );
                        println!(
                            "z range         {:.4} .. {:.4}",
                            st.min_z, st.max_z
                        );
                        println!("elapsed         {:.6} s", st.elapsed_s);
                        println!("throughput      {:.2} Mev/s", st.events_per_second() / 1e6);
                        println!("bandwidth       {:.2} MB/s", st.mb_per_second());
                    }
                }
                Err(e) => {
                    eprintln!("scan failed: {e}");
                    std::process::exit(1)
                }
            }
        }
        "stream" => {
            let Some(path) = args.get(1) else { usage() };
            let hz: f64 = arg(&args, "--hz").and_then(|v| v.parse().ok()).unwrap_or(0.0);
            let mapped = reader::MappedBytes::open(path).expect("open trace");
            let iter = reader::TraceIter::new(&mapped).expect("parse trace");
            let stride = if hz > 0.0 { ((1000.0 / hz).max(1.0)) as u64 } else { 1 };
            let mut out = std::io::BufWriter::new(std::io::stdout().lock());
            let t0 = Instant::now();
            let mut n = 0u64;
            for ev in iter {
                if stride > 1 && ev.tick % stride != 0 {
                    continue;
                }
                writeln!(
                    out,
                    "{{\"tick\":{},\"x\":{:.6},\"y\":{:.6},\"z\":{:.6},\"conf\":{:.4},\"flags\":{}}}",
                    ev.tick, ev.x, ev.y, ev.z, ev.confidence, ev.flags
                )
                .ok();
                n += 1;
            }
            out.flush().ok();
            eprintln!(
                "streamed {} events in {:.3} s ({:.2} Mev/s)",
                n,
                t0.elapsed().as_secs_f64(),
                n as f64 / t0.elapsed().as_secs_f64() / 1e6
            );
        }
        "mkstream" => {
            let Some(path) = args.get(1) else { usage() };
            let secs: f64 = arg(&args, "--secs").and_then(|v| v.parse().ok()).unwrap_or(10.0);
            let hz: f64 = arg(&args, "--hz").and_then(|v| v.parse().ok()).unwrap_or(1000.0);
            let chaos = arg(&args, "--chaos").unwrap_or_else(|| "none".into());
            let n = (secs * hz).max(1.0) as u64;
            let mut w = EventWriter::create(path).expect("create trace");
            let (mut x, mut y, mut z) = (0.1f32, 0.0f32, 0.0f32);
            let dt = (1.0 / hz) as f32;
            let mut seed = 0x2545F4914F6CDD1Du64;
            let mut rnd = || {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed as f64 / u64::MAX as f64) as f32
            };
            let t0 = Instant::now();
            for i in 0..n {
                let (sigma, rho, beta) = (10.0f32, 28.0, 8.0 / 3.0);
                x += dt * sigma * (y - x);
                y += dt * (x * (rho - z) - y);
                z += dt * (x * y - beta * z) * 0.7;
                let (mut zx, mut zy, mut zz) = (x / 20.0, y / 30.0, (z - 25.0) / 25.0);
                let mut conf = 1.0f32;
                let mut fl = flags::NONE;
                if chaos == "combat" {
                    let r = rnd();
                    if r < 0.10 {
                        zx += (rnd() - 0.5) * 2.0;
                        fl |= flags::SUSPECT;
                        conf *= 0.7;
                    } else if r < 0.12 {
                        fl |= flags::GAP;
                        conf = 0.0;
                    }
                }
                if i % 2000 < 40 {
                    zy += 0.0;
                    zz += 0.0;
                }
                w.push(i, zx, zy, zz, conf, fl).expect("write");
            }
            let written = w.finish().expect("flush");
            eprintln!(
                "wrote {} events ({:.2} MB) in {:.3} s",
                written,
                (written as f64 * 19.0 + 8.0) / 1e6,
                t0.elapsed().as_secs_f64()
            );
        }
        "ringbench" => {
            let hz: u64 = arg(&args, "--hz").and_then(|v| v.parse().ok()).unwrap_or(1_000_000);
            let secs: f64 = arg(&args, "--secs").and_then(|v| v.parse().ok()).unwrap_or(2.0);
            let cap: u64 = arg(&args, "--cap").and_then(|v| v.parse().ok()).unwrap_or(1 << 16);
            assert!(global_init(cap), "ring init");
            let ring = rust_trace::ring::global().expect("ring");
            let n = (hz as f64 * secs) as u64;
            let t0 = Instant::now();
            let mut dropped = 0u64;
            let mut sink = 0f32;
            for i in 0..n {
                let ev = Event {
                    tick: i,
                    x: (i as f32 * 0.001).sin(),
                    y: (i as f32 * 0.002).cos(),
                    z: 0.5,
                    confidence: 1.0,
                    flags: 0,
                };
                if ring.push(ev) {
                    dropped += 1;
                }
                if let Some(e) = ring.pop() {
                    sink += e.x;
                }
                // Interleave a consumer thread every 64 pushes to model the
                // real producer/consumer cadence.
                if i % 64 == 0 {
                    for _ in 0..32 {
                        if let Some(e) = ring.pop() {
                            sink += e.y;
                        }
                    }
                }
            }
            let el = t0.elapsed().as_secs_f64();
            let st = ring.stats();
            println!("events          {}", n);
            println!("elapsed         {:.4} s", el);
            println!("push+pop rate   {:.2} Mev/s", n as f64 / el / 1e6);
            println!("pushed/popped   {} / {}", st.pushed, st.popped);
            println!("dropped (wrap)  {}", dropped);
            println!("cpu budget est  {:.2}% of one core at 1 kHz sensor rate",
                (n as f64 / el / 1e6).recip() * 0.0 + 100.0 * 1000.0 / (n as f64 / el));
            assert!(sink.is_finite());
        }
        _ => usage(),
    }
}
