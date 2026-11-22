//! Builds the two-sided bearoff database (see src/bearoff.rs).
//!
//!   cargo run --release --bin make_bearoff -- [--checkers 15] [--threads N] [--out data/bearoff-6x15.bd]
//!
//! The full database (15 checkers) needs about 11 GB of memory while generating and writes 5.5 GB.
use std::path::PathBuf;
use std::time::Instant;

use backgammon_rust::bearoff;

fn main() {
    let mut checkers = bearoff::MAX_CHECKERS;
    let mut threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let mut out: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args.next().unwrap_or_else(|| panic!("{} needs a value", flag));
        match flag.as_str() {
            "--checkers" => checkers = value.parse().expect("--checkers"),
            "--threads" => threads = value.parse().expect("--threads"),
            "--out" => out = Some(PathBuf::from(value)),
            _ => panic!("usage: make_bearoff [--checkers 15] [--threads N] [--out PATH]"),
        }
    }
    let out = out.unwrap_or_else(|| PathBuf::from(format!("data/bearoff-6x{}.bd", checkers)));
    let n = bearoff::Indexer::new(checkers).n;
    println!("bearoff database 6 points x {} checkers: {} positions per side, {} pairs, {:.2} GB, {} threads",
             checkers, n, n * n, (2 * n * n) as f64 / 1e9, threads);
    let start = Instant::now();
    let values = bearoff::generate(checkers, threads, |layer, layers| {
        if layer % 10 == 0 || layer == layers {
            println!("  layer {:3}/{}  {:7.1}s", layer, layers, start.elapsed().as_secs_f64());
        }
    });
    println!("generated in {:.1}s, writing {}", start.elapsed().as_secs_f64(), out.display());
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir).expect("create output directory");
    }
    bearoff::write(&out, checkers, &values).expect("write database");
    println!("done in {:.1}s", start.elapsed().as_secs_f64());
}
