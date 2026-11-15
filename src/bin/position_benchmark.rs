//! Compares SimplePosition and CacheFriendlyPosition on identical searches (same positions, depths
//! and results): search time, position size and cache-key size. Use it to decide which one the
//! engine should use (see `EnginePosition` in lib.rs). Run with:
//!     cargo run --bin position_benchmark --release [cache size in MB, default 1024]
use std::time::{Duration, Instant};
use backgammon_rust::{CacheFriendlyPosition, Player, Position, RustAnalyzer, SimplePosition};

fn setup<P: Position>(p: &mut P, me: &[(usize, u8)], opp: &[(usize, u8)]) {
    for &(i, c) in me { p.set_checkers(Player::Me, i, c).unwrap(); }
    for &(i, c) in opp { p.set_checkers(Player::Opponent, i, c).unwrap(); }
    p.set_turn(Player::Me);
}

/// Average time over `runs` searches, each with an empty cache. Returns (equity, time, cache size).
fn time<P: Position + Clone>(pos: &P, depth: u32, runs: u32) -> (f64, Duration, usize) {
    let analyzer = RustAnalyzer::<P>::with_cache_mb(cache_mb());
    let mut total = Duration::ZERO;
    let mut equity = 0.0;
    for _ in 0..runs {
        analyzer.clear_cache();
        let t = Instant::now();
        equity = analyzer.equity(pos, Player::Me, depth);
        total += t.elapsed();
    }
    (equity, total / runs, analyzer.cache_len())
}

/// Cache size from the first command-line argument.
fn cache_mb() -> usize {
    std::env::args().nth(1).map(|a| a.parse().expect("cache size in MB")).unwrap_or(1024)
}

fn compare(name: &str, me: &[(usize, u8)], opp: &[(usize, u8)], depth: u32, runs: u32) {
    let (mut simple, mut fast) = (SimplePosition::new(), CacheFriendlyPosition::new());
    setup(&mut simple, me, opp);
    setup(&mut fast, me, opp);
    let (es, ts, entries) = time(&simple, depth, runs);
    let (ef, tf, _) = time(&fast, depth, runs);
    assert!((es - ef).abs() < 1e-12, "the two position types must give the same equity");
    println!("{name:<28} depth {depth}: Simple {ts:>9.2?}  CacheFriendly {tf:>9.2?}  CacheFriendly is {:.2}x as fast  ({entries} cache entries)",
             ts.as_secs_f64() / tf.as_secs_f64());
}

fn main() {
    println!("cache size: {} MB", cache_mb());
    println!("cache key size: Simple {} bytes, CacheFriendly {} bytes",
             std::mem::size_of::<<SimplePosition as Position>::CacheKey>(),
             std::mem::size_of::<<CacheFriendlyPosition as Position>::CacheKey>());
    println!("position size:  Simple {} bytes, CacheFriendly {} bytes\n",
             std::mem::size_of::<SimplePosition>(), std::mem::size_of::<CacheFriendlyPosition>());
    let start = [(6, 5), (8, 3), (13, 5), (24, 2)];
    compare("bear-off, 2 checkers each", &[(6, 2), (0, 13)], &[(6, 2), (0, 13)], 6, 5);
    compare("bear-off, 4 checkers each", &[(6, 2), (5, 2), (0, 11)], &[(6, 2), (5, 2), (0, 11)], 3, 3);
    compare("starting position", &start, &start, 1, 5);
}
