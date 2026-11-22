//! Is it worth caching leaf evaluations? Measures, with the neural net evaluator:
//! - the cost of one evaluation (encode + forward pass) and of one cache lookup;
//! - in real searches: how many leaves are evaluated, and how many of those are repeats
//!   (the hit rate an unbounded leaf cache would get), plus the search time.
//!
//! Result on 2026-11-19 (54-128-64-1 net): 530 ns per evaluation, 40 ns per lookup, but only
//! ~10% repeated leaves; caching leaves in the main table made depth-2 searches 15-17% *slower*,
//! so leaves are not cached. Re-run this when the evaluator gets much slower or the repeat rate
//! changes (e.g. with pruning).
//!
//!     cargo run --release --bin leaf_cache_benchmark [net file, default nets/td.bgnet]
use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::hint::black_box;
use std::path::Path;
use std::rc::Rc;
use std::time::Instant;

use backgammon_rust::evaluator::Evaluator;
use backgammon_rust::neural_net::{NeuralNet, NeuralNetEvaluator};
use backgammon_rust::transposition_table::TranspositionTable;
use backgammon_rust::{build_position, Player, Position, RustAnalyzer, SimplePosition};

type Key = <SimplePosition as Position>::CacheKey;

/// Wraps an evaluator and records every position it is asked about.
struct Counting {
    inner: NeuralNetEvaluator,
    calls: Cell<u64>,
    seen: RefCell<HashSet<Key>>,
}

impl Evaluator<SimplePosition> for Counting {
    fn evaluate(&self, position: &SimplePosition) -> f64 {
        self.calls.set(self.calls.get() + 1);
        self.seen.borrow_mut().insert(position.cache_key(0));
        self.inner.evaluate(position)
    }
    fn name(&self) -> String {
        "counting".into()
    }
}

/// Lets the analyzer own an evaluator that the benchmark can still read afterwards.
struct Shared(Rc<Counting>);

impl Evaluator<SimplePosition> for Shared {
    fn evaluate(&self, position: &SimplePosition) -> f64 {
        self.0.evaluate(position)
    }
    fn name(&self) -> String {
        "shared".into()
    }
}

fn position(me: &[(usize, u8)], opp: &[(usize, u8)]) -> SimplePosition {
    let (mut m, mut o) = (vec![0u8; 26], vec![0u8; 26]);
    for &(i, c) in me { m[i] = c; }
    for &(i, c) in opp { o[i] = c; }
    m[0] = 15 - m.iter().sum::<u8>();
    o[0] = 15 - o.iter().sum::<u8>();
    build_position(m, o, 1, 1, None).unwrap()
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "nets/td.bgnet".into());
    let net = NeuralNet::load(Path::new(&path)).unwrap_or_else(|e| panic!("{}", e));
    println!("net: {}\n", net.description());

    let start = [(6, 5), (8, 3), (13, 5), (24, 2)];
    let middle = position(&[(25, 1), (13, 4), (8, 3), (6, 4), (5, 2), (3, 1)],
                          &[(24, 2), (18, 2), (13, 3), (8, 2), (6, 5), (4, 1)]);
    let starting = position(&start, &start);

    // --- micro costs ---------------------------------------------------------------------------
    let n = 2_000_000;
    let t = Instant::now();
    let mut sum = 0.0;
    for i in 0..n {
        let p = if i % 2 == 0 { &middle } else { &starting };
        sum += net.equity(black_box(p));
    }
    black_box(sum);
    let eval_ns = t.elapsed().as_nanos() as f64 / n as f64;

    // Lookups in a 256 MB table filled with distinct keys, so most lookups miss the CPU cache,
    // as they do in a real search.
    let mut table = TranspositionTable::<Key>::with_size_bytes(256 << 20);
    let mut keys = Vec::new();
    let mut p = middle.clone();
    for i in 0..1_000_000u32 {
        p.set_cube(1 + i, None);   // the cube is part of the key: cheap way to make distinct keys
        keys.push(p.cache_key(0));
    }
    for (i, k) in keys.iter().enumerate() { table.store(*k, 0, i as f64); }
    let t = Instant::now();
    let mut hits = 0u32;
    for round in 0..2 {
        for k in keys.iter().skip(round).step_by(2) {
            if table.get(black_box(k)).is_some() { hits += 1; }
        }
    }
    black_box(hits);
    let lookup_ns = t.elapsed().as_nanos() as f64 / keys.len() as f64;
    println!("one net evaluation: {:7.0} ns", eval_ns);
    println!("one cache lookup:   {:7.0} ns (256 MB table)\n", lookup_ns);

    // --- leaf repeats in real searches ----------------------------------------------------------
    for (name, pos, depth) in [("starting position", &starting, 1), ("starting position", &starting, 2),
                               ("middle game", &middle, 1), ("middle game", &middle, 2)] {
        // Timed with the plain net; the counting run below is slowed down by its bookkeeping.
        let analyzer = RustAnalyzer::with_evaluator(Box::new(NeuralNetEvaluator { net: net.clone() }), 1024);
        // Warm up first: the app reuses one analyzer, so its table's memory is already mapped.
        analyzer.equity(pos, Player::Me, depth);
        analyzer.clear_cache();
        let t = Instant::now();
        let equity = analyzer.equity(pos, Player::Me, depth);
        let elapsed = t.elapsed();

        let counter = Rc::new(Counting { inner: NeuralNetEvaluator { net: net.clone() },
                                         calls: Cell::new(0), seen: RefCell::new(HashSet::new()) });
        RustAnalyzer::with_evaluator(Box::new(Shared(counter.clone())), 1024).equity(pos, Player::Me, depth);
        let calls = counter.calls.get();
        let distinct = counter.seen.borrow().len() as u64;
        println!("{:<18} depth {}: {:>10} leaf evaluations, {:>9} distinct ({:4.1}% repeats), {:>9.2?}, equity {:+.3}",
                 name, depth, calls, distinct, 100.0 * (calls - distinct) as f64 / calls as f64, elapsed, equity);
    }
}
