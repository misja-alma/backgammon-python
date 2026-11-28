//! Speed of the engine's search with an evaluation net, with and without incremental evaluation
//! of the positions after the moves at the search's last level (`Evaluator::evaluate_child_scored`).
//!
//!     cargo run --release --bin search_benchmark [eval net, default nets/td-money-eval.bgnet] [positions, default 200] [depths, default 1] [threads, e.g. 1,4,10: only the parallel search's speed]
//!
//! Positions from self-play games of the net; for each, the best move with a random roll at 1 and
//! 2 roll levels (the cache cleared per position). Also counts how often the two choose another
//! move, and the largest difference of the chosen move's equity.
use std::path::Path;
use std::time::{Duration, Instant};

use backgammon_rust::eval_net::{EvalNet, EvalNetEvaluator};
use backgammon_rust::evaluator::{is_game_over, Evaluator, Scoring, MONEY_OUTPUTS};
use backgammon_rust::game::{opening_roll, Dice};
use backgammon_rust::neural_net::Objective;
use backgammon_rust::{EnginePosition, Position, RustAnalyzer};

/// The same evaluator without incremental evaluation (the trait's default `evaluate_child_scored`).
struct Plain(EvalNetEvaluator);

impl Evaluator<EnginePosition> for Plain {
    fn evaluate(&self, position: &EnginePosition) -> f64 {
        self.0.evaluate(position)
    }
    fn evaluate_scored(&self, position: &EnginePosition, scoring: Scoring) -> f64 {
        self.0.evaluate_scored(position, scoring)
    }
    fn outcome(&self, position: &EnginePosition) -> [f64; MONEY_OUTPUTS] {
        Evaluator::<EnginePosition>::outcome(&self.0, position)
    }
    fn name(&self) -> String {
        format!("{} (not incremental)", Evaluator::<EnginePosition>::name(&self.0))
    }
}

/// A nearly free evaluator that counts its calls: the search's tree doesn't depend on the values
/// (no pruning), so a search with it costs what the search does besides evaluating.
struct Trivial(std::rc::Rc<std::cell::Cell<u64>>);

impl Evaluator<EnginePosition> for Trivial {
    fn evaluate(&self, position: &EnginePosition) -> f64 {
        self.0.set(self.0.get() + 1);
        position.count(position.get_turn(), 6) as f64 * 0.01
    }
    fn name(&self) -> String {
        "trivial".into()
    }
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "nets/td-money-eval.bgnet".into());
    let n: usize = std::env::args().nth(2).map(|g| g.parse().unwrap()).unwrap_or(200);
    let depths: Vec<u32> = std::env::args().nth(3).unwrap_or_else(|| "1".into())
        .split(',').map(|d| d.parse().unwrap()).collect();
    let load = || EvalNet::load(Path::new(&path)).unwrap_or_else(|e| panic!("{}", e));
    let net = load();
    let scoring = if net.objective == Objective::Money { Scoring::Money } else { Scoring::Dmp };
    println!("net {} (incremental: {}), scoring {:?}", net.description(), net.incremental(), scoring);

    // Positions (with a roll) from self-play at depth 0.
    let mut positions = Vec::new();
    let collector = RustAnalyzer::with_evaluator(Box::new(EvalNetEvaluator::new(load())), 64);
    let mut seed = 0;
    while positions.len() < n {
        let mut dice = Dice::new(seed);
        seed += 1;
        let mut position = EnginePosition::new();
        position.setup_starting_position();
        let (starter, mut d1, mut d2) = opening_roll(&mut dice);
        position.set_turn(starter);
        let mut turn = 0;
        while !is_game_over(&position) {
            if turn % 7 == 3 {
                positions.push((position.clone(), d1, d2));
            }
            position = collector.best_move(&position, d1, d2, 0).map(|(p, _)| p).unwrap();
            d1 = dice.roll();
            d2 = dice.roll();
            turn += 1;
        }
    }
    positions.truncate(n);

    if let Some(list) = std::env::args().nth(4) {
        // The parallel search: time per move for each number of threads (same results checked).
        let reference: Vec<_> = {
            let mut a = RustAnalyzer::with_evaluator(Box::new(EvalNetEvaluator::new(load())), 64);
            a.set_scoring(scoring);
            depths.iter().map(|&d| positions.iter().map(|(p, d1, d2)| {
                a.clear_cache();
                a.best_move(p, *d1, *d2, d).unwrap().1
            }).collect::<Vec<f64>>()).collect()
        };
        for threads in list.split(',').map(|t| t.parse::<usize>().unwrap()) {
            let mut a = RustAnalyzer::with_evaluator(Box::new(EvalNetEvaluator::new(load())), 64);
            a.set_scoring(scoring);
            a.set_threads(threads);
            for (k, &depth) in depths.iter().enumerate() {
                let mut elapsed = Duration::ZERO;
                for (j, (p, d1, d2)) in positions.iter().enumerate() {
                    a.clear_cache();
                    let t = Instant::now();
                    let (_, e) = a.best_move(p, *d1, *d2, depth).unwrap();
                    elapsed += t.elapsed();
                    assert_eq!(e, reference[k][j], "{} threads: other result", threads);
                }
                println!("{:2} threads, depth {}: {:8.2} ms per move", a.threads(), depth,
                         elapsed.as_secs_f64() * 1e3 / n as f64);
            }
        }
        return;
    }
    let mut incremental = RustAnalyzer::with_evaluator(Box::new(EvalNetEvaluator::new(load())), 256);
    let mut plain = RustAnalyzer::with_evaluator(Box::new(Plain(EvalNetEvaluator::new(load()))), 256);
    incremental.set_scoring(scoring);
    plain.set_scoring(scoring);
    let calls = std::rc::Rc::new(std::cell::Cell::new(0u64));
    let trivial = RustAnalyzer::with_evaluator(Box::new(Trivial(calls.clone())), 256);
    for &depth in &depths {
        let (mut t_inc, mut t_plain) = (Duration::ZERO, Duration::ZERO);
        let (mut other, mut max_diff) = (0, 0.0f64);
        // The search without evaluation cost, and the number of evaluations.
        let mut t_trivial = Duration::ZERO;
        let mut evaluations = 0u64;
        for (position, d1, d2) in &positions {
            trivial.clear_cache();
            let t = Instant::now();
            std::hint::black_box(trivial.best_move(position, *d1, *d2, depth));
            t_trivial += t.elapsed();
        }
        evaluations += calls.replace(0);
        for (position, d1, d2) in &positions {
            incremental.clear_cache();
            plain.clear_cache();
            let t = Instant::now();
            let (a, ea) = incremental.best_move(position, *d1, *d2, depth).unwrap();
            t_inc += t.elapsed();
            let t = Instant::now();
            let (b, eb) = plain.best_move(position, *d1, *d2, depth).unwrap();
            t_plain += t.elapsed();
            if a.cache_key(0) != b.cache_key(0) {
                other += 1;
            }
            max_diff = max_diff.max((ea - eb).abs());
        }
        println!("depth {}: plain {:.2} ms, incremental {:.2} ms per move ({:.2}x); other move chosen {} of {}, \
                  largest equity difference {:.5}",
                 depth, t_plain.as_secs_f64() * 1e3 / n as f64, t_inc.as_secs_f64() * 1e3 / n as f64,
                 t_plain.as_secs_f64() / t_inc.as_secs_f64(), other, n, max_diff);
        println!("   search without evaluation cost {:.2} ms per move ({:.0}% of incremental); {:.0} evaluations per move, \
                  {:.0} ns each (incremental) / {:.0} ns (plain), the rest of the search {:.0} ns per evaluation",
                 t_trivial.as_secs_f64() * 1e3 / n as f64, 100.0 * t_trivial.as_secs_f64() / t_inc.as_secs_f64(),
                 evaluations as f64 / n as f64,
                 (t_inc - t_trivial).as_secs_f64() * 1e9 / evaluations as f64,
                 (t_plain - t_trivial).as_secs_f64() * 1e9 / evaluations as f64,
                 t_trivial.as_secs_f64() * 1e9 / evaluations as f64);
    }
}

