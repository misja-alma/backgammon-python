//! Where does the time of TD(λ) training go? Replays the training loop of `TdTrainer::train_game`
//! with a timer around each part, and counts the evaluations per move.
//!
//!     cargo run --release --bin train_profile [net file, default nets/td.bgnet] [games, default 300]
//!
//! The net is trained while profiling (as in real training), on a copy; nothing is saved.
//!
//! History (54-128-128-128-1 net, this machine, µs per move):
//! - 2026-11-20, before: forward_with_gradient 7.27, trace update 3.16, weight update 3.82, move
//!   choice 10.48 (15 evaluations; generation alone 0.34): 24.7 µs, 666 games/s.
//! - 2026-11-20, fused td_update (one pass over parameters and traces, no gradient buffer,
//!   vectorized back-propagation dot products): forward 0.99, move choice 10.2, td_update 6.97:
//!   18.2 µs, 896 games/s (real `train`: 862 games/s; 128-64 net 2286 games/s, was ~1570).
//!   td_update costs ~0.22 ns per parameter whether params + traces fit in L1 (128 KB) or not
//!   (measured from 31 KB to 4 MB): the loop is vectorized and limited by its loads, stores and
//!   arithmetic (~5 cycles per 8 parameters), not by the cache.
use std::cell::Cell;
use std::collections::HashSet;
use std::path::Path;
use std::time::{Duration, Instant};

use backgammon_rust::encoding::MAX_INPUTS;
use backgammon_rust::evaluator::is_game_over;
use backgammon_rust::game::{opening_roll, Dice};
use backgammon_rust::move_generator::MoveGenerator;
use backgammon_rust::{EnginePosition, NeuralNet, Player, Position};

fn value_for_me(net: &NeuralNet, p: &EnginePosition) -> f32 {
    if is_game_over(p) {
        return if p.count(Player::Me, 0) == 15 { 1.0 } else { 0.0 };
    }
    let v = net.win_probability(p);
    if p.get_turn() == Player::Me { v } else { 1.0 - v }
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "nets/td.bgnet".into());
    let games: u64 = std::env::args().nth(2).map(|g| g.parse().unwrap()).unwrap_or(300);
    let mut net = NeuralNet::load(Path::new(&path)).unwrap_or_else(|e| panic!("{}", e));
    let n = net.param_count();
    let (alpha, lambda) = (0.01f32, 0.7f32);
    let mut trace = vec![0.0f32; n];
    let mut acts = net.new_activations();
    let mut generator = MoveGenerator::<EnginePosition>::new();

    let (mut t_grad, mut t_choose, mut t_movegen, mut t_update, mut t_total) =
        (Duration::ZERO, Duration::ZERO, Duration::ZERO, Duration::ZERO, Duration::ZERO);
    let (mut moves, mut evaluations, mut distinct) = (0u64, 0u64, 0u64);
    let started = Instant::now();
    for seed in 0..games {
        let mut dice = Dice::new(seed);
        let mut position = EnginePosition::new();
        position.setup_starting_position();
        let (starter, mut d1, mut d2) = opening_roll(&mut dice);
        position.set_turn(starter);
        trace.iter_mut().for_each(|e| *e = 0.0);
        let mut input = [0.0f32; MAX_INPUTS];
        loop {
            let t0 = Instant::now();
            net.encode(&position, &mut input);
            let v = net.forward_keep(&input, &mut acts);
            let (value, sign) = if position.get_turn() == Player::Me { (v, 1.0) } else { (1.0 - v, -1.0) };
            let t1 = Instant::now();
            let t2 = t1;

            // Move choice, as in training; counts evaluations and distinct candidates.
            let calls = Cell::new(0u64);
            let seen = std::cell::RefCell::new(HashSet::new());
            let next = generator
                .best_position(&position, d1, d2, |c| {
                    calls.set(calls.get() + 1);
                    seen.borrow_mut().insert(c.cache_key(0));
                    -net.equity(c)
                })
                .map(|(p, _)| p);
            evaluations += calls.get();
            distinct += seen.borrow().len() as u64;
            // The same move generation with a trivial score: the cost of generating alone.
            let t4 = Instant::now();
            std::hint::black_box(generator.best_position(&position, d1, d2, |c| c.count(Player::Me, 0) as f64));
            let t5 = Instant::now();
            // Timed without the counting overhead: the choice again, plainly.
            std::hint::black_box(generator.best_position(&position, d1, d2, |c| -net.equity(c)));
            let t6 = Instant::now();

            let next = next.unwrap_or_else(|| {
                let mut p = position.clone();
                p.switch_turn();
                p
            });
            let t7 = Instant::now();
            let error = value_for_me(&net, &next) - value;
            net.td_update(&mut acts, &mut trace, sign, lambda, alpha * error);
            let t8 = Instant::now();

            t_grad += t1 - t0;
            t_movegen += t5 - t4;
            t_choose += t6 - t5;
            t_update += t8 - t7;
            t_total += (t2 - t0) + (t6 - t5) + (t8 - t7);
            moves += 1;
            if is_game_over(&next) {
                break;
            }
            position = next;
            d1 = dice.roll();
            d2 = dice.roll();
        }
    }
    let _ = started;
    let per_move = |d: Duration| d.as_nanos() as f64 / moves as f64 / 1000.0;
    let row = |name: &str, d: Duration| {
        println!("  {:<44} {:7.2} µs  {:5.1}%", name, per_move(d), 100.0 * d.as_secs_f64() / t_total.as_secs_f64())
    };
    println!("net {} ({} params), {} games, {} moves ({:.1} per game)\n",
             net.description(), n, games, moves, moves as f64 / games as f64);
    println!("per move (training work only, {:.0} games/s):", games as f64 / t_total.as_secs_f64());
    row("forward pass keeping activations", t_grad);
    row("move choice (generation + evaluations)", t_choose);
    row("   of which move generation alone", t_movegen);
    row("backward pass + trace + weights (td_update)", t_update);
    println!("\nevaluations per move: {:.1}, distinct positions among them: {:.1} ({:.0}% repeats)",
             evaluations as f64 / moves as f64, distinct as f64 / moves as f64,
             100.0 * (evaluations - distinct) as f64 / evaluations as f64);
}
