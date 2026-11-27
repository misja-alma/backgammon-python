//! Where does the time of money TD(λ) training go? Replays `TdTrainer::train_game_money` with a
//! timer around each part, and counts the evaluations per move.
//!
//!     cargo run --release --bin train_profile_money [net, default nets/td-money.bgnet] [games, default 200]
//!
//! The net is trained while profiling (as in real training: offline TD(λ), see td_training.rs), on
//! a copy; nothing is saved. One thread, with the bearoff database when data/bearoff-6x15.bd
//! exists. A third argument `f16` chooses the moves with the net's f16 copy (`train --search f16`).
//!
//! History (256-512-256-5 money net, this machine, µs per move):
//! - 2026-11-27, online traces: forward 3.9, move choice 53.6 (18.4 evaluations), evaluation of the
//!   chosen position 2.8, backward + 5 dense traces + weights 223: 283 µs, 63 games/s.
//! - 2026-11-27, offline TD(λ) (one sparse backward pass per position at the end of the game).
use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::path::Path;
use std::time::{Duration, Instant};

use backgammon_rust::bearoff::BearoffDb;
use backgammon_rust::encoding::MAX_INPUTS;
use backgammon_rust::evaluator::{flip_outcome, money_equity, MONEY_OUTPUTS};
use backgammon_rust::game::{self, Dice, Variant, MAX_TURNS};
use backgammon_rust::move_generator::MoveGenerator;
use backgammon_rust::td_training::{exact_outcome, learn_offline_from_errors, ME_SIGNS, ME_SOURCE};
use backgammon_rust::{EnginePosition, NeuralNet, Player, Position};

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "nets/td-money.bgnet".into());
    let games: u64 = std::env::args().nth(2).map(|g| g.parse().unwrap()).unwrap_or(200);
    let mut net = NeuralNet::load(Path::new(&path)).unwrap_or_else(|e| panic!("{}", e));
    let bearoff = BearoffDb::open(Path::new("data/bearoff-6x15.bd")).ok();
    let bearoff = bearoff.as_ref();
    let n = net.param_count();
    let (alpha, lambda) = (0.0025f32, 0.7f32);
    let search_f16 = std::env::args().nth(3).as_deref() == Some("f16");
    let mut grad = vec![0.0f32; n];
    let mut game_acts: Vec<backgammon_rust::neural_net::Activations> = Vec::new();
    let (mut errors, mut sides, mut learns) = (Vec::new(), Vec::new(), Vec::new());
    let mut generator = MoveGenerator::<EnginePosition>::new();

    let (mut t_fwd, mut t_choose, mut t_movegen, mut t_after, mut t_update, mut t_total) =
        (Duration::ZERO, Duration::ZERO, Duration::ZERO, Duration::ZERO, Duration::ZERO, Duration::ZERO);
    let (mut moves, mut evaluations, mut distinct, mut exact_evals) = (0u64, 0u64, 0u64, 0u64);
    let wall = Instant::now();
    for seed in 0..games {
        let mut dice = Dice::new(seed);
        let variant = if seed % 2 == 1 { Variant::Backgammon } else { Variant::Nackgammon };
        let mut position = variant.starting_position();
        let (starter, mut d1, mut d2) = game::opening_roll(&mut dice);
        position.set_turn(starter);
        let copy = search_f16.then(|| {
            backgammon_rust::td_training::quantized_copy(&net, backgammon_rust::eval_net::Kind::F16)
        });
        errors.clear();
        sides.clear();
        learns.clear();
        let mut values: Vec<[f32; MONEY_OUTPUTS]> = Vec::new();
        let mut input = [0.0f32; MAX_INPUTS];
        for _ in 1..=MAX_TURNS {
            let t0 = Instant::now();
            net.encode(&position, &mut input);
            let t = values.len();
            if game_acts.len() <= t {
                game_acts.push(net.new_activations());
            }
            net.forward_keep(&input, &mut game_acts[t]);
            let mut y = [0.0; MONEY_OUTPUTS];
            y.copy_from_slice(game_acts[t].outputs());
            let side = (position.get_turn() != Player::Me) as usize;
            values.push(if side == 0 { y } else { flip_outcome(&y) });
            sides.push(side);
            learns.push(Some(t));
            let t1 = Instant::now();

            // Counting pass (not timed): evaluations, distinct candidates, exact ones.
            let calls = Cell::new(0u64);
            let exacts = Cell::new(0u64);
            let seen = RefCell::new(HashSet::new());
            generator.best_position(&position, d1, d2, |c| {
                calls.set(calls.get() + 1);
                seen.borrow_mut().insert(c.cache_key(0));
                match exact_outcome(bearoff, c) {
                    Some(o) => { exacts.set(exacts.get() + 1); -money_equity(&o) }
                    None => -money_equity(&match &copy { Some(c16) => c16.outcome(c), None => net.outcome(c) }),
                }
            });
            evaluations += calls.get();
            exact_evals += exacts.get();
            distinct += seen.borrow().len() as u64;
            let t2 = Instant::now();
            std::hint::black_box(generator.best_position(&position, d1, d2, |c| c.count(Player::Me, 0) as f64));
            let t3 = Instant::now();
            let next = generator
                .best_position(&position, d1, d2, |c| {
                    -money_equity(&exact_outcome(bearoff, c).unwrap_or_else(|| match &copy {
                        Some(c16) => c16.outcome(c),
                        None => net.outcome(c),
                    }))
                })
                .map(|(p, _)| p)
                .unwrap_or_else(|| {
                    let mut p = position.clone();
                    p.switch_turn();
                    p
                });
            let t4 = Instant::now();
            // The target is the next turn's estimate (fixed weights) or the exact outcome.
            let exact = exact_outcome(bearoff, &next);
            let t5 = Instant::now();
            if let Some(exact) = exact {
                values.push(if next.get_turn() == Player::Me { exact } else { flip_outcome(&exact) });
                for t in 0..sides.len() {
                    errors.push(std::array::from_fn(|v| values[t + 1][v] - values[t][v]));
                }
                learn_offline_from_errors(&mut net, &mut game_acts, &errors, &sides, &learns, &ME_SOURCE, &ME_SIGNS,
                                          MONEY_OUTPUTS, alpha, lambda, &mut grad);
            }
            let t6 = Instant::now();
            t_fwd += t1 - t0;
            t_movegen += t3 - t2;
            t_choose += t4 - t3;
            t_after += t5 - t4;
            t_update += t6 - t5;
            t_total += (t1 - t0) + (t6 - t3);
            moves += 1;
            if exact.is_some() {
                break;
            }
            position = next;
            d1 = dice.roll();
            d2 = dice.roll();
        }
    }
    let _ = wall;
    let per_move = |d: Duration| d.as_nanos() as f64 / moves as f64 / 1000.0;
    let row = |name: &str, d: Duration| {
        println!("  {:<44} {:8.2} µs  {:5.1}%", name, per_move(d), 100.0 * d.as_secs_f64() / t_total.as_secs_f64())
    };
    println!("net {} ({} params), {} games, {} moves ({:.1} per game), bearoff db {}\n",
             net.description(), n, games, moves, moves as f64 / games as f64, bearoff.is_some());
    println!("per move (training work only, {:.0} games/s on one thread{}):", games as f64 / t_total.as_secs_f64(),
             if search_f16 { ", moves chosen by the f16 copy" } else { "" });
    row("forward pass keeping activations", t_fwd);
    row("move choice (generation + evaluations)", t_choose);
    row("   of which move generation alone", t_movegen);
    row("exact outcome of the chosen position", t_after);
    row("offline TD(λ) at the end of the game", t_update);
    println!("\nevaluations per move: {:.1} ({:.1} exact), distinct positions among them: {:.1} ({:.0}% repeats)",
             evaluations as f64 / moves as f64, exact_evals as f64 / moves as f64, distinct as f64 / moves as f64,
             100.0 * (evaluations - distinct) as f64 / evaluations.max(1) as f64);
}
