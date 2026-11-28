//! Where does the time of an incremental evaluation in the search go? For the positions of
//! self-play games of an f16 net, all candidates of each position (as the search's last level
//! evaluates them), timed stage by stage (see `EvalNet::profile_incremental`), plus encoding and
//! the evaluator call as the search makes it.
//!
//!     cargo run --release --bin incremental_profile [f16 net, default nets/td-money-eval.bgnet] [games, default 30]
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

use backgammon_rust::encoding::MAX_INPUTS;
use backgammon_rust::eval_net::{EvalNet, EvalNetEvaluator, HalfBase};
use backgammon_rust::evaluator::{is_game_over, Evaluator, Scoring};
use backgammon_rust::game::{opening_roll, Dice};
use backgammon_rust::move_generator::MoveGenerator;
use backgammon_rust::{EnginePosition, Position};

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "nets/td-money-eval.bgnet".into());
    let games: u64 = std::env::args().nth(2).map(|g| g.parse().unwrap()).unwrap_or(30);
    let net = EvalNet::load(Path::new(&path)).unwrap_or_else(|e| panic!("{}", e));
    let evaluator = EvalNetEvaluator::new(EvalNet::load(Path::new(&path)).unwrap());
    let encoding = net.encoding();
    let mut generator = MoveGenerator::<EnginePosition>::new();
    // (parent, its non-terminal children)
    let mut sets: Vec<(EnginePosition, Vec<EnginePosition>)> = Vec::new();
    for seed in 0..games {
        let mut dice = Dice::new(seed);
        let mut position = EnginePosition::new();
        position.setup_starting_position();
        let (starter, mut d1, mut d2) = opening_roll(&mut dice);
        position.set_turn(starter);
        while !is_game_over(&position) {
            let children: Vec<EnginePosition> =
                generator.legal_positions(&position, d1, d2).into_iter().filter(|c| !is_game_over(c)).collect();
            if !children.is_empty() {
                sets.push((position.clone(), children));
            }
            position = generator.best_position(&position, d1, d2, |c| -evaluator.evaluate_scored(c, Scoring::Money))
                .map(|(p, _)| p)
                .unwrap_or_else(|| { let mut p = position.clone(); p.switch_turn(); p });
            d1 = dice.roll();
            d2 = dice.roll();
        }
    }
    let total: usize = sets.iter().map(|(_, c)| c.len()).sum();
    println!("net {}, {} positions, {} candidates ({:.1} per position)\n", net.description(), sets.len(), total,
             total as f64 / sets.len() as f64);

    let repeats = 7;
    let (mut parts, mut t_encode, mut t_base, mut t_call, mut t_encode_diff) = ([0.0f64; 6], 0.0, 0.0, 0.0, 0.0);
    let mut base = HalfBase::default();
    for (parent, children) in &sets {
        let k = children.len() as f64;
        let mut flipped = parent.clone();
        flipped.switch_turn();
        let mut base_input = [0.0f32; MAX_INPUTS];
        encoding.encode(&flipped, &mut base_input);
        net.layer1_base(&base_input, &mut base);
        let inputs: Vec<[f32; MAX_INPUTS]> = children.iter().map(|c| {
            let mut x = [0.0f32; MAX_INPUTS];
            encoding.encode(c, &mut x);
            x
        }).collect();
        let p = net.profile_incremental(&base, &inputs, repeats);
        for i in 0..6 {
            parts[i] += p[i] * k;
        }
        let best = |f: &mut dyn FnMut() -> f32| -> f64 {
            let mut best = f64::MAX;
            for _ in 0..repeats {
                let t = Instant::now();
                black_box(f());
                best = best.min(t.elapsed().as_nanos() as f64);
            }
            best
        };
        t_encode += best(&mut || {
            let mut sum = 0.0;
            for c in children {
                let mut x = [0.0f32; MAX_INPUTS];
                encoding.encode(c, &mut x);
                sum += x[7];
            }
            sum
        });
        let counts = backgammon_rust::encoding::block_counts(&flipped);
        let (mut idx, mut diff) = ([0u16; MAX_INPUTS], [0.0f32; MAX_INPUTS]);
        t_encode_diff += best(&mut || {
            let mut sum = 0usize;
            for c in children {
                sum += encoding.encode_diff(c, &counts, &base_input, &mut idx, &mut diff).unwrap();
            }
            sum as f32
        });
        t_base += best(&mut || {
            let mut b = HalfBase::default();
            let mut x = [0.0f32; MAX_INPUTS];
            encoding.encode(&flipped, &mut x);
            net.layer1_base(&x, &mut b);
            0.0
        });
        t_call += best(&mut || {
            let mut sum = 0.0;
            for c in children {
                sum += evaluator.evaluate_child_scored(parent, c, Scoring::Money) as f32;
            }
            sum
        });
    }
    let n = total as f64;
    let names = ["collect the differing inputs from the encoding (no longer in the call)", "layer 1 (incremental)", "layer 2",
                 "layer 2, rows already in L1 (same input repeated)", "layer 2 with prefetching", "output layer + sigmoids"];
    println!("per candidate (ns):");
    println!("  {:<52} {:6.0}", "evaluator call as in the search (all of it)", t_call / n);
    println!("  {:<52} {:6.0}", "encoding (no longer in the call)", t_encode / n);
    println!("  {:<52} {:6.0}", "differing inputs from the checker counts (encode_diff)", t_encode_diff / n);
    println!("  {:<52} {:6.0}", "base for the parent (per candidate)", t_base / n);
    for (name, t) in names.iter().zip(parts) {
        println!("  {:<52} {:6.0}", name, t / n);
    }
    let stages = t_encode_diff / n + parts[1] / n + parts[2] / n + parts[5] / n + t_base / n;
    println!("  {:<52} {:6.0}", "rest of the call (identity check, equity, ...)", t_call / n - stages);
}
