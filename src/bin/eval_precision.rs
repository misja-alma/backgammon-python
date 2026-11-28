//! How much speed and accuracy lower precision gives for evaluation (see src/eval_net.rs).
//!
//!     cargo run --release --bin eval_precision [net file, default nets/td.bgnet] [games, default 100]
//!
//! Positions: the decisions of self-play games of the f32 net at depth 0, and all their candidate
//! moves. Per variant: time per forward pass (pre-encoded), mean and max |difference| in winning
//! chance against f32 over all candidates, and per decision whether it picks another move than
//! f32, with the average loss of those moves in the f32 net's own equity.
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

use backgammon_rust::encoding::MAX_INPUTS;
use backgammon_rust::eval_net::{fake_quantized, EvalNet, EvalNetEvaluator, Kind};
use backgammon_rust::evaluator::is_game_over;
use backgammon_rust::game::{opening_roll, Dice};
use backgammon_rust::move_generator::MoveGenerator;
use backgammon_rust::{NeuralNet, NeuralNetEvaluator, Player, Position, RustAnalyzer, SimplePosition};

struct Decision {
    position: SimplePosition,
    dice: (i32, i32),
    /// Indices into the candidate list.
    first: usize,
    count: usize,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args.first().map(String::as_str).unwrap_or("nets/td.bgnet");
    let games: u64 = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(100);
    let net = NeuralNet::load(Path::new(path)).unwrap_or_else(|e| panic!("{}", e));

    // Self-play of the f32 net; alternating backgammon and nackgammon starts.
    let mut candidates: Vec<SimplePosition> = Vec::new();
    let mut decisions: Vec<Decision> = Vec::new();
    let mut generator = MoveGenerator::new();
    for seed in 0..games {
        let mut dice = Dice::new(1000 + seed);
        let mut position = SimplePosition::new();
        position.setup_starting_position();
        if seed % 2 == 1 {
            for player in [Player::Me, Player::Opponent] {
                position.set_checkers(player, 13, 4).unwrap();
                position.set_checkers(player, 6, 4).unwrap();
                position.set_checkers(player, 23, 2).unwrap();
            }
        }
        let (starter, mut d1, mut d2) = opening_roll(&mut dice);
        position.set_turn(starter);
        while !is_game_over(&position) {
            let moves = generator.legal_positions(&position, d1, d2);
            if moves.len() > 1 && moves.iter().all(|m| !is_game_over(m)) {
                decisions.push(Decision { position: position.clone(), dice: (d1, d2), first: candidates.len(), count: moves.len() });
                candidates.extend(moves.iter().cloned());
            }
            let next = generator.best_position(&position, d1, d2, |c| -net.equity(c)).map(|(p, _)| p);
            position = next.unwrap_or_else(|| {
                let mut p = position.clone();
                p.switch_turn();
                p
            });
            d1 = dice.roll();
            d2 = dice.roll();
        }
    }
    let inputs: Vec<[f32; MAX_INPUTS]> = candidates.iter().map(|p| {
        let mut x = [0.0; MAX_INPUTS];
        net.encode(p, &mut x);
        x
    }).collect();
    println!("net {}: {} games, {} decisions, {} candidate positions\n", net.description(), games, decisions.len(), inputs.len());

    if std::env::var("SEARCH_ONLY").is_ok() {
        let sample: Vec<(SimplePosition, (i32, i32))> =
            decisions.iter().step_by(5).map(|d| (d.position.clone(), d.dice)).collect();
        search_speed(&net, &sample);
        return;
    }
    let reference: Vec<f32> = inputs.iter().map(|x| net.forward(x)).collect();
    let scales = EvalNet::calibrate(&net, &inputs[..inputs.len().min(50_000)], 0.9999);
    println!("i8 input scales per hidden layer: {:?}\n", scales);

    let n_hidden = net.sizes().len() - 2;
    let all = |k: Kind| vec![k; n_hidden];
    let mut first_f16 = all(Kind::I8);
    first_f16[0] = Kind::F16;
    let mut first_i8 = all(Kind::F16);
    first_i8[0] = Kind::I8;
    let mut variants: Vec<(String, Box<dyn Fn(&[f32]) -> f32>)> = Vec::new();
    let f32_net = net.clone();
    variants.push(("f32 NeuralNet::forward".into(), Box::new(move |x| f32_net.forward(x))));
    for kinds in [all(Kind::F32), all(Kind::F16Acc32), all(Kind::F16), all(Kind::I8), first_f16, first_i8] {
        let e = EvalNet::from_net(&net, &kinds, &scales).unwrap();
        let name: Vec<&str> = kinds.iter().map(|k| k.name()).collect();
        variants.push((format!("EvalNet [{}]", name.join(" ")), Box::new(move |x| e.forward(x))));
    }
    for bits in [8, 6, 5, 4, 3] {
        let q = fake_quantized(&net, bits);
        variants.push((format!("f32 compute, {}-bit weights", bits), Box::new(move |x| q.forward(x))));
    }

    println!("{:<36} {:>8} {:>10} {:>9} {:>11} {:>12}", "variant", "ns/eval", "mean |dp|", "max |dp|", "other move", "loss/decis.");
    for (name, f) in &variants {
        let mut best = f64::MAX;
        for _ in 0..5 {
            let t = Instant::now();
            let mut s = 0.0f32;
            for x in &inputs {
                s += f(black_box(x));
            }
            black_box(s);
            best = best.min(t.elapsed().as_nanos() as f64 / inputs.len() as f64);
        }
        let out: Vec<f32> = inputs.iter().map(|x| f(x)).collect();
        let diffs: Vec<f64> = out.iter().zip(&reference).map(|(a, b)| (a - b).abs() as f64).collect();
        let mean = diffs.iter().sum::<f64>() / diffs.len() as f64;
        let max = diffs.iter().cloned().fold(0.0, f64::max);
        // Candidates are evaluated for the opponent (who is on roll there): best = lowest.
        let (mut other, mut loss) = (0usize, 0.0f64);
        for d in &decisions {
            let range = d.first..d.first + d.count;
            let pick = |v: &[f32]| range.clone().min_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap()).unwrap();
            let (r, q) = (pick(&reference), pick(&out));
            if r != q {
                other += 1;
                loss += 2.0 * (reference[q] - reference[r]) as f64;
            }
        }
        println!("{:<36} {:>8.0} {:>10.6} {:>9.5} {:>10.3}% {:>12.6}", name, best, mean, max,
                 100.0 * other as f64 / decisions.len() as f64, loss / decisions.len() as f64);
    }
}

/// Whole searches: time per move at depths 1 and 2 with the f32 net and its f16 copy (fresh
/// cache per move), and how often they pick different moves.
pub fn search_speed(net: &NeuralNet, positions: &[(SimplePosition, (i32, i32))]) {
    println!("\nsearch (best_move, fresh 256 MB cache per move), {} positions:", positions.len());
    let f32_analyzer = RustAnalyzer::with_evaluator(Box::new(NeuralNetEvaluator { net: net.clone() }), 256);
    let half = EvalNet::from_net(net, &vec![Kind::F16; net.sizes().len() - 2], &[]).unwrap();
    let f16_analyzer = RustAnalyzer::with_evaluator(Box::new(EvalNetEvaluator::new(half)), 256);
    for depth in [1u32, 2] {
        let n = if depth == 1 { positions.len() } else { positions.len().min(300) };
        let mut times = [0.0f64; 2];
        let mut picks: [Vec<SimplePosition>; 2] = [Vec::new(), Vec::new()];
        for (k, analyzer) in [&f32_analyzer, &f16_analyzer].into_iter().enumerate() {
            for (p, (d1, d2)) in &positions[..n] {
                analyzer.clear_cache();
                let t = Instant::now();
                let (best, _) = analyzer.best_move(p, *d1, *d2, depth).unwrap();
                times[k] += t.elapsed().as_secs_f64();
                picks[k].push(best);
            }
        }
        let differ = picks[0].iter().zip(&picks[1]).filter(|(a, b)| a != b).count();
        println!("  depth {}: f32 {:8.3} ms/move, f16 {:8.3} ms/move ({:.2}x), different move in {} of {}",
                 depth, 1e3 * times[0] / n as f64, 1e3 * times[1] / n as f64, times[0] / times[1], differ, n);
    }
}
