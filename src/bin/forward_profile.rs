//! Where does the time of one neural net evaluation go? Times each part of `NeuralNet::forward`
//! separately on realistic positions (collected from self-play games of the net itself), using
//! the same layer functions as `forward`. Also counts how many inputs and hidden units are
//! non-zero, since hidden layers only add in the non-zero inputs.
//!
//!     cargo run --release --bin forward_profile [net file, default nets/td.bgnet]
//!
//! History (54-128-64-1 net, this machine):
//! - 2026-11-19, first version: 615 ns per evaluation. Layer 2 took half (each active input
//!   re-read and re-wrote all 64 outputs in memory), the 64x1 output layer 70 ns (an inner loop of
//!   length 1 can't be vectorized), zeroing two 4 KB stack buffers 60-80 ns. Dense layers without
//!   zero-skipping were slower (layer 1: 363 vs 135 ns, layer 2: 561 vs 315 ns).
//! - 2026-11-19, chunked registers + FMA, dot-product output, reused scratch: 265 ns.
//! - 2026-11-19, default net now 54-128-128-128-1: 645 ns (128-64 net: 249 ns). Layer 2 (128x128,
//!   ~70 active inputs) takes 40%, collecting the active inputs of three layers 22%.
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

use backgammon_rust::encoding::MAX_INPUTS;
use backgammon_rust::evaluator::is_game_over;
use backgammon_rust::game::{opening_roll, Dice};
use backgammon_rust::move_generator::MoveGenerator;
use backgammon_rust::neural_net::{dot, sparse_layer, NeuralNet};
use backgammon_rust::{Position, SimplePosition};

/// Positions from self-play games of the net (depth 0), including every candidate it considered,
/// since those are what the search evaluates.
fn collect_positions(net: &NeuralNet, games: u64) -> Vec<SimplePosition> {
    let mut positions = Vec::new();
    let mut generator = MoveGenerator::new();
    for seed in 0..games {
        let mut dice = Dice::new(seed);
        let mut position = SimplePosition::new();
        position.setup_starting_position();
        let (starter, mut d1, mut d2) = opening_roll(&mut dice);
        position.set_turn(starter);
        while !is_game_over(&position) {
            let mut candidates = generator.legal_positions(&position, d1, d2);
            candidates.retain(|c| !is_game_over(c));
            let next = generator.best_position(&position, d1, d2, |c| -net.equity(c)).map(|(p, _)| p);
            positions.extend(candidates);
            position = next.unwrap_or_else(|| {
                let mut p = position.clone();
                p.switch_turn();
                p
            });
            d1 = dice.roll();
            d2 = dice.roll();
        }
    }
    positions
}

/// Time per call of `f` over all items, in ns, best of a few repeats.
fn time_ns<T>(items: &[T], repeats: usize, mut f: impl FnMut(&T) -> f32) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..repeats {
        let t = Instant::now();
        let mut sum = 0.0f32;
        for item in items {
            sum += f(black_box(item));
        }
        black_box(sum);
        best = best.min(t.elapsed().as_nanos() as f64 / items.len() as f64);
    }
    best
}

/// The non-zero values of `values` with their indices, as `forward` collects them.
fn active(values: &[f32]) -> Vec<(u32, f32)> {
    values.iter().enumerate().filter(|(_, &x)| x != 0.0).map(|(i, &x)| (i as u32, x)).collect()
}

/// Weights and biases of a layer.
fn layer_params(net: &NeuralNet, layer: usize) -> (&[f32], &[f32]) {
    let (n_in, n_out) = (net.sizes()[layer], net.sizes()[layer + 1]);
    let start = net.layer_offset(layer);
    let params = net.params();
    (&params[start..start + n_in * n_out], &params[start + n_in * n_out..start + n_in * n_out + n_out])
}

fn hidden(net: &NeuralNet, layer: usize, input: &[f32]) -> Vec<f32> {
    let (weights, biases) = layer_params(net, layer);
    let mut out = vec![0.0; biases.len()];
    sparse_layer(weights, biases, &active(input), &mut out);
    out.iter_mut().for_each(|o| *o = o.max(0.0));
    out
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "nets/td.bgnet".into());
    let net = NeuralNet::load(Path::new(&path)).unwrap_or_else(|e| panic!("{}", e));
    let sizes = net.sizes().to_vec();
    let n_hidden = sizes.len() - 2;

    let positions = collect_positions(&net, 40);
    let inputs: Vec<[f32; MAX_INPUTS]> = positions.iter().map(|p| {
        let mut x = [0.0; MAX_INPUTS];
        net.encode(p, &mut x);
        x
    }).collect();
    // layer_inputs[l][i]: the input of layer l for position i.
    let mut layer_inputs: Vec<Vec<Vec<f32>>> = vec![inputs.iter().map(|x| x[..sizes[0]].to_vec()).collect()];
    for l in 0..n_hidden {
        let next = layer_inputs[l].iter().map(|x| hidden(&net, l, x)).collect();
        layer_inputs.push(next);
    }
    let actives: Vec<Vec<Vec<(u32, f32)>>> =
        layer_inputs[..n_hidden].iter().map(|layer| layer.iter().map(|x| active(x)).collect()).collect();

    let n = positions.len() as f64;
    println!("net {}, {} positions from 40 self-play games\n", net.description(), positions.len());
    for (l, layer) in layer_inputs.iter().enumerate() {
        let non_zero = layer.iter().map(|x| x.iter().filter(|&&v| v != 0.0).count() as f64).sum::<f64>() / n;
        let name = if l == 0 { "non-zero inputs:".to_string() } else { format!("active hidden-{} units:", l) };
        println!("{:<24}{:5.1} of {}", name, non_zero, sizes[l]);
    }
    // Units that are zero for every position: "dead" ReLUs, which get no gradient and never learn.
    for l in 1..=n_hidden {
        let dead = (0..sizes[l]).filter(|&j| layer_inputs[l].iter().all(|x| x[j] == 0.0)).count();
        println!("dead hidden-{} units:    {:5} of {} (zero in every position)", l, dead, sizes[l]);
    }
    println!();

    let reps = 5;
    let whole = time_ns(&positions, reps, |p| net.equity(p) as f32);
    let forward = time_ns(&inputs, reps, |x| net.forward(x));
    let encoding = time_ns(&positions, reps, |p| {
        let mut x = [0.0; MAX_INPUTS];
        net.encode(p, &mut x);
        x[3]
    });
    let row = |name: &str, ns: f64| println!("  {:<34} {:6.0} ns  {:5.1}%", name, ns, 100.0 * ns / whole);
    println!("whole evaluation (net.equity): {:6.0} ns", whole);
    row("encoding", encoding);
    row("forward pass (pre-encoded)", forward);
    println!("parts of the forward pass, timed separately:");
    let mut parts = 0.0;
    for l in 0..n_hidden {
        let (w, b) = layer_params(&net, l);
        let mut out = vec![0.0; sizes[l + 1]];
        let t = time_ns(&actives[l], reps, |a| {
            sparse_layer(w, b, a, &mut out);
            out.iter_mut().for_each(|o| *o = o.max(0.0));
            out[0]
        });
        row(&format!("layer {} ({}x{}) + ReLU", l + 1, sizes[l], sizes[l + 1]), t);
        parts += t;
    }
    let (w, b) = layer_params(&net, n_hidden);
    let t = time_ns(&layer_inputs[n_hidden], reps, |h| dot(h, w, b[0]));
    row(&format!("layer {} ({}x1), dot product", n_hidden + 1, sizes[n_hidden]), t);
    parts += t;
    row("rest (collecting active inputs etc.)", forward - parts);
}
