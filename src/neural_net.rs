//! A small fully connected neural net that estimates the winning chance of the player on roll.
//!
//! Architecture: the inputs of the net's `Encoding` (see `encoding`), any number of ReLU hidden layers, and one
//! sigmoid output. The layer sizes are stored in the weight file, so nets of different shapes can
//! be swapped in without code changes; new nets default to hidden layers of 128 and 64.
//!
//! Parameters live in one flat vector, layer after layer: the weights of a layer as
//! `[input][output]` (the row for one input is contiguous, so an input that is zero, which most
//! checker counts are, is skipped entirely), followed by its biases. Training code (TD(λ), see
//! `td_training`) works directly on that flat vector and on gradients with the same layout.
//!
//! Inference (`forward`) is tuned for speed, measured with `cargo run --release --bin
//! forward_profile`: hidden layers are computed in chunks of `CHUNK` outputs that stay in vector
//! registers while the active (non-zero) inputs are added in with fused multiply-adds, and the
//! single-output layer is a plain dot product. Scratch buffers are reused per thread instead of
//! being zeroed for every call.
use std::cell::RefCell;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::encoding::{Encoding, MAX_INPUTS};
use crate::evaluator::{terminal_equity, Evaluator};
use crate::position::Position;

pub const DEFAULT_HIDDEN: [usize; 3] = [128, 128, 128];
/// Upper bound on layer width (the size of the inference scratch buffers).
pub const MAX_WIDTH: usize = 1024;
/// Outputs computed together in `forward`: 32 floats are 8 NEON registers, so the partial sums of a
/// chunk stay in registers while all active inputs are added in.
const CHUNK: usize = 32;

/// Reused buffers for `forward`, one set per thread.
struct Scratch {
    current: Vec<f32>,
    next: Vec<f32>,
    /// The non-zero values of `current` and their indices.
    active: Vec<(u32, f32)>,
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch {
        current: vec![0.0; MAX_WIDTH],
        next: vec![0.0; MAX_WIDTH],
        active: vec![(0, 0.0); MAX_WIDTH],
    });
}

/// `out = biases + Σ x * weights[i]` over the active inputs (i, x); weights are `[input][output]`.
/// Public for `forward_profile`.
#[inline]
pub fn sparse_layer(weights: &[f32], biases: &[f32], active: &[(u32, f32)], out: &mut [f32]) {
    let n_out = out.len();
    let mut start = 0;
    while start + CHUNK <= n_out {
        let mut acc = [0.0f32; CHUNK];
        acc.copy_from_slice(&biases[start..start + CHUNK]);
        for &(i, x) in active {
            let row = i as usize * n_out + start;
            let w = &weights[row..row + CHUNK];
            for k in 0..CHUNK {
                acc[k] = x.mul_add(w[k], acc[k]);
            }
        }
        out[start..start + CHUNK].copy_from_slice(&acc);
        start += CHUNK;
    }
    // Widths that aren't a multiple of CHUNK: the rest one output at a time.
    for j in start..n_out {
        let mut sum = biases[j];
        for &(i, x) in active {
            sum = x.mul_add(weights[i as usize * n_out + j], sum);
        }
        out[j] = sum;
    }
}

/// `bias + Σ input[i] * weights[i]`, with 8 independent partial sums so it vectorizes.
/// Public for `forward_profile`.
#[inline]
pub fn dot(input: &[f32], weights: &[f32], bias: f32) -> f32 {
    let mut acc = [0.0f32; 8];
    let chunks = input.len() / 8 * 8;
    for (xs, ws) in input[..chunks].chunks_exact(8).zip(weights[..chunks].chunks_exact(8)) {
        for k in 0..8 {
            acc[k] = xs[k].mul_add(ws[k], acc[k]);
        }
    }
    let mut sum = bias + acc.iter().sum::<f32>();
    for (x, w) in input[chunks..].iter().zip(&weights[chunks..]) {
        sum = x.mul_add(*w, sum);
    }
    sum
}

/// Per-layer inputs of one forward pass plus back-propagation buffers, for `td_update`.
pub struct Activations {
    /// layers[l]: the input of layer l (the net's input, then the hidden layers' ReLU outputs).
    layers: Vec<Vec<f32>>,
    active: Vec<(u32, f32)>,
    delta: Vec<f32>,
    delta_prev: Vec<f32>,
    /// The net's output (winning chance).
    output: f32,
}

const MAGIC: &str = "BGNET 1";

#[derive(Clone)]
pub struct NeuralNet {
    /// How positions are turned into inputs; `sizes[0]` is its input count.
    encoding: Encoding,
    /// Layer widths from input to output, e.g. [54, 128, 64, 1].
    sizes: Vec<usize>,
    params: Vec<f32>,
    /// Start of each layer's weights in `params`; its biases follow the weights.
    offsets: Vec<usize>,
    /// Number of training games behind these weights (informational, stored in the file).
    pub games_trained: u64,
}

/// SplitMix64, for reproducible weight initialization without a dependency.
struct Rng(u64);

impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        (z >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Standard normal sample (Box-Muller).
    fn normal(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-7);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl NeuralNet {
    /// A freshly initialized net with the given hidden layer sizes (He initialization).
    pub fn new_random(encoding: Encoding, hidden: &[usize], seed: u64) -> Self {
        let mut sizes = vec![encoding.n_inputs()];
        sizes.extend_from_slice(hidden);
        sizes.push(1);
        let mut net = Self::with_sizes(encoding, sizes).expect("valid layer sizes");
        let mut rng = Rng(seed);
        for layer in 0..net.layer_count() {
            let (n_in, n_out) = (net.sizes[layer], net.sizes[layer + 1]);
            let limit = (6.0 / n_in as f32).sqrt();
            let start = net.offsets[layer];
            for w in &mut net.params[start..start + n_in * n_out] {
                *w = (2.0 * rng.next_f32() - 1.0) * limit;
            }
        }
        net
    }

    fn with_sizes(encoding: Encoding, sizes: Vec<usize>) -> Result<Self, String> {
        let n_inputs = encoding.n_inputs();
        if sizes.len() < 2 || sizes[0] != n_inputs || *sizes.last().unwrap() != 1 {
            return Err(format!("layer sizes must run from {} inputs ({}) to 1 output, got {:?}",
                               n_inputs, encoding.name(), sizes));
        }
        if sizes.iter().any(|&s| s == 0 || s > MAX_WIDTH) {
            return Err(format!("layer sizes must be between 1 and {}, got {:?}", MAX_WIDTH, sizes));
        }
        let mut offsets = Vec::new();
        let mut total = 0;
        for pair in sizes.windows(2) {
            offsets.push(total);
            total += pair[0] * pair[1] + pair[1];
        }
        Ok(NeuralNet { encoding, sizes, params: vec![0.0; total], offsets, games_trained: 0 })
    }

    pub fn sizes(&self) -> &[usize] {
        &self.sizes
    }

    pub fn encoding(&self) -> Encoding {
        self.encoding
    }

    /// The inputs for `position` in this net's encoding, in `buf[..sizes()[0]]`; pass `buf` to
    /// `forward` / `forward_keep` / `forward_with_gradient`.
    #[inline]
    pub fn encode<P: Position>(&self, position: &P, buf: &mut [f32; MAX_INPUTS]) {
        self.encoding.encode(position, buf);
    }

    fn layer_count(&self) -> usize {
        self.sizes.len() - 1
    }

    pub fn param_count(&self) -> usize {
        self.params.len()
    }

    /// All parameters, in the layout described at the top of this file.
    pub fn params(&self) -> &[f32] {
        &self.params
    }

    /// Start of layer `layer`'s weights in `params()`; its biases follow the weights.
    pub fn layer_offset(&self, layer: usize) -> usize {
        self.offsets[layer]
    }

    /// The parameters, for training: `params[k] += step[k]`.
    pub fn params_mut(&mut self) -> &mut [f32] {
        &mut self.params
    }

    /// One-line description, e.g. "54-128-64-1 counts-pips-v1, 100000 games".
    pub fn description(&self) -> String {
        let shape: Vec<String> = self.sizes.iter().map(|s| s.to_string()).collect();
        format!("{} {}, {} games", shape.join("-"), self.encoding.name(), self.games_trained)
    }

    /// Winning chance of the player on roll, for inputs from `encode` (only the first
    /// `sizes()[0]` values of `input` are used).
    pub fn forward(&self, input: &[f32]) -> f32 {
        SCRATCH.with(|scratch| {
            let Scratch { current, next, active } = &mut *scratch.borrow_mut();
            let n = self.sizes[0];
            current[..n].copy_from_slice(&input[..n]);
            let layers = self.layer_count();
            for layer in 0..layers {
                let (n_in, n_out) = (self.sizes[layer], self.sizes[layer + 1]);
                let start = self.offsets[layer];
                let weights = &self.params[start..start + n_in * n_out];
                let biases = &self.params[start + n_in * n_out..start + n_in * n_out + n_out];
                if layer + 1 == layers {
                    // The output layer has a single unit: a dense dot product is fastest.
                    return sigmoid(dot(&current[..n_in], weights, biases[0]));
                }
                // Collect the non-zero inputs without branching: always write, advance if non-zero.
                let mut count = 0;
                for (i, &x) in current[..n_in].iter().enumerate() {
                    active[count] = (i as u32, x);
                    count += (x != 0.0) as usize;
                }
                let out = &mut next[..n_out];
                sparse_layer(weights, biases, &active[..count], out);
                for o in out.iter_mut() {
                    *o = o.max(0.0); // ReLU (only hidden layers get here)
                }
                std::mem::swap(current, next);
            }
            unreachable!("a net always ends with its output layer")
        })
    }

    /// Like `forward`, and writes the gradient of the output with respect to every parameter
    /// into `grad` (same layout as the parameters, length `param_count()`).
    pub fn forward_with_gradient(&self, input: &[f32], grad: &mut [f32]) -> f32 {
        let layers = self.layer_count();
        // activations[l] is the input of layer l: the net's input, then the ReLU outputs of the
        // hidden layers; activations[layers] holds the output layer's pre-sigmoid value.
        let mut activations: Vec<Vec<f32>> = Vec::with_capacity(layers + 1);
        activations.push(input[..self.sizes[0]].to_vec());
        for layer in 0..layers {
            let (n_in, n_out) = (self.sizes[layer], self.sizes[layer + 1]);
            let start = self.offsets[layer];
            let mut out = self.params[start + n_in * n_out..start + n_in * n_out + n_out].to_vec();
            for (i, &x) in activations[layer].iter().enumerate() {
                if x != 0.0 {
                    for (o, &w) in out.iter_mut().zip(&self.params[start + i * n_out..start + (i + 1) * n_out]) {
                        *o += x * w;
                    }
                }
            }
            if layer + 1 < layers {
                for o in out.iter_mut() {
                    *o = o.max(0.0);
                }
            }
            activations.push(out);
        }
        let y = sigmoid(activations[layers][0]);

        // delta = d(output) / d(pre-activation of the current layer's units).
        let mut delta = vec![y * (1.0 - y)];
        for layer in (0..layers).rev() {
            let (n_in, n_out) = (self.sizes[layer], self.sizes[layer + 1]);
            let start = self.offsets[layer];
            let input = &activations[layer];
            for i in 0..n_in {
                let row = &mut grad[start + i * n_out..start + (i + 1) * n_out];
                for (g, &d) in row.iter_mut().zip(&delta) {
                    *g = input[i] * d;
                }
            }
            grad[start + n_in * n_out..start + n_in * n_out + n_out].copy_from_slice(&delta);
            if layer > 0 {
                delta = (0..n_in)
                    .map(|i| {
                        if input[i] <= 0.0 {
                            return 0.0; // ReLU was inactive
                        }
                        let row = &self.params[start + i * n_out..start + (i + 1) * n_out];
                        row.iter().zip(&delta).map(|(w, d)| w * d).sum()
                    })
                    .collect();
            }
        }
        y
    }

    /// This net with a larger encoding that starts with the current one (see `Encoding::extends`):
    /// the extra inputs get zero weights, so the new net gives exactly the same outputs until it is
    /// trained further.
    pub fn extend_inputs(&self, encoding: Encoding) -> Result<NeuralNet, String> {
        if !encoding.extends(self.encoding) {
            return Err(format!("encoding {} does not extend {}", encoding.name(), self.encoding.name()));
        }
        let mut sizes = self.sizes.clone();
        sizes[0] = encoding.n_inputs();
        let mut net = Self::with_sizes(encoding, sizes)?;
        net.games_trained = self.games_trained;
        let (old_in, n_out) = (self.sizes[0], self.sizes[1]);
        // Layer 0: the old input rows, zero rows for the new inputs (already zero), the biases.
        net.params[..old_in * n_out].copy_from_slice(&self.params[..old_in * n_out]);
        let new_bias = net.sizes[0] * n_out;
        net.params[new_bias..new_bias + n_out].copy_from_slice(&self.params[old_in * n_out..old_in * n_out + n_out]);
        // The other layers are unchanged.
        let rest = self.offsets.get(1).copied().unwrap_or(self.params.len());
        let new_rest = net.offsets.get(1).copied().unwrap_or(net.params.len());
        net.params[new_rest..].copy_from_slice(&self.params[rest..]);
        Ok(net)
    }

    /// "Shrink and perturb": every parameter becomes `(1 - shrink)·w + sigma·std·N(0, 1)`, where
    /// `std` is the standard deviation of that layer's weights (so `sigma` is relative to a typical
    /// weight of the layer; biases get the noise of their layer). Used to shake a net off a plateau.
    pub fn perturb(&mut self, shrink: f32, sigma: f32, seed: u64) {
        let mut rng = Rng(seed);
        for layer in 0..self.layer_count() {
            let (n_in, n_out) = (self.sizes[layer], self.sizes[layer + 1]);
            let start = self.offsets[layer];
            let weights = &self.params[start..start + n_in * n_out];
            let mean = weights.iter().map(|&w| w as f64).sum::<f64>() / weights.len() as f64;
            let var = weights.iter().map(|&w| (w as f64 - mean).powi(2)).sum::<f64>() / weights.len() as f64;
            let scale = sigma * var.sqrt() as f32;
            for p in &mut self.params[start..start + n_in * n_out + n_out] {
                *p = (1.0 - shrink) * *p + scale * rng.normal();
            }
        }
    }

    /// Buffers for `forward_keep` and `td_update`, sized for this net.
    pub fn new_activations(&self) -> Activations {
        let widest = *self.sizes.iter().max().unwrap();
        Activations {
            layers: self.sizes[..self.layer_count()].iter().map(|&n| vec![0.0; n]).collect(),
            active: vec![(0, 0.0); widest],
            delta: vec![0.0; widest],
            delta_prev: vec![0.0; widest],
            output: 0.0,
        }
    }

    /// Like `forward`, and keeps the input of every layer in `acts` for a later `td_update`.
    pub fn forward_keep(&self, input: &[f32], acts: &mut Activations) -> f32 {
        let layers = self.layer_count();
        acts.layers[0].copy_from_slice(&input[..self.sizes[0]]);
        for layer in 0..layers {
            let (n_in, n_out) = (self.sizes[layer], self.sizes[layer + 1]);
            let start = self.offsets[layer];
            let weights = &self.params[start..start + n_in * n_out];
            let biases = &self.params[start + n_in * n_out..start + n_in * n_out + n_out];
            if layer + 1 == layers {
                acts.output = sigmoid(dot(&acts.layers[layer], weights, biases[0]));
                return acts.output;
            }
            let mut count = 0;
            for (i, &x) in acts.layers[layer].iter().enumerate() {
                acts.active[count] = (i as u32, x);
                count += (x != 0.0) as usize;
            }
            let out = &mut acts.layers[layer + 1];
            sparse_layer(weights, biases, &acts.active[..count], out);
            for o in out.iter_mut() {
                *o = o.max(0.0);
            }
        }
        unreachable!("a net always ends with its output layer")
    }

    /// One TD(λ) step in a single pass over the parameters, for the input last given to
    /// `forward_keep` (whose activations are in `acts`; the parameters must not have changed since):
    /// for every parameter, `e = λ·e + sign·∂output/∂w`, then `w += step·e`. Equivalent to
    /// `forward_with_gradient` followed by separate trace and weight updates, without a gradient
    /// buffer. Back-propagation through a layer uses its weights before they are updated.
    pub fn td_update(&mut self, acts: &mut Activations, trace: &mut [f32], sign: f32, lambda: f32, step: f32) {
        let layers = self.layer_count();
        let y = acts.output;
        acts.delta[0] = y * (1.0 - y);
        for layer in (0..layers).rev() {
            let (n_in, n_out) = (self.sizes[layer], self.sizes[layer + 1]);
            let start = self.offsets[layer];
            let input = &acts.layers[layer];
            let delta = &acts.delta[..n_out];
            if layer > 0 {
                for i in 0..n_in {
                    acts.delta_prev[i] = if input[i] <= 0.0 {
                        0.0 // ReLU was inactive
                    } else {
                        dot(&self.params[start + i * n_out..start + (i + 1) * n_out], delta, 0.0)
                    };
                }
            }
            let params = &mut self.params[start..start + n_in * n_out + n_out];
            let traces = &mut trace[start..start + n_in * n_out + n_out];
            for i in 0..n_in {
                let x = sign * input[i];
                let range = i * n_out..(i + 1) * n_out;
                for ((p, e), &d) in params[range.clone()].iter_mut().zip(&mut traces[range]).zip(delta) {
                    *e = lambda * *e + x * d;
                    *p += step * *e;
                }
            }
            let range = n_in * n_out..n_in * n_out + n_out;
            for ((p, e), &d) in params[range.clone()].iter_mut().zip(&mut traces[range]).zip(delta) {
                *e = lambda * *e + sign * d;
                *p += step * *e;
            }
            if layer > 0 {
                std::mem::swap(&mut acts.delta, &mut acts.delta_prev);
            }
        }
    }

    /// Winning chance of the player on roll in `position` (the game must not be over).
    pub fn win_probability<P: Position>(&self, position: &P) -> f32 {
        let mut input = [0.0; MAX_INPUTS];
        self.encode(position, &mut input);
        self.forward(&input)
    }

    /// Equity for the side to move: exact when the game is over, otherwise 2 * winning chance - 1.
    pub fn equity<P: Position>(&self, position: &P) -> f64 {
        terminal_equity(position).unwrap_or_else(|| 2.0 * self.win_probability(position) as f64 - 1.0)
    }

    // --- weight files -----------------------------------------------------------------------
    //
    // A short text header, then the parameters as little-endian f32:
    //   BGNET 1
    //   encoding counts-pips-v1
    //   layers 54 128 64 1
    //   activation relu sigmoid
    //   games 100000
    //   data
    //   <param_count() * 4 bytes>

    pub fn save(&self, path: &Path) -> Result<(), String> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {}", dir.display(), e))?;
        }
        // Write to a temporary file and rename, so a crash never leaves a half-written net.
        let tmp = path.with_extension("tmp");
        let write = || -> std::io::Result<()> {
            let mut f = BufWriter::new(File::create(&tmp)?);
            let shape: Vec<String> = self.sizes.iter().map(|s| s.to_string()).collect();
            write!(f, "{}\nencoding {}\nlayers {}\nactivation relu sigmoid\ngames {}\ndata\n",
                   MAGIC, self.encoding.name(), shape.join(" "), self.games_trained)?;
            for p in &self.params {
                f.write_all(&p.to_le_bytes())?;
            }
            f.flush()
        };
        write().map_err(|e| format!("{}: {}", tmp.display(), e))?;
        std::fs::rename(&tmp, path).map_err(|e| format!("{}: {}", path.display(), e))
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("{}: {}", path.display(), e))?;
        let mut reader = BufReader::new(file);
        let mut fields = std::collections::HashMap::new();
        let mut first = true;
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line).map_err(|e| e.to_string())? == 0 {
                return Err(format!("{}: missing data section", path.display()));
            }
            let line = line.trim_end();
            if first {
                if line != MAGIC {
                    return Err(format!("{}: not a net file (expected '{}')", path.display(), MAGIC));
                }
                first = false;
                continue;
            }
            if line == "data" {
                break;
            }
            if let Some((key, value)) = line.split_once(' ') {
                fields.insert(key.to_string(), value.to_string());
            }
        }
        let field = |key: &str| fields.get(key).ok_or_else(|| format!("{}: missing '{}'", path.display(), key));
        let encoding = Encoding::from_name(field("encoding")?)
            .ok_or_else(|| format!("{}: unknown encoding '{}'", path.display(), fields["encoding"]))?;
        if field("activation")? != "relu sigmoid" {
            return Err(format!("{}: unsupported activation '{}'", path.display(), field("activation")?));
        }
        let sizes = field("layers")?
            .split_whitespace()
            .map(|s| s.parse::<usize>().map_err(|e| format!("{}: bad layers: {}", path.display(), e)))
            .collect::<Result<Vec<_>, _>>()?;
        let mut net = Self::with_sizes(encoding, sizes).map_err(|e| format!("{}: {}", path.display(), e))?;
        net.games_trained = field("games")?.parse().map_err(|e| format!("{}: bad games: {}", path.display(), e))?;

        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).map_err(|e| e.to_string())?;
        if bytes.len() != net.params.len() * 4 {
            return Err(format!("{}: expected {} parameters, found {} bytes", path.display(), net.params.len(), bytes.len()));
        }
        for (p, chunk) in net.params.iter_mut().zip(bytes.chunks_exact(4)) {
            *p = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        Ok(net)
    }
}

/// Plugs a `NeuralNet` into the analyzer.
pub struct NeuralNetEvaluator {
    pub net: NeuralNet,
}

impl<P: Position> Evaluator<P> for NeuralNetEvaluator {
    fn evaluate(&self, position: &P) -> f64 {
        self.net.equity(position)
    }

    fn name(&self) -> String {
        format!("neural net ({})", self.net.description())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::Player;
    use crate::simple_position::SimplePosition;

    fn sample_input(encoding: Encoding) -> [f32; MAX_INPUTS] {
        let mut p = SimplePosition::new();
        p.setup_starting_position();
        p.set_turn(Player::Me);
        let mut input = [0.0; MAX_INPUTS];
        encoding.encode(&p, &mut input);
        input
    }

    #[test]
    fn test_output_is_a_probability_and_gradient_matches_forward() {
        for encoding in Encoding::ALL {
            let net = NeuralNet::new_random(encoding, &DEFAULT_HIDDEN, 1);
            let input = sample_input(encoding);
            let mut grad = vec![0.0; net.param_count()];
            let y = net.forward(&input);
            assert!(y > 0.0 && y < 1.0);
            assert!((y - net.forward_with_gradient(&input, &mut grad)).abs() < 1e-6);
        }
    }

    /// The straightforward forward pass, to check the optimized one against.
    fn forward_reference(net: &NeuralNet, input: &[f32]) -> f32 {
        let mut activations = input[..net.sizes[0]].to_vec();
        let layers = net.layer_count();
        for layer in 0..layers {
            let (n_in, n_out) = (net.sizes[layer], net.sizes[layer + 1]);
            let start = net.offsets[layer];
            let out: Vec<f32> = (0..n_out)
                .map(|j| {
                    let sum = net.params[start + n_in * n_out + j]
                        + (0..n_in).map(|i| activations[i] * net.params[start + i * n_out + j]).sum::<f32>();
                    if layer + 1 < layers { sum.max(0.0) } else { sum }
                })
                .collect();
            activations = out;
        }
        sigmoid(activations[0])
    }

    #[test]
    fn test_forward_matches_reference() {
        // Default shape, and widths that aren't multiples of the chunk size.
        let shapes = [(Encoding::CountsPips, vec![128, 64]), (Encoding::CountsPips, vec![37, 70, 5]),
                      (Encoding::CountsPips, vec![200]), (Encoding::TdCounts, vec![128, 128]),
                      (Encoding::TdCounts, vec![37, 70])];
        for (encoding, hidden) in shapes {
            let net = NeuralNet::new_random(encoding, &hidden, 11);
            let mut p = SimplePosition::new();
            p.setup_starting_position();
            for (turn, extra) in [(Player::Me, 0), (Player::Opponent, 3)] {
                p.set_turn(turn);
                p.set_checkers(Player::Me, 25, extra).unwrap();
                let mut input = [0.0; MAX_INPUTS];
                net.encode(&p, &mut input);
                let (fast, slow) = (net.forward(&input), forward_reference(&net, &input));
                assert!((fast - slow).abs() < 1e-5, "{:?}: {} vs {}", hidden, fast, slow);
            }
        }
    }

    #[test]
    fn test_fused_td_update_matches_gradient_trace_and_update() {
        // A few steps of trace + weight updates, done the plain way and with td_update.
        let shapes = [(Encoding::CountsPips, vec![128, 128, 128]), (Encoding::CountsPips, vec![37, 70]),
                      (Encoding::TdCounts, vec![128, 128])];
        for (encoding, hidden) in shapes {
            let mut plain = NeuralNet::new_random(encoding, &hidden, 4);
            let mut fused = plain.clone();
            let n = plain.param_count();
            let (mut trace_plain, mut trace_fused, mut grad) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
            let mut acts = fused.new_activations();
            let mut input = sample_input(encoding);
            for step in 0..5 {
                input[(step * 7) % encoding.n_inputs()] += 0.2;
                let (sign, lambda, rate) = (if step % 2 == 0 { 1.0 } else { -1.0 }, 0.7, 0.05);
                let v_plain = plain.forward_with_gradient(&input, &mut grad);
                for (e, g) in trace_plain.iter_mut().zip(&grad) {
                    *e = lambda * *e + sign * g;
                }
                for (p, e) in plain.params.iter_mut().zip(&trace_plain) {
                    *p += rate * e;
                }
                let v_fused = fused.forward_keep(&input, &mut acts);
                fused.td_update(&mut acts, &mut trace_fused, sign, lambda, rate);
                assert!((v_plain - v_fused).abs() < 1e-5, "{:?} step {}: {} vs {}", hidden, step, v_plain, v_fused);
            }
            for k in 0..n {
                assert!((plain.params[k] - fused.params[k]).abs() < 1e-5, "{:?} param {}", hidden, k);
                assert!((trace_plain[k] - trace_fused[k]).abs() < 1e-5, "{:?} trace {}", hidden, k);
            }
        }
    }

    #[test]
    fn test_perturb_scales_noise_per_layer() {
        let original = NeuralNet::new_random(Encoding::CountsPips, &[128, 128, 128], 6);
        let mut net = original.clone();
        net.perturb(0.0, 0.0, 1);
        assert_eq!(net.params, original.params);
        net.perturb(0.0, 0.1, 1);
        for layer in 0..original.layer_count() {
            let (n_in, n_out) = (original.sizes[layer], original.sizes[layer + 1]);
            let range = original.offsets[layer]..original.offsets[layer] + n_in * n_out;
            let rms = |v: &mut dyn Iterator<Item = f32>| {
                let (sum, n) = v.fold((0.0f64, 0usize), |(s, n), x| (s + (x as f64).powi(2), n + 1));
                (sum / n as f64).sqrt()
            };
            let weight_rms = rms(&mut original.params[range.clone()].iter().copied());
            let noise_rms = rms(&mut range.clone().map(|k| net.params[k] - original.params[k]));
            let ratio = noise_rms / weight_rms;
            assert!((0.09..0.11).contains(&ratio), "layer {}: noise/weights {}", layer, ratio);
        }
        let mut shrunk = original.clone();
        shrunk.perturb(0.5, 0.0, 1);
        assert!((shrunk.params[10] - 0.5 * original.params[10]).abs() < 1e-7);
    }

    #[test]
    fn test_gradient_matches_finite_differences() {
        let mut net = NeuralNet::new_random(Encoding::TdCounts, &[8, 4], 3);
        let input = sample_input(Encoding::TdCounts);
        let mut grad = vec![0.0; net.param_count()];
        net.forward_with_gradient(&input, &mut grad);
        let eps = 1e-3;
        let mut checked = 0;
        for k in (0..net.param_count()).step_by(7) {
            let original = net.params[k];
            net.params[k] = original + eps;
            let up = net.forward(&input);
            net.params[k] = original - eps;
            let down = net.forward(&input);
            net.params[k] = original;
            let numeric = (up - down) / (2.0 * eps);
            assert!((numeric - grad[k]).abs() < 2e-3, "param {}: numeric {} vs analytic {}", k, numeric, grad[k]);
            checked += 1;
        }
        assert!(checked > 50);
    }

    #[test]
    fn test_save_and_load_round_trip() {
        for encoding in Encoding::ALL {
            let mut net = NeuralNet::new_random(encoding, &DEFAULT_HIDDEN, 5);
            net.games_trained = 1234;
            let path = std::env::temp_dir().join(format!("bgnet_test_{}_{}.bgnet", std::process::id(), encoding.name()));
            net.save(&path).unwrap();
            let loaded = NeuralNet::load(&path).unwrap();
            std::fs::remove_file(&path).ok();
            assert_eq!(loaded.encoding(), encoding);
            assert_eq!(loaded.sizes(), &[encoding.n_inputs(), 128, 128, 128, 1]);
            assert_eq!(loaded.games_trained, 1234);
            assert_eq!(loaded.forward(&sample_input(encoding)), net.forward(&sample_input(encoding)));
        }
    }

    #[test]
    fn test_extend_inputs_keeps_outputs() {
        let net = NeuralNet::new_random(Encoding::TdCounts, &[32, 16], 8);
        let wide = net.extend_inputs(Encoding::TdFlags).unwrap();
        assert_eq!(wide.sizes(), &[257, 32, 16, 1]);
        assert_eq!(wide.encoding(), Encoding::TdFlags);
        let mut p = SimplePosition::new();
        p.setup_starting_position();
        for turn in [Player::Me, Player::Opponent] {
            p.set_turn(turn);
            assert_eq!(net.win_probability(&p), wide.win_probability(&p));
        }
        assert!(net.extend_inputs(Encoding::CountsPips).is_err());
    }

    #[test]
    fn test_load_rejects_other_files() {
        let path = std::env::temp_dir().join(format!("bgnet_bad_{}.bgnet", std::process::id()));
        std::fs::write(&path, "hello\n").unwrap();
        assert!(NeuralNet::load(&path).is_err());
        std::fs::remove_file(&path).ok();
    }
}
