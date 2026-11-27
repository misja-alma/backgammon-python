//! An inference-only copy of a `NeuralNet` in lower precision, for evaluation in the search.
//!
//! Training keeps working on the f32 `NeuralNet`; `EvalNet::from_net` converts a trained net,
//! layer by layer, to one of these kinds:
//!
//! - `F32`: the same arithmetic as `NeuralNet::forward` (the reference).
//! - `F16`: weights, biases and running sums in IEEE half precision, 8 lanes per NEON register,
//!   so twice the multiply-adds per instruction and half the weight memory of f32.
//! - `F16Acc32`: weights stored as f16 (half the memory), widened to f32 for the sums.
//! - `I8`: weights as 8-bit integers (one scale per output unit), inputs as 8-bit integers (one
//!   scale per layer), sums in 32-bit integers with the `sdot` instruction: 16 multiply-adds per
//!   instruction, a quarter of the weight memory. The inputs of layer 1 are all multiples of 1/30
//!   (see `encoding`), so with input scale 1/30 they are exact for up to 11 checkers on a point;
//!   the scale of a hidden layer's inputs is calibrated on sample positions.
//!
//! The output layer (one unit, or 5 for a money net) always stays f32: a dot product per output.
//!
//! Files: the `NeuralNet` format with an extra header line `precision <kind per hidden layer>`,
//! the parameters of f16 layers as little-endian f16 and the others as f32 (`save_net`, `load`);
//! i8 layers are quantized on loading, with the input scales from an `input-scales` line.
//! `load` also reads plain `NeuralNet` files, as all-f32 nets with exactly `NeuralNet`'s results.
//! The f16 and i8 kernels need aarch64 (NEON with fp16 and dotprod, as on Apple silicon).
use std::cell::RefCell;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::encoding::{Encoding, MAX_INPUTS};
use crate::evaluator::{money_equity, terminal_equity, terminal_money_equity, Evaluator, Scoring, MONEY_OUTPUTS};
use crate::neural_net::{dot, sparse_layer, NeuralNet, Objective, MAX_WIDTH};
use crate::position::Position;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    F32,
    F16,
    F16Acc32,
    I8,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::F32 => "f32",
            Kind::F16 => "f16",
            Kind::F16Acc32 => "f16acc32",
            Kind::I8 => "i8",
        }
    }

    pub fn from_name(name: &str) -> Option<Kind> {
        [Kind::F32, Kind::F16, Kind::F16Acc32, Kind::I8].into_iter().find(|k| k.name() == name)
    }
}

/// Scale of the net's inputs for an `I8` first layer: every input is a multiple of 1/30.
pub const INPUT_SCALE: f32 = 1.0 / 30.0;

enum Layer {
    /// Weights `[input][output]`, as in `NeuralNet`.
    F32 { w: Vec<f32>, b: Vec<f32> },
    /// f16 bit patterns, weights `[input][output]`.
    F16 { w: Vec<u16>, b: Vec<u16>, acc32: bool },
    /// Weights `[input / 4][output][4]` (4 consecutive inputs of one output together, for
    /// `sdot`); output j = Σ q_in * q_w * scale[j] + b[j] with q_in = round(x / in_scale).
    I8 { w: Vec<i8>, scale: Vec<f32>, b: Vec<f32>, in_scale: f32 },
}

pub struct EvalNet {
    encoding: Encoding,
    sizes: Vec<usize>,
    kinds: Vec<Kind>,
    hidden: Vec<Layer>,
    /// Output layer, one row of `sizes[-2]` weights per output.
    out_w: Vec<f32>,
    out_b: Vec<f32>,
    /// Every hidden layer is `F16` (the fast path `forward_f16`).
    all_f16: bool,
    pub games_trained: u64,
    pub objective: Objective,
}

struct Scratch {
    current: Vec<f32>,
    next: Vec<f32>,
    active: Vec<(u32, f32)>,
    groups: Vec<(u32, u32)>,
    /// Active inputs as (index, f16 bits), for the all-f16 path: input and output of a layer.
    half_in: HalfActive,
    half_out: HalfActive,
}

/// Non-zero activations of a layer for the all-f16 path: indices and f16 bits, in separate arrays
/// so that 8 of each can be stored with one vector store (with 8 entries of slack at the end).
struct HalfActive {
    idx: Vec<u16>,
    val: Vec<u16>,
}

impl HalfActive {
    fn new() -> Self {
        HalfActive { idx: vec![0; MAX_WIDTH + 8], val: vec![0; MAX_WIDTH + 8] }
    }
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch {
        current: vec![0.0; MAX_WIDTH],
        next: vec![0.0; MAX_WIDTH],
        active: vec![(0, 0.0); MAX_WIDTH],
        groups: vec![(0, 0); MAX_WIDTH / 4],
        half_in: HalfActive::new(),
        half_out: HalfActive::new(),
    });
}

/// f32 to f16 bits, rounding to nearest even (finite values; overflow gives infinity).
pub fn f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let abs = f32::from_bits(bits & 0x7fff_ffff);
    if abs >= 65520.0 {
        return sign | 0x7c00;
    }
    if abs < 6.103_515_6e-5 {
        // Subnormal f16: multiples of 2^-24.
        return sign | (abs * 16_777_216.0).round_ties_even() as u16;
    }
    let b = abs.to_bits();
    let exp = ((b >> 23) as i32) - 127 + 15;
    let mant = b & 0x7f_ffff;
    let mut h = ((exp as u32) << 10) | (mant >> 13);
    let rest = mant & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && (h & 1) == 1) {
        h += 1; // may carry into the exponent, which is still correct
    }
    sign | h as u16
}

pub fn f16_value(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let mant = (h & 0x3ff) as f32;
    sign * if exp == 0 {
        mant * 2f32.powi(-24)
    } else if exp == 31 {
        f32::INFINITY
    } else {
        (1.0 + mant / 1024.0) * 2f32.powi(exp - 15)
    }
}

/// Weights of a layer quantized to `bits`-bit integers with one scale per output unit
/// (symmetric, max |w| maps to the largest integer). Returns (integers, scales), `[input][output]`.
pub fn quantize_columns(w: &[f32], n_in: usize, n_out: usize, bits: u32) -> (Vec<i32>, Vec<f32>) {
    let top = ((1 << (bits - 1)) - 1) as f32;
    let scales: Vec<f32> = (0..n_out)
        .map(|j| {
            let m = (0..n_in).map(|i| w[i * n_out + j].abs()).fold(0.0f32, f32::max);
            if m > 0.0 { m / top } else { 1.0 }
        })
        .collect();
    let q = (0..n_in * n_out).map(|k| (w[k] / scales[k % n_out]).round().clamp(-top, top) as i32).collect();
    (q, scales)
}

/// A copy of `net` with every weight (not the biases) rounded to `bits`-bit integers with one
/// scale per output unit, still computed in f32: shows what weight precision alone costs.
pub fn fake_quantized(net: &NeuralNet, bits: u32) -> NeuralNet {
    let mut copy = net.clone();
    let sizes = net.sizes().to_vec();
    for layer in 0..sizes.len() - 1 {
        let (n_in, n_out) = (sizes[layer], sizes[layer + 1]);
        let start = net.layer_offset(layer);
        let w = &net.params()[start..start + n_in * n_out];
        let (q, scales) = quantize_columns(w, n_in, n_out, bits);
        for (k, p) in copy.params_mut()[start..start + n_in * n_out].iter_mut().enumerate() {
            *p = q[k] as f32 * scales[k % n_out];
        }
    }
    copy
}

impl EvalNet {
    /// Converts `net`: `kinds[l]` for hidden layer l. `input_scales[l]` is the input scale of an
    /// `I8` hidden layer l > 0 (see `calibrate`); layer 0 uses `INPUT_SCALE`.
    pub fn from_net(net: &NeuralNet, kinds: &[Kind], input_scales: &[f32]) -> Result<EvalNet, String> {
        let sizes = net.sizes().to_vec();
        let n_hidden = sizes.len() - 2;
        if kinds.len() != n_hidden {
            return Err(format!("{} hidden layers but {} kinds", n_hidden, kinds.len()));
        }
        let params = net.params();
        let mut hidden = Vec::new();
        for (l, &kind) in kinds.iter().enumerate() {
            let (n_in, n_out) = (sizes[l], sizes[l + 1]);
            let start = net.layer_offset(l);
            let w = &params[start..start + n_in * n_out];
            let b = &params[start + n_in * n_out..start + n_in * n_out + n_out];
            if kind != Kind::F32 && !cfg!(target_arch = "aarch64") {
                return Err(format!("{} layers need aarch64", kind.name()));
            }
            hidden.push(match kind {
                Kind::F32 => Layer::F32 { w: w.to_vec(), b: b.to_vec() },
                Kind::F16 | Kind::F16Acc32 => {
                    if n_out % 8 != 0 {
                        return Err(format!("f16 layers need a multiple of 8 outputs, layer {} has {}", l + 1, n_out));
                    }
                    Layer::F16 {
                        w: w.iter().map(|&x| f16_bits(x)).collect(),
                        b: b.iter().map(|&x| f16_bits(x)).collect(),
                        acc32: kind == Kind::F16Acc32,
                    }
                }
                Kind::I8 => {
                    if n_out % 4 != 0 {
                        return Err(format!("i8 layers need a multiple of 4 outputs, layer {} has {}", l + 1, n_out));
                    }
                    let in_scale = if l == 0 { INPUT_SCALE } else { input_scales[l] };
                    let (q, scales) = quantize_columns(w, n_in, n_out, 8);
                    let groups = n_in.div_ceil(4);
                    let mut packed = vec![0i8; groups * n_out * 4];
                    for i in 0..n_in {
                        for j in 0..n_out {
                            packed[((i / 4) * n_out + j) * 4 + i % 4] = q[i * n_out + j] as i8;
                        }
                    }
                    Layer::I8 { w: packed, scale: scales.iter().map(|s| s * in_scale).collect(), b: b.to_vec(), in_scale }
                }
            });
        }
        let start = net.layer_offset(n_hidden);
        let (n_in, k_out) = (sizes[n_hidden], sizes[n_hidden + 1]);
        let out_w = (0..k_out).flat_map(|j| (0..n_in).map(move |i| params[start + i * k_out + j])).collect();
        Ok(EvalNet {
            encoding: net.encoding(),
            sizes,
            kinds: kinds.to_vec(),
            hidden,
            out_w,
            out_b: params[start + n_in * k_out..start + n_in * k_out + k_out].to_vec(),
            all_f16: cfg!(target_arch = "aarch64") && kinds.iter().all(|&k| k == Kind::F16),
            games_trained: net.games_trained,
            objective: net.objective,
        })
    }

    /// Input scales for `I8` hidden layers after the first: the `quantile` (e.g. 0.9999) of the
    /// non-zero inputs of each layer over `inputs`, mapped to 127. Entry 0 is `INPUT_SCALE`.
    pub fn calibrate(net: &NeuralNet, inputs: &[[f32; MAX_INPUTS]], quantile: f64) -> Vec<f32> {
        let sizes = net.sizes().to_vec();
        let n_hidden = sizes.len() - 2;
        let mut values: Vec<Vec<f32>> = vec![Vec::new(); n_hidden];
        let params = net.params();
        for x in inputs {
            let mut current = x[..sizes[0]].to_vec();
            for l in 0..n_hidden {
                let (n_in, n_out) = (sizes[l], sizes[l + 1]);
                let start = net.layer_offset(l);
                let active: Vec<(u32, f32)> =
                    current.iter().enumerate().filter(|(_, &v)| v != 0.0).map(|(i, &v)| (i as u32, v)).collect();
                let mut out = vec![0.0; n_out];
                sparse_layer(&params[start..start + n_in * n_out],
                             &params[start + n_in * n_out..start + n_in * n_out + n_out], &active, &mut out);
                out.iter_mut().for_each(|o| *o = o.max(0.0));
                if l + 1 < n_hidden {
                    values[l + 1].extend(out.iter().filter(|&&v| v > 0.0));
                }
                current = out;
            }
        }
        let mut scales = vec![INPUT_SCALE; n_hidden];
        for l in 1..n_hidden {
            let v = &mut values[l];
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let top = if v.is_empty() { 1.0 } else { v[((v.len() - 1) as f64 * quantile) as usize] };
            scales[l] = top / 127.0;
        }
        scales
    }

    pub fn encoding(&self) -> Encoding {
        self.encoding
    }

    pub fn description(&self) -> String {
        let shape: Vec<String> = self.sizes.iter().map(|s| s.to_string()).collect();
        let kinds: Vec<&str> = self.kinds.iter().map(|k| k.name()).collect();
        format!("{} {} {} [{}], {} games", shape.join("-"), self.encoding.name(), self.objective.name(), kinds.join(" "),
                self.games_trained)
    }

    /// Winning chance of the player on roll, for inputs from `Encoding::encode`.
    pub fn forward(&self, input: &[f32]) -> f32 {
        let mut out = [0.0];
        self.forward_into(input, &mut out);
        out[0]
    }

    /// All outputs (`objective.outputs()` of them) into `out`.
    pub fn forward_all(&self, input: &[f32], out: &mut [f32]) {
        let k = self.objective.outputs();
        self.forward_into(input, &mut out[..k]);
    }

    /// The first `out.len()` outputs.
    fn forward_into(&self, input: &[f32], out: &mut [f32]) {
        #[cfg(target_arch = "aarch64")]
        if self.all_f16 {
            return unsafe { self.forward_f16(input, out) };
        }
        SCRATCH.with(|scratch| {
            let Scratch { current, next, active, groups, .. } = &mut *scratch.borrow_mut();
            let n = self.sizes[0];
            current[..n].copy_from_slice(&input[..n]);
            for (l, layer) in self.hidden.iter().enumerate() {
                let (n_in, n_out) = (self.sizes[l], self.sizes[l + 1]);
                let out = &mut next[..n_out];
                match layer {
                    Layer::F32 { w, b } => {
                        let count = collect(&current[..n_in], active);
                        sparse_layer(w, b, &active[..count], out);
                        out.iter_mut().for_each(|o| *o = o.max(0.0));
                    }
                    #[cfg(target_arch = "aarch64")]
                    Layer::F16 { w, b, acc32 } => {
                        let count = collect(&current[..n_in], active);
                        unsafe {
                            if *acc32 {
                                f16_acc32_layer(w, b, &active[..count], out)
                            } else if n_out % 128 == 0 {
                                f16_layer::<16>(w, b, &active[..count], out)
                            } else if n_out % 64 == 0 {
                                f16_layer::<8>(w, b, &active[..count], out)
                            } else {
                                f16_layer::<1>(w, b, &active[..count], out)
                            }
                        }
                    }
                    #[cfg(target_arch = "aarch64")]
                    Layer::I8 { w, scale, b, in_scale } => {
                        let count = collect_groups(&current[..n_in], 1.0 / in_scale, groups);
                        unsafe {
                            if n_out % 64 == 0 {
                                i8_layer::<16>(w, scale, b, &groups[..count], out)
                            } else {
                                i8_layer::<1>(w, scale, b, &groups[..count], out)
                            }
                        }
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    _ => unreachable!("only f32 layers off aarch64"),
                }
                std::mem::swap(current, next);
            }
            let n_in = self.sizes[self.sizes.len() - 2];
            for (j, y) in out.iter_mut().enumerate() {
                *y = sigmoid(dot(&current[..n_in], &self.out_w[j * n_in..(j + 1) * n_in], self.out_b[j]));
            }
        })
    }

    /// `forward` when every hidden layer is `F16`: activations stay f16, and each layer hands the
    /// next only its non-zero outputs, found with vector compares instead of a scan.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "fp16")]
    unsafe fn forward_f16(&self, input: &[f32], out: &mut [f32]) {
        SCRATCH.with(|scratch| {
            let Scratch { half_in, half_out, .. } = &mut *scratch.borrow_mut();
            let mut count = collect_f16(&input[..self.sizes[0]], half_in);
            for (l, layer) in self.hidden.iter().enumerate() {
                let Layer::F16 { w, b, .. } = layer else { unreachable!() };
                let n_out = self.sizes[l + 1];
                count = if n_out % 128 == 0 {
                    f16_layer_sparse::<16>(w, b, n_out, half_in, count, half_out)
                } else if n_out % 64 == 0 {
                    f16_layer_sparse::<8>(w, b, n_out, half_in, count, half_out)
                } else {
                    f16_layer_sparse::<1>(w, b, n_out, half_in, count, half_out)
                };
                std::mem::swap(half_in, half_out);
            }
            let n_in = self.sizes[self.sizes.len() - 2];
            for (j, y) in out.iter_mut().enumerate() {
                let w = &self.out_w[j * n_in..(j + 1) * n_in];
                let mut sum = self.out_b[j];
                for k in 0..count {
                    sum += w[half_in.idx[k] as usize] * f16_to_f32(half_in.val[k]);
                }
                *y = sigmoid(sum);
            }
        })
    }

    /// Writes the evaluation copy of `net` with these kinds and input scales (see `from_net`):
    /// f16 layers as f16, f32 and i8 layers as f32 (an i8 layer is quantized again on loading,
    /// exactly as `from_net` does, with the scales stored in the header).
    pub fn save_net(net: &NeuralNet, kinds: &[Kind], input_scales: &[f32], path: &Path) -> Result<(), String> {
        let sizes = net.sizes();
        if kinds.len() + 2 != sizes.len() {
            return Err(format!("{} kinds for {} hidden layers", kinds.len(), sizes.len() - 2));
        }
        let i8 = kinds.contains(&Kind::I8);
        if i8 && input_scales.len() != kinds.len() {
            return Err("i8 layers need an input scale per hidden layer".into());
        }
        let tmp = path.with_extension("tmp");
        let write = || -> std::io::Result<()> {
            let mut f = BufWriter::new(File::create(&tmp)?);
            let shape: Vec<String> = sizes.iter().map(|s| s.to_string()).collect();
            let names: Vec<&str> = kinds.iter().map(|k| k.name()).collect();
            write!(f, "BGNET 1\nencoding {}\nlayers {}\nactivation relu sigmoid\ngames {}\nobjective {}\nprecision {}\n",
                   net.encoding().name(), shape.join(" "), net.games_trained, net.objective.name(), names.join(" "))?;
            if i8 {
                let scales: Vec<String> = input_scales.iter().map(|s| format!("{:e}", s)).collect();
                writeln!(f, "input-scales {}", scales.join(" "))?;
            }
            writeln!(f, "data")?;
            for layer in 0..sizes.len() - 1 {
                let start = net.layer_offset(layer);
                let end = start + sizes[layer] * sizes[layer + 1] + sizes[layer + 1];
                let half = layer < kinds.len() && matches!(kinds[layer], Kind::F16 | Kind::F16Acc32);
                for &x in &net.params()[start..end] {
                    if half {
                        f.write_all(&f16_bits(x).to_le_bytes())?;
                    } else {
                        f.write_all(&x.to_le_bytes())?;
                    }
                }
            }
            f.flush()
        };
        write().map_err(|e| format!("{}: {}", tmp.display(), e))?;
        std::fs::rename(&tmp, path).map_err(|e| format!("{}: {}", path.display(), e))
    }

    /// Reads a net written by `save_net`, or a plain `NeuralNet` file (as an all-f32 net).
    pub fn load(path: &Path) -> Result<EvalNet, String> {
        let err = |e: String| format!("{}: {}", path.display(), e);
        let mut reader = BufReader::new(File::open(path).map_err(|e| err(e.to_string()))?);
        let mut fields = std::collections::HashMap::new();
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line).map_err(|e| err(e.to_string()))? == 0 {
                return Err(err("missing data section".into()));
            }
            let line = line.trim_end();
            if line == "data" {
                break;
            }
            if let Some((key, value)) = line.split_once(' ') {
                fields.insert(key.to_string(), value.to_string());
            }
        }
        let Some(precision) = fields.get("precision") else {
            let net = NeuralNet::load(path)?;
            let kinds = vec![Kind::F32; net.sizes().len() - 2];
            return EvalNet::from_net(&net, &kinds, &[]);
        };
        let kinds = precision.split_whitespace()
            .map(|k| Kind::from_name(k).ok_or_else(|| err(format!("unknown precision '{}'", k))))
            .collect::<Result<Vec<_>, _>>()?;
        // Rebuild an f32 net with the stored values (f16 ones widened) and convert it back: f16
        // values survive the round trip exactly.
        let encoding = Encoding::from_name(fields.get("encoding").ok_or_else(|| err("missing encoding".into()))?)
            .ok_or_else(|| err("unknown encoding".into()))?;
        let sizes = fields.get("layers").ok_or_else(|| err("missing layers".into()))?
            .split_whitespace().map(|s| s.parse::<usize>().map_err(|e| err(e.to_string())))
            .collect::<Result<Vec<_>, _>>()?;
        if kinds.len() + 2 != sizes.len() {
            return Err(err(format!("{} precisions for {} hidden layers", kinds.len(), sizes.len().saturating_sub(2))));
        }
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).map_err(|e| err(e.to_string()))?;
        let objective = Objective::from_fields(&fields).map_err(err)?;
        let mut net = NeuralNet::new_random_for(encoding, &sizes[1..sizes.len() - 1], objective, 0);
        if net.sizes() != sizes.as_slice() {
            return Err(err(format!("bad layer sizes {:?}", sizes)));
        }
        let mut pos = 0;
        let mut values = Vec::with_capacity(net.param_count());
        for layer in 0..sizes.len() - 1 {
            let count = sizes[layer] * sizes[layer + 1] + sizes[layer + 1];
            let half = layer < kinds.len() && matches!(kinds[layer], Kind::F16 | Kind::F16Acc32);
            let width = if half { 2 } else { 4 };
            let chunk = bytes.get(pos..pos + count * width).ok_or_else(|| err("data too short".into()))?;
            if half {
                values.extend(chunk.chunks_exact(2).map(|c| f16_value(u16::from_le_bytes([c[0], c[1]]))));
            } else {
                values.extend(chunk.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])));
            }
            pos += count * width;
        }
        if pos != bytes.len() {
            return Err(err("data too long".into()));
        }
        net.params_mut().copy_from_slice(&values);
        let scales = match fields.get("input-scales") {
            Some(v) => v.split_whitespace().map(|s| s.parse::<f32>().map_err(|e| err(e.to_string())))
                .collect::<Result<Vec<_>, _>>()?,
            None if kinds.contains(&Kind::I8) => return Err(err("i8 layers need input-scales".into())),
            None => Vec::new(),
        };
        let mut eval = EvalNet::from_net(&net, &kinds, &scales)?;
        eval.games_trained = fields.get("games").and_then(|g| g.parse().ok()).unwrap_or(0);
        Ok(eval)
    }

    pub fn win_probability<P: Position>(&self, position: &P) -> f32 {
        let mut input = [0.0; MAX_INPUTS];
        self.encoding.encode(position, &mut input);
        self.forward(&input)
    }

    /// The 5 outcome probabilities of a money net for the player on roll (the game must not be over).
    pub fn outcome<P: Position>(&self, position: &P) -> [f32; MONEY_OUTPUTS] {
        let mut input = [0.0; MAX_INPUTS];
        self.encoding.encode(position, &mut input);
        let mut out = [0.0; MONEY_OUTPUTS];
        self.forward_all(&input, &mut out);
        out
    }

    /// Equity for the side to move, for the net's objective (see `NeuralNet::equity`).
    pub fn equity<P: Position>(&self, position: &P) -> f64 {
        match self.objective {
            Objective::Dmp => terminal_equity(position)
                .unwrap_or_else(|| 2.0 * self.win_probability(position) as f64 - 1.0),
            Objective::Money => terminal_money_equity(position)
                .unwrap_or_else(|| money_equity(&self.outcome(position))),
        }
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// The non-zero values of `values` with their indices into `active`; returns how many.
#[inline]
fn collect(values: &[f32], active: &mut [(u32, f32)]) -> usize {
    let mut count = 0;
    for (i, &x) in values.iter().enumerate() {
        active[count] = (i as u32, x);
        count += (x != 0.0) as usize;
    }
    count
}

/// `values` quantized to 0..=127 (values * inv_scale, rounded, clamped), packed 4 per u32; the
/// groups of 4 that aren't all zero go into `groups` as (group index, packed). Returns how many.
#[inline]
fn collect_groups(values: &[f32], inv_scale: f32, groups: &mut [(u32, u32)]) -> usize {
    let mut count = 0;
    for (g, chunk) in values.chunks(4).enumerate() {
        let mut packed = 0u32;
        for (k, &x) in chunk.iter().enumerate() {
            let q = (x * inv_scale).round().clamp(-127.0, 127.0) as i8;
            packed |= (q as u8 as u32) << (8 * k);
        }
        groups[count] = (g as u32, packed);
        count += (packed != 0) as usize;
    }
    count
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn load_f16(p: *const u16) -> float16x8_t {
    vreinterpretq_f16_u16(vld1q_u16(p))
}

/// f16 version of `sparse_layer` plus ReLU: sums in f16, C registers of 8 outputs at a time.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "fp16")]
unsafe fn f16_layer<const C: usize>(w: &[u16], b: &[u16], active: &[(u32, f32)], out: &mut [f32]) {
    let n_out = out.len();
    let zero = vreinterpretq_f16_u16(vdupq_n_u16(0));
    let mut start = 0;
    while start < n_out {
        let mut acc = [zero; C];
        for c in 0..C {
            acc[c] = load_f16(b.as_ptr().add(start + 8 * c));
        }
        for &(i, x) in active {
            let half = vcvt_f16_f32(vdupq_n_f32(x));
            let xv = vcombine_f16(half, half);
            let row = w.as_ptr().add(i as usize * n_out + start);
            for c in 0..C {
                acc[c] = vfmaq_f16(acc[c], load_f16(row.add(8 * c)), xv);
            }
        }
        for c in 0..C {
            let r = vmaxq_f16(acc[c], zero);
            let o = out.as_mut_ptr().add(start + 8 * c);
            vst1q_f32(o, vcvt_f32_f16(vget_low_f16(r)));
            vst1q_f32(o.add(4), vcvt_high_f32_f16(r));
        }
        start += 8 * C;
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn f16_to_f32(h: u16) -> f32 {
    vgetq_lane_f32::<0>(vcvt_f32_f16(vreinterpret_f16_u16(vdup_n_u16(h))))
}

/// For an 8-bit lane mask: the `tbl` byte indices that move the selected u16 lanes to the front.
const COMPACT: [[u8; 16]; 256] = {
    let mut table = [[0u8; 16]; 256];
    let mut m = 0;
    while m < 256 {
        let mut k = 0;
        let mut lane = 0;
        while lane < 8 {
            if m & (1 << lane) != 0 {
                table[m][2 * k] = 2 * lane as u8;
                table[m][2 * k + 1] = 2 * lane as u8 + 1;
                k += 1;
            }
            lane += 1;
        }
        m += 1;
    }
    table
};

/// Appends the lanes of `h` where `mask` is set, as indices base + lane and f16 bits, at
/// position `count` of `out`, without branches: shuffle them to the front with a table lookup,
/// store all 8 lanes, and advance by the number selected. Returns the new count.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn push_nonzero(base: usize, h: float16x8_t, mask: uint16x8_t, out: &mut HalfActive, count: usize) -> usize {
    const BITS: [u16; 8] = [1, 2, 4, 8, 16, 32, 64, 128];
    const LANES: [u16; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
    let m = vaddvq_u16(vandq_u16(mask, vld1q_u16(BITS.as_ptr()))) as usize;
    let shuffle = vld1q_u8(COMPACT.get_unchecked(m).as_ptr());
    let values = vqtbl1q_u8(vreinterpretq_u8_f16(h), shuffle);
    let indices = vqtbl1q_u8(vreinterpretq_u8_u16(vaddq_u16(vdupq_n_u16(base as u16), vld1q_u16(LANES.as_ptr()))), shuffle);
    vst1q_u16(out.val.as_mut_ptr().add(count), vreinterpretq_u16_u8(values));
    vst1q_u16(out.idx.as_mut_ptr().add(count), vreinterpretq_u16_u8(indices));
    count + m.count_ones() as usize
}

/// The non-zero values of `values` as indices and f16 bits in `out`, 8 at a time. Returns how many.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "fp16")]
unsafe fn collect_f16(values: &[f32], out: &mut HalfActive) -> usize {
    let mut count = 0;
    let n = values.len();
    let mut start = 0;
    while start + 8 <= n {
        let p = values.as_ptr().add(start);
        let h = vcombine_f16(vcvt_f16_f32(vld1q_f32(p)), vcvt_f16_f32(vld1q_f32(p.add(4))));
        count = push_nonzero(start, h, vmvnq_u16(vceqzq_f16(h)), out, count);
        start += 8;
    }
    for i in start..n {
        if values[i] != 0.0 {
            out.idx[count] = i as u16;
            out.val[count] = f16_bits(values[i]);
            count += 1;
        }
    }
    count
}

/// `f16_layer` with f16 inputs (the first `count` of `active`), writing the non-zero outputs
/// (after ReLU) into `out`; returns how many.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "fp16")]
unsafe fn f16_layer_sparse<const C: usize>(w: &[u16], b: &[u16], n_out: usize, active: &HalfActive, count: usize,
                                           out: &mut HalfActive) -> usize {
    let zero = vreinterpretq_f16_u16(vdupq_n_u16(0));
    let mut n = 0;
    let mut start = 0;
    while start < n_out {
        let mut acc = [zero; C];
        for c in 0..C {
            acc[c] = load_f16(b.as_ptr().add(start + 8 * c));
        }
        for k in 0..count {
            let xv = vreinterpretq_f16_u16(vdupq_n_u16(*active.val.get_unchecked(k)));
            let row = w.as_ptr().add(*active.idx.get_unchecked(k) as usize * n_out + start);
            for c in 0..C {
                acc[c] = vfmaq_f16(acc[c], load_f16(row.add(8 * c)), xv);
            }
        }
        for c in 0..C {
            let r = vmaxq_f16(acc[c], zero);
            n = push_nonzero(start + 8 * c, r, vcgtq_f16(r, zero), out, n);
        }
        start += 8 * C;
    }
    n
}

/// f16 weights widened to f32 for the sums, 64 outputs (16 f32 registers) at a time.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "fp16")]
unsafe fn f16_acc32_layer(w: &[u16], b: &[u16], active: &[(u32, f32)], out: &mut [f32]) {
    const C: usize = 8; // registers of 8 f16 weights = 2 f32 accumulators each
    let n_out = out.len();
    let mut start = 0;
    while start < n_out {
        let width = (8 * C).min(n_out - start) / 8;
        let mut acc = [vdupq_n_f32(0.0); 2 * C];
        for c in 0..width {
            let h = load_f16(b.as_ptr().add(start + 8 * c));
            acc[2 * c] = vcvt_f32_f16(vget_low_f16(h));
            acc[2 * c + 1] = vcvt_high_f32_f16(h);
        }
        for &(i, x) in active {
            let xv = vdupq_n_f32(x);
            let row = w.as_ptr().add(i as usize * n_out + start);
            for c in 0..width {
                let h = load_f16(row.add(8 * c));
                acc[2 * c] = vfmaq_f32(acc[2 * c], vcvt_f32_f16(vget_low_f16(h)), xv);
                acc[2 * c + 1] = vfmaq_f32(acc[2 * c + 1], vcvt_high_f32_f16(h), xv);
            }
        }
        for c in 0..2 * width {
            vst1q_f32(out.as_mut_ptr().add(start + 4 * c), vmaxq_f32(acc[c], vdupq_n_f32(0.0)));
        }
        start += 8 * width;
    }
}

/// `acc += Σ a[k] * b[k]` per 32-bit lane, over the 4 bytes of the lane (the `sdot` instruction;
/// its intrinsic isn't stable in Rust yet).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn sdot(mut acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    std::arch::asm!(".arch_extension dotprod", "sdot {0:v}.4s, {1:v}.16b, {2:v}.16b",
                    inout(vreg) acc, in(vreg) a, in(vreg) b, options(pure, nomem, nostack));
    acc
}

/// Integer layer plus ReLU: for each non-zero group of 4 inputs, `sdot` with 4 outputs per
/// register, C registers (4C outputs) at a time.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn i8_layer<const C: usize>(w: &[i8], scale: &[f32], b: &[f32], groups: &[(u32, u32)], out: &mut [f32]) {
    let n_out = out.len();
    let mut start = 0;
    while start < n_out {
        let mut acc = [vdupq_n_s32(0); C];
        for &(g, packed) in groups {
            let a = vreinterpretq_s8_u32(vdupq_n_u32(packed));
            let row = w.as_ptr().add((g as usize * n_out + start) * 4);
            for c in 0..C {
                acc[c] = sdot(acc[c], vld1q_s8(row.add(16 * c)), a);
            }
        }
        for c in 0..C {
            let k = start + 4 * c;
            let v = vfmaq_f32(vld1q_f32(b.as_ptr().add(k)), vcvtq_f32_s32(acc[c]), vld1q_f32(scale.as_ptr().add(k)));
            vst1q_f32(out.as_mut_ptr().add(k), vmaxq_f32(v, vdupq_n_f32(0.0)));
        }
        start += 4 * C;
    }
}

/// Plugs an `EvalNet` into the analyzer.
pub struct EvalNetEvaluator {
    pub net: EvalNet,
}

impl<P: Position> Evaluator<P> for EvalNetEvaluator {
    fn evaluate(&self, position: &P) -> f64 {
        self.net.equity(position)
    }

    /// DMP: the winning chance only. Money: a money net's money equity; a DMP net's 2·win - 1
    /// (it knows no gammons), with the money value of finished games.
    fn evaluate_scored(&self, position: &P, scoring: Scoring) -> f64 {
        let dmp = || terminal_equity(position).unwrap_or_else(|| 2.0 * self.net.win_probability(position) as f64 - 1.0);
        match (scoring, self.net.objective) {
            (Scoring::Dmp, _) => dmp(),
            (Scoring::Money, Objective::Money) => self.net.equity(position),
            (Scoring::Money, Objective::Dmp) => terminal_money_equity(position).unwrap_or_else(dmp),
        }
    }

    fn outcome(&self, position: &P) -> [f64; MONEY_OUTPUTS] {
        if let Some(o) = crate::evaluator::terminal_outcome(position) {
            return o.map(|x| x as f64);
        }
        match self.net.objective {
            Objective::Money => self.net.outcome(position).map(|x| x as f64),
            Objective::Dmp => [self.net.win_probability(position) as f64, f64::NAN, f64::NAN, f64::NAN, f64::NAN],
        }
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

    fn inputs(net: &NeuralNet) -> Vec<[f32; MAX_INPUTS]> {
        let mut p = SimplePosition::new();
        p.setup_starting_position();
        p.set_turn(Player::Me);
        let mut x = [0.0; MAX_INPUTS];
        net.encode(&p, &mut x);
        let mut q = p.clone();
        q.set_checkers(Player::Me, 24, 0).unwrap();
        q.set_checkers(Player::Me, 25, 2).unwrap();
        let mut y = [0.0; MAX_INPUTS];
        net.encode(&q, &mut y);
        vec![x, y]
    }

    #[test]
    fn test_money_net_copies_match() {
        let net = NeuralNet::new_random_for(Encoding::TdNack, &[64, 32], Objective::Money, 5);
        let mut kinds = vec![vec![Kind::F32, Kind::F32]];
        if cfg!(target_arch = "aarch64") {
            kinds.push(vec![Kind::F16, Kind::F16]);
        }
        for kinds in kinds {
            let eval = EvalNet::from_net(&net, &kinds, &[]).unwrap();
            for x in &inputs(&net) {
                let (mut a, mut b) = ([0.0; MONEY_OUTPUTS], [0.0; MONEY_OUTPUTS]);
                net.forward_all(x, &mut a);
                eval.forward_all(x, &mut b);
                for k in 0..MONEY_OUTPUTS {
                    assert!((a[k] - b[k]).abs() < 0.01, "{:?} output {}: {} vs {}", kinds, k, a[k], b[k]);
                }
                assert_eq!(eval.forward(x), b[0]);
            }
        }
        let path = std::env::temp_dir().join(format!("money_eval_{}.bgnet", std::process::id()));
        EvalNet::save_net(&net, &[Kind::F32, Kind::F32], &[], &path).unwrap();
        let loaded = EvalNet::load(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(loaded.objective, Objective::Money);
    }

    #[test]
    fn test_f16_conversion() {
        for x in [0.0f32, 1.0, -2.5, 0.1, 1.0 / 15.0, 1e-6, 65504.0, -0.000_061_035_156] {
            let back = f16_value(f16_bits(x));
            assert!((back - x).abs() <= x.abs() / 1024.0 + 6e-8, "{} -> {}", x, back);
        }
        assert_eq!(f16_bits(1.0), 0x3c00);
        assert_eq!(f16_bits(-2.0), 0xc000);
        assert_eq!(f16_bits(1.0 + 1.0 / 2048.0), 0x3c00); // tie rounds to even
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_save_and_load_round_trip() {
        let mut net = NeuralNet::new_random(Encoding::TdNack, &[64, 32], 5);
        net.games_trained = 1234;
        let dir = std::env::temp_dir();
        let scales = [INPUT_SCALE, 0.05];
        for kinds in [[Kind::F16, Kind::F16], [Kind::F32, Kind::F16Acc32], [Kind::I8, Kind::I8], [Kind::F16, Kind::I8]] {
            let eval = EvalNet::from_net(&net, &kinds, &scales).unwrap();
            let path = dir.join(format!("eval_net_test_{}.bgnet", std::process::id()));
            EvalNet::save_net(&net, &kinds, &scales, &path).unwrap();
            let loaded = EvalNet::load(&path).unwrap();
            std::fs::remove_file(&path).unwrap();
            assert_eq!(loaded.description(), eval.description());
            for x in inputs(&net) {
                assert_eq!(loaded.forward(&x), eval.forward(&x));
            }
            assert!(NeuralNet::load(&path).is_err());
        }
        // A plain NeuralNet file loads as all-f32, with the same results.
        let path = dir.join(format!("eval_net_plain_{}.bgnet", std::process::id()));
        net.save(&path).unwrap();
        let loaded = EvalNet::load(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        for x in inputs(&net) {
            assert_eq!(loaded.forward(&x), net.forward(&x));
        }
    }

    #[test]
    fn test_f32_kind_matches_neural_net() {
        let net = NeuralNet::new_random(Encoding::TdNack, &[64, 32], 7);
        let eval = EvalNet::from_net(&net, &[Kind::F32, Kind::F32], &[]).unwrap();
        for x in inputs(&net) {
            assert_eq!(eval.forward(&x), net.forward(&x));
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_low_precision_kinds_are_close() {
        let net = NeuralNet::new_random(Encoding::TdNack, &[128, 64], 3);
        let xs = inputs(&net);
        let scales = EvalNet::calibrate(&net, &xs, 1.0);
        for kinds in [[Kind::F16, Kind::F16], [Kind::F16Acc32, Kind::F16Acc32], [Kind::I8, Kind::I8], [Kind::F16, Kind::I8]] {
            let eval = EvalNet::from_net(&net, &kinds, &scales).unwrap();
            for x in &xs {
                let (a, b) = (net.forward(x), eval.forward(x));
                assert!((a - b).abs() < 0.01, "{:?}: {} vs {}", kinds, a, b);
            }
        }
    }

    /// Widths that are not a multiple of 128 take the 64-wide (C=8) or 8-wide kernels.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_f16_matches_f32_for_other_widths() {
        for sizes in [[576, 192], [192, 64], [256, 256], [64, 576], [136, 72]] {
            let net = NeuralNet::new_random(Encoding::TdNack, &sizes, 11);
            let eval = EvalNet::from_net(&net, &[Kind::F16, Kind::F16], &[]).unwrap();
            for x in &inputs(&net) {
                let (a, b) = (net.forward(x), eval.forward(x));
                assert!((a - b).abs() < 0.01, "{:?}: {} vs {}", sizes, a, b);
            }
        }
    }
}
