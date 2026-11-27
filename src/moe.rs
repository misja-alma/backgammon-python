//! Mixture of experts: one net per game type, a generic net for every other position, and a
//! gateway net that judges which experts suit a position.
//!
//! **Game types** (`GameTypes`, file nets/moe/types.txt from analysis/export_types.py): groups of
//! layer-2 neurons of a fixed reference net that fire together for one kind of position (a blitz,
//! a holding game, a back game, ...). A contact position belongs to a type when the mean z-scored
//! activation of the type's neurons reaches the type's threshold; it can belong to several types.
//! Races, and contact positions of no type, belong to the generic expert (the last one).
//!
//! **Training**, in stages (benchmarks/moe_schedule.py):
//! 1. Collect (src/bin/moe_collect.rs): self-play of the reference net, every position classified
//!    and stored in a large set per expert.
//! 2. Experts (src/bin/moe_expert.rs), one at a time: TD(λ) self-play games starting from positions
//!    of the expert's set; positions of its type are judged by the expert, others by a frozen
//!    value net (the reference net, or for money experts a money net), and only the expert learns,
//!    only from positions of its type.
//! 3. Gateway (`GateNet`, src/bin/moe_gate.rs), with the experts frozen: one hidden layer, one
//!    sigmoid output per expert, trained to predict how well that expert evaluates a self-play
//!    position: exp(-(error / σ)²), the error measured against a 1-ply lookahead.
//!
//! **Evaluator** (`MoeNet`, a `.bgmoe` manifest): `mode gate` asks the gateway for its ratings,
//! skips the experts rated below `threshold` (default 0.25; the best one is always used),
//! normalizes the remaining ratings and averages those experts' winning chances with them
//! (money experts: their 5 outcome probabilities, so the money equity is averaged too);
//! `mode router` averages the experts of the position's game types (no gateway needed). All
//! experts have the same objective (double match point or money).
//!
//! Manifest (paths relative to the manifest's directory):
//!     BGMOE 1
//!     mode gate
//!     threshold 0.25
//!     gate gate.bggate
//!     types types.txt          (needed for mode router)
//!     expert holding-ahead holding-ahead.bgnet
//!     ...
//!     expert generic generic.bgnet
use std::cell::RefCell;
use std::path::{Path, PathBuf};

use crate::encoding::{is_race, Encoding, MAX_INPUTS};
use crate::eval_net::EvalNet;
use crate::evaluator::{money_equity, terminal_equity, terminal_money_equity, terminal_outcome, Evaluator, Scoring, MONEY_OUTPUTS};
use crate::neural_net::{Activations, NeuralNet, Objective};
use crate::position::Position;

/// Name of the expert for positions of no game type.
pub const GENERIC: &str = "generic";
/// The most experts a type set (bit mask) can hold.
pub const MAX_EXPERTS: usize = 16;

#[derive(Clone, Debug)]
pub struct TypeDef {
    pub name: String,
    pub threshold: f32,
    pub neurons: Vec<usize>,
    pub mu: Vec<f32>,
    pub sd: Vec<f32>,
}

/// The game types, defined by neuron groups of the reference net.
#[derive(Clone)]
pub struct GameTypes {
    pub reference: NeuralNet,
    pub layer: usize,
    pub types: Vec<TypeDef>,
}

impl GameTypes {
    pub fn load(path: &Path) -> Result<GameTypes, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path.display(), e))?;
        let (mut reference, mut layer, mut types) = (None, 2usize, Vec::<TypeDef>::new());
        let floats = |rest: &[&str]| -> Result<Vec<f32>, String> {
            rest.iter().map(|s| s.parse::<f32>().map_err(|e| e.to_string())).collect()
        };
        for line in text.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            match parts.as_slice() {
                [] => {}
                ["reference", p] => reference = Some(PathBuf::from(p)),
                ["layer", l] => layer = l.parse().map_err(|_| "bad layer".to_string())?,
                ["type", name, threshold, _n] => types.push(TypeDef {
                    name: name.to_string(),
                    threshold: threshold.parse().map_err(|_| "bad threshold".to_string())?,
                    neurons: vec![], mu: vec![], sd: vec![],
                }),
                ["neurons", rest @ ..] => {
                    let t = types.last_mut().ok_or("neurons before type")?;
                    t.neurons = rest.iter().map(|s| s.parse::<usize>().map_err(|e| e.to_string())).collect::<Result<_, _>>()?;
                }
                ["mu", rest @ ..] => types.last_mut().ok_or("mu before type")?.mu = floats(rest)?,
                ["sd", rest @ ..] => types.last_mut().ok_or("sd before type")?.sd = floats(rest)?,
                _ => return Err(format!("{}: unexpected line {:?}", path.display(), line)),
            }
        }
        let reference = reference.ok_or("no reference net")?;
        // A relative reference path is relative to the working directory (the project root).
        let reference = NeuralNet::load(&reference).map_err(|e| format!("{}: {}", reference.display(), e))?;
        if types.len() + 1 > MAX_EXPERTS {
            return Err(format!("{} types: at most {}", types.len(), MAX_EXPERTS - 1));
        }
        for t in &types {
            if t.neurons.len() != t.mu.len() || t.neurons.len() != t.sd.len() || t.neurons.is_empty() {
                return Err(format!("type {}: neurons, mu and sd differ in length", t.name));
            }
        }
        Ok(GameTypes { reference, layer, types })
    }

    /// Experts: one per type, then the generic one.
    pub fn n_experts(&self) -> usize {
        self.types.len() + 1
    }

    pub fn generic(&self) -> usize {
        self.types.len()
    }

    pub fn names(&self) -> Vec<String> {
        self.types.iter().map(|t| t.name.clone()).chain([GENERIC.to_string()]).collect()
    }

    /// Each type's score: the mean z-scored activation of its neurons (computes the reference net
    /// on `acts`, which must come from `reference.new_activations()`).
    pub fn scores<P: Position>(&self, position: &P, acts: &mut Activations) -> Vec<f32> {
        let mut input = [0.0f32; MAX_INPUTS];
        self.reference.encode(position, &mut input);
        self.reference.forward_keep(&input, acts);
        let layer = acts.layer(self.layer);
        self.types.iter().map(|t| {
            let sum: f32 = t.neurons.iter().zip(&t.mu).zip(&t.sd).map(|((&n, &m), &s)| (layer[n] - m) / s).sum();
            sum / t.neurons.len() as f32
        }).collect()
    }

    /// The position's experts as a bit mask: its types, or the generic expert for a race or a
    /// position of no type.
    pub fn classify<P: Position>(&self, position: &P, acts: &mut Activations) -> u16 {
        self.classify_with_value(position, acts).0
    }

    /// `classify`, and the reference net's winning chance for the player on roll (computed on the
    /// way for contact positions; for races too, then without classifying).
    pub fn classify_with_value<P: Position>(&self, position: &P, acts: &mut Activations) -> (u16, f32) {
        let scores = self.scores(position, acts);
        let value = acts.outputs()[0];
        if is_race(position) {
            return (1 << self.generic(), value);
        }
        let mask = self.types.iter().zip(&scores).enumerate()
            .fold(0u16, |m, (i, (t, &s))| if s >= t.threshold { m | (1 << i) } else { m });
        (if mask == 0 { 1 << self.generic() } else { mask }, value)
    }
}

/// Iterates the experts in a type mask.
pub fn experts_in(mask: u16) -> impl Iterator<Item = usize> {
    (0..MAX_EXPERTS).filter(move |&i| mask & (1 << i) != 0)
}

// --- gateway net ---------------------------------------------------------------------------------

const GATE_MAGIC: &str = "BGGATE 1";

/// The gateway: td-nack-v1 inputs, one ReLU hidden layer, one sigmoid output per expert.
#[derive(Clone)]
pub struct GateNet {
    pub n_in: usize,
    pub n_hidden: usize,
    pub n_out: usize,
    /// w1[i * n_hidden + j]: input i to hidden j.
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    /// w2[j * n_out + k]: hidden j to output k.
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
}

pub const GATE_ENCODING: Encoding = Encoding::TdNack;

impl GateNet {
    pub fn new_random(n_hidden: usize, n_out: usize, seed: u64) -> GateNet {
        let n_in = GATE_ENCODING.n_inputs();
        let mut state = seed.wrapping_mul(0x9e3779b97f4a7c15) | 1;
        let mut uniform = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0
        };
        let s1 = (6.0 / (n_in + n_hidden) as f32).sqrt();
        let s2 = (6.0 / (n_hidden + n_out) as f32).sqrt();
        GateNet {
            n_in, n_hidden, n_out,
            w1: (0..n_in * n_hidden).map(|_| uniform() * s1).collect(),
            b1: vec![0.0; n_hidden],
            w2: (0..n_hidden * n_out).map(|_| uniform() * s2).collect(),
            b2: vec![0.0; n_out],
        }
    }

    pub fn encode<P: Position>(position: &P, input: &mut [f32; MAX_INPUTS]) {
        GATE_ENCODING.encode(position, input);
    }

    /// Hidden activations into `hidden`, outputs (sigmoid) into `out`.
    pub fn forward(&self, input: &[f32], hidden: &mut [f32], out: &mut [f32]) {
        hidden.copy_from_slice(&self.b1);
        for (i, &x) in input[..self.n_in].iter().enumerate() {
            if x != 0.0 {
                for (h, &w) in hidden.iter_mut().zip(&self.w1[i * self.n_hidden..(i + 1) * self.n_hidden]) {
                    *h += x * w;
                }
            }
        }
        out.copy_from_slice(&self.b2);
        for (j, h) in hidden.iter_mut().enumerate() {
            *h = h.max(0.0);
            if *h > 0.0 {
                for (o, &w) in out.iter_mut().zip(&self.w2[j * self.n_out..(j + 1) * self.n_out]) {
                    *o += *h * w;
                }
            }
        }
        for o in out.iter_mut() {
            *o = 1.0 / (1.0 + (-*o).exp());
        }
    }

    pub fn evaluate<P: Position>(&self, position: &P) -> Vec<f32> {
        let mut input = [0.0f32; MAX_INPUTS];
        Self::encode(position, &mut input);
        let (mut hidden, mut out) = (vec![0.0; self.n_hidden], vec![0.0; self.n_out]);
        self.forward(&input, &mut hidden, &mut out);
        out
    }

    /// One SGD step on the cross-entropy between the outputs and `target` (each in [0, 1]).
    /// Returns the outputs before the step.
    pub fn train_step(&mut self, input: &[f32], target: &[f32], lr: f32) -> Vec<f32> {
        let (mut hidden, mut out) = (vec![0.0; self.n_hidden], vec![0.0; self.n_out]);
        self.forward(input, &mut hidden, &mut out);
        let d_out: Vec<f32> = out.iter().zip(target).map(|(y, t)| y - t).collect();
        let mut d_hidden = vec![0.0f32; self.n_hidden];
        for j in 0..self.n_hidden {
            if hidden[j] > 0.0 {
                let row = &mut self.w2[j * self.n_out..(j + 1) * self.n_out];
                let mut sum = 0.0;
                for (w, &d) in row.iter_mut().zip(&d_out) {
                    sum += *w * d;
                    *w -= lr * hidden[j] * d;
                }
                d_hidden[j] = sum;
            }
        }
        for (b, &d) in self.b2.iter_mut().zip(&d_out) {
            *b -= lr * d;
        }
        for (i, &x) in input[..self.n_in].iter().enumerate() {
            if x != 0.0 {
                for (w, &d) in self.w1[i * self.n_hidden..(i + 1) * self.n_hidden].iter_mut().zip(&d_hidden) {
                    *w -= lr * x * d;
                }
            }
        }
        for (b, &d) in self.b1.iter_mut().zip(&d_hidden) {
            *b -= lr * d;
        }
        out
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        let mut bytes = format!("{}\nencoding {}\nsizes {} {} {}\ndata\n", GATE_MAGIC, GATE_ENCODING.name(),
                                self.n_in, self.n_hidden, self.n_out).into_bytes();
        for v in [&self.w1, &self.b1, &self.w2, &self.b2] {
            for x in v.iter() {
                bytes.extend_from_slice(&x.to_le_bytes());
            }
        }
        std::fs::write(path, bytes).map_err(|e| format!("{}: {}", path.display(), e))
    }

    pub fn load(path: &Path) -> Result<GateNet, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {}", path.display(), e))?;
        let marker = b"\ndata\n";
        let at = bytes.windows(marker.len()).position(|w| w == marker).ok_or("no data section")?;
        let header = String::from_utf8_lossy(&bytes[..at]).to_string();
        if !header.starts_with(GATE_MAGIC) {
            return Err(format!("{}: not a gateway net", path.display()));
        }
        let sizes: Vec<usize> = header.lines().find_map(|l| l.strip_prefix("sizes "))
            .ok_or("no sizes")?.split_whitespace().map(|s| s.parse().unwrap_or(0)).collect();
        let (n_in, n_hidden, n_out) = (sizes[0], sizes[1], sizes[2]);
        if n_in != GATE_ENCODING.n_inputs() {
            return Err(format!("{}: {} inputs, expected {}", path.display(), n_in, GATE_ENCODING.n_inputs()));
        }
        let data: Vec<f32> = bytes[at + marker.len()..].chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let n = [n_in * n_hidden, n_hidden, n_hidden * n_out, n_out];
        if data.len() != n.iter().sum::<usize>() {
            return Err(format!("{}: wrong data size", path.display()));
        }
        let mut at = 0;
        let mut take = |k: usize| { let v = data[at..at + k].to_vec(); at += k; v };
        Ok(GateNet { n_in, n_hidden, n_out, w1: take(n[0]), b1: take(n[1]), w2: take(n[2]), b2: take(n[3]) })
    }
}

// --- the evaluator -------------------------------------------------------------------------------

/// The experts to use and their weights for the gateway's `ratings`: those rated at least
/// `threshold` (the best one if none is), with their ratings normalized to sum to 1.
pub fn gate_weights(ratings: &[f32], threshold: f32) -> Vec<(usize, f32)> {
    let mut chosen: Vec<usize> = (0..ratings.len()).filter(|&i| ratings[i] >= threshold).collect();
    if chosen.is_empty() {
        chosen.push((0..ratings.len()).max_by(|&a, &b| ratings[a].total_cmp(&ratings[b])).unwrap());
    }
    let total: f32 = chosen.iter().map(|&i| ratings[i]).sum::<f32>().max(1e-12);
    chosen.into_iter().map(|i| (i, ratings[i] / total)).collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeMode {
    Gate,
    Router,
}

/// The mixture of experts as an evaluator (see the module documentation).
pub struct MoeNet {
    pub names: Vec<String>,
    pub experts: Vec<EvalNet>,
    pub gate: Option<GateNet>,
    pub types: Option<GameTypes>,
    pub mode: MoeMode,
    /// Gate mode: experts rated below this are skipped (the best one is always used).
    pub threshold: f32,
    /// The experts' objective.
    pub objective: Objective,
    path: PathBuf,
    acts: RefCell<Option<Activations>>,
}

impl MoeNet {
    pub fn load(path: &Path) -> Result<MoeNet, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path.display(), e))?;
        let dir = path.parent().unwrap_or(Path::new("."));
        let mut lines = text.lines();
        if lines.next().map(str::trim) != Some("BGMOE 1") {
            return Err(format!("{}: not a mixture-of-experts manifest", path.display()));
        }
        let (mut mode, mut threshold, mut gate, mut types) = (MoeMode::Gate, 0.25f32, None, None);
        let (mut names, mut experts) = (Vec::new(), Vec::new());
        for line in lines {
            let parts: Vec<&str> = line.split_whitespace().collect();
            match parts.as_slice() {
                [] => {}
                ["mode", "gate"] => mode = MoeMode::Gate,
                ["mode", "router"] => mode = MoeMode::Router,
                ["threshold", t] => threshold = t.parse().map_err(|_| "bad threshold".to_string())?,
                ["gate", p] => gate = Some(GateNet::load(&dir.join(p))?),
                ["types", p] => types = Some(GameTypes::load(&dir.join(p))?),
                ["expert", name, p] => {
                    names.push(name.to_string());
                    experts.push(EvalNet::load(&dir.join(p)).map_err(|e| format!("{}: {}", p, e))?);
                }
                _ => return Err(format!("{}: unexpected line {:?}", path.display(), line)),
            }
        }
        if experts.is_empty() {
            return Err(format!("{}: no experts", path.display()));
        }
        let objective = experts[0].objective;
        if experts.iter().any(|e| e.objective != objective) {
            return Err(format!("{}: the experts must all have the same objective", path.display()));
        }
        match mode {
            MoeMode::Gate => {
                let g = gate.as_ref().ok_or("mode gate needs a gate")?;
                if g.n_out != experts.len() {
                    return Err(format!("gate has {} outputs for {} experts", g.n_out, experts.len()));
                }
            }
            MoeMode::Router => {
                let t = types.as_ref().ok_or("mode router needs types")?;
                if t.names() != names {
                    return Err("the experts must be the types in order, then generic".to_string());
                }
            }
        }
        Ok(MoeNet { names, experts, gate, types, mode, threshold, objective, path: path.to_path_buf(), acts: RefCell::new(None) })
    }

    /// The experts used for `position` and their weights (summing to 1).
    pub fn weights<P: Position>(&self, position: &P) -> Vec<(usize, f32)> {
        match self.mode {
            MoeMode::Gate => gate_weights(&self.gate.as_ref().unwrap().evaluate(position), self.threshold),
            MoeMode::Router => {
                let types = self.types.as_ref().unwrap();
                let mut acts = self.acts.borrow_mut();
                let acts = acts.get_or_insert_with(|| types.reference.new_activations());
                let mask = types.classify(position, acts);
                let n = mask.count_ones() as f32;
                experts_in(mask).map(|i| (i, 1.0 / n)).collect()
            }
        }
    }

    /// Winning chance of the player on roll.
    pub fn win_probability<P: Position>(&self, position: &P) -> f32 {
        self.weights(position).into_iter().map(|(i, w)| w * self.experts[i].win_probability(position)).sum()
    }

    /// The 5 outcome probabilities of the player on roll (money experts; the game must not be over).
    pub fn outcome<P: Position>(&self, position: &P) -> [f32; MONEY_OUTPUTS] {
        let mut total = [0.0f32; MONEY_OUTPUTS];
        for (i, w) in self.weights(position) {
            for (t, o) in total.iter_mut().zip(self.experts[i].outcome(position)) {
                *t += w * o;
            }
        }
        total
    }

    pub fn description(&self) -> String {
        let mode = match self.mode {
            MoeMode::Gate => format!("gate, threshold {}", self.threshold),
            MoeMode::Router => "router".to_string(),
        };
        format!("mixture of {} {} experts ({}) from {}", self.experts.len(), self.objective.name(), mode, self.path.display())
    }
}

impl<P: Position> Evaluator<P> for MoeNet {
    fn evaluate(&self, position: &P) -> f64 {
        self.evaluate_scored(position, Scoring::Dmp)
    }

    /// As a net of the experts' objective (see NeuralNetEvaluator): DMP experts know only the
    /// winning chance (money scoring: 2·win - 1, with the money value of finished games), money
    /// experts the money equity of their averaged outcome probabilities.
    fn evaluate_scored(&self, position: &P, scoring: Scoring) -> f64 {
        let dmp = || terminal_equity(position).unwrap_or_else(|| 2.0 * self.win_probability(position) as f64 - 1.0);
        match (scoring, self.objective) {
            (Scoring::Dmp, _) => dmp(),
            (Scoring::Money, Objective::Money) => terminal_money_equity(position)
                .unwrap_or_else(|| money_equity(&self.outcome(position))),
            (Scoring::Money, Objective::Dmp) => terminal_money_equity(position).unwrap_or_else(dmp),
        }
    }

    fn outcome(&self, position: &P) -> [f64; MONEY_OUTPUTS] {
        if let Some(o) = terminal_outcome(position) {
            return o.map(|x| x as f64);
        }
        match self.objective {
            Objective::Money => self.outcome(position).map(|x| x as f64),
            Objective::Dmp => [self.win_probability(position) as f64, f64::NAN, f64::NAN, f64::NAN, f64::NAN],
        }
    }

    fn name(&self) -> String {
        self.description()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EnginePosition;

    #[test]
    fn test_gate_learns_and_survives_save_and_load() {
        let mut position = EnginePosition::new();
        position.setup_starting_position();
        let mut input = [0.0f32; MAX_INPUTS];
        GateNet::encode(&position, &mut input);
        let target = [0.9f32, 0.1, 0.5];
        let mut gate = GateNet::new_random(16, 3, 7);
        let error = |g: &GateNet| -> f32 {
            let (mut h, mut o) = (vec![0.0; 16], vec![0.0; 3]);
            g.forward(&input, &mut h, &mut o);
            o.iter().zip(&target).map(|(a, b)| (a - b).abs()).sum()
        };
        let before = error(&gate);
        for _ in 0..200 {
            gate.train_step(&input, &target, 0.05);
        }
        assert!(error(&gate) < 0.1 * before, "{} -> {}", before, error(&gate));
        let path = std::env::temp_dir().join("test_gate.bggate");
        gate.save(&path).unwrap();
        let loaded = GateNet::load(&path).unwrap();
        assert_eq!(gate.evaluate(&position), loaded.evaluate(&position));
    }

    #[test]
    fn test_gate_weights_skip_low_ratings_and_normalize() {
        let w = gate_weights(&[0.1, 0.6, 0.3, 0.2], 0.25);
        assert_eq!(vec![1, 2], w.iter().map(|x| x.0).collect::<Vec<_>>());
        assert!((w[0].1 - 0.6 / 0.9).abs() < 1e-6 && (w[1].1 - 0.3 / 0.9).abs() < 1e-6);
        assert_eq!(vec![(2, 1.0)], gate_weights(&[0.1, 0.05, 0.2], 0.25));   // none passes: the best
    }

    #[test]
    fn test_experts_in_mask() {
        assert_eq!(vec![0, 3, 11], experts_in(0b1000_0000_1001).collect::<Vec<_>>());
    }
}
