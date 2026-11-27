//! Stage 3 of the mixture of experts (see src/moe.rs): trains the gateway (classifier) with the
//! experts frozen, and writes the manifests.
//!
//!     moe_gate --dir nets/moe/experts --positions 200000 --threads 12 --bearoff data/bearoff-6x15.bd
//!
//! Values are winning chances for double match point experts and cubeless money equities for money
//! experts (then σ defaults to 0.06, twice 0.03, as equity = 2·win - 1 without gammons; the
//! comparison net is --value-net, a money net, instead of the game types' reference net).
//!
//! 1. Data: self-play games of the experts (a position's value is the mean value of the
//!    experts of its game types, moves chosen greedily) on --threads threads; each position is kept
//!    with probability --sample (not finished or bearoff-database positions). For each kept
//!    position: the target, a 1-ply lookahead (all 21 rolls, best move each), and every expert's
//!    own value. Saved in DIR/gate-data.bin, reused by later runs (--fresh to regenerate).
//! 2. Gateway: one hidden layer (--hidden, 512), trained (SGD, cross-entropy, --epochs) to output for
//!    each expert exp(-((expert - target) / σ)²) (--sigma), i.e. how well that expert judges the
//!    position. Only the gateway learns. 10% of the positions are held out; on them the mean error
//!    against the target is reported for the reference net, the experts picked by the game types,
//!    the gateway as the evaluator uses it (experts rated below --threshold, 0.25, skipped, the
//!    others weighted by their normalized ratings), its top 1 / top 2 / all experts, and the best
//!    expert per position (oracle); and per game type, on the held-out positions of that type, the
//!    reference net against the type's own expert and the gateway.
//! 3. Writes DIR/gate.bggate, DIR/moe.bgmoe (mode gate, --threshold) and DIR/router.bgmoe (mode
//!    router), using the experts DIR/<name><--suffix>.bgnet (default -avg) and a copy of the types.
use std::cell::RefCell;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use backgammon_rust::bearoff::BearoffDb;
use backgammon_rust::encoding::MAX_INPUTS;
use backgammon_rust::evaluator::{gammon_possible, money_equity, terminal_money_equity, MONEY_OUTPUTS};
use backgammon_rust::game::{opening_roll, Dice, Variant};
use backgammon_rust::moe::{experts_in, gate_weights, GameTypes, GateNet};
use backgammon_rust::move_generator::MoveGenerator;
use backgammon_rust::neural_net::{Activations, NeuralNet, Objective};
use backgammon_rust::{EnginePosition, Player, Position};

struct Sample {
    checkers: [u8; 52],
    target: f32,
    values: Vec<f32>,
}

/// The exact value for the player on roll when known (money: the equity of a finished game, or of
/// a bearoff position once no gammon is possible; DMP: the winning chance).
fn exact(money: bool, bearoff: Option<&BearoffDb>, position: &EnginePosition) -> Option<f32> {
    if money {
        if let Some(e) = terminal_money_equity(position) {
            return Some(e as f32);
        }
        if gammon_possible(position) {
            return None;
        }
        return bearoff.and_then(|db| db.win_probability(position)).map(|p| 2.0 * p as f32 - 1.0);
    }
    for p in [Player::Me, Player::Opponent] {
        if position.count(p, 0) == 15 {
            return Some(if p == position.get_turn() { 1.0 } else { 0.0 });
        }
    }
    bearoff.and_then(|db| db.win_probability(position)).map(|p| p as f32)
}

/// A net's value for the player on roll, from the encoded inputs.
fn net_value(money: bool, net: &NeuralNet, input: &[f32]) -> f32 {
    if money {
        let mut out = [0.0f32; MONEY_OUTPUTS];
        net.forward_all(input, &mut out);
        money_equity(&out) as f32
    } else {
        net.forward(input)
    }
}

/// The value for the other side.
fn flip(money: bool, v: f32) -> f32 {
    if money { -v } else { 1.0 - v }
}

struct System<'a> {
    money: bool,
    types: &'a GameTypes,
    experts: &'a [NeuralNet],
    router: RefCell<Activations>,
    bearoff: Option<&'a BearoffDb>,
}

impl<'a> System<'a> {
    fn value(&self, position: &EnginePosition) -> f32 {
        if let Some(v) = exact(self.money, self.bearoff, position) {
            return v;
        }
        let mask = self.types.classify(position, &mut self.router.borrow_mut());
        let mut input = [0.0f32; MAX_INPUTS];
        self.experts[0].encode(position, &mut input);
        experts_in(mask).map(|e| net_value(self.money, &self.experts[e], &input)).sum::<f32>() / mask.count_ones() as f32
    }

    /// The mover's best move with these dice (a pass when there is none).
    fn best(&self, generator: &mut MoveGenerator<EnginePosition>, position: &EnginePosition, d1: i32, d2: i32) -> EnginePosition {
        generator.best_position(position, d1, d2, |c| -self.value(c) as f64).map(|(p, _)| p)
            .unwrap_or_else(|| { let mut p = position.clone(); p.switch_turn(); p })
    }

    /// 1-ply lookahead: the value for the player on roll, averaged over the 21 rolls.
    fn lookahead(&self, generator: &mut MoveGenerator<EnginePosition>, position: &EnginePosition) -> f32 {
        let mut total = 0.0;
        for d1 in 1..=6 {
            for d2 in d1..=6 {
                let weight = if d1 == d2 { 1.0 } else { 2.0 };
                total += weight * flip(self.money, self.value(&self.best(generator, position, d1, d2)));
            }
        }
        total / 36.0
    }
}

fn to_position(checkers: &[u8; 52]) -> EnginePosition {
    let mut p = EnginePosition::new();
    for point in 0..26 {
        p.set_checkers(Player::Me, point, checkers[point]).unwrap();
        p.set_checkers(Player::Opponent, point, checkers[26 + point]).unwrap();
    }
    p.set_turn(Player::Me);
    p
}

fn save_data(path: &Path, samples: &[Sample]) {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    let n = samples.first().map_or(0, |s| s.values.len());
    f.write_all(&(samples.len() as u64).to_le_bytes()).unwrap();
    f.write_all(&(n as u64).to_le_bytes()).unwrap();
    for s in samples {
        f.write_all(&s.checkers).unwrap();
        f.write_all(&s.target.to_le_bytes()).unwrap();
        for v in &s.values {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
    }
}

fn load_data(path: &Path, n_experts: usize) -> Option<Vec<Sample>> {
    let bytes = std::fs::read(path).ok()?;
    let count = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let n = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    if n != n_experts {
        return None;
    }
    let f = |b: &[u8]| f32::from_le_bytes(b.try_into().unwrap());
    let size = 52 + 4 + 4 * n;
    Some((0..count).map(|i| {
        let r = &bytes[16 + i * size..16 + (i + 1) * size];
        Sample {
            checkers: r[..52].try_into().unwrap(),
            target: f(&r[52..56]),
            values: (0..n).map(|k| f(&r[56 + 4 * k..60 + 4 * k])).collect(),
        }
    }).collect())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut dir, mut types_path) = (PathBuf::from("nets/moe/run"), PathBuf::from("nets/moe/types.txt"));
    let (mut positions, mut sample, mut threads, mut seed) = (200_000usize, 0.1f64, 12usize, 77u64);
    let mut value_path = None::<PathBuf>;
    let (mut sigma, mut epochs, mut hidden, mut threshold, mut lr) = (f32::NAN, 12usize, 512usize, 0.25f32, 0.02f32);
    let (mut suffix, mut bearoff_path, mut fresh) = ("-avg".to_string(), None::<PathBuf>, false);
    let mut i = 0;
    while i < args.len() {
        let v = args.get(i + 1).cloned().unwrap_or_default();
        let mut step = 2;
        match args[i].as_str() {
            "--dir" => dir = v.into(),
            "--types" => types_path = v.into(),
            "--value-net" => value_path = Some(v.into()),
            "--positions" => positions = v.parse().unwrap(),
            "--sample" => sample = v.parse().unwrap(),
            "--threads" => threads = v.parse().unwrap(),
            "--seed" => seed = v.parse().unwrap(),
            "--sigma" => sigma = v.parse().unwrap(),
            "--epochs" => epochs = v.parse().unwrap(),
            "--hidden" => hidden = v.parse().unwrap(),
            "--threshold" => threshold = v.parse().unwrap(),
            "--lr" => lr = v.parse().unwrap(),
            "--suffix" => suffix = if v == "none" { String::new() } else { v },
            "--bearoff" => bearoff_path = if v == "none" { None } else { Some(v.into()) },
            "--fresh" => { fresh = true; step = 1; }
            other => panic!("unknown option {other}"),
        }
        i += step;
    }
    let types = GameTypes::load(&types_path).unwrap_or_else(|e| panic!("{e}"));
    let names = types.names();
    let n = names.len();
    let experts: Vec<NeuralNet> = names.iter()
        .map(|nm| { let p = dir.join(format!("{nm}{suffix}.bgnet")); NeuralNet::load(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display())) })
        .collect();
    let money = experts[0].objective() == Objective::Money;
    assert!(experts.iter().all(|e| e.objective() == experts[0].objective()), "the experts must have the same objective");
    if sigma.is_nan() {
        sigma = if money { 0.06 } else { 0.03 };
    }
    // The single net to compare with: the game types' reference net, or --value-net (needed for
    // money experts: a money net).
    let value_net = value_path.as_ref().map(|p| NeuralNet::load(p).unwrap_or_else(|e| panic!("{e}")));
    let compare = value_net.as_ref().unwrap_or(&types.reference);
    assert_eq!(compare.objective(), experts[0].objective(), "--value-net must have the experts' objective");
    let unit = if money { "hundredths of cubeless money equity" } else { "winning-chance points" };
    eprintln!("{} experts ({}), sigma {sigma}", n, experts[0].objective().name());
    let bearoff = bearoff_path.as_ref().map(|p| BearoffDb::open(p).unwrap_or_else(|e| panic!("{e}")));
    let data_path = dir.join("gate-data.bin");

    // 1. data
    let samples = match (fresh, load_data(&data_path, n)) {
        (false, Some(s)) if s.len() >= positions => { eprintln!("{} positions from {}", s.len(), data_path.display()); s }
        _ => {
            let started = Instant::now();
            let collected = Mutex::new(Vec::<Sample>::new());
            let (next_game, kept) = (AtomicU64::new(0), AtomicUsize::new(0));
            std::thread::scope(|scope| {
                for _ in 0..threads {
                    let (types, experts, bearoff, collected, next_game, kept) =
                        (&types, &experts, bearoff.as_ref(), &collected, &next_game, &kept);
                    scope.spawn(move || {
                        let system = System { money, types, experts, router: RefCell::new(types.reference.new_activations()), bearoff };
                        let mut generator = MoveGenerator::new();
                        let mut input = [0.0f32; MAX_INPUTS];
                        while kept.load(Ordering::Relaxed) < positions {
                            let game = next_game.fetch_add(1, Ordering::Relaxed);
                            let mut dice = Dice::new(seed ^ game.wrapping_mul(0xd1b54a32d192ed03));
                            let mut keep_rng = Dice::new(seed ^ 0x5eed ^ game.wrapping_mul(0x9e3779b97f4a7c15));
                            let variant = if game % 2 == 0 { Variant::Backgammon } else { Variant::Nackgammon };
                            let mut position = variant.starting_position();
                            let (starter, mut d1, mut d2) = opening_roll(&mut dice);
                            position.set_turn(starter);
                            loop {
                                let r = (keep_rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
                                if r < sample && exact(money, bearoff, &position).is_none() {
                                    let target = system.lookahead(&mut generator, &position);
                                    experts[0].encode(&position, &mut input);
                                    let values: Vec<f32> = experts.iter().map(|e| net_value(money, e, &input)).collect();
                                    let on_roll = position.get_turn();
                                    let mut checkers = [0u8; 52];
                                    for (block, player) in [on_roll, on_roll.other_player()].into_iter().enumerate() {
                                        for point in 0..26 {
                                            checkers[block * 26 + point] = position.count(player, point);
                                        }
                                    }
                                    collected.lock().unwrap().push(Sample { checkers, target, values });
                                    let k = kept.fetch_add(1, Ordering::Relaxed) + 1;
                                    if k % 20000 == 0 {
                                        eprintln!("{k} positions, {:.0} s", started.elapsed().as_secs_f64());
                                    }
                                }
                                let next = system.best(&mut generator, &position, d1, d2);
                                if exact(money, bearoff, &next).is_some() {
                                    break;
                                }
                                position = next;
                                d1 = dice.roll();
                                d2 = dice.roll();
                            }
                        }
                    });
                }
            });
            let mut s = collected.into_inner().unwrap();
            s.truncate(positions);
            save_data(&data_path, &s);
            eprintln!("{} positions in {:.0} s, saved to {}", s.len(), started.elapsed().as_secs_f64(), data_path.display());
            s
        }
    };

    // 2. gateway
    let held_out = samples.len() / 10;
    let (test, train) = samples.split_at(held_out);
    let quality = |s: &Sample| -> Vec<f32> { s.values.iter().map(|v| (-((v - s.target) / sigma).powi(2)).exp()).collect() };
    let inputs: Vec<[f32; MAX_INPUTS]> = samples.iter().map(|s| {
        let mut x = [0.0f32; MAX_INPUTS];
        GateNet::encode(&to_position(&s.checkers), &mut x);
        x
    }).collect();
    let (test_x, train_x) = inputs.split_at(held_out);
    let train_q: Vec<Vec<f32>> = train.iter().map(quality).collect();
    let mut gate = GateNet::new_random(hidden, n, seed);
    let mut order: Vec<usize> = (0..train.len()).collect();
    let mut rng = Dice::new(seed);
    for epoch in 0..epochs {
        for i in (1..order.len()).rev() {
            order.swap(i, (rng.next_u64() % (i as u64 + 1)) as usize);
        }
        let rate = lr * (1.0 - epoch as f32 / epochs as f32).max(0.1);
        let mut loss = 0.0f64;
        for &j in &order {
            let y = gate.train_step(&train_x[j], &train_q[j], rate);
            loss += y.iter().zip(&train_q[j]).map(|(&y, &t)| {
                let y = y.clamp(1e-6, 1.0 - 1e-6) as f64;
                -(t as f64 * y.ln() + (1.0 - t as f64) * (1.0 - y).ln())
            }).sum::<f64>() / n as f64;
        }
        eprintln!("epoch {:2}: rate {:.4}, cross-entropy {:.4}", epoch + 1, rate, loss / order.len() as f64);
    }

    // Evaluation on the held-out positions.
    let mut router = types.reference.new_activations();
    let mut errors = [0.0f64; 8];
    let mut experts_used = 0usize;
    let mut best_hit = 0usize;
    let mut used = vec![0usize; n];
    // Per expert, on held-out positions of its own type: count, error of the reference net, of the
    // expert itself and of the gateway (top 2).
    let mut per_type = vec![(0usize, 0.0f64, 0.0f64, 0.0f64); n];
    let (mut h, mut out) = (vec![0.0f32; hidden], vec![0.0f32; n]);
    for (s, x) in test.iter().zip(test_x) {
        let position = to_position(&s.checkers);
        gate.forward(x, &mut h, &mut out);
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| out[b].total_cmp(&out[a]));
        let weighted = |k: usize| -> f32 {
            let total: f32 = order[..k].iter().map(|&i| out[i]).sum::<f32>().max(1e-12);
            order[..k].iter().map(|&i| out[i] * s.values[i]).sum::<f32>() / total
        };
        let chosen = gate_weights(&out, threshold);
        experts_used += chosen.len();
        let gated: f32 = chosen.iter().map(|&(i, w)| w * s.values[i]).sum();
        let mut input = [0.0f32; MAX_INPUTS];
        compare.encode(&position, &mut input);
        let reference = net_value(money, compare, &input);
        let mask = types.classify(&position, &mut router);
        let routed = experts_in(mask).map(|e| s.values[e]).sum::<f32>() / mask.count_ones() as f32;
        let best = (0..n).min_by(|&a, &b| (s.values[a] - s.target).abs().total_cmp(&(s.values[b] - s.target).abs())).unwrap();
        best_hit += (order[0] == best) as usize;
        for e in experts_in(mask) {
            let row = &mut per_type[e];
            row.0 += 1;
            row.1 += (reference - s.target).abs() as f64;
            row.2 += (s.values[e] - s.target).abs() as f64;
            row.3 += (gated - s.target).abs() as f64;
        }
        used[order[0]] += 1;
        for (k, v) in [reference, routed, gated, weighted(1), weighted(2), weighted(n), s.values[best], s.values[n - 1]].iter().enumerate() {
            errors[k] += (v - s.target).abs() as f64;
        }
    }
    let t = test.len() as f64;
    println!("held-out positions: {}; mean |value - 1-ply lookahead| ({unit}):", test.len());
    let gated_label = format!("gateway, ratings >= {threshold}");
    for (k, label) in [if value_net.is_some() { "value net" } else { "reference net" }, "routed system (types)", gated_label.as_str(), "gateway top 1", "gateway top 2",
                       "gateway all (weighted)", "best expert (oracle)", "generic expert only"].iter().enumerate() {
        println!("    {:24} {:.3}", label, 100.0 * errors[k] / t);
    }
    println!("gateway with threshold {threshold}: {:.2} experts per position on average", experts_used as f64 / t);
    println!("gateway's first choice is the best expert in {:.0}% of positions; first choices: {}", 100.0 * best_hit as f64 / t,
             names.iter().zip(&used).map(|(nm, &u)| format!("{nm} {:.0}%", 100.0 * u as f64 / t)).collect::<Vec<_>>().join(", "));

    println!("per game type, on held-out positions of that type: {} / own expert / gateway (threshold {threshold})",
             if value_net.is_some() { "value net" } else { "reference net" });
    for (nm, &(count, r, e, g)) in names.iter().zip(&per_type) {
        let c = count.max(1) as f64;
        println!("    {:24} {:6} positions  {:.3} / {:.3} / {:.3}", nm, count, 100.0 * r / c, 100.0 * e / c, 100.0 * g / c);
    }

    // 3. files
    gate.save(&dir.join("gate.bggate")).unwrap();
    std::fs::copy(&types_path, dir.join("types.txt")).unwrap();
    let experts_lines: String = names.iter().map(|nm| format!("expert {nm} {nm}{suffix}.bgnet\n")).collect();
    std::fs::write(dir.join("moe.bgmoe"), format!("BGMOE 1\nmode gate\nthreshold {threshold}\ngate gate.bggate\n{experts_lines}")).unwrap();
    std::fs::write(dir.join("router.bgmoe"), format!("BGMOE 1\nmode router\ntypes types.txt\n{experts_lines}")).unwrap();
    println!("wrote {0}/gate.bggate, {0}/moe.bgmoe, {0}/router.bgmoe", dir.display());
}
