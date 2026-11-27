//! Records a net's hidden-layer activations over self-play positions, for interpretability
//! analysis (analysis/activations.py).
//!
//!     activation_dump --net nets/td.bgnet --games 20000 --sample 0.25 --out data/activations/td
//!
//! Plays greedy 1-ply self-play games with the net (backgammon and nackgammon starts alternately,
//! played to the end without bearoff database) and keeps each position before a move with
//! probability --sample. Writes, for N kept positions:
//! - OUT.hidden.f32: N rows of all hidden activations (layer 1, then layer 2, ...), f32 little endian;
//! - OUT.positions.u8: N rows of 52 checker counts, the player on roll's points 0-25 (0 = borne off,
//!   25 = bar, own numbering), then the other player's;
//! - OUT.meta.csv: per row game, variant, turn, net output (winning chance of the player on roll),
//!   and whether the player on roll won the game in the end (and with how many points: 1, 2, 3).
//! - OUT.json: the layer sizes and counts.
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use backgammon_rust::encoding::MAX_INPUTS;
use backgammon_rust::game::{opening_roll, Dice, Variant};
use backgammon_rust::move_generator::MoveGenerator;
use backgammon_rust::neural_net::NeuralNet;
use backgammon_rust::{EnginePosition, Player, Position};

const MAX_TURNS: u32 = 10_000;

struct Row {
    hidden: Vec<f32>,
    checkers: [u8; 52],
    variant: usize,
    turn: u32,
    output: f32,
    on_roll: Player,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut net_path = PathBuf::from("nets/td.bgnet");
    let (mut games, mut sample, mut seed) = (20_000u64, 0.25f64, 1u64);
    let mut out = PathBuf::from("data/activations/td");
    let mut i = 0;
    while i < args.len() {
        let value = args.get(i + 1).cloned().unwrap_or_default();
        match args[i].as_str() {
            "--net" => net_path = PathBuf::from(value),
            "--games" => games = value.parse().expect("--games N"),
            "--sample" => sample = value.parse().expect("--sample P"),
            "--seed" => seed = value.parse().expect("--seed N"),
            "--out" => out = PathBuf::from(value),
            other => panic!("unknown option {other}"),
        }
        i += 2;
    }
    let net = NeuralNet::load(&net_path).expect("load net");
    let sizes = net.sizes().to_vec();
    let hidden_sizes = &sizes[1..sizes.len() - 1];
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir).unwrap();
    }
    let path = |ext: &str| PathBuf::from(format!("{}.{ext}", out.display()));
    let mut f_hidden = BufWriter::new(File::create(path("hidden.f32")).unwrap());
    let mut f_pos = BufWriter::new(File::create(path("positions.u8")).unwrap());
    let mut f_meta = BufWriter::new(File::create(path("meta.csv")).unwrap());
    writeln!(f_meta, "game,variant,turn,output,won,points").unwrap();

    let mut dice = Dice::new(seed);
    let mut rng = Dice::new(seed ^ 0x5eed);
    let mut generator = MoveGenerator::<EnginePosition>::new();
    let mut acts = net.new_activations();
    let mut input = [0.0f32; MAX_INPUTS];
    let mut kept = 0u64;
    for game in 0..games {
        let variant = if game % 2 == 0 { Variant::Backgammon } else { Variant::Nackgammon };
        let mut position = variant.starting_position();
        let (starter, mut d1, mut d2) = opening_roll(&mut dice);
        position.set_turn(starter);
        let mut rows: Vec<Row> = Vec::new();
        let mut winner = None;
        for turn in 1..=MAX_TURNS {
            // Uniform [0, 1) from two dice-generator draws is too coarse: use 3 rolls (216 values).
            let r = ((rng.roll() - 1) * 36 + (rng.roll() - 1) * 6 + (rng.roll() - 1)) as f64 / 216.0;
            if r < sample {
                net.encode(&position, &mut input);
                let output = net.forward_keep(&input, &mut acts);
                let mut hidden = Vec::with_capacity(hidden_sizes.iter().sum());
                for l in 1..=hidden_sizes.len() {
                    hidden.extend_from_slice(acts.layer(l));
                }
                let on_roll = position.get_turn();
                let mut checkers = [0u8; 52];
                for (block, player) in [on_roll, on_roll.other_player()].into_iter().enumerate() {
                    for point in 0..26 {
                        checkers[block * 26 + point] = position.count(player, point);
                    }
                }
                rows.push(Row { hidden, checkers, variant: variant as usize, turn, output, on_roll });
            }
            let next = generator
                .best_position(&position, d1, d2, |child| {
                    let mut buf = [0.0f32; MAX_INPUTS];
                    net.encode(child, &mut buf);
                    1.0 - net.forward(&buf) as f64 // the child is seen from the opponent
                })
                .map(|(p, _)| p)
                .unwrap_or_else(|| {
                    let mut passed = position.clone();
                    passed.switch_turn();
                    passed
                });
            if let Some(w) = [Player::Me, Player::Opponent].into_iter().find(|&p| next.count(p, 0) == 15) {
                let loser = w.other_player();
                let points = if next.count(loser, 0) > 0 {
                    1
                } else if (19..26).any(|p| next.count(loser, p) > 0) {
                    3 // the loser still has a checker in the winner's home board or on the bar
                } else {
                    2
                };
                winner = Some((w, points));
                break;
            }
            position = next;
            d1 = dice.roll();
            d2 = dice.roll();
        }
        let (w, points) = winner.expect("game finished");
        for row in rows {
            for v in &row.hidden {
                f_hidden.write_all(&v.to_le_bytes()).unwrap();
            }
            f_pos.write_all(&row.checkers).unwrap();
            writeln!(f_meta, "{game},{},{},{:.5},{},{points}", row.variant, row.turn, row.output,
                     (row.on_roll == w) as u8).unwrap();
            kept += 1;
        }
        if (game + 1) % 1000 == 0 {
            eprintln!("{} games, {kept} positions", game + 1);
        }
    }
    let json = format!(
        "{{\"net\": \"{}\", \"hidden\": {:?}, \"positions\": {kept}, \"games\": {games}, \"sample\": {sample}}}\n",
        net_path.display(), hidden_sizes);
    std::fs::write(path("json"), json).unwrap();
    eprintln!("wrote {kept} positions to {}.*", out.display());
}
