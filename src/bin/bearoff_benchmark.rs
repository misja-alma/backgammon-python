//! Speed of the bearoff database: random lookups, compared with a net evaluation, and the effect
//! on a search of a bearoff position.
//!
//!   cargo run --release --bin bearoff_benchmark -- [data/bearoff-6x15.bd] [nets/td.bgnet]
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use backgammon_rust::bearoff::{BearoffDb, Indexer, Side};
use backgammon_rust::evaluator::Evaluator;
use backgammon_rust::{NeuralNet, NeuralNetEvaluator, Player, Position, RustAnalyzer, SimplePosition};

fn position(me: &Side, opponent: &Side) -> SimplePosition {
    let mut p = SimplePosition::new();
    for (player, side) in [(Player::Me, me), (Player::Opponent, opponent)] {
        for (i, &v) in side.iter().enumerate() {
            if v > 0 {
                p.set_checkers(player, i + 1, v).unwrap();
            }
        }
        let on_board: u8 = side.iter().sum();
        p.set_checkers(player, 0, 15 - on_board).unwrap();
    }
    p.set_turn(Player::Me);
    p
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let db_path = args.first().map(String::as_str).unwrap_or("data/bearoff-6x15.bd");
    let net_path = args.get(1).map(String::as_str).unwrap_or("nets/td.bgnet");

    let start = Instant::now();
    let db = Arc::new(BearoffDb::open(Path::new(db_path)).expect("open database"));
    println!("open (memory-map): {:.3} ms", start.elapsed().as_secs_f64() * 1e3);

    // Random positions (xorshift), excluding empty sides.
    let all: Vec<Side> = Indexer::new(15).all().into_iter().skip(1).collect();
    let mut state = 0x9E3779B97F4A7C15u64;
    let mut next = || { state ^= state << 13; state ^= state >> 7; state ^= state << 17; state };
    let positions: Vec<SimplePosition> = (0..1_000_000)
        .map(|_| position(&all[next() as usize % all.len()], &all[next() as usize % all.len()]))
        .collect();

    for pass in ["first pass", "second pass"] {
        let start = Instant::now();
        let mut sum = 0.0;
        for p in &positions {
            sum += db.win_probability(p).unwrap();
        }
        let s = start.elapsed().as_secs_f64();
        println!("{} lookups ({}): {:.0} ns each, {:.1} M/s   (checksum {:.1})",
                 positions.len(), pass, s / positions.len() as f64 * 1e9, positions.len() as f64 / s / 1e6, sum);
    }

    let net = NeuralNet::load(Path::new(net_path)).expect("load net");
    println!("net: {}", net.description());
    let evaluator = NeuralNetEvaluator { net };
    let n = 100_000;
    let start = Instant::now();
    let mut sum = 0.0;
    for p in &positions[..n] {
        sum += Evaluator::<SimplePosition>::evaluate(&evaluator, p);
    }
    let s = start.elapsed().as_secs_f64();
    println!("{} net evaluations: {:.0} ns each   (checksum {:.1})", n, s / n as f64 * 1e9, sum);

    // A search from a bearoff position, with and without the database.
    let p = position(&[2, 2, 3, 2, 3, 3], &[3, 2, 2, 3, 2, 3]);
    for depth in [1u32, 2] {
        for with_db in [false, true] {
            let net = NeuralNet::load(Path::new(net_path)).unwrap();
            let mut analyzer = RustAnalyzer::with_evaluator(Box::new(NeuralNetEvaluator { net }), 256);
            if with_db {
                analyzer.set_bearoff(Some(db.clone()));
            }
            let start = Instant::now();
            let (_, equity) = analyzer.best_move(&p, 6, 5, depth).unwrap();
            println!("best move 6-5 at depth {} {}: {:8.2} ms, equity {:+.4}", depth,
                     if with_db { "with database   " } else { "net only        " },
                     start.elapsed().as_secs_f64() * 1e3, equity);
        }
    }
}
