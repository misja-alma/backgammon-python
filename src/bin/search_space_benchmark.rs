//! Cost and speed of search space settings (the move filter, see `rust_analyzer::MoveFilter`): the
//! engine's choice at a given depth, with each setting, on bgsage's reference decisions.
//!
//!     cargo run --release --bin search_space_benchmark -- [--net FILE] [--depth N] [--decisions N]
//!         [--threads N] [--set money|pasko] [--no-full] SETTING...
//!
//! A SETTING is `margin:moves` (e.g. `0.1:5`; `all` for no limit, e.g. `0.1:all`, `all:3`), or
//! `full` (every move). Money scoring, with the bearoff database when data/bearoff-6x15.bd exists.
//!
//! The reference is the full search at the same depth (or, with `--no-full` for depths where the
//! full search takes too long, the first SETTING): its value of every move. Per setting:
//!
//! - `same`: how often the setting chooses the reference's move;
//! - `loss`: the average equity (millipoints) its choice loses by the reference's own values,
//!   which is the cost of the filter itself;
//! - `bgsage`: the average loss of its choice by bgsage's reference values (millipoints), for the
//!   choices among bgsage's candidates, and `other`: how many choices weren't (bgsage judged those
//!   clearly worse at a lower depth);
//! - `ms`: the time per decision, the cache cleared before each.
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use backgammon_rust::eval_net::{EvalNet, EvalNetEvaluator};
use backgammon_rust::evaluator::Scoring;
use backgammon_rust::rust_analyzer::MoveFilter;
use backgammon_rust::{Player, Position, RustAnalyzer, SimplePosition};

struct Decision {
    position: SimplePosition,
    dice: (i32, i32),
    /// (position after the move, bgsage's cubeless money equity for the mover)
    candidates: Vec<(SimplePosition, f64)>,
}

/// A bgsage board (mover positive, opponent negative, opponent's bar at 0, mover's at 25) as a
/// position with `turn` on roll; the mover is `Me`. (As in decision_benchmark.)
fn position(board: &[i32], turn: Player) -> SimplePosition {
    let mut p = SimplePosition::new();
    let (mut me, mut opponent) = (0, 0);
    for i in 1..=24 {
        let v = board[i];
        if v > 0 {
            p.set_checkers(Player::Me, i, v as u8).unwrap();
            me += v;
        } else if v < 0 {
            p.set_checkers(Player::Opponent, 25 - i, (-v) as u8).unwrap();
            opponent -= v;
        }
    }
    p.set_checkers(Player::Me, 25, board[25] as u8).unwrap();
    p.set_checkers(Player::Opponent, 25, board[0] as u8).unwrap();
    me += board[25];
    opponent += board[0];
    p.set_checkers(Player::Me, 0, (15 - me) as u8).unwrap();
    p.set_checkers(Player::Opponent, 0, (15 - opponent) as u8).unwrap();
    p.set_turn(turn);
    p
}

fn load(path: &Path) -> Vec<Decision> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{}: {} (run .venv/bin/python -m benchmarks.extract_decisions)", path.display(), e));
    let mut decisions: Vec<Decision> = Vec::new();
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let fields: Vec<&str> = line.split_whitespace().collect();
        match fields[0] {
            "D" => {
                let numbers: Vec<i32> = fields[3..31].iter().map(|s| s.parse().unwrap()).collect();
                decisions.push(Decision {
                    position: position(&numbers[2..28], Player::Me),
                    dice: (numbers[0], numbers[1]),
                    candidates: Vec::new(),
                });
            }
            "C" => {
                let board: Vec<i32> = fields[1..27].iter().map(|s| s.parse().unwrap()).collect();
                let p: Vec<f64> = fields[27..32].iter().map(|s| s.parse().unwrap()).collect();
                let equity = 2.0 * p[0] - 1.0 + p[1] - p[3] + p[2] - p[4];
                decisions.last_mut().expect("candidate after a decision").candidates
                    .push((position(&board, Player::Opponent), equity));
            }
            _ => panic!("unexpected line: {}", line),
        }
    }
    decisions
}

fn parse_setting(s: &str) -> (String, Option<MoveFilter>) {
    if s == "full" {
        return (s.to_string(), None);
    }
    let (margin, moves) = s.split_once(':').unwrap_or_else(|| panic!("setting {}: expected margin:moves or full", s));
    let margin = if margin == "all" { f64::INFINITY } else { margin.parse().expect("margin") };
    let max_moves = if moves == "all" { usize::MAX } else { moves.parse().expect("moves") };
    (s.to_string(), Some(MoveFilter { margin, max_moves }))
}

type Key = <SimplePosition as Position>::CacheKey;

#[derive(Default)]
struct Score {
    same: usize,
    loss: f64,
    bgsage_loss: f64,
    bgsage_n: usize,
    other: usize,
    seconds: f64,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut net, mut depth, mut count, mut threads, mut set, mut full) =
        ("nets/td-money-eval.bgnet".to_string(), 2u32, 300usize, std::thread::available_parallelism().map_or(1, |n| n.get()), "money".to_string(), true);
    let mut settings = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--net" => { net = args[i + 1].clone(); i += 1; }
            "--depth" => { depth = args[i + 1].parse().unwrap(); i += 1; }
            "--decisions" => { count = args[i + 1].parse().unwrap(); i += 1; }
            "--threads" => { threads = args[i + 1].parse().unwrap(); i += 1; }
            "--set" => { set = args[i + 1].clone(); i += 1; }
            "--no-full" => full = false,
            s => settings.push(parse_setting(s)),

        }
        i += 1;
    }
    if settings.is_empty() {
        settings = ["0.1:5"].iter().map(|s| parse_setting(s)).collect();
    }
    let reference = if full { ("full".to_string(), None) } else { settings[0].clone() };

    let all = load(&Path::new("data/decisions").join(format!("{}.txt", set)));
    // Spread over the whole set (it is ordered by game plan).
    let step = (all.len() / count.max(1)).max(1);
    let decisions: Vec<&Decision> = all.iter().step_by(step).take(count).collect();

    let mut analyzer: RustAnalyzer<SimplePosition> = RustAnalyzer::with_evaluator(
        Box::new(EvalNetEvaluator::new(EvalNet::load(Path::new(&net)).expect("net"))), 1024);
    analyzer.set_scoring(Scoring::Money);
    let bearoff = Path::new("data/bearoff-6x15.bd");
    if bearoff.exists() {
        analyzer.set_bearoff(Some(std::sync::Arc::new(backgammon_rust::bearoff::BearoffDb::open(bearoff).expect("bearoff"))));
    }
    analyzer.set_threads(threads);
    println!("{} decisions of the {} set, {}-ply (depth {}), {} threads, reference: {}",
             decisions.len(), set, depth + 1, depth, analyzer.threads(), reference.0);

    // The reference's value of every move it searched.
    analyzer.set_move_filter(reference.1);
    let mut seconds = 0.0;
    let reference_values: Vec<HashMap<Key, f64>> = decisions.iter().map(|d| {
        analyzer.clear_cache();
        let start = Instant::now();
        let moves = analyzer.analyze_moves(&d.position, d.dice.0, d.dice.1, depth);
        seconds += start.elapsed().as_secs_f64();
        moves.into_iter().filter(|m| m.3 == depth).map(|m| (m.0.cache_key(0), m.1)).collect()
    }).collect();
    println!("reference: {:.1} ms per decision (all its moves analyzed: as long as choosing one)",
             seconds * 1000.0 / decisions.len() as f64);

    println!("{:>12} {:>7} {:>8} {:>8} {:>6} {:>9}", "setting", "same", "loss", "bgsage", "other", "ms");
    for (name, filter) in &settings {
        analyzer.set_move_filter(*filter);
        let mut score = Score::default();
        for (d, values) in decisions.iter().zip(&reference_values) {
            analyzer.clear_cache();
            let start = Instant::now();
            let (chosen, _) = analyzer.best_move(&d.position, d.dice.0, d.dice.1, depth).unwrap();
            score.seconds += start.elapsed().as_secs_f64();
            let best = values.values().copied().fold(f64::NEG_INFINITY, f64::max);
            let key = chosen.cache_key(0);
            match values.get(&key) {
                Some(&v) => {
                    score.loss += best - v;
                    if best - v < 1e-12 {
                        score.same += 1;
                    }
                }
                // Only with a filtered reference: a move it didn't search counts as its worst.
                None => score.loss += best - values.values().copied().fold(f64::INFINITY, f64::min),
            }
            let bgsage_best = d.candidates.iter().map(|c| c.1).fold(f64::NEG_INFINITY, f64::max);
            match d.candidates.iter().find(|c| c.0.cache_key(0) == key) {
                Some(c) => {
                    score.bgsage_loss += bgsage_best - c.1;
                    score.bgsage_n += 1;
                }
                None => score.other += 1,
            }
        }
        let n = decisions.len() as f64;
        println!("{:>12} {:>6.1}% {:>8.2} {:>8.2} {:>6} {:>9.1}", name, 100.0 * score.same as f64 / n,
                 1000.0 * score.loss / n, 1000.0 * score.bgsage_loss / score.bgsage_n.max(1) as f64,
                 score.other, 1000.0 * score.seconds / n);
    }
}
