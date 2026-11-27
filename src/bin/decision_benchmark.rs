//! Checker play against bgsage's reference decisions, per game plan: a fast, noise-free
//! progress measure for training (the final strength measure stays games against bgsage).
//!
//!     .venv/bin/python -m benchmarks.extract_decisions          (once: writes data/decisions/)
//!     cargo run --release --bin decision_benchmark -- [--set money|pasko] [--scoring dmp|money] [--types FILE] NET...
//!
//! Decisions where all candidates are worth the same under the scoring are left out (since
//! 2026-11-26; before, the money set had 52 of them under dmp scoring, scored 0 by every net).
//!
//! A NET may also be a mixture of experts (a .bgmoe manifest, see moe.rs). `--types FILE` (game
//! types of a mixture, nets/moe/types.txt) adds the scores per game type: lines `type <name> ...`,
//! each decision counted for the game types of its position (generic: races and no type).
//!
//! For every decision (position, dice and the candidates bgsage evaluated at 3-ply or better,
//! see benchmarks/extract_decisions.py) the net picks the candidate with the highest value for
//! the mover, and the reference best is the candidate with the highest reference value. The value:
//!
//! - `--scoring dmp` (default): the winning chance (double match point: gammons don't count);
//! - `--scoring money`: the cubeless money equity, with gammons and backgammons, from the 5
//!   reference probabilities. A money net picks by its own money equity, a DMP net by its winning
//!   chance (it ignores gammons).
//!
//! Per group:
//!
//! - `best`: how often the net picks the reference best;
//! - `loss`: the average value its pick loses against the best: winning chance in percentage
//!   points (dmp), or equity in millipoints (money: 1000 = one point);
//! - `error`: the average |net - reference| value over all candidates, in the same units;
//! - `other`: how often the net's favourite among *all* legal moves is none of the candidates
//!   (a move bgsage judged clearly worse at a lower depth).
use std::path::Path;
use std::time::Instant;

use backgammon_rust::eval_net::EvalNet;
use backgammon_rust::moe::{experts_in, GameTypes, MoeNet};

/// A net, or a mixture of experts (a .bgmoe manifest, see moe.rs; double match point only).
enum Model {
    Net(EvalNet),
    Moe(MoeNet),
}

impl Model {
    fn load(path: &str) -> Model {
        let p = Path::new(path);
        let model = if path.ends_with(".bgmoe") { MoeNet::load(p).map(Model::Moe) } else { EvalNet::load(p).map(Model::Net) };
        model.unwrap_or_else(|e| panic!("{}: {}", path, e))
    }

    fn win_probability(&self, p: &SimplePosition) -> f32 {
        match self {
            Model::Net(n) => n.win_probability(p),
            Model::Moe(m) => m.win_probability(p),
        }
    }

    fn equity(&self, p: &SimplePosition) -> f64 {
        match self {
            Model::Net(n) => n.equity(p),
            Model::Moe(m) => 2.0 * m.win_probability(p) as f64 - 1.0,
        }
    }

    fn description(&self) -> String {
        match self {
            Model::Net(n) => n.description(),
            Model::Moe(m) => m.description(),
        }
    }
}
use backgammon_rust::evaluator::terminal_equity;
use backgammon_rust::move_generator::MoveGenerator;
use backgammon_rust::{Player, Position, SimplePosition};

/// bgsage's game plans, without pure races (the bearoff database's and the search's domain).
/// bgsage's game plans, the groups of the money and pasko sets; other sets (e.g. the game types of
/// the mixture of experts, moe-types) bring their own group names, listed after these.
const GROUPS: [&str; 4] = ["attacking", "priming", "anchoring", "racing"];

struct Candidate {
    position: SimplePosition,
    /// Reference value for the mover (winning chance, or cubeless money equity).
    win: f64,
}

#[derive(Clone, Copy, PartialEq)]
enum Scoring {
    Dmp,
    Money,
}

struct Decision {
    group: usize,
    position: SimplePosition,
    dice: (i32, i32),
    candidates: Vec<Candidate>,
}

/// A bgsage board (mover positive, opponent negative, opponent's bar at 0, mover's at 25) as a
/// position with `turn` on roll; the mover is `Me`.
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

fn load(path: &Path, scoring: Scoring, groups: &mut Vec<String>) -> Vec<Decision> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{}: {} (run .venv/bin/python -m benchmarks.extract_decisions)", path.display(), e));
    let mut decisions: Vec<Decision> = Vec::new();
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let fields: Vec<&str> = line.split_whitespace().collect();
        match fields[0] {
            "D" => {
                let group = groups.iter().position(|g| g == fields[1]).unwrap_or_else(|| {
                    groups.push(fields[1].to_string());
                    groups.len() - 1
                });
                let numbers: Vec<i32> = fields[3..31].iter().map(|s| s.parse().unwrap()).collect();
                decisions.push(Decision {
                    group,
                    position: position(&numbers[2..28], Player::Me),
                    dice: (numbers[0], numbers[1]),
                    candidates: Vec::new(),
                });
            }
            "C" => {
                let board: Vec<i32> = fields[1..27].iter().map(|s| s.parse().unwrap()).collect();
                let p: Vec<f64> = fields[27..32].iter().map(|s| s.parse().unwrap()).collect();
                let win = match scoring {
                    Scoring::Dmp => p[0],
                    Scoring::Money => 2.0 * p[0] - 1.0 + p[1] - p[3] + p[2] - p[4],
                };
                let d = decisions.last_mut().expect("candidate after a decision");
                d.candidates.push(Candidate { position: position(&board, Player::Opponent), win });
            }
            _ => panic!("unexpected line: {}", line),
        }
    }
    // Decisions where every candidate is worth the same under this scoring (e.g. moves that differ
    // only in gammons, under dmp scoring) don't measure anything: every net scores 0 there.
    let before = decisions.len();
    decisions.retain(|d| {
        let (lo, hi) = d.candidates.iter().fold((f64::MAX, f64::MIN), |(lo, hi), c| (lo.min(c.win), hi.max(c.win)));
        hi - lo > 1e-9
    });
    if decisions.len() < before {
        println!("{}: {} decisions without a difference between the candidates left out", path.display(), before - decisions.len());
    }
    decisions
}

#[derive(Default, Clone, Copy)]
struct Score {
    decisions: usize,
    best: usize,
    loss: f64,
    error: f64,
    candidates: usize,
    other: usize,
}

impl Score {
    fn add(&mut self, s: &Score) {
        self.decisions += s.decisions;
        self.best += s.best;
        self.loss += s.loss;
        self.error += s.error;
        self.candidates += s.candidates;
        self.other += s.other;
    }

    /// `unit`: 100 for winning chance in percentage points, 1000 for equity in millipoints.
    fn row(&self, name: &str, unit: f64) -> String {
        let n = self.decisions as f64;
        format!("{:<10} {:>6} {:>7.2}% {:>8.3} {:>8.3} {:>7.2}%", name, self.decisions, 100.0 * self.best as f64 / n,
                unit * self.loss / n, unit * self.error / self.candidates as f64, 100.0 * self.other as f64 / n)
    }
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut set = "money".to_string();
    if let Some(i) = args.iter().position(|a| a == "--set") {
        set = args[i + 1].clone();
        args.drain(i..i + 2);
    }
    let mut scoring = Scoring::Dmp;
    if let Some(i) = args.iter().position(|a| a == "--scoring") {
        scoring = match args[i + 1].as_str() {
            "dmp" => Scoring::Dmp,
            "money" => Scoring::Money,
            other => panic!("--scoring dmp or money, not {}", other),
        };
        args.drain(i..i + 2);
    }
    // --types FILE: also the scores per game type of a mixture of experts (moe.rs), each decision
    // counted for the types of its position (contact positions; races and positions of no type
    // count for generic).
    let mut game_types = None;
    if let Some(i) = args.iter().position(|a| a == "--types") {
        game_types = Some(GameTypes::load(Path::new(&args[i + 1])).unwrap_or_else(|e| panic!("{}", e)));
        args.drain(i..i + 2);
    }
    let unit = if scoring == Scoring::Money { 1000.0 } else { 100.0 };
    if args.is_empty() {
        args.push("nets/td-eval.bgnet".into());
    }
    let file = Path::new("data/decisions").join(format!("{}.txt", set));
    let mut groups: Vec<String> = GROUPS.iter().map(|g| g.to_string()).collect();
    let decisions = load(&file, scoring, &mut groups);

    // The candidates must be legal moves from the decision's position: checks the conversion.
    let mut generator = MoveGenerator::new();
    let legal: Vec<Vec<SimplePosition>> = decisions.iter()
        .map(|d| generator.legal_positions(&d.position, d.dice.0, d.dice.1)).collect();
    let illegal = decisions.iter().zip(&legal)
        .map(|(d, l)| d.candidates.iter().filter(|c| !l.contains(&c.position)).count()).sum::<usize>();
    let n_candidates: usize = decisions.iter().map(|d| d.candidates.len()).sum();
    println!("{}: {} decisions, {} candidates, {} not legal", file.display(), decisions.len(), n_candidates, illegal);
    if illegal > 0 {
        for (i, (d, l)) in decisions.iter().zip(&legal).enumerate() {
            for c in d.candidates.iter().filter(|c| !l.contains(&c.position)) {
                eprintln!("decision {} (dice {:?}): {:?} -> {:?}; legal: {}", i, d.dice, d.position, c.position, l.len());
            }
        }
    }
    assert!(illegal == 0, "candidates that are not legal moves: board conversion is wrong");

    let masks: Vec<u16> = match &game_types {
        Some(t) => {
            let mut acts = t.reference.new_activations();
            decisions.iter().map(|d| t.classify(&d.position, &mut acts)).collect()
        }
        None => vec![0; decisions.len()],
    };
    for path in &args {
        let net = Model::load(path);
        let start = Instant::now();
        // Value for the mover after a move (the opponent is on roll there): the winning chance
        // (dmp), or the money equity (the net's own equity: a DMP net's ignores gammons).
        let mover = |p: &SimplePosition| match scoring {
            Scoring::Dmp => 1.0 - terminal_equity(p).map_or_else(|| net.win_probability(p) as f64, |e| (e + 1.0) / 2.0),
            Scoring::Money => -net.equity(p),
        };
        let mut scores = vec![Score::default(); groups.len()];
        let mut by_type = vec![Score::default(); game_types.as_ref().map_or(0, |t| t.n_experts())];
        for ((d, legal), &mask) in decisions.iter().zip(&legal).zip(&masks) {
            let mut s = Score::default();
            let values: Vec<f64> = d.candidates.iter().map(|c| mover(&c.position)).collect();
            let pick = (0..values.len()).max_by(|&a, &b| values[a].total_cmp(&values[b])).unwrap();
            let best = d.candidates.iter().map(|c| c.win).fold(f64::MIN, f64::max);
            s.decisions += 1;
            s.best += (d.candidates[pick].win == best) as usize;
            s.loss += best - d.candidates[pick].win;
            s.error += d.candidates.iter().zip(&values).map(|(c, v)| (v - c.win).abs()).sum::<f64>();
            s.candidates += d.candidates.len();
            let favourite = legal.iter().max_by(|a, b| mover(a).total_cmp(&mover(b))).unwrap();
            s.other += d.candidates.iter().all(|c| &c.position != favourite) as usize;
            scores[d.group].add(&s);
            for e in experts_in(mask) {
                by_type[e].add(&s);
            }
        }
        println!("\n{} ({}), {:.1} s", path, net.description(), start.elapsed().as_secs_f64());
        println!("{:<10} {:>6} {:>8} {:>8} {:>8} {:>8}", "group", "n", "best", "loss", "error", "other");
        let mut total = Score::default();
        for (g, s) in groups.iter().zip(&scores).filter(|(_, s)| s.decisions > 0) {
            println!("{}", s.row(g, unit));
            total.add(s);
        }
        println!("{}", total.row("all", unit));
        if let Some(t) = &game_types {
            for (name, s) in t.names().iter().zip(&by_type) {
                println!("type {}", s.row(name, unit));
            }
        }
    }
}
