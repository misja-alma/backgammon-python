//! Stage 2 of the mixture of experts (src/moe.rs): train one expert on its own game type.
//!
//!     moe_expert --expert blitz-attacker --starts nets/moe/positions/blitz-attacker.pos \
//!                --dir nets/moe/experts --init nets/moe/base.bgnet --value-net nets/moe/value.bgnet \
//!                --games 2000000 --alpha 0.005 --threads 12 --average-last 250000 --bearoff data/bearoff-6x15.bd
//!
//! TD(λ) self-play games, each starting from a random position of the expert's set (collected by
//! moe_collect; the player on roll moves first, with fresh dice). Positions of the expert's type
//! (as classified by the game types; for the generic expert: races and positions of no type) are
//! judged by the expert, all others by the frozen value net (--value-net; default the game types'
//! reference net), and moves are chosen greedily by those values. Only the expert learns, and only
//! from positions of its type: its TD update at such a position, with its eligibility trace
//! decaying (λ per move) through the positions in between. After 16 moves in a row off its type
//! the trace is negligible and the game ends there (the expert would learn nothing more from it).
//! Offline TD(λ) as in td_training.rs: the weights stay fixed during a game, and change at its end
//! by what the traces would add up to (`learn_offline_from_errors`).
//!
//! The expert's objective is --init's: a double match point net (1 output, the winning chance) or
//! a money net (5 outputs, moves chosen by cubeless money equity, a trace per output as in
//! td_training.rs; a game stops at a bearoff position only when no gammon is possible any more).
//! The value net must have the same objective. The game types always classify with their own
//! reference net.
//!
//! The expert is DIR/<expert>.bgnet, started as a copy of --init the first time; with
//! --average-last N the average of the weights over the last N games is saved as
//! DIR/<expert>-avg.bgnet. Threads as in train.rs (own copy per thread, changes merged after each
//! game).
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use backgammon_rust::bearoff::BearoffDb;
use backgammon_rust::encoding::MAX_INPUTS;
use backgammon_rust::evaluator::{flip_outcome, money_equity, terminal_outcome, MONEY_OUTPUTS};
use backgammon_rust::game::Dice;
use backgammon_rust::moe::GameTypes;
use backgammon_rust::move_generator::MoveGenerator;
use backgammon_rust::neural_net::{NeuralNet, Objective};
use backgammon_rust::td_training::{exact_outcome, learn_offline_from_errors, DMP_SIGNS, DMP_SOURCE, ME_SIGNS, ME_SOURCE};
use backgammon_rust::{EnginePosition, Player, Position};

const MAX_TURNS: u32 = 10_000;
/// Moves off the expert's type after which its trace (λ^moves) is dropped and the game ends: the
/// expert can't learn from the rest of it (the next game starts from its own set again).
const TRACE_STEPS: u32 = 16;

type Outcome = [f32; MONEY_OUTPUTS];

/// The exact value for the player on roll when known: the game is over, or the bearoff database
/// covers the position (money: only once no gammon is possible). DMP values use only element 0.
fn exact(money: bool, bearoff: Option<&BearoffDb>, position: &EnginePosition) -> Option<Outcome> {
    if money {
        return exact_outcome(bearoff, position);
    }
    if let Some(o) = terminal_outcome(position) {
        return Some([o[0], 0.0, 0.0, 0.0, 0.0]);
    }
    bearoff.and_then(|db| db.win_probability(position)).map(|p| [p as f32, 0.0, 0.0, 0.0, 0.0])
}

/// The value seen from the side to move as an equity to maximize (DMP: the winning chance).
fn score(money: bool, o: &Outcome) -> f64 {
    if money { money_equity(o) } else { o[0] as f64 }
}

/// The value for the other side.
fn flip(money: bool, o: &Outcome) -> Outcome {
    if money { flip_outcome(o) } else { [1.0 - o[0], 0.0, 0.0, 0.0, 0.0] }
}

fn to_position(c: &[u8]) -> EnginePosition {
    let mut p = EnginePosition::new();
    for point in 0..26 {
        p.set_checkers(Player::Me, point, c[point]).unwrap();
        p.set_checkers(Player::Opponent, point, c[26 + point]).unwrap();
    }
    p.set_turn(Player::Me);
    p
}

struct Master {
    net: NeuralNet,
    average: Vec<f64>,
    games: u64,
    turns: u64,
    own: u64,
    moves: u64,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut types_path, mut dir, mut init) = (PathBuf::from("nets/moe/types.txt"), PathBuf::from("nets/moe/experts"),
                                               PathBuf::from("nets/moe/reference.bgnet"));
    let (mut expert, mut starts_path, mut value_path) = (String::new(), None::<PathBuf>, None::<PathBuf>);
    let (mut games, mut alpha, mut lambda, mut threads, mut seed) = (100_000u64, 0.005f32, 0.7f32, 12usize, 1u64);
    let (mut report, mut average_last, mut bearoff_path) = (500_000u64, 0u64, None::<PathBuf>);
    let mut i = 0;
    while i < args.len() {
        let v = args.get(i + 1).cloned().unwrap_or_default();
        match args[i].as_str() {
            "--types" => types_path = v.into(),
            "--dir" => dir = v.into(),
            "--init" => init = v.into(),
            "--value-net" => value_path = Some(v.into()),
            "--expert" => expert = v,
            "--starts" => starts_path = Some(v.into()),
            "--games" => games = v.parse().expect("--games N"),
            "--alpha" => alpha = v.parse().expect("--alpha A"),
            "--lambda" => lambda = v.parse().expect("--lambda L"),
            "--threads" => threads = v.parse().expect("--threads N"),
            "--seed" => seed = v.parse().expect("--seed N"),
            "--report" => report = v.parse().expect("--report N"),
            "--average-last" => average_last = v.parse().expect("--average-last N"),
            "--bearoff" => bearoff_path = if v == "none" { None } else { Some(v.into()) },
            other => panic!("unknown option {other}"),
        }
        i += 2;
    }
    let types = GameTypes::load(&types_path).unwrap_or_else(|e| panic!("{e}"));
    let e = types.names().iter().position(|n| *n == expert)
        .unwrap_or_else(|| panic!("--expert must be one of {}", types.names().join(", ")));
    let starts_path = starts_path.unwrap_or_else(|| PathBuf::from(format!("nets/moe/positions/{expert}.pos")));
    let starts = std::fs::read(&starts_path).unwrap_or_else(|err| panic!("{}: {err}", starts_path.display()));
    let n_starts = starts.len() / 52;
    assert!(n_starts > 0, "no start positions in {}", starts_path.display());
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{expert}.bgnet"));
    let net = NeuralNet::load(if path.exists() { &path } else { &init }).unwrap_or_else(|err| panic!("{err}"));
    let objective = net.objective();
    let money = objective == Objective::Money;
    // The judge of positions off the expert's type; None: the game types' reference net (computed
    // anyway when classifying).
    let value_net = value_path.as_ref().map(|p| NeuralNet::load(p).unwrap_or_else(|err| panic!("{err}")));
    let off_objective = value_net.as_ref().map_or(types.reference.objective(), |n| n.objective());
    assert_eq!(off_objective, objective, "the value net for positions off the type must have the expert's objective \
               ({}); give one with --value-net", objective.name());
    let start_games = net.games_trained;
    let bearoff = bearoff_path.as_ref().map(|p| BearoffDb::open(p).unwrap_or_else(|err| panic!("{err}")));
    eprintln!("expert {expert} ({}): {} start positions, net {} ({} games so far), off its type {}; alpha {alpha}, \
               lambda {lambda}, {threads} threads", objective.name(),
              n_starts, if path.exists() { path.display().to_string() } else { init.display().to_string() }, start_games,
              value_path.as_ref().map_or("the reference net".to_string(), |p| p.display().to_string()));

    let average_from = games.saturating_sub(average_last) + 1;
    let master = Mutex::new(Master { average: vec![], net, games: 0, turns: 0, own: 0, moves: 0 });
    let next_game = AtomicU64::new(1);
    let started = Instant::now();
    let bit = 1u16 << e;
    let n_values = objective.outputs();
    // Values seen from Me and their sources (see td_training.rs): DMP the winning chance, with
    // sign -1 when the opponent is on roll.
    let (sources, signs) = if money { (&ME_SOURCE, &ME_SIGNS) } else { (&DMP_SOURCE, &DMP_SIGNS) };

    std::thread::scope(|scope| {
        for _ in 0..threads {
            let (master, next_game, types, starts, bearoff, path, value_net) =
                (&master, &next_game, &types, &starts, bearoff.as_ref(), &path, value_net.as_ref());
            scope.spawn(move || {
                let mut net = master.lock().unwrap().net.clone();
                let n_params = net.params().len();
                let mut base = net.params().to_vec();
                let mut grad = vec![0.0f32; n_params];
                // The game so far: the activations of the expert's own positions, and per move
                // the TD error (seen from Me), the side on roll, and which activations it has.
                let mut acts: Vec<backgammon_rust::neural_net::Activations> = Vec::new();
                let (mut errors, mut sides, mut learns) = (Vec::new(), Vec::new(), Vec::new());
                let router = std::cell::RefCell::new(types.reference.new_activations());
                let mut generator = MoveGenerator::<EnginePosition>::new();
                let mut input = [0.0f32; MAX_INPUTS];
                // Whether the position is of the expert's type, and the value net's outcome when it
                // isn't (the expert's is computed by the caller).
                let classify = |p: &EnginePosition| -> (bool, Outcome) {
                    let (mask, reference) = types.classify_with_value(p, &mut router.borrow_mut());
                    if mask & bit != 0 {
                        (true, [0.0; MONEY_OUTPUTS])
                    } else if let Some(v) = value_net {
                        (false, if money { v.outcome(p) } else { [v.win_probability(p), 0.0, 0.0, 0.0, 0.0] })
                    } else {
                        (false, [reference, 0.0, 0.0, 0.0, 0.0])
                    }
                };
                loop {
                    let game = next_game.fetch_add(1, Ordering::Relaxed);
                    if game > games {
                        break;
                    }
                    let mut dice = Dice::new(seed ^ (start_games + game).wrapping_mul(0xd1b54a32d192ed03));
                    let pick = (dice.next_u64() % n_starts as u64) as usize;
                    let mut position = to_position(&starts[pick * 52..pick * 52 + 52]);
                    errors.clear();
                    sides.clear();
                    learns.clear();
                    let mut n_acts = 0;
                    let mut idle = TRACE_STEPS;
                    let (mut own, mut moves, mut turns) = (0u64, 0u64, 0u32);
                    for turn in 1..=MAX_TURNS {
                        turns = turn;
                        let (d1, d2) = (dice.roll(), dice.roll());
                        let (active, off) = classify(&position);
                        let y: Outcome = if active {
                            net.encode(&position, &mut input);
                            if acts.len() <= n_acts {
                                acts.push(net.new_activations());
                            }
                            net.forward_keep(&input, &mut acts[n_acts]);
                            let mut y = [0.0; MONEY_OUTPUTS];
                            y[..n_values].copy_from_slice(acts[n_acts].outputs());
                            y
                        } else {
                            off
                        };
                        let side = (position.get_turn() != Player::Me) as usize;
                        let value = if side == 0 { y } else { flip(money, &y) };
                        let judge = |p: &EnginePosition| -> Outcome {
                            if let Some(x) = exact(money, bearoff, p) {
                                return x;
                            }
                            match classify(p) {
                                (true, _) if money => net.outcome(p),
                                (true, _) => [net.win_probability(p), 0.0, 0.0, 0.0, 0.0],
                                (false, off) => off,
                            }
                        };
                        let next = generator.best_position(&position, d1, d2, |c| -score(money, &judge(c))).map(|(p, _)| p)
                            .unwrap_or_else(|| { let mut p = position.clone(); p.switch_turn(); p });
                        let after = judge(&next);
                        let target = if next.get_turn() == Player::Me { after } else { flip(money, &after) };
                        let mut error = [0.0f32; MONEY_OUTPUTS];
                        for k in 0..n_values {
                            error[k] = target[k] - value[k];
                        }
                        moves += 1;
                        if active {
                            errors.push(error);
                            sides.push(side);
                            learns.push(Some(n_acts));
                            n_acts += 1;
                            idle = 0;
                            own += 1;
                        } else if idle < TRACE_STEPS {
                            idle += 1;
                            if idle == TRACE_STEPS {
                                break;
                            }
                            errors.push(error);
                            sides.push(side);
                            learns.push(None);
                        }
                        if exact(money, bearoff, &next).is_some() {
                            break;
                        }
                        position = next;
                    }

                    learn_offline_from_errors(&mut net, &mut acts, &errors, &sides, &learns, sources, signs,
                                              n_values, alpha, lambda, &mut grad);
                    let mut m = master.lock().unwrap();
                    for ((w, &local), &before) in m.net.params_mut().iter_mut().zip(net.params()).zip(&base) {
                        *w += local - before;
                    }
                    m.games += 1;
                    m.turns += turns as u64;
                    m.own += own;
                    m.moves += moves;
                    let done = m.games;
                    m.net.games_trained = start_games + done;
                    if average_last > 0 && done >= average_from {
                        let Master { net, average, .. } = &mut *m;
                        if average.is_empty() {
                            *average = vec![0.0; net.params().len()];
                        }
                        for (a, &p) in average.iter_mut().zip(net.params()) {
                            *a += p as f64;
                        }
                    }
                    net.params_mut().copy_from_slice(m.net.params());
                    base.copy_from_slice(m.net.params());
                    if done % report == 0 || done == games {
                        m.net.save(path).unwrap();
                        eprintln!("{:>9} games ({:>9} total)  {:.0} games/s  {:.1} turns/game, {:.0}% of the moves on its own type  saved",
                                  done, start_games + done, done as f64 / started.elapsed().as_secs_f64(),
                                  m.turns as f64 / done as f64, 100.0 * m.own as f64 / m.moves.max(1) as f64);
                    }
                }
            });
        }
    });
    let m = master.into_inner().unwrap();
    m.net.save(&path).unwrap();
    if !m.average.is_empty() {
        let k = (m.games + 1 - average_from) as f64;
        let mut avg = m.net.clone();
        for (p, &a) in avg.params_mut().iter_mut().zip(&m.average) {
            *p = (a / k) as f32;
        }
        let avg_path = dir.join(format!("{expert}-avg.bgnet"));
        avg.save(&avg_path).unwrap();
        eprintln!("average of the last {} games saved to {}", average_last, avg_path.display());
    }
}
