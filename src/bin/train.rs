//! Train a neural net evaluator with TD(λ) self-play.
//!
//!     cargo run --release --bin train -- --games 200000 --out nets/td.bgnet
//!     cargo run --release --bin train -- --resume nets/td.bgnet --games 100000
//!     cargo run --release --bin train -- --resume nets/td.bgnet --games 1000000 --alpha 0.01 --alpha-final 0.001
//!
//! With `--alpha-final` the learning rate decays exponentially from `--alpha` (first game) to
//! `--alpha-final` (last game); without it the rate stays at `--alpha`.
//!
//! `--keep-checkpoints` also keeps a copy of every report's net, named after the total number of
//! games trained (e.g. nets/td-250k.bgnet), so the learning curve can be benchmarked afterwards.
//!
//! `--optimizer adam` applies the TD updates with Adam instead of plain SGD (see td_training.rs);
//! use a much smaller `--alpha`, e.g. 0.0001.
//!
//! `--quantize f16|i8`: quantization-aware training; every game is played and judged by the net's
//! f16 or int8 evaluation copy, made at the start of the game (see td_training.rs). The saved
//! nets are f32; make their evaluation copy with `make_eval_net --precision f16|i8`.
//!
//! `--search f16`: money nets only; the net's f16 evaluation copy (made at the start of every
//! game) chooses the moves, about 2.5x faster than the f32 net, while the values, targets and
//! gradients of the TD updates stay the f32 net's (see td_training.rs).
//!
//! `--objective money` trains a new net with 5 outputs (win, win gammon, win backgammon, lose
//! gammon, lose backgammon) for cubeless money play; the default `dmp` has one output, the
//! winning chance. A resumed net keeps its own objective.
//!
//! `--bearoff PATH` stops every game at its first bearoff position and uses the exact winning
//! chance from that two-sided bearoff database as the final target (see td_training.rs).
//!
//! `--threads N` plays games on N threads in parallel, each on its own copy of the net, merging its
//! weight changes into the shared net after every game (see `train_parallel`). Much faster, but
//! not reproducible; with the default of 1 thread a run is reproducible for a given seed.
//!
//! `--perturb SIGMA` / `--shrink S` change the starting weights first ("shrink and perturb"):
//! w = (1 - S)·w + SIGMA·std(layer)·N(0, 1), with the noise drawn from `--seed`. For branching a
//! plateaued net into copies that are trained further.
//!
//! `--encoding counts|td|flags|nack` picks the input encoding of a new net (see encoding.rs; default
//! td since 2026-11-20). A resumed net keeps its own, unless `--encoding` names one that extends it
//! (td -> flags): then the extra inputs are added with zero weights.
//!
//! `--variant nackgammon` starts the self-play games from the nackgammon position instead of the
//! backgammon one; `--variant mixed` alternates the two (odd games backgammon, even nackgammon) (the progress games against the pip-count heuristic stay backgammon).
//!
//! `--average-last N` also keeps the running mean of the weights over the last N games and saves
//! it as `<out stem>-avg.bgnet` at the end (Polyak averaging). With a constant learning rate the
//! weights keep drifting, so a single snapshot's strength is partly luck; the average is steadier.
//!
//! Every `--report` games the net is saved to `--out` and played against the pip-count heuristic
//! (`--eval-pairs` pairs of games with mirrored dice) to show progress. For a real strength
//! measurement use the benchmark against bgsage: `make strength ARGS="--net nets/td.bgnet"`.
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Mutex};
use std::time::Instant;

use backgammon_rust::encoding::Encoding;
use backgammon_rust::evaluator::PipCountEvaluator;
use backgammon_rust::game::{Dice, Variant};
use backgammon_rust::neural_net::{NeuralNet, NeuralNetEvaluator, Objective, DEFAULT_HIDDEN};
use backgammon_rust::td_training::{win_rate, Optimizer, TdTrainer};

struct Options {
    games: u64,
    alpha: f32,
    alpha_final: Option<f32>,
    lambda: f32,
    seed: u64,
    hidden: Vec<usize>,
    objective: Objective,
    encoding: Option<Encoding>,
    /// None: mixed, alternating backgammon and nackgammon.
    variant: Option<Variant>,
    out: PathBuf,
    resume: Option<PathBuf>,
    report: u64,
    eval_pairs: u32,
    keep_checkpoints: bool,
    average_last: u64,
    optimizer: Optimizer,
    threads: usize,
    bearoff: Option<PathBuf>,
    quantize: Option<backgammon_rust::eval_net::Kind>,
    search: Option<backgammon_rust::eval_net::Kind>,
    /// The opened `bearoff` database.
    bearoff_db: Option<std::sync::Arc<backgammon_rust::bearoff::BearoffDb>>,
    perturb: f32,
    shrink: f32,
}

const USAGE: &str = "usage: train [--games N] [--alpha A] [--alpha-final A] [--lambda L] [--seed S] [--hidden 128,128,128]
             [--objective dmp|money] [--encoding counts|td|flags|nack] [--variant backgammon|nackgammon|mixed]
             [--out FILE] [--resume FILE] [--report N] [--eval-pairs N] [--keep-checkpoints]
             [--average-last N] [--optimizer sgd|adam] [--threads N] [--bearoff data/bearoff-6x15.bd]
             [--quantize f16|i8] [--search f16] [--perturb SIGMA] [--shrink S]";

fn parse_args() -> Result<Options, String> {
    let mut o = Options {
        games: 100_000,
        alpha: 0.01,
        alpha_final: None,
        lambda: 0.7,
        seed: 1,
        hidden: DEFAULT_HIDDEN.to_vec(),
        objective: Objective::Dmp,
        encoding: None,
        variant: Some(Variant::Backgammon),
        out: PathBuf::from("nets/td.bgnet"),
        resume: None,
        report: 5_000,
        eval_pairs: 500,
        keep_checkpoints: false,
        average_last: 0,
        optimizer: Optimizer::Sgd,
        threads: 1,
        bearoff: None,
        quantize: None,
        search: None,
        bearoff_db: None,
        perturb: 0.0,
        shrink: 0.0,
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        if flag == "--help" || flag == "-h" {
            return Err(USAGE.to_string());
        }
        if flag == "--keep-checkpoints" {
            o.keep_checkpoints = true;
            i += 1;
            continue;
        }
        let value = args.get(i + 1).ok_or_else(|| format!("{} needs a value\n{}", flag, USAGE))?;
        let bad = |e: &dyn std::fmt::Display| format!("bad value for {}: {}", flag, e);
        match flag {
            "--games" => o.games = value.parse().map_err(|e| bad(&e))?,
            "--alpha" => o.alpha = value.parse().map_err(|e| bad(&e))?,
            "--alpha-final" => o.alpha_final = Some(value.parse().map_err(|e| bad(&e))?),
            "--lambda" => o.lambda = value.parse().map_err(|e| bad(&e))?,
            "--seed" => o.seed = value.parse().map_err(|e| bad(&e))?,
            "--hidden" => {
                o.hidden = value.split(',').map(|s| s.trim().parse::<usize>()).collect::<Result<_, _>>().map_err(|e| bad(&e))?
            }
            "--objective" => o.objective = Objective::from_name(value)
                .ok_or_else(|| format!("bad value for --objective: {} (dmp or money)", value))?,
            "--encoding" => o.encoding = Some(match value.as_str() {
                "counts" => Encoding::CountsPips,
                "td" => Encoding::TdCounts,
                "flags" => Encoding::TdFlags,
                "nack" => Encoding::TdNack,
                name => Encoding::from_name(name).ok_or_else(|| format!("unknown encoding '{}'\n{}", name, USAGE))?,
            }),
            "--variant" => o.variant = match value.as_str() {
                "backgammon" => Some(Variant::Backgammon),
                "nackgammon" => Some(Variant::Nackgammon),
                "mixed" => None,
                name => return Err(format!("unknown variant '{}'\n{}", name, USAGE)),
            },
            "--out" => o.out = PathBuf::from(value),
            "--resume" => o.resume = Some(PathBuf::from(value)),
            "--report" => o.report = value.parse().map_err(|e| bad(&e))?,
            "--optimizer" => o.optimizer = match value.as_str() {
                "sgd" => Optimizer::Sgd,
                "adam" => Optimizer::Adam,
                _ => return Err(format!("bad value for --optimizer: {} (sgd or adam)", value)),
            },
            "--perturb" => o.perturb = value.parse().map_err(|e| bad(&e))?,
            "--shrink" => o.shrink = value.parse().map_err(|e| bad(&e))?,
            "--threads" => o.threads = value.parse().map_err(|e| bad(&e))?,
            "--bearoff" => o.bearoff = Some(PathBuf::from(value)),
            "--quantize" => o.quantize = Some(match value.as_str() {
                "f16" => backgammon_rust::eval_net::Kind::F16,
                "i8" => backgammon_rust::eval_net::Kind::I8,
                _ => return Err(format!("--quantize f16 or i8, not {}", value)),
            }),
            "--search" => o.search = Some(match value.as_str() {
                "f16" => backgammon_rust::eval_net::Kind::F16,
                _ => return Err(format!("--search f16, not {}", value)),
            }),
            "--average-last" => o.average_last = value.parse().map_err(|e| bad(&e))?,
            "--eval-pairs" => o.eval_pairs = value.parse().map_err(|e| bad(&e))?,
            _ => return Err(format!("unknown option {}\n{}", flag, USAGE)),
        }
        i += 2;
    }
    Ok(o)
}

/// Saves, checkpoints and progress lines at every `--report` games, shared by both modes.
struct Reporter<'a> {
    o: &'a Options,
    started: Instant,
}

impl Reporter<'_> {
    fn report(&self, net: &NeuralNet, game: u64, alpha: f32, turns_total: u64) {
        let o = self.o;
        let elapsed = self.started.elapsed().as_secs_f64();
        net.save(&o.out).unwrap_or_else(|e| eprintln!("could not save: {}", e));
        if o.keep_checkpoints {
            let total = net.games_trained;
            let label = if total % 1000 == 0 { format!("{}k", total / 1000) } else { total.to_string() };
            net.save(&self.sibling(&label)).unwrap_or_else(|e| eprintln!("could not save: {}", e));
        }
        let candidate = NeuralNetEvaluator { net: net.clone() };
        let vs_pips = win_rate(&candidate, &PipCountEvaluator, o.eval_pairs, 12345).unwrap_or(f64::NAN);
        println!("{:>9} games ({:>9} total)  alpha {:.5}  {:6.0} games/s  {:5.1} turns/game  vs pip count {:5.1}%  saved",
                 game, net.games_trained, alpha, game as f64 / elapsed,
                 turns_total as f64 / game as f64, 100.0 * vs_pips);
    }

    fn save_average(&self, net: &NeuralNet, average: &[f64], games: u64) {
        let mut averaged = net.clone();
        for (w, &a) in averaged.params_mut().iter_mut().zip(average) {
            *w = a as f32;
        }
        let path = self.sibling("avg");
        averaged.save(&path).unwrap_or_else(|e| eprintln!("could not save: {}", e));
        let vs_pips = win_rate(&NeuralNetEvaluator { net: averaged }, &PipCountEvaluator, self.o.eval_pairs, 12345)
            .unwrap_or(f64::NAN);
        println!("average of the last {} games saved to {}  vs pip count {:5.1}%", games, path.display(), 100.0 * vs_pips);
    }

    /// `<out stem>-<suffix>.bgnet` next to `--out`.
    fn sibling(&self, suffix: &str) -> PathBuf {
        let stem = self.o.out.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        self.o.out.with_file_name(format!("{}-{}.bgnet", stem, suffix))
    }
}

/// Learning rate for game `game` (1-based) of the run.
fn alpha_for(o: &Options, game: u64) -> f32 {
    match o.alpha_final {
        Some(last) => {
            let progress = if o.games > 1 { (game - 1) as f32 / (o.games - 1) as f32 } else { 1.0 };
            o.alpha * (last / o.alpha).powf(progress)
        }
        None => o.alpha,
    }
}

/// Folds `params` into the running mean `average` as its `k`-th value.
fn add_to_average(average: &mut Vec<f64>, params: &[f32], k: u64) {
    if average.is_empty() {
        *average = vec![0.0; params.len()];
    }
    for (a, &w) in average.iter_mut().zip(params) {
        *a += (w as f64 - *a) / k as f64;
    }
}

fn exit_on_error<T>(result: Result<T, String>) -> T {
    result.unwrap_or_else(|e| {
        eprintln!("{}", e);
        std::process::exit(1);
    })
}

fn main() {
    let mut o = match parse_args() {
        Ok(o) => o,
        Err(message) => {
            eprintln!("{}", message);
            std::process::exit(2);
        }
    };
    let mut net = match &o.resume {
        Some(path) => {
            let net = exit_on_error(NeuralNet::load(path));
            match o.encoding {
                Some(encoding) if encoding != net.encoding() => {
                    println!("Extending the inputs from {} to {} (new inputs start at zero weight)",
                             net.encoding().name(), encoding.name());
                    exit_on_error(net.extend_inputs(encoding))
                }
                _ => net,
            }
        }
        None => NeuralNet::new_random_for(o.encoding.unwrap_or(Encoding::TdCounts), &o.hidden, o.objective, o.seed),
    };
    if o.perturb > 0.0 || o.shrink > 0.0 {
        net.perturb(o.shrink, o.perturb, o.seed ^ 0x5eed);
        println!("Perturbed the starting weights: shrink {}, noise {} x layer std (seed {})", o.shrink, o.perturb, o.seed);
    }
    let schedule = match o.alpha_final {
        Some(last) => format!("alpha {} -> {} (exponential decay)", o.alpha, last),
        None => format!("alpha {}", o.alpha),
    };
    println!("Training {} for {} {} games: {:?}, {}, lambda {}, seed {}, {} thread{} -> {}",
             net.description(), o.games, o.variant.map_or("mixed".to_string(), |v| format!("{:?}", v)), o.optimizer, schedule, o.lambda, o.seed, o.threads,
             if o.threads == 1 { "" } else { "s" }, o.out.display());
    if let Some(path) = &o.bearoff {
        o.bearoff_db = Some(std::sync::Arc::new(exit_on_error(backgammon_rust::bearoff::BearoffDb::open(path))));
        println!("Games stop at bearoff positions, with the exact value from {}", path.display());
    }
    if let Some(kind) = o.quantize {
        println!("Quantization-aware: games played and judged by the {} evaluation copy", kind.name());
    }
    if let Some(kind) = o.search {
        println!("Moves chosen by the {} evaluation copy (made per game); TD updates in f32", kind.name());
    }
    let o = o;
    let reporter = Reporter { o: &o, started: Instant::now() };
    if o.threads <= 1 {
        train_serial(&o, net, &reporter);
    } else {
        train_parallel(&o, net, &reporter);
    }
}

/// Starting position of training game number `game` (counted over all runs of the net).
fn variant_for(o: &Options, game: u64) -> Variant {
    o.variant.unwrap_or(if game % 2 == 1 { Variant::Backgammon } else { Variant::Nackgammon })
}

/// One thread, one dice stream: reproducible for a given seed.
fn train_serial(o: &Options, net: NeuralNet, reporter: &Reporter) {
    let start_games = net.games_trained;
    let mut trainer = TdTrainer::with_optimizer(net, o.alpha, o.lambda, o.optimizer);
    trainer.bearoff = o.bearoff_db.clone();
    trainer.quantize = o.quantize;
    trainer.search = o.search;
    // Continue the dice where an earlier run stopped, so a resumed run doesn't replay its games.
    let mut dice = Dice::new(o.seed.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(start_games));
    let mut turns_total = 0u64;
    // Running mean of the weights over the last `average_last` games.
    let average_from = o.games.saturating_sub(o.average_last) + 1;
    let mut average: Vec<f64> = Vec::new();
    for game in 1..=o.games {
        trainer.alpha = alpha_for(o, game);
        trainer.variant = variant_for(o, start_games + game);
        let (_, turns) = exit_on_error(trainer.train_game(&mut dice));
        turns_total += turns as u64;
        if o.average_last > 0 && game >= average_from {
            add_to_average(&mut average, trainer.net.params(), game - average_from + 1);
        }
        if game % o.report == 0 || game == o.games {
            reporter.report(&trainer.net, game, trainer.alpha, turns_total);
        }
    }
    if !average.is_empty() {
        reporter.save_average(&trainer.net, &average, o.games - average_from + 1);
    }
}

/// The master net and what is kept about it, shared by the worker threads.
struct Master {
    net: NeuralNet,
    games_done: u64,
    turns_total: u64,
    average: Vec<f64>,
}

/// `--threads N`: every thread plays whole games on its own copy of the net, learning as usual
/// during the game. After each game it adds its weight changes (local weights minus the weights it
/// started the game with) to the master net and continues from a fresh copy of the master. Other
/// threads' games land in the master meanwhile, so a game is learned on weights that are at most a
/// few games old. Not reproducible: the order in which games reach the master varies.
///
/// Each game gets its own dice from (seed, game number), independent of which thread plays it.
fn train_parallel(o: &Options, net: NeuralNet, reporter: &Reporter) {
    let start_games = net.games_trained;
    let average_from = o.games.saturating_sub(o.average_last) + 1;
    let master = Mutex::new(Master { net, games_done: 0, turns_total: 0, average: Vec::new() });
    let next_game = AtomicU64::new(1);
    let (snapshots, reports) = mpsc::channel::<(NeuralNet, u64, f32, u64)>();

    std::thread::scope(|scope| {
        for _ in 0..o.threads {
            let snapshots = snapshots.clone();
            let (master, next_game) = (&master, &next_game);
            scope.spawn(move || {
                let start = master.lock().unwrap().net.clone();
                let mut base = start.params().to_vec();
                let mut trainer = TdTrainer::with_optimizer(start, o.alpha, o.lambda, o.optimizer);
                trainer.bearoff = o.bearoff_db.clone();
                trainer.quantize = o.quantize;
                trainer.search = o.search;
                loop {
                    let game = next_game.fetch_add(1, Ordering::Relaxed);
                    if game > o.games {
                        break;
                    }
                    trainer.alpha = alpha_for(o, game);
                    trainer.variant = variant_for(o, start_games + game);
                    let mut dice = Dice::new(game_seed(o.seed, start_games + game));
                    let (_, turns) = exit_on_error(trainer.train_game(&mut dice));

                    let mut m = master.lock().unwrap();
                    for ((w, &local), &before) in m.net.params_mut().iter_mut().zip(trainer.net.params()).zip(&base) {
                        *w += local - before;
                    }
                    m.net.games_trained += 1;
                    m.games_done += 1;
                    m.turns_total += turns as u64;
                    let done = m.games_done;
                    if o.average_last > 0 && done >= average_from {
                        let Master { net, average, .. } = &mut *m;
                        add_to_average(average, net.params(), done - average_from + 1);
                    }
                    trainer.net.params_mut().copy_from_slice(m.net.params());
                    base.copy_from_slice(m.net.params());
                    if done % o.report == 0 || done == o.games {
                        let _ = snapshots.send((m.net.clone(), done, trainer.alpha, m.turns_total));
                    }
                }
            });
        }
        drop(snapshots);
        // Reports (saving, playing against the pip count) run here while the workers go on.
        for (net, done, alpha, turns_total) in reports {
            reporter.report(&net, done, alpha, turns_total);
        }
    });

    let m = master.into_inner().unwrap();
    m.net.save(&o.out).unwrap_or_else(|e| eprintln!("could not save: {}", e));
    if !m.average.is_empty() {
        reporter.save_average(&m.net, &m.average, o.games - average_from + 1);
    }
}

/// Dice seed for one game: a hash of the run's seed and the game's number, so that games get
/// unrelated dice (seeding with seed + game would give shifted copies of one stream).
fn game_seed(seed: u64, game: u64) -> u64 {
    Dice::new(seed ^ game.wrapping_mul(0xd1b54a32d192ed03)).next_u64()
}
