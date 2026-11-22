//! TD(λ) self-play training of a `NeuralNet` (as in TD-Gammon).
//!
//! The net plays both sides, choosing each move greedily by its own evaluation (depth 0); the dice
//! provide the variety. After every move the net is nudged so that its estimate for the position
//! before the move moves towards its estimate after the move (or the actual result when the move
//! ends the game). Eligibility traces spread each correction back over the earlier positions of
//! the game, decaying by λ per move: λ = 0 only corrects the last estimate, λ = 1 corrects every
//! earlier estimate towards the final result.
//!
//! The net always estimates the winning chance of the player on roll, so the side it speaks for
//! alternates every move. The update works with the value from a fixed side (Me) instead:
//! W(s) = V(s) when Me is on roll, 1 - V(s) otherwise, and ∇W = ±∇V accordingly. That keeps a
//! single trace for the whole game.
//!
//! With a bearoff database (`TdTrainer::bearoff`), a game stops as soon as a move reaches a
//! bearoff position: the exact winning chance from the database is the final target, like the
//! result of a finished game, and moves into bearoff positions are chosen by the database values.
//! The net then never trains on bearoff positions (it doesn't need to: the engine looks them up),
//! but learns the positions before them against exact values instead of its own estimates.
//!
//! Two ways to apply the update `δ·e` (TD error times trace) to the weights:
//! - `Optimizer::Sgd` (classic TD(λ)): w += α·δ·e, the same step size for every weight.
//! - `Optimizer::Adam`: δ·e is fed to Adam as if it were a (negated) gradient, giving every weight
//!   its own step size from running averages of its updates (β1 0.9, β2 0.999, ε 1e-8). The moments
//!   are not saved in the weight file; a resumed run starts them from zero. Needs a much smaller α
//!   (around 1e-4) than SGD.
use std::sync::Arc;

use crate::bearoff::BearoffDb;
use crate::evaluator::Evaluator;
use crate::game::{self, Dice, Variant, MAX_TURNS};
use crate::move_generator::MoveGenerator;
use crate::neural_net::{Activations, NeuralNet};
use crate::encoding::MAX_INPUTS;
use crate::position::{Player, Position};
use crate::EnginePosition;

const CHECKERS_PER_PLAYER: u8 = 15;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Optimizer {
    Sgd,
    Adam,
}

const ADAM_BETA1: f32 = 0.9;
const ADAM_BETA2: f32 = 0.999;
const ADAM_EPSILON: f32 = 1e-8;

/// Adam's running averages of the updates and of their squares, plus the step count.
struct AdamState {
    m: Vec<f32>,
    v: Vec<f32>,
    t: i32,
    beta1_power: f32,
    beta2_power: f32,
}

pub struct TdTrainer {
    pub net: NeuralNet,
    /// Learning rate.
    pub alpha: f32,
    /// Trace decay per move.
    pub lambda: f32,
    /// Starting position of the self-play games.
    pub variant: Variant,
    /// When set, games stop at the first bearoff position, with its exact value as the target.
    pub bearoff: Option<Arc<BearoffDb>>,
    trace: Vec<f32>,
    grad: Vec<f32>,
    generator: MoveGenerator<EnginePosition>,
    adam: Option<AdamState>,
    acts: Activations,
}

/// The exact winning chance for the player on roll, when the game is over or the bearoff
/// database covers the position.
fn exact_win_probability(bearoff: Option<&BearoffDb>, position: &EnginePosition) -> Option<f32> {
    if let Some(winner) = winner(position) {
        return Some(if winner == position.get_turn() { 1.0 } else { 0.0 });
    }
    bearoff.and_then(|db| db.win_probability(position)).map(|p| p as f32)
}

/// Value of a position for Me (winning chance): exact when known, else the net's estimate for
/// the player on roll.
fn value_for_me(net: &NeuralNet, bearoff: Option<&BearoffDb>, position: &EnginePosition) -> f32 {
    let v = exact_win_probability(bearoff, position).unwrap_or_else(|| net.win_probability(position));
    if position.get_turn() == Player::Me { v } else { 1.0 - v }
}

fn winner<P: Position>(position: &P) -> Option<Player> {
    [Player::Me, Player::Opponent].into_iter().find(|&p| position.count(p, 0) == CHECKERS_PER_PLAYER)
}

/// The mover's choice with the given dice, choosing by `score` (equity of the child for its side
/// to move, i.e. the opponent); passes when no move is legal.
fn choose_move<F: Fn(&EnginePosition) -> f64>(
    generator: &mut MoveGenerator<EnginePosition>, position: &EnginePosition, die1: i32, die2: i32, equity: F,
) -> EnginePosition {
    generator
        .best_position(position, die1, die2, |child| -equity(child))
        .map(|(p, _)| p)
        .unwrap_or_else(|| {
            let mut passed = position.clone();
            passed.switch_turn();
            passed
        })
}

impl TdTrainer {
    pub fn new(net: NeuralNet, alpha: f32, lambda: f32) -> Self {
        Self::with_optimizer(net, alpha, lambda, Optimizer::Sgd)
    }

    pub fn with_optimizer(net: NeuralNet, alpha: f32, lambda: f32, optimizer: Optimizer) -> Self {
        let n = net.param_count();
        let adam = (optimizer == Optimizer::Adam).then(|| AdamState {
            m: vec![0.0; n], v: vec![0.0; n], t: 0, beta1_power: 1.0, beta2_power: 1.0,
        });
        let acts = net.new_activations();
        TdTrainer { net, alpha, lambda, variant: Variant::Backgammon, bearoff: None, trace: vec![0.0; n], grad: vec![0.0; n], generator: MoveGenerator::new(), adam, acts }
    }

    pub fn optimizer(&self) -> Optimizer {
        if self.adam.is_some() { Optimizer::Adam } else { Optimizer::Sgd }
    }

    /// Play one self-play game and learn from it. Returns (winner, number of turns); a game stopped
    /// at a bearoff position counts as won by the side with the better chance.
    pub fn train_game(&mut self, dice: &mut Dice) -> Result<(Player, u32), String> {
        let TdTrainer { net, alpha, lambda, variant, bearoff, trace, grad, generator, adam, acts } = self;
        let bearoff = bearoff.as_deref();
        trace.iter_mut().for_each(|e| *e = 0.0);

        let mut position = variant.starting_position();
        let (starter, mut die1, mut die2) = game::opening_roll(dice);
        position.set_turn(starter);
        let mut input = [0.0f32; MAX_INPUTS];

        for turn in 1..=MAX_TURNS {
            // Estimate and gradient for the position before the move, as seen by Me.
            net.encode(&position, &mut input);
            // SGD: activations now, trace and weights in one pass once the TD error is known.
            // Adam: the full gradient now (its step needs the updated trace).
            let v = match adam {
                None => net.forward_keep(&input, acts),
                Some(_) => net.forward_with_gradient(&input, grad),
            };
            let (value, sign) = if position.get_turn() == Player::Me { (v, 1.0) } else { (1.0 - v, -1.0) };
            if adam.is_some() {
                for (e, g) in trace.iter_mut().zip(grad.iter()) {
                    *e = *lambda * *e + sign * g;
                }
            }

            let next = choose_move(generator, &position, die1, die2, |child| {
                exact_win_probability(bearoff, child).map_or_else(|| net.equity(child), |p| 2.0 * p as f64 - 1.0)
            });

            // Move the estimate towards the one after the move (the exact value, if the game ended
            // or reached the bearoff database).
            let target = value_for_me(net, bearoff, &next);
            let error = target - value;
            match adam {
                None => net.td_update(acts, trace, sign, *lambda, *alpha * error),
                Some(state) => adam_step(state, net.params_mut(), trace, error, *alpha),
            }

            if winner(&next).is_some() || bearoff.is_some_and(|db| db.win_probability(&next).is_some()) {
                net.games_trained += 1;
                return Ok((if target >= 0.5 { Player::Me } else { Player::Opponent }, turn));
            }
            position = next;
            die1 = dice.roll();
            die2 = dice.roll();
        }
        Err(format!("training game did not finish within {} turns", MAX_TURNS))
    }
}

/// One Adam step with update direction `error·trace` (the TD update; ascent, hence `+=`).
fn adam_step(state: &mut AdamState, params: &mut [f32], trace: &[f32], error: f32, alpha: f32) {
    state.t += 1;
    state.beta1_power *= ADAM_BETA1;
    state.beta2_power *= ADAM_BETA2;
    // Bias correction folded into the step size and epsilon.
    let step = alpha * (1.0 - state.beta2_power).sqrt() / (1.0 - state.beta1_power);
    let epsilon = ADAM_EPSILON * (1.0 - state.beta2_power).sqrt();
    for (((p, &e), m), v) in params.iter_mut().zip(trace).zip(state.m.iter_mut()).zip(state.v.iter_mut()) {
        let u = error * e;
        *m = ADAM_BETA1 * *m + (1.0 - ADAM_BETA1) * u;
        *v = ADAM_BETA2 * *v + (1.0 - ADAM_BETA2) * u * u;
        *p += step * *m / (v.sqrt() + epsilon);
    }
}

/// Play one game between two evaluators at depth 0, `me` playing Me. Returns the winner.
pub fn play_evaluators(
    me: &dyn Evaluator<EnginePosition>,
    opponent: &dyn Evaluator<EnginePosition>,
    generator: &mut MoveGenerator<EnginePosition>,
    dice: &mut Dice,
) -> Result<Player, String> {
    let mut position = EnginePosition::new();
    position.setup_starting_position();
    let (starter, mut die1, mut die2) = game::opening_roll(dice);
    position.set_turn(starter);
    for _ in 0..MAX_TURNS {
        let evaluator = if position.get_turn() == Player::Me { me } else { opponent };
        position = choose_move(generator, &position, die1, die2, |child| evaluator.evaluate(child));
        if let Some(winner) = winner(&position) {
            return Ok(winner);
        }
        die1 = dice.roll();
        die2 = dice.roll();
    }
    Err(format!("game did not finish within {} turns", MAX_TURNS))
}

/// Win rate of `candidate` against `reference` over `pairs` pairs of games: in each pair both
/// play each side once with the same dice seed, which cancels much of the luck.
pub fn win_rate(
    candidate: &dyn Evaluator<EnginePosition>,
    reference: &dyn Evaluator<EnginePosition>,
    pairs: u32,
    seed: u64,
) -> Result<f64, String> {
    let mut generator = MoveGenerator::new();
    let mut wins = 0;
    for pair in 0..pairs {
        let pair_seed = seed.wrapping_mul(1_000_003).wrapping_add(pair as u64);
        if play_evaluators(candidate, reference, &mut generator, &mut Dice::new(pair_seed))? == Player::Me {
            wins += 1;
        }
        if play_evaluators(reference, candidate, &mut generator, &mut Dice::new(pair_seed))? == Player::Opponent {
            wins += 1;
        }
    }
    Ok(wins as f64 / (2 * pairs) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluator::{PipCountEvaluator, TerminalEvaluator};
    use crate::encoding::Encoding;
    use crate::neural_net::{NeuralNetEvaluator, DEFAULT_HIDDEN};

    #[test]
    fn test_training_game_finishes_and_changes_the_net() {
        let net = NeuralNet::new_random(Encoding::CountsPips, &DEFAULT_HIDDEN, 1);
        let before = net.clone();
        let mut trainer = TdTrainer::new(net, 0.01, 0.7);
        let (_, turns) = trainer.train_game(&mut Dice::new(1)).unwrap();
        assert!(turns > 10);
        assert_eq!(trainer.net.games_trained, 1);
        let mut p = EnginePosition::new();
        p.setup_starting_position();
        assert_ne!(before.win_probability(&p), trainer.net.win_probability(&p));
    }

    #[test]
    fn test_games_stop_at_the_bearoff_database() {
        // The real database: with a smaller one the side that hasn't borne off yet nearly always
        // has too many checkers left. Skipped when it hasn't been built.
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/bearoff-6x15.bd");
        let Ok(db) = BearoffDb::open(&path) else {
            eprintln!("skipped: {} not built", path.display());
            return;
        };
        let db = Arc::new(db);
        let net = NeuralNet::new_random(Encoding::TdCounts, &[16], 1);
        let mut plain = TdTrainer::new(net.clone(), 0.01, 0.7);
        let mut stopping = TdTrainer::new(net, 0.01, 0.7);
        stopping.bearoff = Some(db);
        let (mut turns_plain, mut turns_stopping) = (0, 0);
        for seed in 0..20 {
            turns_plain += plain.train_game(&mut Dice::new(seed)).unwrap().1;
            turns_stopping += stopping.train_game(&mut Dice::new(seed)).unwrap().1;
        }
        assert_eq!(stopping.net.games_trained, 20);
        assert!(turns_stopping < turns_plain, "{} vs {}", turns_stopping, turns_plain);
    }

    #[test]
    fn test_nackgammon_training_game_finishes() {
        let mut trainer = TdTrainer::new(NeuralNet::new_random(Encoding::TdCounts, &[16], 1), 0.01, 0.7);
        trainer.variant = Variant::Nackgammon;
        let (_, turns) = trainer.train_game(&mut Dice::new(3)).unwrap();
        assert!(turns > 10);
        assert_eq!(trainer.net.games_trained, 1);
    }

    #[test]
    fn test_short_training_beats_the_placeholder() {
        let mut trainer = TdTrainer::new(NeuralNet::new_random(Encoding::CountsPips, &DEFAULT_HIDDEN, 2), 0.01, 0.7);
        let mut dice = Dice::new(2);
        for _ in 0..300 {
            trainer.train_game(&mut dice).unwrap();
        }
        let candidate = NeuralNetEvaluator { net: trainer.net.clone() };
        assert!(win_rate(&candidate, &TerminalEvaluator, 100, 9).unwrap() > 0.6);
    }

    #[test]
    fn test_short_adam_training_beats_the_placeholder() {
        let net = NeuralNet::new_random(Encoding::CountsPips, &DEFAULT_HIDDEN, 2);
        let mut trainer = TdTrainer::with_optimizer(net, 1e-4, 0.7, Optimizer::Adam);
        assert_eq!(trainer.optimizer(), Optimizer::Adam);
        let mut dice = Dice::new(2);
        for _ in 0..300 {
            trainer.train_game(&mut dice).unwrap();
        }
        assert!(trainer.net.params().iter().all(|w| w.is_finite()));
        let candidate = NeuralNetEvaluator { net: trainer.net.clone() };
        assert!(win_rate(&candidate, &TerminalEvaluator, 100, 9).unwrap() > 0.6);
    }

    #[test]
    fn test_adam_first_step_moves_each_weight_by_alpha() {
        // With bias correction, Adam's first step is α·sign(update) for every non-zero update.
        let mut state = AdamState { m: vec![0.0; 3], v: vec![0.0; 3], t: 0, beta1_power: 1.0, beta2_power: 1.0 };
        let mut params = vec![1.0f32, 1.0, 1.0];
        adam_step(&mut state, &mut params, &[0.5, -2.0, 0.0], 0.1, 0.01);
        assert!((params[0] - 1.01).abs() < 1e-5);
        assert!((params[1] - 0.99).abs() < 1e-5);
        assert_eq!(params[2], 1.0);
    }

    #[test]
    fn test_win_rate_of_an_evaluator_against_itself_is_even() {
        // Paired games with identical players mirror each other exactly.
        assert_eq!(win_rate(&PipCountEvaluator, &PipCountEvaluator, 20, 1).unwrap(), 0.5);
    }
}
