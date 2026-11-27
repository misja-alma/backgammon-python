//! TD(λ) self-play training of a `NeuralNet` (as in TD-Gammon).
//!
//! The net plays both sides, choosing each move greedily by its own evaluation (depth 0); the dice
//! provide the variety. The net's estimate for each position should move towards its estimate
//! after the move (or the actual result when the move ends the game). Eligibility traces spread
//! each correction back over the earlier positions of the game, decaying by λ per move: λ = 0 only
//! corrects the last estimate, λ = 1 corrects every earlier estimate towards the final result.
//!
//! Offline TD(λ) (SGD, since 2026-11-27): the weights stay fixed during a game, and change at its
//! end by exactly what online TD(λ) with fixed weights would add up to (see `learn_offline`):
//! per position one sparse back-propagation of its λ-return error, instead of updating dense
//! traces of all parameters (one plane per output) after every move. For a 512-256 money net that
//! made training 5x faster (the traces were 80% of the time); the next turn's estimate is also
//! the previous position's target, so the chosen move needs no extra evaluation. TD-Gammon updated
//! online; with α 0.0025 a game's updates are far too small for the difference to matter.
//! Adam keeps the online form.
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
//! Quantization-aware training (`TdTrainer::quantize`): the weights stay f32 (TD steps are far
//! smaller than the resolution of f16 or int8 weights), but at the start of every game the net is
//! converted to its low-precision evaluation copy (see `eval_net`), and that copy chooses the
//! moves and gives both the estimate and the target of every TD error. The gradient is still the
//! f32 net's (a straight-through estimate of the copy's). The errors are computed with the copy
//! from the start of the game. The net is thus trained to be good *as its evaluation copy*.
//!
//! Money nets can choose their moves with an f16 copy instead (`TdTrainer::search`, made at the
//! start of every game, about 2.5x faster than the f32 net): only the move choice changes; the
//! values, targets and gradients stay the f32 net's.
//!
//! Money nets (`Objective::Money`, 5 outputs: win, win gammon, win backgammon, lose gammon, lose
//! backgammon for the player on roll) are trained the same way, with a trace per output: seen
//! from Me, the 5 values are the outputs when Me is on roll and (1 - win, lose gammon, lose
//! backgammon, win gammon, win backgammon) otherwise. Moves are chosen by cubeless money equity,
//! a finished game's target is its result (e.g. a gammon win: 1, 1, 0, 0, 0), and a game stops at
//! a bearoff position only when no gammon is possible any more (both sides have borne off a
//! checker): the target is then the database's winning chance with no gammons. SGD only, without
//! quantization-aware training.
//!
//! Two ways to apply the update `δ·e` (TD error times trace) to the weights:
//! - `Optimizer::Sgd` (classic TD(λ)): w += α·δ·e, the same step size for every weight.
//! - `Optimizer::Adam`: δ·e is fed to Adam as if it were a (negated) gradient, giving every weight
//!   its own step size from running averages of its updates (β1 0.9, β2 0.999, ε 1e-8). The moments
//!   are not saved in the weight file; a resumed run starts them from zero. Needs a much smaller α
//!   (around 1e-4) than SGD.
use std::sync::Arc;

use crate::bearoff::BearoffDb;
use crate::eval_net::{EvalNet, Kind, INPUT_SCALE};
use crate::evaluator::{
    flip_outcome, gammon_possible, money_equity, terminal_equity, terminal_outcome, Evaluator, MONEY_OUTPUTS,
};
use crate::game::{self, Dice, Variant, MAX_TURNS};
use crate::move_generator::MoveGenerator;
use crate::neural_net::{Activations, NeuralNet, Objective};
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
    /// Quantization-aware training with this kind of evaluation copy (`F16` or `I8`), see above.
    pub quantize: Option<Kind>,
    /// Money nets: this evaluation copy of the net (made at the start of every game) chooses the
    /// moves; the TD updates stay f32 (see the top of the file).
    pub search: Option<Kind>,
    /// Adam only (online TD(λ)): the eligibility trace and the gradient of the current position.
    trace: Vec<f32>,
    /// SGD: the game's accumulated weight change (offline TD(λ)); Adam: the current gradient.
    grad: Vec<f32>,
    generator: MoveGenerator<EnginePosition>,
    adam: Option<AdamState>,
    /// SGD (offline TD(λ)): the activations of every position of the current game, ...
    game_acts: Vec<Activations>,
    /// ... its values as seen from Me (and the final target last), ...
    game_values: Vec<[f32; MONEY_OUTPUTS]>,
    /// ... and which side was on roll (0: Me, 1: Opponent).
    game_sides: Vec<usize>,
}

/// The exact winning chance for the player on roll, when the game is over or the bearoff
/// database covers the position.
fn exact_win_probability(bearoff: Option<&BearoffDb>, position: &EnginePosition) -> Option<f32> {
    if let Some(winner) = winner(position) {
        return Some(if winner == position.get_turn() { 1.0 } else { 0.0 });
    }
    bearoff.and_then(|db| db.win_probability(position)).map(|p| p as f32)
}

/// Input scale of the hidden layers of int8 evaluation copies in quantization-aware training:
/// fixed (activations up to 6.35), so the net learns to live with it.
pub const I8_HIDDEN_SCALE: f32 = 0.05;

/// The evaluation copy used by quantization-aware training.
pub fn quantized_copy(net: &NeuralNet, kind: Kind) -> EvalNet {
    let n_hidden = net.sizes().len() - 2;
    let mut scales = vec![I8_HIDDEN_SCALE; n_hidden];
    scales[0] = INPUT_SCALE;
    EvalNet::from_net(net, &vec![kind; n_hidden], &scales).expect("quantizable net")
}

/// The net that evaluates positions during a training game: the f32 net, or its evaluation copy.
enum Judge<'a> {
    Full(&'a NeuralNet),
    Copy(&'a EvalNet),
}

impl Judge<'_> {
    fn win_probability(&self, position: &EnginePosition) -> f32 {
        match self {
            Judge::Full(net) => net.win_probability(position),
            Judge::Copy(net) => net.win_probability(position),
        }
    }

    fn equity(&self, position: &EnginePosition) -> f64 {
        terminal_equity(position).unwrap_or_else(|| 2.0 * self.win_probability(position) as f64 - 1.0)
    }
}

/// Value of a position for Me (winning chance): exact when known, else the judge's estimate for
/// the player on roll.
fn value_for_me(judge: &Judge, bearoff: Option<&BearoffDb>, position: &EnginePosition) -> f32 {
    let v = exact_win_probability(bearoff, position).unwrap_or_else(|| judge.win_probability(position));
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
        let trace = if adam.is_some() { vec![0.0; n] } else { Vec::new() };
        TdTrainer {
            net, alpha, lambda, variant: Variant::Backgammon, bearoff: None, quantize: None, search: None, trace, grad: vec![0.0; n],
            generator: MoveGenerator::new(), adam, game_acts: Vec::new(), game_values: Vec::new(), game_sides: Vec::new(),
        }
    }

    pub fn optimizer(&self) -> Optimizer {
        if self.adam.is_some() { Optimizer::Adam } else { Optimizer::Sgd }
    }

    /// Play one self-play game and learn from it. Returns (winner, number of turns); a game stopped
    /// at a bearoff position counts as won by the side with the better chance.
    pub fn train_game(&mut self, dice: &mut Dice) -> Result<(Player, u32), String> {
        if self.net.objective() == Objective::Money {
            return self.train_game_money(dice);
        }
        if self.search.is_some() {
            return Err("--search: money nets only (DMP nets have --quantize)".into());
        }
        if self.adam.is_some() {
            return self.train_game_adam(dice);
        }
        let TdTrainer { net, alpha, lambda, variant, bearoff, quantize, grad, generator, game_acts, game_values, game_sides, .. } = self;
        let bearoff = bearoff.as_deref();
        let copy = quantize.map(|kind| quantized_copy(net, kind));
        game_values.clear();
        game_sides.clear();

        let mut position = variant.starting_position();
        let (starter, mut die1, mut die2) = game::opening_roll(dice);
        position.set_turn(starter);
        let mut input = [0.0f32; MAX_INPUTS];

        for turn in 1..=MAX_TURNS {
            // The estimate for the position before the move, as seen by Me; its activations are
            // kept for the gradient at the end of the game.
            net.encode(&position, &mut input);
            let t = turn as usize - 1;
            if game_acts.len() <= t {
                game_acts.push(net.new_activations());
            }
            let v = net.forward_keep(&input, &mut game_acts[t]);
            let v = match &copy {
                Some(copy) => copy.forward(&input),
                None => v,
            };
            let side = (position.get_turn() != Player::Me) as usize;
            game_values.push([if side == 0 { v } else { 1.0 - v }, 0.0, 0.0, 0.0, 0.0]);
            game_sides.push(side);

            let judge = match &copy {
                Some(copy) => Judge::Copy(copy),
                None => Judge::Full(net),
            };
            let next = choose_move(generator, &position, die1, die2, |child| {
                exact_win_probability(bearoff, child).map_or_else(|| judge.equity(child), |p| 2.0 * p as f64 - 1.0)
            });

            // The game ended or reached the bearoff database: its exact value is the last target.
            // Otherwise the next turn's estimate is (the weights don't change during the game).
            if winner(&next).is_some() || bearoff.is_some_and(|db| db.win_probability(&next).is_some()) {
                let target = value_for_me(&judge, bearoff, &next);
                game_values.push([target, 0.0, 0.0, 0.0, 0.0]);
                learn_offline(net, game_acts, game_values, game_sides, &DMP_SOURCE, &DMP_SIGNS, 1, *alpha, *lambda, grad);
                net.games_trained += 1;
                return Ok((if target >= 0.5 { Player::Me } else { Player::Opponent }, turn));
            }
            position = next;
            die1 = dice.roll();
            die2 = dice.roll();
        }
        Err(format!("training game did not finish within {} turns", MAX_TURNS))
    }

    /// `train_game` with Adam: online TD(λ), with a trace and an Adam step after every move.
    fn train_game_adam(&mut self, dice: &mut Dice) -> Result<(Player, u32), String> {
        let TdTrainer { net, alpha, lambda, variant, bearoff, quantize, trace, grad, generator, adam, .. } = self;
        let state = adam.as_mut().expect("Adam");
        let bearoff = bearoff.as_deref();
        let copy = quantize.map(|kind| quantized_copy(net, kind));
        trace.iter_mut().for_each(|e| *e = 0.0);

        let mut position = variant.starting_position();
        let (starter, mut die1, mut die2) = game::opening_roll(dice);
        position.set_turn(starter);
        let mut input = [0.0f32; MAX_INPUTS];

        for turn in 1..=MAX_TURNS {
            net.encode(&position, &mut input);
            let v = net.forward_with_gradient(&input, grad);
            let v = match &copy {
                Some(copy) => copy.forward(&input),
                None => v,
            };
            let (value, sign) = if position.get_turn() == Player::Me { (v, 1.0) } else { (1.0 - v, -1.0) };
            for (e, g) in trace.iter_mut().zip(grad.iter()) {
                *e = *lambda * *e + sign * g;
            }
            let judge = match &copy {
                Some(copy) => Judge::Copy(copy),
                None => Judge::Full(net),
            };
            let next = choose_move(generator, &position, die1, die2, |child| {
                exact_win_probability(bearoff, child).map_or_else(|| judge.equity(child), |p| 2.0 * p as f64 - 1.0)
            });
            let target = value_for_me(&judge, bearoff, &next);
            adam_step(state, net.params_mut(), trace, target - value, *alpha);
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

/// DMP nets: the single value seen from Me is the output (Me on roll) or 1 - the output.
pub const DMP_SOURCE: [[usize; MONEY_OUTPUTS]; 2] = [[0; MONEY_OUTPUTS]; 2];
pub const DMP_SIGNS: [[f32; MONEY_OUTPUTS]; 2] = [[1.0; MONEY_OUTPUTS], [-1.0; MONEY_OUTPUTS]];

/// Offline TD(λ) at the end of a game: the weights stayed fixed during the game, and change now
/// by the sum of the updates of online TD(λ), `Σ_t α·δ_t·e_t` with `e_t = Σ_{s≤t} λ^(t-s)·∇V_s`,
/// rearranged per position: `Σ_s ∇V_s · α·Σ_{t≥s} λ^(t-s)·δ_t`. The inner sum (the λ-return error
/// of position s) is computed backwards over the game, and each position then needs a single
/// sparse back-propagation (`add_gradient`), instead of decaying and adding dense eligibility
/// traces (a full pass over all parameters per value) after every move.
///
/// `values[t]`: the values of position t seen from Me (`n_values` of them), with the final target
/// as the last entry (one more than positions); value v of position t is
/// `signs[side][v]·output_{source[side][v]}` + a constant, `side = sides[t]`.
#[allow(clippy::too_many_arguments)]
fn learn_offline(net: &mut NeuralNet, acts: &mut [Activations], values: &[[f32; MONEY_OUTPUTS]], sides: &[usize],
                 source: &[[usize; MONEY_OUTPUTS]; 2], signs: &[[f32; MONEY_OUTPUTS]; 2], n_values: usize,
                 alpha: f32, lambda: f32, grad: &mut [f32]) {
    let positions = sides.len();
    assert_eq!(values.len(), positions + 1);
    let errors: Vec<[f32; MONEY_OUTPUTS]> = (0..positions)
        .map(|t| std::array::from_fn(|v| values[t + 1][v] - values[t][v]))
        .collect();
    let learns: Vec<Option<usize>> = (0..positions).map(Some).collect();
    learn_offline_from_errors(net, acts, &errors, sides, &learns, source, signs, n_values, alpha, lambda, grad);
}

/// The core of `learn_offline`, from the TD errors of the moves (`errors[t]`: the value after
/// move t minus the value before it, seen from Me), where only some positions learn: position t
/// with `learns[t] = Some(i)` has its activations in `acts[i]`; the errors of the others still
/// count for the earlier positions (decaying by λ), as with traces that only some positions add
/// to (the mixture of experts' trainer, where an expert learns only on its own game type).
#[allow(clippy::too_many_arguments)]
pub fn learn_offline_from_errors(net: &mut NeuralNet, acts: &mut [Activations], errors: &[[f32; MONEY_OUTPUTS]],
                                 sides: &[usize], learns: &[Option<usize>], source: &[[usize; MONEY_OUTPUTS]; 2],
                                 signs: &[[f32; MONEY_OUTPUTS]; 2], n_values: usize, alpha: f32, lambda: f32,
                                 grad: &mut [f32]) {
    grad.iter_mut().for_each(|g| *g = 0.0);
    let k_out = net.objective().outputs();
    let mut lambda_error = [0.0f32; MONEY_OUTPUTS];
    for t in (0..errors.len()).rev() {
        for v in 0..n_values {
            lambda_error[v] = errors[t][v] + lambda * lambda_error[v];
        }
        let Some(i) = learns[t] else { continue };
        let side = sides[t];
        let mut out_grad = [0.0f32; MONEY_OUTPUTS];
        for v in 0..n_values {
            out_grad[source[side][v]] += alpha * lambda_error[v] * signs[side][v];
        }
        net.add_gradient(&mut acts[i], &out_grad[..k_out], grad);
    }
    for (w, g) in net.params_mut().iter_mut().zip(grad.iter()) {
        *w += g;
    }
}

/// The exact outcome probabilities for the player on roll when the game is over, or when the
/// bearoff database covers the position and no gammon is possible any more.
pub fn exact_outcome(bearoff: Option<&BearoffDb>, position: &EnginePosition) -> Option<[f32; MONEY_OUTPUTS]> {
    terminal_outcome(position).or_else(|| {
        if gammon_possible(position) {
            return None;
        }
        bearoff.and_then(|db| db.win_probability(position)).map(|p| [p as f32, 0.0, 0.0, 0.0, 0.0])
    })
}

/// The outcome probabilities for the player on roll: exact when known, else the net's.
fn outcome(net: &NeuralNet, bearoff: Option<&BearoffDb>, position: &EnginePosition) -> [f32; MONEY_OUTPUTS] {
    exact_outcome(bearoff, position).unwrap_or_else(|| net.outcome(position))
}

/// Per value seen from Me: which output it comes from and with which sign (see the top of the file).
pub const ME_SOURCE: [[usize; MONEY_OUTPUTS]; 2] = [[0, 1, 2, 3, 4], [0, 3, 4, 1, 2]];
pub const ME_SIGNS: [[f32; MONEY_OUTPUTS]; 2] = [[1.0; MONEY_OUTPUTS], [-1.0, 1.0, 1.0, 1.0, 1.0]];

impl TdTrainer {
    /// `train_game` for a money net (offline TD(λ), see `learn_offline`).
    fn train_game_money(&mut self, dice: &mut Dice) -> Result<(Player, u32), String> {
        if self.adam.is_some() || self.quantize.is_some() {
            return Err("money nets: SGD without quantization-aware training only".into());
        }
        let TdTrainer { net, alpha, lambda, variant, bearoff, search, grad, generator, game_acts, game_values, game_sides, .. } = self;
        let bearoff = bearoff.as_deref();
        let copy = search.map(|kind| quantized_copy(net, kind));
        game_values.clear();
        game_sides.clear();

        let mut position = variant.starting_position();
        let (starter, mut die1, mut die2) = game::opening_roll(dice);
        position.set_turn(starter);
        let mut input = [0.0f32; MAX_INPUTS];

        for turn in 1..=MAX_TURNS {
            net.encode(&position, &mut input);
            let t = turn as usize - 1;
            if game_acts.len() <= t {
                game_acts.push(net.new_activations());
            }
            net.forward_keep(&input, &mut game_acts[t]);
            let mut y = [0.0; MONEY_OUTPUTS];
            y.copy_from_slice(game_acts[t].outputs());
            let side = (position.get_turn() != Player::Me) as usize;
            game_values.push(if side == 0 { y } else { flip_outcome(&y) });
            game_sides.push(side);

            let next = match &copy {
                Some(copy) => choose_move(generator, &position, die1, die2, |child| {
                    money_equity(&exact_outcome(bearoff, child).unwrap_or_else(|| copy.outcome(child)))
                }),
                None => choose_move(generator, &position, die1, die2, |child| money_equity(&outcome(net, bearoff, child))),
            };
            if let Some(exact) = exact_outcome(bearoff, &next) {
                let target = if next.get_turn() == Player::Me { exact } else { flip_outcome(&exact) };
                game_values.push(target);
                learn_offline(net, game_acts, game_values, game_sides, &ME_SOURCE, &ME_SIGNS, MONEY_OUTPUTS, *alpha, *lambda, grad);
                net.games_trained += 1;
                return Ok((if target[0] >= 0.5 { Player::Me } else { Player::Opponent }, turn));
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
        // Alpha 0: both nets stay the same, so each stopping game is a prefix of the plain one.
        let mut plain = TdTrainer::new(net.clone(), 0.0, 0.7);
        let mut stopping = TdTrainer::new(net, 0.0, 0.7);
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
    fn test_quantization_aware_games_finish_and_change_the_net() {
        for kind in [Kind::F16, Kind::I8] {
            let net = NeuralNet::new_random(Encoding::TdNack, &[32, 32], 1);
            let before = net.params().to_vec();
            let mut trainer = TdTrainer::new(net, 0.01, 0.7);
            trainer.quantize = Some(kind);
            for seed in 0..3 {
                trainer.train_game(&mut Dice::new(seed)).unwrap();
            }
            assert_eq!(trainer.net.games_trained, 3);
            assert_ne!(trainer.net.params(), before.as_slice());
        }
    }

    #[test]
    fn test_money_training_games_finish_and_change_the_net() {
        let net = NeuralNet::new_random_for(Encoding::TdNack, &[32, 32], Objective::Money, 1);
        let before = net.params().to_vec();
        let mut trainer = TdTrainer::new(net, 0.01, 0.7);
        for seed in 0..5 {
            let (_, turns) = trainer.train_game(&mut Dice::new(seed)).unwrap();
            assert!(turns > 10);
        }
        assert_eq!(trainer.net.games_trained, 5);
        assert_ne!(trainer.net.params(), before.as_slice());
        assert!(trainer.net.params().iter().all(|w| w.is_finite()));
    }

    #[test]
    fn test_offline_update_equals_online_traces_with_fixed_weights() {
        // Σ_t α·δ_t·e_t with e_t = λ·e_{t-1} + sign_t·∇V_t, all gradients at the same weights.
        let net = NeuralNet::new_random(Encoding::TdNack, &[24, 12], 3);
        let n = net.param_count();
        let (alpha, lambda) = (0.1, 0.7);
        let mut generator = MoveGenerator::<EnginePosition>::new();
        let mut dice = Dice::new(5);
        let mut position = Variant::Backgammon.starting_position();
        position.set_turn(Player::Me);
        let mut input = [0.0; MAX_INPUTS];
        let (mut acts, mut values, mut sides, mut grads) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for _ in 0..8 {
            net.encode(&position, &mut input);
            let mut a = net.new_activations();
            let v = net.forward_keep(&input, &mut a);
            let mut g = vec![0.0; n];
            net.forward_with_gradient(&input, &mut g);
            let side = (position.get_turn() != Player::Me) as usize;
            values.push([if side == 0 { v } else { 1.0 - v }, 0.0, 0.0, 0.0, 0.0]);
            sides.push(side);
            acts.push(a);
            grads.push(g);
            position = choose_move(&mut generator, &position, dice.roll(), dice.roll(), |c| net.equity(c));
        }
        values.push([0.9, 0.0, 0.0, 0.0, 0.0]);

        let mut expected = net.params().to_vec();
        let mut trace = vec![0.0f32; n];
        for t in 0..sides.len() {
            let sign = if sides[t] == 0 { 1.0 } else { -1.0 };
            let delta = values[t + 1][0] - values[t][0];
            for k in 0..n {
                trace[k] = lambda * trace[k] + sign * grads[t][k];
                expected[k] += alpha * delta * trace[k];
            }
        }
        let mut offline = net.clone();
        let mut grad = vec![0.0; n];
        learn_offline(&mut offline, &mut acts, &values, &sides, &DMP_SOURCE, &DMP_SIGNS, 1, alpha, lambda, &mut grad);
        for (a, b) in offline.params().iter().zip(&expected) {
            assert!((a - b).abs() < 1e-5, "{} vs {}", a, b);
        }
        assert_ne!(offline.params(), net.params());
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
