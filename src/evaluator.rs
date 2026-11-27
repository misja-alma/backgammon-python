//! Static evaluation of positions, used by the analyzer where the search stops.
//!
//! Evaluators are pluggable: the analyzer holds a `Box<dyn Evaluator<P>>`, so the search doesn't
//! know whether it is using the placeholder, a heuristic or a neural net
//! (`neural_net::NeuralNetEvaluator`).
use crate::position::Position;
use crate::utils::Utils;

const CHECKERS_PER_PLAYER: u8 = 15;

pub const WIN: f64 = 1.0;
pub const LOSS: f64 = -1.0;

/// Equity is always from the point of view of the side to move, before it rolls: +1 is a certain
/// win, -1 a certain loss. Gammons and backgammons are not scored yet, so equity is in [-1, 1]
/// (equity = 2 * winning chance - 1).
pub trait Evaluator<P: Position> {
    fn evaluate(&self, position: &P) -> f64;

    /// Equity under `scoring`. By default `evaluate` (which knows only single wins), with the
    /// money value of finished games; nets override it.
    fn evaluate_scored(&self, position: &P, scoring: Scoring) -> f64 {
        match scoring {
            Scoring::Dmp => self.evaluate(position),
            Scoring::Money => terminal_money_equity(position).unwrap_or_else(|| self.evaluate(position)),
        }
    }

    /// The 5 outcome probabilities for the side to move (win, win gammon, win backgammon, lose
    /// gammon, lose backgammon), NaN for those the evaluator doesn't know. By default only the
    /// winning chance, from the double match point equity; money nets know them all.
    fn outcome(&self, position: &P) -> [f64; MONEY_OUTPUTS] {
        if let Some(o) = terminal_outcome(position) {
            return o.map(|x| x as f64);
        }
        let win = (self.evaluate_scored(position, Scoring::Dmp) + 1.0) / 2.0;
        [win, f64::NAN, f64::NAN, f64::NAN, f64::NAN]
    }

    /// Short description, e.g. for logging which evaluator is in use.
    fn name(&self) -> String;
}

pub fn is_game_over<P: Position>(position: &P) -> bool {
    let turn = position.get_turn();
    position.count(turn, 0) == CHECKERS_PER_PLAYER
        || position.count(turn.other_player(), 0) == CHECKERS_PER_PLAYER
}

/// WIN or LOSS for the side to move when the game is over, None otherwise.
#[inline]
pub fn terminal_equity<P: Position>(position: &P) -> Option<f64> {
    let turn = position.get_turn();
    if position.count(turn, 0) == CHECKERS_PER_PLAYER {
        Some(WIN)
    } else if position.count(turn.other_player(), 0) == CHECKERS_PER_PLAYER {
        Some(LOSS)
    } else {
        None
    }
}

// --- gammons (cubeless money) ---------------------------------------------------------------
//
// A money net estimates 5 probabilities for the side to move, cumulative as in GNU Backgammon:
// [win, win gammon (incl. backgammon), win backgammon, lose gammon (incl. backgammon),
// lose backgammon]. A gammon: the loser has borne off no checker; a backgammon: a gammon where the
// loser still has a checker on the bar or in the winner's home board.

/// What a game is played for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scoring {
    /// Double match point: every win counts 1.
    Dmp,
    /// Cubeless money: a gammon counts 2, a backgammon 3.
    Money,
}

/// Number of outputs of a money net.
pub const MONEY_OUTPUTS: usize = 5;

/// Points won by `winner` in a finished game: 1, 2 (gammon) or 3 (backgammon).
pub fn points_won<P: Position>(position: &P, winner: crate::position::Player) -> u8 {
    let loser = winner.other_player();
    if position.count(loser, 0) > 0 {
        return 1;
    }
    // The winner's home board (its points 1-6) is the loser's 19-24; 25 is the loser's bar.
    if (19..=25).any(|point| position.count(loser, point) > 0) { 3 } else { 2 }
}

/// The 5 outcome probabilities for the side to move when the game is over, None otherwise.
pub fn terminal_outcome<P: Position>(position: &P) -> Option<[f32; MONEY_OUTPUTS]> {
    let turn = position.get_turn();
    let (won, winner) = if position.count(turn, 0) == CHECKERS_PER_PLAYER {
        (true, turn)
    } else if position.count(turn.other_player(), 0) == CHECKERS_PER_PLAYER {
        (false, turn.other_player())
    } else {
        return None;
    };
    let points = points_won(position, winner);
    let (g, bg) = ((points >= 2) as u8 as f32, (points == 3) as u8 as f32);
    Some(if won { [1.0, g, bg, 0.0, 0.0] } else { [0.0, 0.0, 0.0, g, bg] })
}

/// Cubeless money equity for the side to move when the game is over (±1, ±2, ±3), None otherwise.
pub fn terminal_money_equity<P: Position>(position: &P) -> Option<f64> {
    terminal_outcome(position).map(|p| money_equity(&p))
}

/// `flip_outcome` for f64 values (unknown ones stay NaN).
#[inline]
pub fn flip_outcome_f64(p: &[f64; MONEY_OUTPUTS]) -> [f64; MONEY_OUTPUTS] {
    [1.0 - p[0], p[3], p[4], p[1], p[2]]
}

/// The same probabilities from the other side's point of view.
#[inline]
pub fn flip_outcome(p: &[f32; MONEY_OUTPUTS]) -> [f32; MONEY_OUTPUTS] {
    [1.0 - p[0], p[3], p[4], p[1], p[2]]
}

/// Cubeless money equity of 5 outcome probabilities, after making them consistent (a gammon
/// can't be likelier than a win, a backgammon not likelier than a gammon).
#[inline]
pub fn money_equity(p: &[f32; MONEY_OUTPUTS]) -> f64 {
    let win = p[0].clamp(0.0, 1.0);
    let wg = p[1].clamp(0.0, win);
    let wbg = p[2].clamp(0.0, wg);
    let lg = p[3].clamp(0.0, 1.0 - win);
    let lbg = p[4].clamp(0.0, lg);
    (2.0 * win - 1.0 + wg - lg + wbg - lbg) as f64
}

/// Whether a gammon is still possible: one side hasn't borne off any checker yet.
pub fn gammon_possible<P: Position>(position: &P) -> bool {
    position.count(crate::position::Player::Me, 0) == 0 || position.count(crate::position::Player::Opponent, 0) == 0
}

/// The placeholder: knows only finished games; every other position evaluates to 0.
pub struct TerminalEvaluator;

impl TerminalEvaluator {
    pub const UNKNOWN: f64 = 0.0;
}

impl<P: Position> Evaluator<P> for TerminalEvaluator {
    fn evaluate(&self, position: &P) -> f64 {
        terminal_equity(position).unwrap_or(Self::UNKNOWN)
    }

    fn name(&self) -> String {
        "terminal (placeholder)".to_string()
    }
}

/// A simple heuristic: the side with the smaller pip count is ahead. Used as a yardstick while
/// training nets, since any useful net should beat it easily.
pub struct PipCountEvaluator;

impl<P: Position> Evaluator<P> for PipCountEvaluator {
    fn evaluate(&self, position: &P) -> f64 {
        terminal_equity(position).unwrap_or_else(|| {
            let turn = position.get_turn();
            let lead = Utils::pipcount(position, turn.other_player()) - Utils::pipcount(position, turn);
            (lead as f64 / 40.0).tanh()
        })
    }

    fn name(&self) -> String {
        "pip count".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache_friendly_position::CacheFriendlyPosition;
    use crate::position::Player;

    fn with_off(me_off: u8, opp_off: u8, turn: Player) -> CacheFriendlyPosition {
        let mut p = CacheFriendlyPosition::new();
        p.set_checkers(Player::Me, 0, me_off).unwrap();
        p.set_checkers(Player::Opponent, 0, opp_off).unwrap();
        if me_off < 15 { p.set_checkers(Player::Me, 1, 15 - me_off).unwrap(); }
        if opp_off < 15 { p.set_checkers(Player::Opponent, 1, 15 - opp_off).unwrap(); }
        p.set_turn(turn);
        p
    }

    #[test]
    fn test_win_and_loss_are_from_side_to_move() {
        assert_eq!(TerminalEvaluator.evaluate(&with_off(15, 3, Player::Me)), 1.0);
        assert_eq!(TerminalEvaluator.evaluate(&with_off(15, 3, Player::Opponent)), -1.0);
        assert!(is_game_over(&with_off(15, 3, Player::Opponent)));
    }

    #[test]
    fn test_gammons_and_backgammons() {
        // Me has borne off all; Opponent has 3 off: a single win.
        assert_eq!(terminal_money_equity(&with_off(15, 3, Player::Me)), Some(1.0));
        assert_eq!(terminal_money_equity(&with_off(15, 3, Player::Opponent)), Some(-1.0));
        // Opponent has none off, all on its 1-point (its home board): a gammon.
        assert_eq!(terminal_money_equity(&with_off(15, 0, Player::Me)), Some(2.0));
        assert_eq!(terminal_outcome(&with_off(15, 0, Player::Opponent)), Some([0.0, 0.0, 0.0, 1.0, 0.0]));
        // A checker in the winner's home board (Opponent's 19-point): a backgammon.
        let mut p = with_off(15, 0, Player::Me);
        p.set_checkers(Player::Opponent, 1, 14).unwrap();
        p.set_checkers(Player::Opponent, 19, 1).unwrap();
        assert_eq!(terminal_money_equity(&p), Some(3.0));
        assert_eq!(terminal_outcome(&p), Some([1.0, 1.0, 1.0, 0.0, 0.0]));
        // On the bar counts too.
        p.set_checkers(Player::Opponent, 19, 0).unwrap();
        p.set_checkers(Player::Opponent, 25, 1).unwrap();
        assert_eq!(terminal_money_equity(&p), Some(3.0));
        assert!(gammon_possible(&with_off(3, 0, Player::Me)));
        assert!(!gammon_possible(&with_off(3, 1, Player::Me)));
    }

    #[test]
    fn test_money_equity() {
        assert_eq!(money_equity(&[0.5, 0.0, 0.0, 0.0, 0.0]), 0.0);
        assert!((money_equity(&[0.6, 0.2, 0.05, 0.1, 0.01]) - (0.2 + 0.2 - 0.1 + 0.05 - 0.01)).abs() < 1e-6);
        // Inconsistent estimates are clipped: a gammon chance above the winning chance counts as it.
        assert!((money_equity(&[0.3, 0.5, 0.0, 0.0, 0.0]) - (-0.4 + 0.3)).abs() < 1e-6);
        let p = [0.6, 0.2, 0.05, 0.1, 0.01];
        assert!((money_equity(&flip_outcome(&p)) + money_equity(&p)).abs() < 1e-6);
    }

    #[test]
    fn test_unfinished_position_is_unknown() {
        let mut p = CacheFriendlyPosition::new();
        p.setup_starting_position();
        assert_eq!(TerminalEvaluator.evaluate(&p), 0.0);
        assert!(!is_game_over(&p));
    }

    #[test]
    fn test_pip_count_favours_the_side_ahead_in_the_race() {
        // Me: 14 off and one checker on the 1-point (1 pip); Opponent: 15 checkers on its 1-point.
        let me_ahead = with_off(14, 0, Player::Me);
        assert!(PipCountEvaluator.evaluate(&me_ahead) > 0.0);
        let mut opp_to_move = me_ahead.clone();
        opp_to_move.set_turn(Player::Opponent);
        assert!(PipCountEvaluator.evaluate(&opp_to_move) < 0.0);
    }
}
