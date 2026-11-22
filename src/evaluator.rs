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
