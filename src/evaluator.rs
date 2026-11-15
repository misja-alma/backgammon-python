use crate::position::Position;

const CHECKERS_PER_PLAYER: u8 = 15;

/// Static evaluation of a position, used by the analyzer where the search stops.
///
/// Equity is always from the point of view of the side to move: +1 is a certain win, -1 a
/// certain loss. Gammons and backgammons are not scored yet, so equity is in [-1, 1].
/// Positions that are not finished currently evaluate to 0 (no opinion).
pub struct Evaluator;

impl Evaluator {
    pub const WIN: f64 = 1.0;
    pub const LOSS: f64 = -1.0;
    pub const UNKNOWN: f64 = 0.0;

    pub fn is_game_over<P: Position>(position: &P) -> bool {
        let turn = position.get_turn();
        position.count(turn, 0) == CHECKERS_PER_PLAYER
            || position.count(turn.other_player(), 0) == CHECKERS_PER_PLAYER
    }

    pub fn evaluate<P: Position>(position: &P) -> f64 {
        let turn = position.get_turn();
        if position.count(turn, 0) == CHECKERS_PER_PLAYER {
            Self::WIN
        } else if position.count(turn.other_player(), 0) == CHECKERS_PER_PLAYER {
            Self::LOSS
        } else {
            Self::UNKNOWN
        }
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
        assert_eq!(Evaluator::evaluate(&with_off(15, 3, Player::Me)), 1.0);
        assert_eq!(Evaluator::evaluate(&with_off(15, 3, Player::Opponent)), -1.0);
        assert!(Evaluator::is_game_over(&with_off(15, 3, Player::Opponent)));
    }

    #[test]
    fn test_unfinished_position_is_unknown() {
        let mut p = CacheFriendlyPosition::new();
        p.setup_starting_position();
        assert_eq!(Evaluator::evaluate(&p), 0.0);
        assert!(!Evaluator::is_game_over(&p));
    }
}
