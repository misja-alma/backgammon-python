use crate::position::{Position, Player};
use crate::utils::Utils;

pub struct Analyzer;

impl Analyzer {
    // Static array of all possible dice rolls (1,1) through (6,6) with unique combinations
    const DICE_ROLLS: [(i32, i32); 21] = [
        (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
        (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
        (3, 3), (3, 4), (3, 5), (3, 6),
        (4, 4), (4, 5), (4, 6),
        (5, 5), (5, 6),
        (6, 6)
    ];

    pub fn dice_rolls() -> &'static [(i32, i32)] {
        &Self::DICE_ROLLS
    }

    pub fn find_known_pw(position: &Position, player: Player) -> Option<f64> {
        if position.get_checkers(player, 0).unwrap_or(0) == 15 {
            return Some(1.0);
        }
        if position.get_checkers(player.other_player(), 0).unwrap_or(0) == 15 {
            return Some(0.0);
        }
        None
    }

    pub fn winning_chances(position: &Position, player: Player) -> f64 {
        Analyzer::winning_chances_with_depth(position, player, 0)
    }

    fn winning_chances_with_depth(position: &Position, player: Player, depth: u32) -> f64 {
        if let Some(pw_known) = Analyzer::find_known_pw(position, player) {
            return pw_known;
        }

        // Prevent infinite recursion by limiting depth
        if depth > 3 {
            return 0.5; // TODO Return neutral probability for deep positions
        }

        let mut total = 0.0;
        for &(die1, die2) in Analyzer::dice_rolls() {
            if let Some((_, pw)) = Analyzer::best_move_with_depth(position, die1, die2, depth + 1) {
                let multiplier = if die1 != die2 { 2.0 } else { 1.0 };
                total += multiplier * pw;
            }
        }

        if player == position.get_turn() {
            total / 36.0
        } else {
            1.0 - total / 36.0
        }
    }

    // best move for player on roll:
    // generate all legal moves
    // get winning chances for each for player on roll
    // return the resulting position with highest winning chances
    pub fn best_move(position: &Position, die1: i32, die2: i32) -> Option<(Position, f64)> {
        Analyzer::best_move_with_depth(position, die1, die2, 0)
    }

    fn best_move_with_depth(position: &Position, die1: i32, die2: i32, depth: u32) -> Option<(Position, f64)> {
        let mut best: Option<(Position, f64)> = None;
        let current_player = position.get_turn();
        
        for half_moves in Utils::valid_possible_moves(position, die1, die2) {
            if let Ok(new_position) = Utils::apply_move(position, &half_moves.into_iter().collect::<Vec<_>>()) {
                let pw = Analyzer::winning_chances_with_depth(&new_position, current_player, depth);
                
                match &best {
                    None => {
                        best = Some((new_position, pw));
                    }
                    Some((_, best_pw)) => {
                        if pw > *best_pw {
                            best = Some((new_position, pw));
                        }
                    }
                }
            }
        }

        if best.is_none() {
            // no move, return winning chances after passing
            let mut after_pass = position.clone();
            after_pass.switch_turn();
            let pw = Analyzer::winning_chances_with_depth(&after_pass, current_player, depth);
            return Some((after_pass, pw));
        }

        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::{Position, Player};

    #[test]
    fn test_dice_rolls() {
        let rolls = Analyzer::dice_rolls();
        
        // Should have 21 unique combinations: (1,1), (1,2), ..., (6,6)
        assert_eq!(rolls.len(), 21);
        
        // Check specific combinations exist
        assert!(rolls.contains(&(1, 1)));
        assert!(rolls.contains(&(1, 6)));
        assert!(rolls.contains(&(6, 6)));
        
        // Should not have duplicate reversed pairs
        assert!(!rolls.contains(&(2, 1))); // Should only have (1, 2)
    }

    #[test]
    fn test_find_known_pw_win() {
        let mut position = Position::new();
        position.set_checkers(Player::Me, 0, 15).unwrap(); // All checkers borne off
        
        let pw = Analyzer::find_known_pw(&position, Player::Me);
        assert_eq!(pw, Some(1.0));
    }

    #[test]
    fn test_find_known_pw_loss() {
        let mut position = Position::new();
        position.set_checkers(Player::Opponent, 0, 15).unwrap(); // Opponent has all checkers borne off
        
        let pw = Analyzer::find_known_pw(&position, Player::Me);
        assert_eq!(pw, Some(0.0));
    }

    #[test]
    fn test_find_known_pw_unknown() {
        let mut position = Position::new();
        position.setup_starting_position();
        
        let pw = Analyzer::find_known_pw(&position, Player::Me);
        assert_eq!(pw, None);
    }

    #[test]
    fn test_winning_chances_known_win() {
        let mut position = Position::new();
        position.set_checkers(Player::Me, 0, 15).unwrap();
        
        let pw = Analyzer::winning_chances(&position, Player::Me);
        assert_eq!(pw, 1.0);
    }

    #[test]
    fn test_winning_chances_known_loss() {
        let mut position = Position::new();
        position.set_checkers(Player::Opponent, 0, 15).unwrap();
        
        let pw = Analyzer::winning_chances(&position, Player::Me);
        assert_eq!(pw, 0.0);
    }

    #[test]
    fn test_best_move_no_legal_moves() {
        let mut position = Position::new();
        // Create position where opponent has won
        position.set_checkers(Player::Opponent, 0, 15).unwrap();
        position.set_turn(Player::Me);
        
        let result = Analyzer::best_move(&position, 6, 5);
        assert!(result.is_some());
        
        // Should return position after passing (turn switched)
        if let Some((new_pos, pw)) = result {
            assert_eq!(new_pos.get_turn(), Player::Opponent);
            // Should have 0 winning probability since opponent has won
            assert_eq!(pw, 0.0);
        }
    }
}