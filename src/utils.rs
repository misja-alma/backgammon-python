use crate::position::{Position, Player};
use itertools::Itertools;
use std::collections::HashSet;

pub struct Utils;

impl Utils {
    pub fn pipcount(position: &Position, player: Player) -> i32 {
        let mut total = 0;
        for point in 1..26 {  // Points 1-25 (exclude 0 as those are off the board)
            let checkers = position.get_checkers(player, point).unwrap_or(0) as i32;
            total += point as i32 * checkers;
        }
        total
    }

    pub fn can_bear_off(position: &Position) -> bool {
        let player = position.get_turn();
        // Check if player has any checkers outside home board (points 7-24) or on bar (point 25)
        for point in 7..26 {
            if position.get_checkers(player, point).unwrap_or(0) > 0 {
                return false;
            }
        }
        true
    }

    pub fn can_move(position: &Position, start_point: usize, target_point: usize) -> bool {
        let player = position.get_turn();
        let opponent = player.other_player();

        // Check if player has checkers on start point
        if position.get_checkers(player, start_point).unwrap_or(0) == 0 {
            return false;
        }

        // Check if player has checkers on the bar and trying to move from elsewhere
        if position.get_checkers(player, 25).unwrap_or(0) > 0 && start_point != 25 {
            return false;
        }

        // Check if target is bearing off (point 0)
        if target_point == 0 {
            if !Utils::can_bear_off(position) {
                return false;
            }
        }

        // Check if target point is occupied by opponent (2 or more opponent checkers)
        // Opponent's point numbering is reversed: opponent's point = 25 - target_point
        if target_point > 0 && position.get_checkers(opponent, 25 - target_point).unwrap_or(0) >= 2 {
            return false;
        }

        true
    }

    pub fn apply_half_move(pos: &Position, start: usize, target: usize, player: Player) -> Result<Position, String> {
        let mut new_pos = pos.clone();
        
        // Get current checker count at start position
        let current_at_start = new_pos.get_checkers(player, start)?;
        if current_at_start == 0 {
            return Err("No checkers at start position".to_string());
        }
        
        // Move checker from start to target
        new_pos.set_checkers(player, start, current_at_start - 1)?;
        let current_at_target = new_pos.get_checkers(player, target)?;
        new_pos.set_checkers(player, target, current_at_target + 1)?;
        
        Ok(new_pos)
    }

    pub fn apply_move(pos: &Position, half_moves: &[(usize, usize)]) -> Result<Position, String> {
        let mut new_pos = pos.clone();
        let player = pos.get_turn();
        
        for &(start, target) in half_moves {
            new_pos = Utils::apply_half_move(&new_pos, start, target, player)?;
        }
        
        new_pos.switch_turn();
        Ok(new_pos)
    }

    pub fn possible_moves(position: &Position, dice: &[i32]) -> HashSet<Vec<(usize, usize)>> {
        let player = position.get_turn();
        let mut all_moves_set = HashSet::new();

        fn find_moves_recursive(
            pos: &Position,
            remaining_dice: &[i32],
            current_moves: Vec<(usize, usize)>,
            all_moves_set: &mut HashSet<Vec<(usize, usize)>>,
            player: Player
        ) {
            if remaining_dice.is_empty() {
                all_moves_set.insert(current_moves);
                return;
            }

            let die = remaining_dice[0];
            let mut found_move = false;

            // Try all possible starting points
            for start in 1..26 {
                if pos.get_checkers(player, start).unwrap_or(0) == 0 {
                    continue;
                }

                let target = if start != 25 {
                    if start as i32 - die <= 0 {
                        0  // Bear off
                    } else {
                        start - die as usize
                    }
                } else {
                    // From bar, move to point 25 - die
                    25 - die as usize
                };

                // Overshoot bear-off is only legal when no checker sits on a higher home-board point
                if target == 0 && die as usize > start {
                    if (start + 1..7).any(|p| pos.get_checkers(player, p).unwrap_or(0) > 0) {
                        continue;
                    }
                }

                if Utils::can_move(pos, start, target) {
                    found_move = true;
                    if let Ok(new_pos) = Utils::apply_half_move(pos, start, target, player) {
                        let mut new_current_moves = current_moves.clone();
                        new_current_moves.push((start, target));
                        find_moves_recursive(&new_pos, &remaining_dice[1..], new_current_moves, all_moves_set, player);
                    }
                }
            }

            // If no moves possible with remaining dice, add current partial sequence
            if !found_move && !current_moves.is_empty() {
                all_moves_set.insert(current_moves);
            }
        }

        // Try all permutations of dice
        for dice_perm in dice.iter().permutations(dice.len()) {
            let dice_vec: Vec<i32> = dice_perm.into_iter().cloned().collect();
            find_moves_recursive(position, &dice_vec, Vec::new(), &mut all_moves_set, player);
        }

        all_moves_set
    }

    pub fn valid_possible_moves(position: &Position, die1: i32, die2: i32) -> HashSet<Vec<(usize, usize)>> {
        // Convert to dice list, handling doubles
        let dice = if die1 == die2 {
            vec![die1, die1, die1, die1]  // Four moves for doubles
        } else {
            vec![die1, die2]
        };

        // Get all possible moves
        let all_moves = Utils::possible_moves(position, &dice);

        if all_moves.is_empty() {
            return HashSet::new();
        }

        // Filter to keep only moves with maximum number of dice used
        let max_dice_used = all_moves.iter().map(|moves| moves.len()).max().unwrap_or(0);
        let max_dice_moves: HashSet<Vec<(usize, usize)>> = all_moves
            .into_iter()
            .filter(|moves| moves.len() == max_dice_used)
            .collect();

        // For non-doubles, if only one die could be played, keep only moves using the higher die
        if die1 != die2 && max_dice_used == 1 {
            let higher_die = die1.max(die2);
            let highest_die_moves: HashSet<Vec<(usize, usize)>> = max_dice_moves
                .iter()
                .filter(|moves| {
                    moves.iter().any(|(start, target)| {
                        (*start as i32 - *target as i32).abs() == higher_die
                    })
                })
                .cloned()
                .collect();

            // If no moves with highest die, return original filtered moves
            if !highest_die_moves.is_empty() {
                return highest_die_moves;
            }
        }

        max_dice_moves
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::{Position, Player};

    #[test]
    fn test_pipcount() {
        let mut position = Position::new();
        position.setup_starting_position();
        
        // Starting position pipcount should be 167 for both players
        // (2*24 + 5*13 + 3*8 + 5*6 = 48 + 65 + 24 + 30 = 167)
        let pipcount_me = Utils::pipcount(&position, Player::Me);
        let pipcount_opponent = Utils::pipcount(&position, Player::Opponent);
        
        assert_eq!(pipcount_me, 167);
        assert_eq!(pipcount_opponent, 167);
    }

    #[test]
    fn test_can_bear_off_false() {
        let mut position = Position::new();
        position.setup_starting_position();
        
        // Starting position should not allow bearing off
        assert!(!Utils::can_bear_off(&position));
    }

    #[test]
    fn test_can_bear_off_true() {
        let mut position = Position::new();
        
        // Put all checkers in home board (points 1-6) for current player
        position.set_checkers(Player::Me, 1, 2).unwrap();
        position.set_checkers(Player::Me, 6, 13).unwrap();
        position.set_turn(Player::Me);
        
        assert!(Utils::can_bear_off(&position));
    }

    #[test]
    fn test_can_move_basic() {
        let mut position = Position::new();
        position.set_checkers(Player::Me, 6, 1).unwrap();
        position.set_turn(Player::Me);
        
        // Should be able to move from 6 to 4 (die value 2)
        assert!(Utils::can_move(&position, 6, 4));
        
        // Should not be able to move from empty point
        assert!(!Utils::can_move(&position, 5, 3));
    }

    #[test]
    fn test_apply_half_move() {
        let mut position = Position::new();
        position.set_checkers(Player::Me, 6, 2).unwrap();
        
        let result = Utils::apply_half_move(&position, 6, 4, Player::Me);
        assert!(result.is_ok());
        
        let new_pos = result.unwrap();
        assert_eq!(new_pos.get_checkers(Player::Me, 6).unwrap(), 1);
        assert_eq!(new_pos.get_checkers(Player::Me, 4).unwrap(), 1);
    }

    #[test]
    fn test_apply_move() {
        let mut position = Position::new();
        position.set_checkers(Player::Me, 6, 1).unwrap();
        position.set_checkers(Player::Me, 8, 1).unwrap();
        position.set_turn(Player::Me);
        
        let half_moves = vec![(6, 4), (8, 5)];
        let result = Utils::apply_move(&position, &half_moves);
        assert!(result.is_ok());
        
        let new_pos = result.unwrap();
        assert_eq!(new_pos.get_checkers(Player::Me, 6).unwrap(), 0);
        assert_eq!(new_pos.get_checkers(Player::Me, 8).unwrap(), 0);
        assert_eq!(new_pos.get_checkers(Player::Me, 4).unwrap(), 1);
        assert_eq!(new_pos.get_checkers(Player::Me, 5).unwrap(), 1);
        // Turn should be switched
        assert_eq!(new_pos.get_turn(), Player::Opponent);
    }

    #[test]
    fn test_valid_possible_moves_doubles() {
        let mut position = Position::new();
        position.set_checkers(Player::Me, 6, 4).unwrap();
        position.set_turn(Player::Me);
        
        let moves = Utils::valid_possible_moves(&position, 2, 2);
        
        // Should have some valid moves for doubles
        assert!(!moves.is_empty());
        
        // All moves should use maximum dice (4 for doubles)
        for move_seq in &moves {
            assert!(move_seq.len() <= 4);
        }
    }

    #[test]
    fn test_valid_possible_moves_bearoff_overshoot() {
        let mut position = Position::new();
        position.set_checkers(Player::Me, 2, 1).unwrap();
        position.set_checkers(Player::Me, 9, 1).unwrap();
        position.set_turn(Player::Me);

        let moves = Utils::valid_possible_moves(&position, 4, 3);

        // Bearing off from point 2 with die 3 is illegal while checker is on point 9
        // Only valid moves are 9->5 then 5->2, or 9->6 then 6->2
        let expected: HashSet<Vec<(usize, usize)>> = [
            vec![(9, 5), (5, 2)],
            vec![(9, 6), (6, 2)],
        ].into_iter().collect();

        assert_eq!(moves, expected);
    }

    #[test]
    fn test_valid_possible_moves_non_doubles() {
        let mut position = Position::new();
        position.set_checkers(Player::Me, 6, 2).unwrap();
        position.set_turn(Player::Me);
        
        let moves = Utils::valid_possible_moves(&position, 2, 3);
        
        // Should have some valid moves
        assert!(!moves.is_empty());
        
        // All moves should use at most 2 dice for non-doubles
        for move_seq in &moves {
            assert!(move_seq.len() <= 2);
        }
    }
}