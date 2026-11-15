use crate::position::{Position, Player};
use itertools::Itertools;
use crate::move_generator::MoveGenerator;
use std::collections::HashSet;

pub struct Utils;

impl Utils {
    pub fn pipcount<P: Position>(position: &P, player: Player) -> i32 {
        let mut total = 0;
        for point in 1..26 {
            let checkers = position.get_checkers(player, point).unwrap_or(0) as i32;
            total += point as i32 * checkers;
        }
        total
    }

    #[inline]
    pub fn can_bear_off<P: Position>(position: &P) -> bool {
        let player = position.get_turn();
        for point in 7..26 {
            if position.get_checkers(player, point).unwrap_or(0) > 0 {
                return false;
            }
        }
        true
    }

    #[inline]
    pub fn can_move<P: Position>(position: &P, start_point: usize, target_point: usize) -> bool {
        let player = position.get_turn();
        let opponent = player.other_player();

        if position.get_checkers(player, start_point).unwrap_or(0) == 0 {
            return false;
        }

        if position.get_checkers(player, 25).unwrap_or(0) > 0 && start_point != 25 {
            return false;
        }

        if target_point == 0 {
            if !Utils::can_bear_off(position) {
                return false;
            }
        }

        if target_point > 0 && position.get_checkers(opponent, 25 - target_point).unwrap_or(0) >= 2 {
            return false;
        }

        true
    }

    #[inline]
    pub fn apply_half_move<P: Position + Clone>(pos: &P, start: usize, target: usize, player: Player) -> Result<P, String> {
        let mut new_pos = pos.clone();

        let current_at_start = new_pos.get_checkers(player, start)?;
        if current_at_start == 0 {
            return Err("No checkers at start position".to_string());
        }

        new_pos.set_checkers(player, start, current_at_start - 1)?;

        // Hit: a single opponent checker on the target point goes to the opponent's bar.
        // This must happen before placing our checker, because in a bit-packed position both
        // players share the physical slot.
        if (1..=24).contains(&target) {
            let opponent = player.other_player();
            let opponent_point = 25 - target;
            if new_pos.get_checkers(opponent, opponent_point)? == 1 {
                new_pos.set_checkers(opponent, opponent_point, 0)?;
                let on_bar = new_pos.get_checkers(opponent, 25)?;
                new_pos.set_checkers(opponent, 25, on_bar + 1)?;
            }
        }

        let current_at_target = new_pos.get_checkers(player, target)?;
        new_pos.set_checkers(player, target, current_at_target + 1)?;

        Ok(new_pos)
    }

    pub fn apply_move<P: Position + Clone>(pos: &P, half_moves: &[(usize, usize)]) -> Result<P, String> {
        let mut new_pos = pos.clone();
        let player = pos.get_turn();

        for &(start, target) in half_moves {
            new_pos = Utils::apply_half_move(&new_pos, start, target, player)?;
        }

        new_pos.switch_turn();
        Ok(new_pos)
    }

    pub fn possible_moves<P: Position + Clone>(position: &P, dice: &[i32]) -> HashSet<Vec<(usize, usize)>> {
        let player = position.get_turn();
        let mut all_moves_set = HashSet::new();

        fn find_moves_recursive<P: Position + Clone>(
            pos: &P,
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

            for start in 1..26 {
                if pos.get_checkers(player, start).unwrap_or(0) == 0 {
                    continue;
                }

                let target = if start != 25 {
                    if start as i32 - die <= 0 {
                        0
                    } else {
                        start - die as usize
                    }
                } else {
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

            if !found_move && !current_moves.is_empty() {
                all_moves_set.insert(current_moves);
            }
        }

        // Only distinct orderings: [d, d, d, d] has 24 permutations that are all the same, and
        // searching each of them would repeat the identical search 24 times.
        for dice_vec in dice.iter().cloned().permutations(dice.len()).unique() {
            find_moves_recursive(position, &dice_vec, Vec::new(), &mut all_moves_set, player);
        }

        all_moves_set
    }

    /// The distinct positions that can result from playing `die1`/`die2`, with the turn passed to
    /// the opponent, in a deterministic order; empty when no move is legal. Same rules as
    /// `valid_possible_moves`, but much faster (see `MoveGenerator`). For repeated calls, keep a
    /// `MoveGenerator` and call it directly to reuse its tables.
    pub fn legal_positions<P: Position + Clone>(position: &P, die1: i32, die2: i32) -> Vec<P> {
        MoveGenerator::new().legal_positions(position, die1, die2)
    }

    pub fn valid_possible_moves<P: Position + Clone>(position: &P, die1: i32, die2: i32) -> HashSet<Vec<(usize, usize)>> {
        let dice = if die1 == die2 {
            vec![die1, die1, die1, die1]
        } else {
            vec![die1, die2]
        };

        let all_moves = Utils::possible_moves(position, &dice);

        if all_moves.is_empty() {
            return HashSet::new();
        }

        let max_dice_used = all_moves.iter().map(|moves| moves.len()).max().unwrap_or(0);
        let max_dice_moves: HashSet<Vec<(usize, usize)>> = all_moves
            .into_iter()
            .filter(|moves| moves.len() == max_dice_used)
            .collect();

        if die1 != die2 && max_dice_used == 1 {
            let higher_die = die1.max(die2);
            // A move "uses the higher die" if it is legal with that die alone in this position
            // (a bear-off with a larger die than needed is only legal in some positions).
            let playable_with_higher = Utils::possible_moves(position, &[higher_die]);
            let highest_die_moves: HashSet<Vec<(usize, usize)>> = max_dice_moves
                .iter()
                .filter(|moves| playable_with_higher.contains(*moves))
                .cloned()
                .collect();

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
    use crate::simple_position::SimplePosition;
    use crate::position::Player;

    #[test]
    fn test_pipcount() {
        let mut position = SimplePosition::new();
        position.setup_starting_position();

        let pipcount_me = Utils::pipcount(&position, Player::Me);
        let pipcount_opponent = Utils::pipcount(&position, Player::Opponent);

        assert_eq!(pipcount_me, 167);
        assert_eq!(pipcount_opponent, 167);
    }

    #[test]
    fn test_can_bear_off_false() {
        let mut position = SimplePosition::new();
        position.setup_starting_position();

        assert!(!Utils::can_bear_off(&position));
    }

    #[test]
    fn test_can_bear_off_true() {
        let mut position = SimplePosition::new();

        position.set_checkers(Player::Me, 1, 2).unwrap();
        position.set_checkers(Player::Me, 6, 13).unwrap();
        position.set_turn(Player::Me);

        assert!(Utils::can_bear_off(&position));
    }

    #[test]
    fn test_can_move_basic() {
        let mut position = SimplePosition::new();
        position.set_checkers(Player::Me, 6, 1).unwrap();
        position.set_turn(Player::Me);

        assert!(Utils::can_move(&position, 6, 4));
        assert!(!Utils::can_move(&position, 5, 3));
    }

    #[test]
    fn test_apply_half_move() {
        let mut position = SimplePosition::new();
        position.set_checkers(Player::Me, 6, 2).unwrap();

        let result = Utils::apply_half_move(&position, 6, 4, Player::Me);
        assert!(result.is_ok());

        let new_pos = result.unwrap();
        assert_eq!(new_pos.get_checkers(Player::Me, 6).unwrap(), 1);
        assert_eq!(new_pos.get_checkers(Player::Me, 4).unwrap(), 1);
    }

    fn positions_from_sequences<P: Position + Clone>(pos: &P, d1: i32, d2: i32) -> HashSet<P::CacheKey> {
        Utils::valid_possible_moves(pos, d1, d2)
            .iter()
            .map(|m| Utils::apply_move(pos, m).unwrap().cache_key(0))
            .collect()
    }

    #[test]
    fn test_legal_positions_matches_sequence_generator() {
        use crate::cache_friendly_position::CacheFriendlyPosition;
        use crate::game::Dice;
        let mut compared = 0;
        for seed in 0..8 {
            let mut dice = Dice::new(seed);
            let mut pos = CacheFriendlyPosition::new();
            pos.setup_starting_position();
            for _ in 0..120 {
                for d1 in 1..=6 {
                    for d2 in d1..=6 {
                        let fast = Utils::legal_positions(&pos, d1, d2);
                        let keys: HashSet<_> = fast.iter().map(|p| p.cache_key(0)).collect();
                        assert_eq!(keys.len(), fast.len(), "positions must be distinct");
                        let expected = positions_from_sequences(&pos, d1, d2);
                        assert_eq!(keys, expected, "roll {}-{}", d1, d2);
                        compared += 1;
                    }
                }
                // Advance the game with a random roll and an arbitrary legal move.
                let (d1, d2) = (dice.roll(), dice.roll());
                let next = Utils::legal_positions(&pos, d1, d2);
                if next.is_empty() {
                    pos.switch_turn();
                } else {
                    pos = next[(dice.roll() as usize) % next.len()].clone();
                }
                let mover = pos.get_turn().other_player();
                if pos.get_checkers(mover, 0).unwrap() == 15 {
                    break;
                }
            }
        }
        assert!(compared > 10_000);
    }

    #[test]
    fn test_single_die_must_be_the_higher_one_even_when_bearing_off() {
        // Me: checkers on the 4- and 2-points, rolling 6-1, with Me's 3- and 1-points blocked.
        // The 1 can't be played (4/3 and 2/1 are blocked), and the 6 can only bear off the
        // highest checker (4/off, overshooting). After that the 1 still can't be played.
        let mut pos = SimplePosition::new();
        pos.set_checkers(Player::Me, 4, 1).unwrap();
        pos.set_checkers(Player::Me, 2, 1).unwrap();
        pos.set_checkers(Player::Me, 0, 13).unwrap();
        pos.set_checkers(Player::Opponent, 22, 2).unwrap(); // Me's 3-point
        pos.set_checkers(Player::Opponent, 24, 2).unwrap(); // Me's 1-point
        pos.set_turn(Player::Me);
        let moves = Utils::valid_possible_moves(&pos, 6, 1);
        let expected: HashSet<Vec<(usize, usize)>> = [vec![(4, 0)]].into_iter().collect();
        assert_eq!(moves, expected);
        assert_eq!(Utils::legal_positions(&pos, 6, 1).len(), 1);
    }

    #[test]
    fn test_single_die_any_move_when_higher_die_is_unplayable() {
        // Me on 5 (x2), 3 (x8) and 2 (x2) rolling 4-2, Me's 1-point blocked: the 4 can't be played
        // (5/1 blocked, overshooting bear-offs not allowed with a checker on the 5), so any move
        // with the 2 is allowed, including 2/off.
        let mut pos = SimplePosition::new();
        pos.set_checkers(Player::Me, 5, 2).unwrap();
        pos.set_checkers(Player::Me, 3, 8).unwrap();
        pos.set_checkers(Player::Me, 2, 2).unwrap();
        pos.set_checkers(Player::Me, 0, 3).unwrap();
        pos.set_checkers(Player::Opponent, 24, 2).unwrap(); // Me's 1-point
        pos.set_turn(Player::Me);
        let moves = Utils::valid_possible_moves(&pos, 4, 2);
        let expected: HashSet<Vec<(usize, usize)>> = [vec![(5, 3)], vec![(2, 0)]].into_iter().collect();
        assert_eq!(moves, expected);
        assert_eq!(Utils::legal_positions(&pos, 4, 2).len(), 2);
    }

    #[test]
    fn test_apply_half_move_hits_blot() {
        let mut pos = SimplePosition::new();
        pos.set_checkers(Player::Me, 8, 1).unwrap();
        pos.set_checkers(Player::Opponent, 20, 1).unwrap(); // Me's point 5

        let new_pos = Utils::apply_half_move(&pos, 8, 5, Player::Me).unwrap();
        assert_eq!(new_pos.get_checkers(Player::Me, 5).unwrap(), 1);
        assert_eq!(new_pos.get_checkers(Player::Opponent, 20).unwrap(), 0);
        assert_eq!(new_pos.get_checkers(Player::Opponent, 25).unwrap(), 1);
    }

    #[test]
    fn test_apply_half_move_hits_blot_cache_friendly_position() {
        let mut pos = crate::cache_friendly_position::CacheFriendlyPosition::new();
        pos.set_checkers(Player::Me, 8, 1).unwrap();
        pos.set_checkers(Player::Opponent, 20, 1).unwrap();

        let new_pos = Utils::apply_half_move(&pos, 8, 5, Player::Me).unwrap();
        assert_eq!(new_pos.get_checkers(Player::Me, 5).unwrap(), 1);
        assert_eq!(new_pos.get_checkers(Player::Opponent, 20).unwrap(), 0);
        assert_eq!(new_pos.get_checkers(Player::Opponent, 25).unwrap(), 1);
    }

    #[test]
    fn test_apply_move() {
        let mut position = SimplePosition::new();
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
        assert_eq!(new_pos.get_turn(), Player::Opponent);
    }

    #[test]
    fn test_valid_possible_moves_doubles() {
        let mut position = SimplePosition::new();
        position.set_checkers(Player::Me, 6, 4).unwrap();
        position.set_turn(Player::Me);

        let moves = Utils::valid_possible_moves(&position, 2, 2);

        assert!(!moves.is_empty());

        for move_seq in &moves {
            assert!(move_seq.len() <= 4);
        }
    }

    #[test]
    fn test_valid_possible_moves_bearoff_overshoot() {
        let mut position = SimplePosition::new();
        position.set_checkers(Player::Me, 2, 1).unwrap();
        position.set_checkers(Player::Me, 9, 1).unwrap();
        position.set_turn(Player::Me);

        let moves = Utils::valid_possible_moves(&position, 4, 3);

        let expected: HashSet<Vec<(usize, usize)>> = [
            vec![(9, 5), (5, 2)],
            vec![(9, 6), (6, 2)],
        ].into_iter().collect();

        assert_eq!(moves, expected);
    }

    #[test]
    fn test_valid_possible_moves_non_doubles() {
        let mut position = SimplePosition::new();
        position.set_checkers(Player::Me, 6, 2).unwrap();
        position.set_turn(Player::Me);

        let moves = Utils::valid_possible_moves(&position, 2, 3);

        assert!(!moves.is_empty());

        for move_seq in &moves {
            assert!(move_seq.len() <= 2);
        }
    }
}