use crate::EnginePosition;
use crate::position::{Player, Position};
use crate::rust_analyzer::RustAnalyzer;

const CHECKERS_PER_PLAYER: u8 = 15;

/// Safety net against a game that can never end (e.g. both players closed out on the bar).
/// A normal game takes well under 200 turns.
pub const MAX_TURNS: u32 = 10_000;

/// Seeded dice (SplitMix64), so a game can be reproduced from its seed.
pub struct Dice {
    state: u64,
}

impl Dice {
    pub fn new(seed: u64) -> Self {
        Dice { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    /// A die from 1 to 6 (the modulo bias over 2^64 values is negligible).
    pub fn roll(&mut self) -> i32 {
        (self.next_u64() % 6) as i32 + 1
    }
}

pub struct GameResult {
    pub winner: Player,
    pub final_position: EnginePosition,
    pub turns: u32,
}

/// Play a complete game between two Rust engines, with the same rules as the Python `Game`:
/// opening roll with one die each (ties rerolled, the higher die starts and plays both dice),
/// then alternating turns until one player has borne off all checkers. No doubling cube.
pub fn play_game(
    analyzer: &RustAnalyzer<EnginePosition>,
    me_depth: u32,
    opponent_depth: u32,
    dice: &mut Dice,
) -> Result<GameResult, String> {
    let mut position = EnginePosition::new();
    position.setup_starting_position();

    let (me_die, opponent_die) = loop {
        let (a, b) = (dice.roll(), dice.roll());
        if a != b {
            break (a, b);
        }
    };
    position.set_turn(if me_die > opponent_die { Player::Me } else { Player::Opponent });
    let (mut die1, mut die2) = (me_die.max(opponent_die), me_die.min(opponent_die));

    for turn in 1..=MAX_TURNS {
        let mover = position.get_turn();
        let depth = if mover == Player::Me { me_depth } else { opponent_depth };
        // best_move also handles "no legal move": it returns the position with the turn passed.
        let (next, _) = analyzer
            .best_move(&position, die1, die2, depth)
            .ok_or_else(|| "engine returned no move".to_string())?;
        position = next;

        if position.get_checkers(mover, 0)? == CHECKERS_PER_PLAYER {
            return Ok(GameResult { winner: mover, final_position: position, turns: turn });
        }
        die1 = dice.roll();
        die2 = dice.roll();
    }
    Err(format!("game did not finish within {} turns", MAX_TURNS))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn total_checkers(position: &EnginePosition, player: Player) -> u32 {
        (0..26).map(|p| position.get_checkers(player, p).unwrap() as u32).sum()
    }

    #[test]
    fn test_dice_are_in_range_and_reproducible() {
        let mut a = Dice::new(42);
        let mut b = Dice::new(42);
        let mut seen = [false; 7];
        for _ in 0..1000 {
            let roll = a.roll();
            assert!((1..=6).contains(&roll));
            assert_eq!(roll, b.roll());
            seen[roll as usize] = true;
        }
        assert!(seen[1..].iter().all(|&s| s));
    }

    #[test]
    fn test_game_finishes_with_a_winner() {
        for seed in 0..5 {
            let analyzer = RustAnalyzer::with_cache_mb(4);
            let result = play_game(&analyzer, 0, 0, &mut Dice::new(seed)).unwrap();
            let loser = result.winner.other_player();
            assert_eq!(result.final_position.get_checkers(result.winner, 0).unwrap(), 15);
            assert!(result.final_position.get_checkers(loser, 0).unwrap() < 15);
            assert_eq!(total_checkers(&result.final_position, Player::Me), 15);
            assert_eq!(total_checkers(&result.final_position, Player::Opponent), 15);
            assert!(result.turns > 10);
        }
    }

    #[test]
    fn test_same_seed_gives_same_game() {
        let a = play_game(&RustAnalyzer::with_cache_mb(4), 0, 0, &mut Dice::new(7)).unwrap();
        let b = play_game(&RustAnalyzer::with_cache_mb(4), 0, 0, &mut Dice::new(7)).unwrap();
        assert_eq!(a.winner, b.winner);
        assert_eq!(a.turns, b.turns);
        assert!(a.final_position == b.final_position);
    }
}
