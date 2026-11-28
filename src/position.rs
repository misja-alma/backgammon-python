#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Player {
    Opponent = 0,
    Me = 1,
}

impl Player {
    pub fn other_player(&self) -> Player {
        match self {
            Player::Me => Player::Opponent,
            Player::Opponent => Player::Me,
        }
    }
}

use std::hash::Hash;

/// The error for a point outside 0-25. Kept out of line (and cold) so that the checker
/// accessors stay small enough to be inlined into the move generator.
#[cold]
#[inline(never)]
pub fn invalid_point_error<T>() -> Result<T, String> {
    Err("Point must be between 0 and 25".to_string())
}

/// Positions are plain data, so they can be shared between and sent to threads (the analyzer's
/// parallel search).
pub trait Position: Send + Sync + 'static {
    /// A compact identity of (position, search depth), used as the analyzer's cache key.
    /// Unlike a hash it is collision-free: two keys are equal only for identical positions.
    type CacheKey: Copy + Eq + Hash + Send + Sync;

    fn cache_key(&self, depth: u16) -> Self::CacheKey;
    fn get_checkers(&self, player: Player, point: usize) -> Result<u8, String>;

    /// Checker count for the move search's hot loops: no range check and no `Result`. `point`
    /// must be 0-25. Override with a direct read where possible.
    #[inline]
    fn count(&self, player: Player, point: usize) -> u8 {
        self.get_checkers(player, point).unwrap_or(0)
    }

    /// Set a checker count in the move search's hot loops: `point` must be 0-25 and the result
    /// must be a valid position. Override with a direct write where possible.
    #[inline]
    fn set_count(&mut self, player: Player, point: usize, count: u8) {
        let _ = self.set_checkers(player, point, count);
    }
    fn set_checkers(&mut self, player: Player, point: usize, count: u8) -> Result<(), String>;
    fn get_turn(&self) -> Player;
    fn set_turn(&mut self, player: Player);
    fn switch_turn(&mut self);
    fn setup_starting_position(&mut self);
    fn double_cube(&mut self, player: Player);
    fn set_cube(&mut self, value: u32, owner: Option<Player>);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_other_player() {
        assert_eq!(Player::Me.other_player(), Player::Opponent);
        assert_eq!(Player::Opponent.other_player(), Player::Me);
    }
}