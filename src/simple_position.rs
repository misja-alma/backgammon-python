use crate::position::{invalid_point_error, Position, Player};

/// Straightforward position: one checker count per point and player. Fast to read and write,
/// 60 bytes. Used by the engine; see `EnginePosition` in lib.rs for when to choose which.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SimplePosition {
    /// Checker counts per player (indexed by `Player as usize`: Opponent = 0, Me = 1) and point.
    checkers: [[u8; 26]; 2],
    pub turn: Player,
    pub cube_value: u32,
    pub cube_owner: Option<Player>,
}

impl SimplePosition {
    pub fn new() -> Self {
        SimplePosition {
            checkers: [[0; 26]; 2],
            turn: Player::Me,
            cube_value: 1,
            cube_owner: None,
        }
    }
}

impl Position for SimplePosition {
    // The whole position plus the depth: simple, but about 64 bytes per key.
    type CacheKey = ([[u8; 26]; 2], Player, u32, Option<Player>, u16);

    #[inline]
    fn cache_key(&self, depth: u16) -> Self::CacheKey {
        (self.checkers, self.turn, self.cube_value, self.cube_owner, depth)
    }

    #[inline(always)]
    fn count(&self, player: Player, point: usize) -> u8 {
        self.checkers[player as usize][point]
    }

    #[inline(always)]
    fn set_count(&mut self, player: Player, point: usize, count: u8) {
        self.checkers[player as usize][point] = count;
    }

    #[inline]
    fn get_checkers(&self, player: Player, point: usize) -> Result<u8, String> {
        if point > 25 {
            return invalid_point_error();
        }
        Ok(self.checkers[player as usize][point])
    }

    #[inline]
    fn set_checkers(&mut self, player: Player, point: usize, count: u8) -> Result<(), String> {
        if point > 25 {
            return invalid_point_error();
        }
        self.checkers[player as usize][point] = count;
        Ok(())
    }

    #[inline]
    fn get_turn(&self) -> Player {
        self.turn
    }

    #[inline]
    fn set_turn(&mut self, player: Player) {
        self.turn = player;
    }

    #[inline]
    fn switch_turn(&mut self) {
        self.turn = self.turn.other_player();
    }


    fn setup_starting_position(&mut self) {
        let mut one_side = [0; 26];
        one_side[24] = 2;
        one_side[13] = 5;
        one_side[8] = 3;
        one_side[6] = 5;
        self.checkers = [one_side; 2];
        self.turn = Player::Me;
        self.cube_value = 1;
        self.cube_owner = None;
    }

    fn double_cube(&mut self, player: Player) {
        self.cube_value *= 2;
        self.cube_owner = Some(player);
    }

    fn set_cube(&mut self, value: u32, owner: Option<Player>) {
        self.cube_value = value;
        self.cube_owner = owner;
    }
}

impl Default for SimplePosition {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_position() {
        let pos = SimplePosition::new();
        assert_eq!(pos.turn, Player::Me);
        assert_eq!(pos.cube_value, 1);
        assert_eq!(pos.cube_owner, None);

        for point in 0..26 {
            assert_eq!(pos.get_checkers(Player::Me, point).unwrap(), 0);
            assert_eq!(pos.get_checkers(Player::Opponent, point).unwrap(), 0);
        }
    }

    #[test]
    fn test_set_get_checkers() {
        let mut pos = SimplePosition::new();

        pos.set_checkers(Player::Me, 5, 3).unwrap();
        pos.set_checkers(Player::Opponent, 10, 2).unwrap();

        assert_eq!(pos.get_checkers(Player::Me, 5).unwrap(), 3);
        assert_eq!(pos.get_checkers(Player::Opponent, 10).unwrap(), 2);
        assert_eq!(pos.get_checkers(Player::Me, 10).unwrap(), 0);
    }

    #[test]
    fn test_invalid_point() {
        let mut pos = SimplePosition::new();

        assert!(pos.set_checkers(Player::Me, 26, 1).is_err());
        assert!(pos.get_checkers(Player::Me, 26).is_err());
    }

    #[test]
    fn test_switch_turn() {
        let mut pos = SimplePosition::new();
        assert_eq!(pos.turn, Player::Me);

        pos.switch_turn();
        assert_eq!(pos.turn, Player::Opponent);

        pos.switch_turn();
        assert_eq!(pos.turn, Player::Me);
    }

    #[test]
    fn test_setup_starting_position() {
        let mut pos = SimplePosition::new();
        pos.setup_starting_position();

        assert_eq!(pos.get_checkers(Player::Me, 24).unwrap(), 2);
        assert_eq!(pos.get_checkers(Player::Opponent, 24).unwrap(), 2);
        assert_eq!(pos.get_checkers(Player::Me, 13).unwrap(), 5);
        assert_eq!(pos.get_checkers(Player::Opponent, 13).unwrap(), 5);
        assert_eq!(pos.get_checkers(Player::Me, 8).unwrap(), 3);
        assert_eq!(pos.get_checkers(Player::Opponent, 8).unwrap(), 3);
        assert_eq!(pos.get_checkers(Player::Me, 6).unwrap(), 5);
        assert_eq!(pos.get_checkers(Player::Opponent, 6).unwrap(), 5);

        assert_eq!(pos.turn, Player::Me);
        assert_eq!(pos.cube_value, 1);
        assert_eq!(pos.cube_owner, None);
    }

    #[test]
    fn test_double_cube() {
        let mut pos = SimplePosition::new();

        pos.double_cube(Player::Me);
        assert_eq!(pos.cube_value, 2);
        assert_eq!(pos.cube_owner, Some(Player::Me));

        pos.double_cube(Player::Opponent);
        assert_eq!(pos.cube_value, 4);
        assert_eq!(pos.cube_owner, Some(Player::Opponent));
    }
}