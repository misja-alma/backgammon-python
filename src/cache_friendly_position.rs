use crate::position::{invalid_point_error, Position, Player};

// Bit-packed physical board: Me's logical point p → physical p;
// Opponent's logical point p → physical 25-p. Each physical slot holds
// at most one player's checkers for valid game states.
//
// board: 5 bits × 24 physical positions, in two u64 words so that a read is a single 64-bit
//   shift: `low` holds physical points 1-12, `high` points 13-24 (60 bits each).
//   Per slot: bits 3:0 = checker count (0-15), bit 4 = player (1=Me, 0=Opponent).
//
// Points 0 (bear-off) and 25 (bar) are stored separately because both
// players can accumulate checkers there simultaneously:
//   lower nibble = Me's count, upper nibble = Opponent's count.
//
// state: bit 0   = turn (1=Me, 0=Opponent)
//        bits 2:1 = cube owner (00=None, 01=Me, 10=Opponent)
//        bits 5:3 = cube exponent (value = 1 << exp, max 1<<7 = 128)
/// Bit-packed position: 24 bytes, with 24-byte cache keys, but slower checker reads and writes
/// than `SimplePosition`. See `EnginePosition` in lib.rs for when to choose which.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheFriendlyPosition {
    low: u64,
    high: u64,
    bearoff: u8,
    bar: u8,
    state: u8,
}

/// Cache key: the packed position (142 bits) plus the search depth, in 24 bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CacheFriendlyKey {
    board_lo: u64,
    board_hi: u64,
    rest: u32, // bearoff | bar << 8 | state << 16
    depth: u16,
}

impl CacheFriendlyPosition {
    pub fn new() -> Self {
        CacheFriendlyPosition {
            low: 0,
            high: 0,
            bearoff: 0,
            bar: 0,
            state: 0x01, // turn=Me, owner=None, exponent=0 (value=1)
        }
    }

    pub fn get_cube_value(&self) -> u32 {
        1u32 << ((self.state >> 3) & 0x07)
    }

    pub fn get_cube_owner(&self) -> Option<Player> {
        match (self.state >> 1) & 0x03 {
            1 => Some(Player::Me),
            2 => Some(Player::Opponent),
            _ => None,
        }
    }

    fn physical(player: Player, logical: usize) -> usize {
        match player {
            Player::Me => logical,
            Player::Opponent => 25 - logical,
        }
    }

    /// The word holding a physical point (1-24) and the bit offset of its slot in that word.
    #[inline(always)]
    fn slot_location(physical: usize) -> (bool, u32) {
        if physical <= 12 {
            (false, ((physical - 1) * 5) as u32)
        } else {
            (true, ((physical - 13) * 5) as u32)
        }
    }

    #[inline(always)]
    fn read_slot(&self, physical: usize) -> u8 {
        let (is_high, shift) = Self::slot_location(physical);
        let word = if is_high { self.high } else { self.low };
        ((word >> shift) & 0x1F) as u8
    }

    #[inline(always)]
    fn write_slot(&mut self, physical: usize, value: u8) {
        let (is_high, shift) = Self::slot_location(physical);
        let word = if is_high { &mut self.high } else { &mut self.low };
        *word = (*word & !(0x1F << shift)) | ((value as u64 & 0x1F) << shift);
    }
}

#[cold]
#[inline(never)]
fn occupied_point_error() -> Result<(), String> {
    Err("Point is occupied by the other player".to_string())
}

#[cold]
#[inline(never)]
fn too_many_checkers_error() -> Result<(), String> {
    Err("At most 15 checkers per point".to_string())
}

impl Position for CacheFriendlyPosition {
    type CacheKey = CacheFriendlyKey;

    #[inline]
    fn cache_key(&self, depth: u16) -> CacheFriendlyKey {
        CacheFriendlyKey {
            board_lo: self.low,
            board_hi: self.high,
            rest: self.bearoff as u32 | (self.bar as u32) << 8 | (self.state as u32) << 16,
            depth,
        }
    }

    #[inline(always)]
    fn get_checkers(&self, player: Player, point: usize) -> Result<u8, String> {
        if point > 25 {
            return invalid_point_error();
        }
        let count = match point {
            0 => match player {
                Player::Me => self.bearoff & 0x0F,
                Player::Opponent => (self.bearoff >> 4) & 0x0F,
            },
            25 => match player {
                Player::Me => self.bar & 0x0F,
                Player::Opponent => (self.bar >> 4) & 0x0F,
            },
            p => {
                let phys = Self::physical(player, p);
                let slot = self.read_slot(phys);
                let count = slot & 0x0F;
                if count == 0 {
                    return Ok(0);
                }
                let slot_player = if slot & 0x10 != 0 { Player::Me } else { Player::Opponent };
                if slot_player == player { count } else { 0 }
            }
        };
        Ok(count)
    }

    #[inline(always)]
    fn set_checkers(&mut self, player: Player, point: usize, count: u8) -> Result<(), String> {
        if point > 25 {
            return invalid_point_error();
        }
        if count > 15 {
            return too_many_checkers_error();
        }
        match point {
            0 => match player {
                Player::Me => self.bearoff = (self.bearoff & 0xF0) | (count & 0x0F),
                Player::Opponent => self.bearoff = (self.bearoff & 0x0F) | ((count & 0x0F) << 4),
            },
            25 => match player {
                Player::Me => self.bar = (self.bar & 0xF0) | (count & 0x0F),
                Player::Opponent => self.bar = (self.bar & 0x0F) | ((count & 0x0F) << 4),
            },
            p => {
                let phys = Self::physical(player, p);
                if count == 0 {
                    // Clearing is only allowed on the player's own (or an empty) point.
                    let slot = self.read_slot(phys);
                    if slot & 0x0F != 0 && (slot & 0x10 != 0) != (player == Player::Me) {
                        return Ok(());
                    }
                    self.write_slot(phys, 0);
                } else {
                    // A point holds one player's checkers. Refuse to silently overwrite the
                    // other player's (a hit must first move them to the bar).
                    let slot = self.read_slot(phys);
                    if slot & 0x0F != 0 && (slot & 0x10 != 0) != (player == Player::Me) {
                        return occupied_point_error();
                    }
                    let player_bit: u8 = if player == Player::Me { 0x10 } else { 0x00 };
                    self.write_slot(phys, (count & 0x0F) | player_bit);
                }
            }
        }
        Ok(())
    }

    #[inline]
    fn get_turn(&self) -> Player {
        if self.state & 0x01 != 0 { Player::Me } else { Player::Opponent }
    }

    #[inline]
    fn set_turn(&mut self, player: Player) {
        self.state = (self.state & !0x01) | if player == Player::Me { 1 } else { 0 };
    }

    #[inline]
    fn switch_turn(&mut self) {
        self.state ^= 0x01;
    }


    fn setup_starting_position(&mut self) {
        self.low = 0;
        self.high = 0;
        self.bearoff = 0;
        self.bar = 0;
        // Write physical slots directly.
        // Me: logical 24→phys 24, 13→phys 13, 8→phys 8, 6→phys 6.
        self.write_slot(24, 2 | 0x10);
        self.write_slot(13, 5 | 0x10);
        self.write_slot(8,  3 | 0x10);
        self.write_slot(6,  5 | 0x10);
        // Opponent: logical 24→phys 1, 13→phys 12, 8→phys 17, 6→phys 19.
        self.write_slot(1,  2);
        self.write_slot(12, 5);
        self.write_slot(17, 3);
        self.write_slot(19, 5);
        self.state = 0x01; // turn=Me, owner=None, exponent=0 (value=1)
    }

    fn double_cube(&mut self, player: Player) {
        let exp = (self.state >> 3) & 0x07;
        let owner_bits: u8 = if player == Player::Me { 1 } else { 2 };
        self.state = (self.state & 0x01) | (owner_bits << 1) | ((exp + 1) << 3);
    }

    fn set_cube(&mut self, value: u32, owner: Option<Player>) {
        let exp = value.trailing_zeros() as u8;
        let owner_bits: u8 = match owner {
            None => 0,
            Some(Player::Me) => 1,
            Some(Player::Opponent) => 2,
        };
        self.state = (self.state & 0x01) | (owner_bits << 1) | (exp << 3);
    }
}

impl Default for CacheFriendlyPosition {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_position() {
        let pos = CacheFriendlyPosition::new();
        assert_eq!(pos.get_turn(), Player::Me);
        assert_eq!(pos.get_cube_value(), 1);
        assert_eq!(pos.get_cube_owner(), None);

        for point in 0..26 {
            assert_eq!(pos.get_checkers(Player::Me, point).unwrap(), 0);
            assert_eq!(pos.get_checkers(Player::Opponent, point).unwrap(), 0);
        }
    }

    #[test]
    fn test_set_get_checkers() {
        let mut pos = CacheFriendlyPosition::new();

        pos.set_checkers(Player::Me, 5, 3).unwrap();
        pos.set_checkers(Player::Opponent, 10, 2).unwrap();

        assert_eq!(pos.get_checkers(Player::Me, 5).unwrap(), 3);
        assert_eq!(pos.get_checkers(Player::Opponent, 10).unwrap(), 2);
        assert_eq!(pos.get_checkers(Player::Me, 10).unwrap(), 0);
    }

    #[test]
    fn test_cannot_overwrite_other_players_checkers() {
        let mut pos = CacheFriendlyPosition::new();
        pos.set_checkers(Player::Opponent, 17, 2).unwrap(); // Me's 8-point
        assert!(pos.set_checkers(Player::Me, 8, 3).is_err());
        assert_eq!(pos.get_checkers(Player::Opponent, 17).unwrap(), 2);
        // Clearing a point for the player who isn't on it leaves the other player's checkers.
        pos.set_checkers(Player::Me, 8, 0).unwrap();
        assert_eq!(pos.get_checkers(Player::Opponent, 17).unwrap(), 2);
        assert!(pos.set_checkers(Player::Me, 5, 16).is_err());
    }

    #[test]
    fn test_invalid_point() {
        let mut pos = CacheFriendlyPosition::new();

        assert!(pos.set_checkers(Player::Me, 26, 1).is_err());
        assert!(pos.get_checkers(Player::Me, 26).is_err());
    }

    #[test]
    fn test_switch_turn() {
        let mut pos = CacheFriendlyPosition::new();
        assert_eq!(pos.get_turn(), Player::Me);

        pos.switch_turn();
        assert_eq!(pos.get_turn(), Player::Opponent);

        pos.switch_turn();
        assert_eq!(pos.get_turn(), Player::Me);
    }

    #[test]
    fn test_setup_starting_position() {
        let mut pos = CacheFriendlyPosition::new();
        pos.setup_starting_position();

        assert_eq!(pos.get_checkers(Player::Me, 24).unwrap(), 2);
        assert_eq!(pos.get_checkers(Player::Opponent, 24).unwrap(), 2);
        assert_eq!(pos.get_checkers(Player::Me, 13).unwrap(), 5);
        assert_eq!(pos.get_checkers(Player::Opponent, 13).unwrap(), 5);
        assert_eq!(pos.get_checkers(Player::Me, 8).unwrap(), 3);
        assert_eq!(pos.get_checkers(Player::Opponent, 8).unwrap(), 3);
        assert_eq!(pos.get_checkers(Player::Me, 6).unwrap(), 5);
        assert_eq!(pos.get_checkers(Player::Opponent, 6).unwrap(), 5);

        assert_eq!(pos.get_turn(), Player::Me);
        assert_eq!(pos.get_cube_value(), 1);
        assert_eq!(pos.get_cube_owner(), None);
    }

    #[test]
    fn test_double_cube() {
        let mut pos = CacheFriendlyPosition::new();

        pos.double_cube(Player::Me);
        assert_eq!(pos.get_cube_value(), 2);
        assert_eq!(pos.get_cube_owner(), Some(Player::Me));

        pos.double_cube(Player::Opponent);
        assert_eq!(pos.get_cube_value(), 4);
        assert_eq!(pos.get_cube_owner(), Some(Player::Opponent));
    }
}