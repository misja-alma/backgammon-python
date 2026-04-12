#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

#[derive(Debug, Clone, PartialEq)]
pub struct Position {
    // Arrays representing checkers for each player on points 0-25
    // Point 0: off the board (bear off), Points 1-24: board positions, Point 25: on the bar
    // Both players use same indexing: move from high numbers to low numbers, bear off to 0
    white_checkers: [u8; 26],
    black_checkers: [u8; 26],
    
    // Whose turn it is
    pub turn: Player,
    
    // Doubling cube state
    pub cube_value: u32,
    pub cube_owner: Option<Player>,
}

impl Position {
    pub fn new() -> Self {
        Position {
            white_checkers: [0; 26],
            black_checkers: [0; 26],
            turn: Player::Me,
            cube_value: 1,
            cube_owner: None,
        }
    }
    
    pub fn set_checkers(&mut self, player: Player, point: usize, count: u8) -> Result<(), String> {
        if point > 25 {
            return Err("Point must be between 0 and 25".to_string());
        }
        
        match player {
            Player::Opponent => self.white_checkers[point] = count,
            Player::Me => self.black_checkers[point] = count,
        }
        Ok(())
    }
    
    pub fn get_checkers(&self, player: Player, point: usize) -> Result<u8, String> {
        if point > 25 {
            return Err("Point must be between 0 and 25".to_string());
        }
        
        Ok(match player {
            Player::Opponent => self.white_checkers[point],
            Player::Me => self.black_checkers[point],
        })
    }
    
    pub fn switch_turn(&mut self) {
        self.turn = self.turn.other_player();
    }
    
    pub fn get_turn(&self) -> Player {
        self.turn
    }
    
    pub fn set_turn(&mut self, player: Player) {
        self.turn = player;
    }
    
    pub fn double_cube(&mut self, player: Player) {
        self.cube_value *= 2;
        self.cube_owner = Some(player);
    }
    
    pub fn set_cube(&mut self, value: u32, owner: Option<Player>) {
        self.cube_value = value;
        self.cube_owner = owner;
    }
    
    pub fn setup_starting_position(&mut self) {
        // Clear all positions
        self.white_checkers = [0; 26];
        self.black_checkers = [0; 26];
        
        // Both players have identical starting positions
        // 2 checkers on point 24
        self.white_checkers[24] = 2;
        self.black_checkers[24] = 2;
        
        // 5 checkers on point 13
        self.white_checkers[13] = 5;
        self.black_checkers[13] = 5;
        
        // 3 checkers on point 8
        self.white_checkers[8] = 3;
        self.black_checkers[8] = 3;
        
        // 5 checkers on point 6
        self.white_checkers[6] = 5;
        self.black_checkers[6] = 5;
        
        // Reset game state
        self.turn = Player::Me;
        self.cube_value = 1;
        self.cube_owner = None;
    }
}

impl Default for Position {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_position() {
        let pos = Position::new();
        assert_eq!(pos.turn, Player::Me);
        assert_eq!(pos.cube_value, 1);
        assert_eq!(pos.cube_owner, None);
        
        // All points should be empty
        for point in 0..26 {
            assert_eq!(pos.get_checkers(Player::Me, point).unwrap(), 0);
            assert_eq!(pos.get_checkers(Player::Opponent, point).unwrap(), 0);
        }
    }
    
    #[test]
    fn test_set_get_checkers() {
        let mut pos = Position::new();
        
        pos.set_checkers(Player::Me, 5, 3).unwrap();
        pos.set_checkers(Player::Opponent, 10, 2).unwrap();
        
        assert_eq!(pos.get_checkers(Player::Me, 5).unwrap(), 3);
        assert_eq!(pos.get_checkers(Player::Opponent, 10).unwrap(), 2);
        assert_eq!(pos.get_checkers(Player::Me, 10).unwrap(), 0);
    }
    
    #[test]
    fn test_invalid_point() {
        let mut pos = Position::new();
        
        assert!(pos.set_checkers(Player::Me, 26, 1).is_err());
        assert!(pos.get_checkers(Player::Me, 26).is_err());
    }
    
    #[test]
    fn test_switch_turn() {
        let mut pos = Position::new();
        assert_eq!(pos.turn, Player::Me);
        
        pos.switch_turn();
        assert_eq!(pos.turn, Player::Opponent);
        
        pos.switch_turn();
        assert_eq!(pos.turn, Player::Me);
    }
    
    #[test]
    fn test_other_player() {
        assert_eq!(Player::Me.other_player(), Player::Opponent);
        assert_eq!(Player::Opponent.other_player(), Player::Me);
    }
    
    #[test]
    fn test_setup_starting_position() {
        let mut pos = Position::new();
        pos.setup_starting_position();
        
        // Check starting position for both players
        assert_eq!(pos.get_checkers(Player::Me, 24).unwrap(), 2);
        assert_eq!(pos.get_checkers(Player::Opponent, 24).unwrap(), 2);
        assert_eq!(pos.get_checkers(Player::Me, 13).unwrap(), 5);
        assert_eq!(pos.get_checkers(Player::Opponent, 13).unwrap(), 5);
        assert_eq!(pos.get_checkers(Player::Me, 8).unwrap(), 3);
        assert_eq!(pos.get_checkers(Player::Opponent, 8).unwrap(), 3);
        assert_eq!(pos.get_checkers(Player::Me, 6).unwrap(), 5);
        assert_eq!(pos.get_checkers(Player::Opponent, 6).unwrap(), 5);
        
        // Check game state
        assert_eq!(pos.turn, Player::Me);
        assert_eq!(pos.cube_value, 1);
        assert_eq!(pos.cube_owner, None);
    }
    
    #[test]
    fn test_double_cube() {
        let mut pos = Position::new();
        
        pos.double_cube(Player::Me);
        assert_eq!(pos.cube_value, 2);
        assert_eq!(pos.cube_owner, Some(Player::Me));
        
        pos.double_cube(Player::Opponent);
        assert_eq!(pos.cube_value, 4);
        assert_eq!(pos.cube_owner, Some(Player::Opponent));
    }
}