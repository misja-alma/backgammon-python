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

pub trait Position {
    fn get_checkers(&self, player: Player, point: usize) -> Result<u8, String>;
    fn set_checkers(&mut self, player: Player, point: usize, count: u8) -> Result<(), String>;
    fn get_turn(&self) -> Player;
    fn set_turn(&mut self, player: Player);
    fn switch_turn(&mut self);
    fn hash_code(&self) -> u64;
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