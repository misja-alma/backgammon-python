pub mod position;
pub mod utils;
pub mod rust_analyzer;

pub use position::{Position, Player};
pub use utils::Utils;
pub use rust_analyzer::RustAnalyzer;

use pyo3::prelude::*;

fn build_position(me_checkers: Vec<u8>, opp_checkers: Vec<u8>, turn: u8) -> Position {
    let mut pos = Position::new();
    for i in 0..26 {
        pos.set_checkers(Player::Me, i, me_checkers[i]).unwrap();
        pos.set_checkers(Player::Opponent, i, opp_checkers[i]).unwrap();
    }
    pos.set_turn(if turn == 1 { Player::Me } else { Player::Opponent });
    pos
}

#[pyfunction]
fn winning_chances(me_checkers: Vec<u8>, opp_checkers: Vec<u8>, turn: u8, player: u8) -> f64 {
    let pos = build_position(me_checkers, opp_checkers, turn);
    let p = if player == 1 { Player::Me } else { Player::Opponent };
    RustAnalyzer::winning_chances(&pos, p)
}

#[pyfunction]
fn best_move(
    me_checkers: Vec<u8>,
    opp_checkers: Vec<u8>,
    turn: u8,
    die1: i32,
    die2: i32,
) -> Option<(Vec<u8>, Vec<u8>, u8, f64)> {
    let pos = build_position(me_checkers, opp_checkers, turn);
    RustAnalyzer::best_move(&pos, die1, die2).map(|(result_pos, pw)| {
        let me_out: Vec<u8> = (0..26).map(|i| result_pos.get_checkers(Player::Me, i).unwrap()).collect();
        let opp_out: Vec<u8> = (0..26).map(|i| result_pos.get_checkers(Player::Opponent, i).unwrap()).collect();
        let turn_out = if result_pos.get_turn() == Player::Me { 1u8 } else { 0u8 };
        (me_out, opp_out, turn_out, pw)
    })
}

#[pymodule]
fn backgammon_rust(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(winning_chances, m)?)?;
    m.add_function(wrap_pyfunction!(best_move, m)?)?;
    Ok(())
}
