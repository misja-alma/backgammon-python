pub mod position;
pub mod utils;
pub mod rust_analyzer;

pub use position::{Position, Player};
pub use utils::Utils;
pub use rust_analyzer::RustAnalyzer;

#[cfg(feature = "extension-module")]
use pyo3::prelude::*;

#[cfg(feature = "extension-module")]
use std::cell::RefCell;

#[cfg(feature = "extension-module")]
// A single RustAnalyzer instance per thread, persisted for the lifetime of the game so that
// its position cache accumulates across analysis calls rather than being discarded after each one.
// thread_local is used because RustAnalyzer contains RefCell, which is not Send.
thread_local! {
    static ANALYZER: RefCell<RustAnalyzer> = RefCell::new(RustAnalyzer::new());
}

#[cfg(feature = "extension-module")]
fn build_position(me_checkers: Vec<u8>, opp_checkers: Vec<u8>, turn: u8) -> Position {
    let mut pos = Position::new();
    for i in 0..26 {
        pos.set_checkers(Player::Me, i, me_checkers[i]).unwrap();
        pos.set_checkers(Player::Opponent, i, opp_checkers[i]).unwrap();
    }
    pos.set_turn(if turn == 1 { Player::Me } else { Player::Opponent });
    pos
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn winning_chances(me_checkers: Vec<u8>, opp_checkers: Vec<u8>, turn: u8, player: u8, max_depth: u32) -> f64 {
    let pos = build_position(me_checkers, opp_checkers, turn);
    let p = if player == 1 { Player::Me } else { Player::Opponent };
    ANALYZER.with(|a| a.borrow().winning_chances(&pos, p, max_depth))
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn best_move(
    me_checkers: Vec<u8>,
    opp_checkers: Vec<u8>,
    turn: u8,
    die1: i32,
    die2: i32,
    max_depth: u32,
) -> Option<(Vec<u8>, Vec<u8>, u8, f64)> {
    let pos = build_position(me_checkers, opp_checkers, turn);
    ANALYZER.with(|a| a.borrow().best_move(&pos, die1, die2, max_depth)).map(|(result_pos, pw)| {
        let me_out: Vec<u8> = (0..26).map(|i| result_pos.get_checkers(Player::Me, i).unwrap()).collect();
        let opp_out: Vec<u8> = (0..26).map(|i| result_pos.get_checkers(Player::Opponent, i).unwrap()).collect();
        let turn_out = if result_pos.get_turn() == Player::Me { 1u8 } else { 0u8 };
        (me_out, opp_out, turn_out, pw)
    })
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn clear_cache() {
    ANALYZER.with(|a| a.borrow().clear_cache());
}

#[cfg(feature = "extension-module")]
#[pymodule]
fn backgammon_rust(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(winning_chances, m)?)?;
    m.add_function(wrap_pyfunction!(best_move, m)?)?;
    m.add_function(wrap_pyfunction!(clear_cache, m)?)?;
    Ok(())
}
