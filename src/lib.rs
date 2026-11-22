pub mod evaluator;
pub mod bearoff;
pub mod encoding;
pub mod neural_net;
pub mod td_training;
pub mod fast_hash;
pub mod transposition_table;
pub mod move_generator;
pub mod game;
pub mod position;
pub mod simple_position;
pub mod cache_friendly_position;
pub mod utils;
pub mod rust_analyzer;

pub use position::{Position, Player};
pub use utils::Utils;
pub use rust_analyzer::RustAnalyzer;
pub use evaluator::{Evaluator, TerminalEvaluator, PipCountEvaluator};
pub use neural_net::{NeuralNet, NeuralNetEvaluator};

#[cfg(feature = "extension-module")]
use pyo3::prelude::*;

#[cfg(feature = "extension-module")]
use std::cell::RefCell;

pub use cache_friendly_position::CacheFriendlyPosition;
pub use simple_position::SimplePosition;

/// The position representation the engine uses (the Python bindings and the Rust-driven game).
/// The search is generic over `Position`, so this is the only line to change to switch.
///
/// How to choose:
/// - `SimplePosition` (two `[u8; 26]` arrays, 60 bytes): a checker read is a plain array lookup.
///   Deep searches spend almost all their time reading and writing checker counts during move
///   generation, so this is the faster choice for searching (about 1.5× faster than
///   `CacheFriendlyPosition` at depth 3; see `cargo run --bin position_benchmark --release`).
/// - `CacheFriendlyPosition` (bit-packed, 24 bytes): every read and write unpacks bits, which
///   costs more, but positions and cache keys are 2.5× smaller (24 vs 60 bytes per key). Choose
///   it when memory is the limit, e.g. a cache that must hold many millions of positions, or when
///   positions are stored in bulk rather than searched.
///
/// Re-run `position_benchmark` after changing move generation or the evaluator: the balance
/// between the two has shifted before (hashing-heavy code favoured `CacheFriendlyPosition`).
pub type EnginePosition = SimplePosition;

#[cfg(feature = "extension-module")]
// A single RustAnalyzer instance per thread, persisted for the lifetime of the game so that
// its position cache accumulates across analysis calls rather than being discarded after each one.
// thread_local is used because RustAnalyzer contains RefCell, which is not Send.
thread_local! {
    static ANALYZER: RefCell<RustAnalyzer<EnginePosition>> = RefCell::new(RustAnalyzer::new());
}

fn player_from_u8(p: u8) -> Player {
    if p == 1 { Player::Me } else { Player::Opponent }
}

/// Builds a position from per-player checker arrays (each player's own perspective, 26 slots),
/// the side to move (1 = Me, 0 = Opponent) and the doubling cube.
/// `cube_owner` is `None` for a centered cube, otherwise 1 = Me, 0 = Opponent.
pub fn build_position(
    me_checkers: Vec<u8>,
    opp_checkers: Vec<u8>,
    turn: u8,
    cube_value: u32,
    cube_owner: Option<u8>,
) -> Result<EnginePosition, String> {
    if me_checkers.len() != 26 || opp_checkers.len() != 26 {
        return Err("Checker arrays must have 26 entries".to_string());
    }
    // Reject positions that can't occur in a game (SimplePosition would store them as given).
    if me_checkers.iter().chain(&opp_checkers).any(|&c| c > 15) {
        return Err("At most 15 checkers per point".to_string());
    }
    if (1..=24).any(|p| me_checkers[p] > 0 && opp_checkers[25 - p] > 0) {
        return Err("A point can't hold checkers of both players".to_string());
    }
    let mut pos = EnginePosition::new();
    for i in 0..26 {
        if me_checkers[i] > 0 { pos.set_checkers(Player::Me, i, me_checkers[i])?; }
        if opp_checkers[i] > 0 { pos.set_checkers(Player::Opponent, i, opp_checkers[i])?; }
    }
    pos.set_turn(player_from_u8(turn));
    pos.set_cube(cube_value, cube_owner.map(player_from_u8));
    Ok(pos)
}

#[cfg(feature = "extension-module")]
/// Equity of the position for `player` (1 = Me, 0 = Opponent), searching `max_depth` roll levels.
#[pyfunction]
#[pyo3(signature = (me_checkers, opp_checkers, turn, cube_value, cube_owner, player, max_depth))]
#[allow(clippy::too_many_arguments)]
fn equity(
    py: Python<'_>,
    me_checkers: Vec<u8>,
    opp_checkers: Vec<u8>,
    turn: u8,
    cube_value: u32,
    cube_owner: Option<u8>,
    player: u8,
    max_depth: u32,
) -> PyResult<f64> {
    let pos = build_position(me_checkers, opp_checkers, turn, cube_value, cube_owner)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let p = player_from_u8(player);
    // Release the GIL so the Python UI thread keeps running while we search.
    Ok(py.allow_threads(move || ANALYZER.with(|a| a.borrow().equity(&pos, p, max_depth))))
}

#[cfg(feature = "extension-module")]
/// Best move with the given dice: (me_checkers, opp_checkers, turn, equity for the mover).
#[pyfunction]
#[pyo3(signature = (me_checkers, opp_checkers, turn, cube_value, cube_owner, die1, die2, max_depth))]
#[allow(clippy::too_many_arguments)]
fn best_move(
    py: Python<'_>,
    me_checkers: Vec<u8>,
    opp_checkers: Vec<u8>,
    turn: u8,
    cube_value: u32,
    cube_owner: Option<u8>,
    die1: i32,
    die2: i32,
    max_depth: u32,
) -> PyResult<Option<(Vec<u8>, Vec<u8>, u8, f64)>> {
    let pos = build_position(me_checkers, opp_checkers, turn, cube_value, cube_owner)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(py.allow_threads(move || ANALYZER.with(|a| a.borrow().best_move(&pos, die1, die2, max_depth))).map(|(result_pos, equity)| {
        let me_out: Vec<u8> = (0..26).map(|i| result_pos.get_checkers(Player::Me, i).unwrap()).collect();
        let opp_out: Vec<u8> = (0..26).map(|i| result_pos.get_checkers(Player::Opponent, i).unwrap()).collect();
        let turn_out = if result_pos.get_turn() == Player::Me { 1u8 } else { 0u8 };
        (me_out, opp_out, turn_out, equity)
    }))
}

#[cfg(feature = "extension-module")]
#[pyfunction]
fn clear_cache() {
    // Note: the analyzer (and its cache) is thread-local, so this only clears the cache of the
    // calling thread. Python calls all engine functions from one dedicated worker thread.
    ANALYZER.with(|a| a.borrow().clear_cache());
}

/// Replace the engine's cache with an empty one of at most `megabytes` (at least 1).
/// Like the cache itself this is per thread: call it on the thread that runs the engine.
#[cfg(feature = "extension-module")]
#[pyfunction]
fn set_cache_size(megabytes: usize) {
    ANALYZER.with(|a| a.borrow_mut().set_cache_mb(megabytes));
}

/// Memory reserved for the engine's cache on this thread, in MB (rounded to the nearest MB).
#[cfg(feature = "extension-module")]
#[pyfunction]
fn cache_size_mb() -> usize {
    ANALYZER.with(|a| (a.borrow().cache_size_bytes() + (1 << 19)) >> 20)
}

/// Play a complete game between two Rust engines, entirely in Rust, from the backgammon or
/// (`nackgammon=True`) the nackgammon starting position.
/// Returns (winner, me_checkers, opp_checkers, turns), with winner 1 = Me, 0 = Opponent.
/// Cannot be interrupted once started.
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (me_depth, opponent_depth, seed, nackgammon=false))]
fn play_game(py: Python<'_>, me_depth: u32, opponent_depth: u32, seed: u64, nackgammon: bool)
             -> PyResult<(u8, Vec<u8>, Vec<u8>, u32)> {
    let variant = if nackgammon { game::Variant::Nackgammon } else { game::Variant::Backgammon };
    let result = py.allow_threads(move || {
        ANALYZER.with(|a| game::play_game(&a.borrow(), variant, me_depth, opponent_depth, &mut game::Dice::new(seed)))
    });
    let result = result.map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    let pos = &result.final_position;
    let me_out: Vec<u8> = (0..26).map(|i| pos.get_checkers(Player::Me, i).unwrap()).collect();
    let opp_out: Vec<u8> = (0..26).map(|i| pos.get_checkers(Player::Opponent, i).unwrap()).collect();
    let winner = if result.winner == Player::Me { 1u8 } else { 0u8 };
    Ok((winner, me_out, opp_out, result.turns))
}

/// Use the neural net in the weight file at `path` as the engine's evaluator, on this thread
/// (like the cache, the analyzer is per thread). Returns a description of the net.
#[cfg(feature = "extension-module")]
#[pyfunction]
fn load_network(path: &str) -> PyResult<String> {
    let net = NeuralNet::load(std::path::Path::new(path)).map_err(pyo3::exceptions::PyValueError::new_err)?;
    let description = net.description();
    ANALYZER.with(|a| a.borrow_mut().set_evaluator(Box::new(NeuralNetEvaluator { net })));
    Ok(description)
}

/// Use the two-sided bearoff database at `path` (see bearoff.rs) for exact values of bearoff
/// positions on this thread, or none (`None`). The file is memory-mapped, so its pages are shared
/// by all threads and processes using it. Returns a description.
#[cfg(feature = "extension-module")]
#[pyfunction]
#[pyo3(signature = (path))]
fn load_bearoff(path: Option<&str>) -> PyResult<String> {
    let db = match path {
        Some(path) => Some(std::sync::Arc::new(
            bearoff::BearoffDb::open(std::path::Path::new(path)).map_err(pyo3::exceptions::PyValueError::new_err)?)),
        None => None,
    };
    let description = db.as_ref().map_or("no bearoff database".to_string(),
                                         |db| format!("two-sided bearoff database 6x{}", db.checkers()));
    ANALYZER.with(|a| a.borrow_mut().set_bearoff(db));
    Ok(description)
}

/// Go back to the placeholder evaluator on this thread.
#[cfg(feature = "extension-module")]
#[pyfunction]
fn use_placeholder_evaluator() {
    ANALYZER.with(|a| a.borrow_mut().set_evaluator(Box::new(TerminalEvaluator)));
}

/// Description of the evaluator in use on this thread.
#[cfg(feature = "extension-module")]
#[pyfunction]
fn evaluator_name() -> String {
    ANALYZER.with(|a| a.borrow().evaluator().name())
}

#[cfg(feature = "extension-module")]
#[pymodule]
fn backgammon_rust(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(equity, m)?)?;
    m.add_function(wrap_pyfunction!(best_move, m)?)?;
    m.add_function(wrap_pyfunction!(clear_cache, m)?)?;
    m.add_function(wrap_pyfunction!(play_game, m)?)?;
    m.add_function(wrap_pyfunction!(set_cache_size, m)?)?;
    m.add_function(wrap_pyfunction!(cache_size_mb, m)?)?;
    m.add_function(wrap_pyfunction!(load_network, m)?)?;
    m.add_function(wrap_pyfunction!(load_bearoff, m)?)?;
    m.add_function(wrap_pyfunction!(use_placeholder_evaluator, m)?)?;
    m.add_function(wrap_pyfunction!(evaluator_name, m)?)?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_position_sets_cube() {
        let mut me = vec![0u8; 26];
        let mut opp = vec![0u8; 26];
        me[6] = 2;
        opp[6] = 2;

        let centered = build_position(me.clone(), opp.clone(), 1, 1, None).unwrap();
        let owned_by_me = build_position(me.clone(), opp.clone(), 1, 4, Some(1)).unwrap();
        let owned_by_opp = build_position(me, opp, 1, 4, Some(0)).unwrap();

        // Cube state is part of the position identity, so it must be part of the cache key.
        assert_ne!(centered.cache_key(1), owned_by_me.cache_key(1));
        assert_ne!(owned_by_me.cache_key(1), owned_by_opp.cache_key(1));
        assert_eq!(owned_by_me.cube_value, 4);
        assert_eq!(owned_by_opp.cube_owner, Some(Player::Opponent));
    }

    #[test]
    fn test_build_position_rejects_impossible_positions() {
        let mut me = vec![0u8; 26];
        let mut opp = vec![0u8; 26];
        me[8] = 3;
        opp[17] = 2; // the same point as Me's 8
        assert!(build_position(me.clone(), opp.clone(), 1, 1, None).is_err());
        opp[17] = 0;
        me[8] = 16;
        assert!(build_position(me, opp, 1, 1, None).is_err());
    }
}
