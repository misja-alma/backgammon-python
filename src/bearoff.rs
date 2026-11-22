//! Two-sided bearoff database: the exact winning chance of every position in which both players
//! have all their remaining checkers on their own six home points (no contact is possible).
//!
//! Cubeless and without gammons: a value is the probability that the side to move, before it
//! rolls, bears off first, when both sides play to maximize their own winning chance. It is the
//! exact solution of
//!
//! ```text
//! V(a, b) = Σ over the 21 rolls of (1 or 2)/36 · max over the moves a → a' of
//!           (1 if a' is empty, else 1 − V(b, a'))
//! ```
//!
//! where `a` is the side to move and `b` the other side. (Every die can always be played in a
//! bearoff, so a move always uses all dice, or bears off the last checker.)
//!
//! How gnubg and bgsage do it, for comparison:
//! - gnubg (`makebearoff.c`) builds two-sided databases of P points × C checkers (C ≤ 11) with the
//!   same recursion (maximizing equity), stored as 16-bit equities, optionally with 3 cubeful
//!   values; it ships a small one (6 × 6) and falls back to one-sided roll distributions beyond it.
//! - bgsage stores only one-sided data (6 × 15): for each side the distribution of the number of
//!   rolls needed, with moves chosen to minimize the expected number of rolls, and combines the two
//!   sides' distributions. That is fast and small but not exact: minimizing expected rolls is not
//!   always the move that maximizes the winning chance against a particular opponent.
//!
//! This database covers all 6 × 15 positions, which is every bearoff position:
//! n = C(21, 6) = 54,264 positions per side, n² ≈ 2.9 billion pairs, 2 bytes each (5.5 GB).
//!
//! **Generation** (`generate`): the one-sided moves are computed once (for each position and roll
//! the distinct resulting positions), so the two-sided pass is only table lookups. Every move
//! lowers the mover's pip count, so `V(a, b)` depends only on pairs with a smaller total pip count:
//! the pairs are processed in layers of equal total pip count, each layer in parallel. Values are
//! computed in f32 (summed in f64) and written as u16 at the end, so rounding happens once
//! (error ≤ 1/131070 per value), not at every step of the recursion.
//!
//! **File** (`write`): a 64-byte header (magic, version, points, checkers, n), then n² u16 little
//! endian, `round(V · 65535)`, at index `a · n + b` (a = side to move). `BearoffDb::open`
//! memory-maps it, so the pages are shared between processes and loaded on demand.
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::position::{Player, Position};

pub const POINTS: usize = 6;
pub const MAX_CHECKERS: usize = 15;
const MAGIC: &[u8; 8] = b"BGBEAROF";
const VERSION: u32 = 1;
const HEADER_BYTES: usize = 64;
const SCALE: f32 = 65535.0;

/// One side's checkers on points 1-6 (index 0 = point 1).
pub type Side = [u8; POINTS];

/// Number of ways to put at most `m` checkers on `k` points: C(k + m, k).
fn count(k: usize, m: usize) -> usize {
    (1..=k).fold(1, |acc, i| acc * (m + i) / i)
}

/// Ranks one-sided positions with at most `checkers` checkers: lexicographic order of
/// (point 1, ..., point 6), so 0 is the empty side (all borne off).
#[derive(Clone)]
pub struct Indexer {
    pub checkers: usize,
    pub n: usize,
    /// cum[p][remaining][v]: positions that come before having v checkers on point p+1, given
    /// `remaining` checkers left for points p+1..6.
    cum: Vec<u32>,
}

impl Indexer {
    pub fn new(checkers: usize) -> Indexer {
        assert!(checkers <= MAX_CHECKERS);
        let w = checkers + 2;
        let mut cum = vec![0u32; POINTS * w * w];
        for p in 0..POINTS {
            for remaining in 0..=checkers {
                let mut total = 0;
                for v in 0..=remaining {
                    cum[(p * w + remaining) * w + v] = total as u32;
                    total += count(POINTS - 1 - p, remaining - v);
                }
            }
        }
        Indexer { checkers, n: count(POINTS, checkers), cum }
    }

    /// The index of `side`, which must have at most `checkers` checkers.
    #[inline]
    pub fn rank(&self, side: &Side) -> usize {
        let w = self.checkers + 2;
        let mut remaining = self.checkers;
        let mut index = 0;
        for (p, &v) in side.iter().enumerate() {
            index += self.cum[(p * w + remaining) * w + v as usize] as usize;
            remaining -= v as usize;
        }
        index
    }

    /// All positions, in index order.
    pub fn all(&self) -> Vec<Side> {
        let mut out = Vec::with_capacity(self.n);
        let mut side = [0u8; POINTS];
        fn fill(p: usize, remaining: usize, side: &mut Side, out: &mut Vec<Side>) {
            if p == POINTS {
                out.push(*side);
                return;
            }
            for v in 0..=remaining {
                side[p] = v as u8;
                fill(p + 1, remaining - v, side, out);
            }
            side[p] = 0;
        }
        fill(0, self.checkers, &mut side, &mut out);
        out
    }
}

pub fn pips(side: &Side) -> usize {
    side.iter().enumerate().map(|(p, &v)| (p + 1) * v as usize).sum()
}

fn is_empty(side: &Side) -> bool {
    side.iter().all(|&v| v == 0)
}

/// The 21 distinct rolls with their weights (out of 36).
pub fn rolls() -> Vec<(u8, u8, u32)> {
    let mut out = Vec::with_capacity(21);
    for d1 in 1..=6u8 {
        for d2 in d1..=6u8 {
            out.push((d1, d2, if d1 == d2 { 1 } else { 2 }));
        }
    }
    out
}

/// The distinct positions after playing `dice` (in order) from `side`. A die moves a checker
/// from point p to p − d, bears it off from point d, or bears off the highest checker when it is
/// below d. Once the side is empty the game is over and the remaining dice are not played.
fn play(side: Side, dice: &[u8], out: &mut Vec<Side>) {
    if dice.is_empty() || is_empty(&side) {
        out.push(side);
        return;
    }
    let d = dice[0] as usize;
    let highest = (0..POINTS).rev().find(|&p| side[p] > 0).unwrap() + 1;
    for point in 1..=POINTS {
        if side[point - 1] == 0 {
            continue;
        }
        let legal = point >= d || point == highest;
        if !legal {
            continue;
        }
        let mut next = side;
        next[point - 1] -= 1;
        if point > d {
            next[point - d - 1] += 1;
        }
        play(next, &dice[1..], out);
    }
}

/// The distinct positions `side` can reach with the roll `d1`-`d2`, sorted.
pub fn moves(side: &Side, d1: u8, d2: u8) -> Vec<Side> {
    let mut out = Vec::new();
    if d1 == d2 {
        play(*side, &[d1; 4], &mut out);
    } else {
        play(*side, &[d1, d2], &mut out);
        play(*side, &[d2, d1], &mut out);
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// For every position and roll, the indices of the distinct resulting positions.
struct MoveTable {
    /// starts[a * 21 + r] .. starts[a * 21 + r + 1] index into `targets`.
    starts: Vec<u32>,
    targets: Vec<u32>,
}

impl MoveTable {
    fn new(indexer: &Indexer, positions: &[Side]) -> MoveTable {
        let rolls = rolls();
        let mut starts = Vec::with_capacity(positions.len() * 21 + 1);
        let mut targets = Vec::new();
        for side in positions {
            for &(d1, d2, _) in &rolls {
                starts.push(targets.len() as u32);
                if !is_empty(side) {
                    targets.extend(moves(side, d1, d2).iter().map(|s| indexer.rank(s) as u32));
                }
            }
        }
        starts.push(targets.len() as u32);
        MoveTable { starts, targets }
    }

    #[inline]
    fn targets(&self, a: usize, roll: usize) -> &[u32] {
        let i = a * 21 + roll;
        &self.targets[self.starts[i] as usize..self.starts[i + 1] as usize]
    }
}

/// Shares the value table between threads. Within a layer every thread writes different cells
/// and reads only cells of earlier layers, so the accesses never overlap.
#[derive(Clone, Copy)]
struct SharedValues(*mut f32);
unsafe impl Send for SharedValues {}
unsafe impl Sync for SharedValues {}

/// The winning chances of all pairs, as `values[a * n + b]` for the side to move `a`. Row 0 (the
/// side to move has borne off) is 1, column 0 (the other side has) is 0. `progress` is called after
/// each layer with (layer, number of layers).
pub fn generate(checkers: usize, threads: usize, mut progress: impl FnMut(usize, usize)) -> Vec<f32> {
    let indexer = Indexer::new(checkers);
    let n = indexer.n;
    let positions = indexer.all();
    let table = MoveTable::new(&indexer, &positions);
    let weights: Vec<f64> = rolls().iter().map(|&(_, _, w)| w as f64 / 36.0).collect();

    let mut values = vec![0f32; n * n];
    values[..n].iter_mut().for_each(|v| *v = 1.0);

    let max_pips = POINTS * checkers;
    let mut by_pips: Vec<Vec<u32>> = vec![Vec::new(); max_pips + 1];
    for (i, side) in positions.iter().enumerate() {
        by_pips[pips(side)].push(i as u32);
    }

    let shared = SharedValues(values.as_mut_ptr());
    let layers = 2 * max_pips;
    for total in 2..=layers {
        // The other side's positions b in this layer; the side to move then has total − pips(b).
        let bs: Vec<u32> = (1..=max_pips.min(total - 1))
            .filter(|&pb| total - pb <= max_pips)
            .flat_map(|pb| by_pips[pb].iter().copied())
            .collect();
        let next = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..threads.max(1) {
                scope.spawn(|| {
                    let shared = shared;
                    const CHUNK: usize = 16;
                    loop {
                        let start = next.fetch_add(CHUNK, Ordering::Relaxed);
                        if start >= bs.len() {
                            break;
                        }
                        for &b in &bs[start..(start + CHUNK).min(bs.len())] {
                            let b = b as usize;
                            let row_b = b * n;
                            for &a in &by_pips[total - pips(&positions[b])] {
                                let a = a as usize;
                                let mut sum = 0.0f64;
                                for (roll, weight) in weights.iter().enumerate() {
                                    let mut best = 0.0f32;
                                    for &t in table.targets(a, roll) {
                                        // t == 0: the move bears off the last checker.
                                        let v = if t == 0 { 1.0 } else {
                                            1.0 - unsafe { *shared.0.add(row_b + t as usize) }
                                        };
                                        best = best.max(v);
                                    }
                                    sum += weight * best as f64;
                                }
                                unsafe { *shared.0.add(a * n + b) = sum as f32 };
                            }
                        }
                    }
                });
            }
        });
        progress(total - 1, layers - 1);
    }
    values
}

/// Writes the values as the database file (see the module docs).
pub fn write(path: &Path, checkers: usize, values: &[f32]) -> std::io::Result<()> {
    let n = Indexer::new(checkers).n;
    assert_eq!(values.len(), n * n);
    let mut out = BufWriter::with_capacity(1 << 24, File::create(path)?);
    let mut header = [0u8; HEADER_BYTES];
    header[..8].copy_from_slice(MAGIC);
    for (i, v) in [VERSION, POINTS as u32, checkers as u32, n as u32].iter().enumerate() {
        header[8 + 4 * i..12 + 4 * i].copy_from_slice(&v.to_le_bytes());
    }
    out.write_all(&header)?;
    let mut buffer = Vec::with_capacity(2 * n);
    for row in values.chunks(n) {
        buffer.clear();
        for &v in row {
            buffer.extend_from_slice(&((v.clamp(0.0, 1.0) * SCALE).round() as u16).to_le_bytes());
        }
        out.write_all(&buffer)?;
    }
    out.flush()
}

/// A database file, memory-mapped.
pub struct BearoffDb {
    map: memmap2::Mmap,
    indexer: Indexer,
}

impl BearoffDb {
    pub fn open(path: &Path) -> Result<BearoffDb, String> {
        let file = File::open(path).map_err(|e| format!("{}: {}", path.display(), e))?;
        let map = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| format!("{}: {}", path.display(), e))?;
        let bad = |what: &str| format!("{}: not a bearoff database ({})", path.display(), what);
        if map.len() < HEADER_BYTES || &map[..8] != MAGIC {
            return Err(bad("magic"));
        }
        let field = |i: usize| u32::from_le_bytes(map[8 + 4 * i..12 + 4 * i].try_into().unwrap()) as usize;
        if field(0) != VERSION as usize || field(1) != POINTS || field(2) > MAX_CHECKERS {
            return Err(bad("version or size"));
        }
        let indexer = Indexer::new(field(2));
        if field(3) != indexer.n || map.len() != HEADER_BYTES + 2 * indexer.n * indexer.n {
            return Err(bad("length"));
        }
        Ok(BearoffDb { map, indexer })
    }

    pub fn checkers(&self) -> usize {
        self.indexer.checkers
    }

    /// The winning chance of the side to move (`a`) against `b`.
    #[inline]
    pub fn win_probability_sides(&self, a: &Side, b: &Side) -> f64 {
        let i = HEADER_BYTES + 2 * (self.indexer.rank(a) * self.indexer.n + self.indexer.rank(b));
        u16::from_le_bytes([self.map[i], self.map[i + 1]]) as f64 / SCALE as f64
    }

    /// The player's checkers as a `Side`, or None when some are outside the home board or there
    /// are more than the database covers.
    fn side<P: Position>(&self, position: &P, player: Player) -> Option<Side> {
        if (POINTS + 1..=25).any(|p| position.count(player, p) > 0) {
            return None;
        }
        let side: Side = std::array::from_fn(|i| position.count(player, i + 1));
        (side.iter().map(|&v| v as usize).sum::<usize>() <= self.indexer.checkers).then_some(side)
    }

    /// The winning chance of the side to move, before it rolls, when the position is in the
    /// database (both sides' checkers all in their home boards), else None.
    #[inline]
    pub fn win_probability<P: Position>(&self, position: &P) -> Option<f64> {
        let turn = position.get_turn();
        let a = self.side(position, turn)?;
        let b = self.side(position, turn.other_player())?;
        Some(self.win_probability_sides(&a, &b))
    }

    /// Equity (2p − 1) of the side to move, when the position is in the database.
    #[inline]
    pub fn equity<P: Position>(&self, position: &P) -> Option<f64> {
        self.win_probability(position).map(|p| 2.0 * p - 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluator::TerminalEvaluator;
    use crate::move_generator::MoveGenerator;
    use crate::rust_analyzer::RustAnalyzer;
    use crate::simple_position::SimplePosition;

    #[test]
    fn test_rank_matches_enumeration_order() {
        for checkers in [0, 1, 3, 15] {
            let indexer = Indexer::new(checkers);
            let all = indexer.all();
            assert_eq!(all.len(), indexer.n);
            for (i, side) in all.iter().enumerate() {
                assert_eq!(indexer.rank(side), i);
            }
        }
        assert_eq!(Indexer::new(15).n, 54264);
    }

    fn to_position(me: &Side, opponent: &Side) -> SimplePosition {
        let mut p = SimplePosition::new();
        for (player, side) in [(Player::Me, me), (Player::Opponent, opponent)] {
            let on_board: u8 = side.iter().sum();
            for (i, &v) in side.iter().enumerate() {
                if v > 0 {
                    p.set_checkers(player, i + 1, v).unwrap();
                }
            }
            if on_board < 15 {
                p.set_checkers(player, 0, 15 - on_board).unwrap();
            }
        }
        p.set_turn(Player::Me);
        p
    }

    #[test]
    fn test_moves_match_the_engine() {
        let indexer = Indexer::new(15);
        let opponent: Side = [1, 0, 0, 0, 0, 0];
        let mut generator = MoveGenerator::<SimplePosition>::new();
        for side in indexer.all().iter().step_by(97).filter(|s| !is_empty(s)) {
            let position = to_position(side, &opponent);
            for (d1, d2, _) in rolls() {
                let mut expected: Vec<Side> = generator.legal_positions(&position, d1 as i32, d2 as i32).iter()
                    .map(|p| std::array::from_fn(|i| p.count(Player::Me, i + 1)))
                    .collect();
                expected.sort_unstable();
                expected.dedup();
                assert_eq!(moves(side, d1, d2), expected, "{:?} {}-{}", side, d1, d2);
            }
        }
    }

    #[test]
    fn test_values_match_full_depth_search() {
        // With at most 3 checkers on points 1-3 every game ends within 3 rolls per side, so a
        // depth-6 search with the placeholder evaluator is exact.
        let checkers = 3;
        let values = generate(checkers, 4, |_, _| {});
        let indexer = Indexer::new(checkers);
        let analyzer: RustAnalyzer<SimplePosition> = RustAnalyzer::with_evaluator(Box::new(TerminalEvaluator), 16);
        let small: Vec<Side> = indexer.all().into_iter()
            .filter(|s| !is_empty(s) && s[3..].iter().all(|&v| v == 0))
            .collect();
        for a in &small {
            for b in &small {
                let exact = 2.0 * values[indexer.rank(a) * indexer.n + indexer.rank(b)] as f64 - 1.0;
                let searched = analyzer.equity(&to_position(a, b), Player::Me, 6);
                assert!((exact - searched).abs() < 1e-6, "{:?} vs {:?}: {} {}", a, b, exact, searched);
            }
        }
    }

    #[test]
    fn test_known_values_and_file_round_trip() {
        let checkers = 4;
        let values = generate(checkers, 2, |_, _| {});
        let path = std::env::temp_dir().join(format!("bearoff_test_{}.bd", std::process::id()));
        write(&path, checkers, &values).unwrap();
        let db = BearoffDb::open(&path).unwrap();
        // One checker on the 1-point: the side to move always wins.
        assert_eq!(db.win_probability_sides(&[1, 0, 0, 0, 0, 0], &[0, 0, 0, 0, 0, 4]), 1.0);
        // One checker on the 6-point against one on the 1-point: the mover must bear off now.
        // Only 4-1, 3-2, 3-1, 2-1 (two ways each) and 1-1 fall short.
        let p = db.win_probability_sides(&[0, 0, 0, 0, 0, 1], &[1, 0, 0, 0, 0, 0]);
        let misses = 2.0 * 4.0 + 1.0;
        assert!((p - (36.0 - misses) / 36.0).abs() < 1e-4, "{}", p);
        // Lookup through a position, from either side.
        let mut position = to_position(&[0, 0, 0, 0, 0, 1], &[1, 0, 0, 0, 0, 0]);
        assert!((db.win_probability(&position).unwrap() - p).abs() < 1e-12);
        position.set_turn(Player::Opponent);
        assert_eq!(db.win_probability(&position), Some(1.0));
        // Not a bearoff position, or more checkers than the database has.
        let mut contact = position.clone();
        contact.set_checkers(Player::Me, 7, 1).unwrap();
        assert_eq!(db.win_probability(&contact), None);
        let big = to_position(&[5, 0, 0, 0, 0, 0], &[1, 0, 0, 0, 0, 0]);
        assert_eq!(db.win_probability(&big), None);
        std::fs::remove_file(&path).unwrap();
    }
}
