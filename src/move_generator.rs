//! Position-based move generation for the analyzer.
//!
//! Generates the positions a roll can lead to, rather than move sequences: no allocation per
//! sequence, and intermediate positions reached by playing the same checkers in a different order
//! (only possible with doubles) are explored once. A `MoveGenerator` keeps its hash tables between
//! calls, so a search doesn't allocate and grow fresh tables for every roll.
//!
//! Two modes:
//! - `legal_positions` returns each distinct resulting position once (for searching deeper).
//! - `best_position` scores each resulting position as soon as it is found and keeps only the
//!   best one. This is for the last level of a search, where only the best child matters: no
//!   duplicate check, no list of children. Evaluating a duplicate twice is harmless for a maximum.
//!
//! Both apply the same rules (use as many dice as possible; if only one die can be played, the
//! higher one when possible) and visit positions in the same deterministic order, so they choose
//! the same move, ties included.
use crate::fast_hash::FastHashMap;
use crate::position::{Player, Position};

struct Collected<P> {
    position: P,
    dice_used: u8,
    used_higher_die: bool,
}

pub struct MoveGenerator<P: Position> {
    /// `legal_positions` only: resulting position → index in `collected`.
    result_index: FastHashMap<P::CacheKey, usize>,
    collected: Vec<Collected<P>>,
}

impl<P: Position + Clone> Default for MoveGenerator<P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: Position + Clone> MoveGenerator<P> {
    pub fn new() -> Self {
        MoveGenerator {
            result_index: FastHashMap::default(),
            collected: Vec::new(),
        }
    }

    /// The orders in which to play the dice: one for doubles, both orders otherwise.
    fn dice_orders(die1: i32, die2: i32) -> ([[i32; 4]; 2], usize, usize) {
        if die1 == die2 {
            ([[die1; 4], [0; 4]], 1, 4)
        } else {
            ([[die1, die2, 0, 0], [die2, die1, 0, 0]], 2, 2)
        }
    }

    /// The distinct positions that can result from playing `die1`/`die2`, with the turn passed to
    /// the opponent, in a deterministic order. Empty when no move is legal.
    pub fn legal_positions(&mut self, position: &P, die1: i32, die2: i32) -> Vec<P> {
        self.result_index.clear();
        self.collected.clear();
        let higher_die = die1.max(die2);
        let (orders, order_count, dice_count) = Self::dice_orders(die1, die2);

        let mut board = Board::new(position);
        let mut working = position.clone();
        let Self { result_index, collected } = self;
        for (order, dice) in orders[..order_count].iter().enumerate() {
            let mut on_result = |pos: &mut P, dice_used: u8, first_die: i32| {
                let used_higher_die = dice_used == 1 && first_die == higher_die;
                let key = pos.cache_key(0);
                match result_index.get(&key) {
                    Some(&i) => {
                        let r = &mut collected[i];
                        r.dice_used = r.dice_used.max(dice_used);
                        r.used_higher_die |= used_higher_die;
                    }
                    None => {
                        result_index.insert(key, collected.len());
                        collected.push(Collected { position: pos.clone(), dice_used, used_higher_die });
                    }
                }
            };
            board.explore(&mut working, &dice[..dice_count], 0, dice[0], die1 == die2, 25, order == 1, NO_MOVE, &mut on_result);
        }

        let max_used = collected.iter().map(|r| r.dice_used).max().unwrap_or(0);
        let higher_only = die1 != die2 && max_used == 1
            && collected.iter().any(|r| r.dice_used == 1 && r.used_higher_die);
        collected
            .iter()
            .filter(|r| r.dice_used == max_used && (!higher_only || r.used_higher_die))
            .map(|r| {
                let mut p = r.position.clone();
                p.switch_turn();
                p
            })
            .collect()
    }

    /// The legal resulting position (turn passed to the opponent) with the highest `score`, and
    /// that score; the first one found wins ties, as with `legal_positions`. None when no move is
    /// legal.
    pub fn best_position<F: Fn(&P) -> f64>(&mut self, position: &P, die1: i32, die2: i32, score: F) -> Option<(P, f64)> {
        let higher_die = die1.max(die2);
        let (orders, order_count, dice_count) = Self::dice_orders(die1, die2);

        // Best result per number of dice used, and the best single-die result using the higher die.
        let mut best: [Option<(P, f64)>; 5] = Default::default();
        let mut best_higher: Option<(P, f64)> = None;
        let mut on_result = |pos: &mut P, dice_used: u8, first_die: i32| {
            // Score the working position from the opponent's side (as the child it becomes),
            // without copying it; copy only when it is the best so far.
            pos.switch_turn();
            let value = score(pos);
            let improves = |slot: &Option<(P, f64)>| slot.as_ref().map_or(true, |(_, v)| value > *v);
            if dice_used == 1 && first_die == higher_die && improves(&best_higher) {
                best_higher = Some((pos.clone(), value));
            }
            if improves(&best[dice_used as usize]) {
                best[dice_used as usize] = Some((pos.clone(), value));
            }
            pos.switch_turn();
        };
        let mut board = Board::new(position);
        let mut working = position.clone();
        for (order, dice) in orders[..order_count].iter().enumerate() {
            board.explore(&mut working, &dice[..dice_count], 0, dice[0], die1 == die2, 25, order == 1, NO_MOVE, &mut on_result);
        }

        let max_used = (1..best.len()).rev().find(|&k| best[k].is_some())?;
        if die1 != die2 && max_used == 1 && best_higher.is_some() {
            return best_higher;
        }
        best[max_used].take()
    }
}

/// Bit for a point in a mask of points 1-25 (bit 0 is unused: borne-off checkers aren't tracked).
#[inline(always)]
fn bit(point: usize) -> u32 {
    1 << point
}

/// A half-move already played in the current move: from `start` to `target`, and whether it hit.
#[derive(Clone, Copy)]
struct HalfMove {
    start: usize,
    target: usize,
    hit: bool,
}

const NO_MOVE: HalfMove = HalfMove { start: 0, target: 0, hit: false };

/// What the move search needs to know about the board besides the position itself, kept as
/// bitmasks so that the questions asked for every candidate are single instructions.
struct Board {
    player: Player,
    opponent: Player,
    /// Points (in the mover's numbering) where the opponent has two or more checkers. Constant
    /// during a move: the mover can only change the opponent's checkers by hitting a single one.
    blocked: u32,
    /// Points 1-25 where the mover has checkers. Updated by `make`; restored after `unmake`.
    occupied: u32,
}

impl Board {
    fn new<P: Position>(position: &P) -> Self {
        let player = position.get_turn();
        let opponent = player.other_player();
        let mut blocked = 0;
        let mut occupied = 0;
        for point in 1..26 {
            if position.count(player, point) > 0 {
                occupied |= bit(point);
            }
            if point < 25 && position.count(opponent, 25 - point) >= 2 {
                blocked |= bit(point);
            }
        }
        Board { player, opponent, blocked, occupied }
    }

    /// Move a checker from `start` to `target` in place (hitting a single opponent checker there).
    /// Returns whether it hit, which `unmake` needs to undo the move.
    #[inline(always)]
    fn make<P: Position>(&mut self, pos: &mut P, start: usize, target: usize) -> bool {
        let (player, opponent) = (self.player, self.opponent);
        let left = pos.count(player, start) - 1;
        pos.set_count(player, start, left);
        if left == 0 {
            self.occupied &= !bit(start);
        }
        let mut hit = false;
        if target != 0 {
            let opponent_point = 25 - target;
            if pos.count(opponent, opponent_point) == 1 {
                pos.set_count(opponent, opponent_point, 0);
                let on_bar = pos.count(opponent, 25);
                pos.set_count(opponent, 25, on_bar + 1);
                hit = true;
            }
            self.occupied |= bit(target);
        }
        let there = pos.count(player, target);
        pos.set_count(player, target, there + 1);
        hit
    }

    /// With two different dice, playing `first` (with the other die) and then `start` → `target`
    /// (with `die`): does playing them the other way round, as the first order does, certainly
    /// give the same position? Then the second order can skip this sequence. Called in the
    /// position after `first`.
    ///
    /// - Two different checkers, without entering from the bar or bearing off: the other order is
    ///   always legal (blocked points don't change during a move) and moves the same checkers to
    ///   the same points, hits included. Same position.
    /// - The same checker twice: the other order passes a different intermediate point, so it is
    ///   the same only if that point isn't blocked and neither intermediate point hits a blot.
    /// - With the bar or bearing off the order can matter (which moves are legal), so keep those.
    #[inline(always)]
    fn first_order_plays_the_same<P: Position>(&self, pos: &P, first: HalfMove, start: usize, target: usize, die: i32) -> bool {
        if first.start == 25 || start == 25 || first.target == 0 || target == 0 {
            return false;
        }
        // The second half-move moves the checker the first one just placed, if that was the only
        // checker there.
        let same_checker = start == first.target && pos.count(self.player, start) == 1;
        if !same_checker {
            return true;
        }
        let other_intermediate = first.start - die as usize;
        !first.hit
            && self.blocked & bit(other_intermediate) == 0
            && pos.count(self.opponent, 25 - other_intermediate) != 1
    }

    /// Undo `make` (the caller restores `occupied`).
    #[inline(always)]
    fn unmake<P: Position>(&self, pos: &mut P, start: usize, target: usize, hit: bool) {
        let (player, opponent) = (self.player, self.opponent);
        let there = pos.count(player, target);
        pos.set_count(player, target, there - 1);
        if hit {
            let on_bar = pos.count(opponent, 25);
            pos.set_count(opponent, 25, on_bar - 1);
            pos.set_count(opponent, 25 - target, 1);
        }
        let left = pos.count(player, start);
        pos.set_count(player, start, left + 1);
    }

    /// Play the remaining `dice` in order from `pos`, calling `on_result(position, dice_used,
    /// first_die)` for every position where the move ends: all dice played, or no further die
    /// playable. `first_die` is the die the move started with. `pos` is changed in place while
    /// searching and is back in its original state on return.
    ///
    /// With doubles, half-moves are only played in descending order of start point (each starts
    /// at or below `max_start`, the previous one's start), so each combination of half-moves is
    /// generated once instead of in every order. That loses no moves: any legal double move can
    /// be played highest checker first (entering from the bar comes first, moves into the home
    /// board come before bearing off, and blocked points never change during a move).
    ///
    /// With two different dice both orders are played. In the second order (`second_order`),
    /// two-move sequences that the first order is sure to generate as well are skipped (see
    /// `first_order_plays_the_same`); `first_move` is the half-move already played.
    #[allow(clippy::too_many_arguments)]
    fn explore<P, F>(
        &mut self,
        pos: &mut P,
        dice: &[i32],
        dice_used: u8,
        first_die: i32,
        is_double: bool,
        max_start: usize,
        second_order: bool,
        first_move: HalfMove,
        on_result: &mut F,
    ) where
        P: Position + Clone,
        F: FnMut(&mut P, u8, i32),
    {
        if dice.is_empty() {
            on_result(pos, dice_used, first_die);
            return;
        }
        let die = dice[0];
        let occupied = self.occupied;
        let mut found_move = false;

        // A checker on the bar must enter first; bearing off requires all checkers in the home
        // board, and bearing off with a larger die than needed only from the highest point.
        let on_bar = occupied & bit(25) != 0;
        let highest = if occupied == 0 { 0 } else { 31 - occupied.leading_zeros() as usize };
        let can_bear_off = occupied != 0 && highest <= 6;

        // Candidate start points, lowest first (the order determines how ties are broken).
        let mut starts = if on_bar { bit(25) } else { occupied };
        while starts != 0 {
            let start = starts.trailing_zeros() as usize;
            starts &= starts - 1;

            let target = if start == 25 {
                25 - die as usize
            } else if start as i32 - die <= 0 {
                0
            } else {
                start - die as usize
            };
            if target == 0 {
                if !can_bear_off || (die as usize > start && start != highest) {
                    continue;
                }
            } else if self.blocked & bit(target) != 0 {
                continue;
            }
            // Legal: so this isn't where the move ends, even if the descending order means this
            // half-move is played in another branch rather than here.
            found_move = true;
            if start > max_start {
                continue;
            }
            if second_order && dice_used == 1 && self.first_order_plays_the_same(pos, first_move, start, target, die) {
                continue;
            }

            let hit = self.make(pos, start, target);
            let next_max_start = if is_double { start } else { 25 };
            let played = HalfMove { start, target, hit };
            self.explore(pos, &dice[1..], dice_used + 1, first_die, is_double, next_max_start, second_order, played, on_result);
            self.unmake(pos, start, target, hit);
            self.occupied = occupied;
        }

        if !found_move && dice_used > 0 {
            on_result(pos, dice_used, first_die);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fast_hash::FxHasher;
    use crate::cache_friendly_position::CacheFriendlyPosition;
    use crate::game::Dice;
    use std::hash::{Hash, Hasher};

    /// A deterministic, position-dependent pseudo-random score.
    fn scrambled_score(p: &CacheFriendlyPosition) -> f64 {
        let mut h = FxHasher::default();
        p.cache_key(0).hash(&mut h);
        (h.finish() % 1000) as f64
    }

    /// The first maximum over `legal_positions`, which is what the analyzer used to compute.
    fn best_by_collecting<F: Fn(&CacheFriendlyPosition) -> f64>(gen: &mut MoveGenerator<CacheFriendlyPosition>, pos: &CacheFriendlyPosition,
                                                       d1: i32, d2: i32, score: F) -> Option<(CacheFriendlyPosition, f64)> {
        let mut best: Option<(CacheFriendlyPosition, f64)> = None;
        for child in gen.legal_positions(pos, d1, d2) {
            let value = score(&child);
            if best.as_ref().map_or(true, |(_, v)| value > *v) {
                best = Some((child, value));
            }
        }
        best
    }

    #[test]
    fn test_best_position_matches_collecting_mode() {
        let mut gen = MoveGenerator::<CacheFriendlyPosition>::new();
        let mut compared = 0;
        for seed in 0..6 {
            let mut dice = Dice::new(seed);
            let mut pos = CacheFriendlyPosition::new();
            pos.setup_starting_position();
            for _ in 0..120 {
                for d1 in 1..=6 {
                    for d2 in d1..=6 {
                        // A varied score checks the maximum; a constant score checks tie-breaking.
                        for constant in [false, true] {
                            let score = |p: &CacheFriendlyPosition| if constant { 0.0 } else { scrambled_score(p) };
                            let fused = gen.best_position(&pos, d1, d2, score);
                            let collected = best_by_collecting(&mut gen, &pos, d1, d2, score);
                            match (fused, collected) {
                                (None, None) => {}
                                (Some((a, va)), Some((b, vb))) => {
                                    assert_eq!(va, vb, "roll {}-{}", d1, d2);
                                    assert!(a == b, "roll {}-{}: different move chosen", d1, d2);
                                }
                                _ => panic!("roll {}-{}: one mode found a move, the other didn't", d1, d2),
                            }
                            compared += 1;
                        }
                    }
                }
                let (d1, d2) = (dice.roll(), dice.roll());
                let next = gen.legal_positions(&pos, d1, d2);
                if next.is_empty() {
                    pos.switch_turn();
                } else {
                    pos = next[(dice.roll() as usize) % next.len()].clone();
                }
                if pos.get_checkers(pos.get_turn().other_player(), 0).unwrap() == 15 {
                    break;
                }
            }
        }
        assert!(compared > 10_000);
    }
}
