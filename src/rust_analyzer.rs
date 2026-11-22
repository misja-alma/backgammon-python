use std::cell::{Cell, RefCell};
use std::sync::Arc;
use crate::bearoff::BearoffDb;
use crate::transposition_table::TranspositionTable;
use crate::evaluator::{self, Evaluator, TerminalEvaluator};
use crate::position::{Position, Player};
use crate::move_generator::MoveGenerator;

/// Expectimax search over dice rolls (port of the Python `PythonAnalyzer`).
///
/// Equity is computed for the side to move and negated when viewed from the other side.
/// `depth` is the number of remaining roll levels: at depth 0 the evaluator gives the answer.
pub struct RustAnalyzer<P: Position + Clone> {
    /// Static evaluation where the search stops. Pluggable: the placeholder `TerminalEvaluator`
    /// by default, or e.g. a `NeuralNetEvaluator` (see `set_evaluator`).
    evaluator: Box<dyn Evaluator<P>>,
    /// (position, remaining depth) → equity for the side to move. Keyed by the position itself
    /// (in compact form, see `Position::cache_key`), so different positions never share an entry.
    /// Fixed size: when full, old entries are replaced (see `TranspositionTable`), so memory stays
    /// bounded however long the analyzer is used.
    cache: RefCell<TranspositionTable<P::CacheKey>>,
    /// Cache lookups and hits since the last `clear_cache`, for measuring the cache.
    lookups: Cell<u64>,
    hits: Cell<u64>,
    /// Reused for every roll, so its hash tables aren't rebuilt from scratch each time.
    generator: RefCell<MoveGenerator<P>>,
    /// Exact values for bearoff positions (see `set_bearoff`), or None.
    bearoff: Option<Arc<BearoffDb>>,
}

/// Default size of the analyzer's cache. The app overrides it from its settings
/// (`Settings.rust_cache_mb` in Python, applied through `set_cache_size`).
pub const DEFAULT_CACHE_MB: usize = 1024;

impl<P: Position + Clone> RustAnalyzer<P> {
    /// An analyzer with a cache of `DEFAULT_CACHE_MB`.
    pub fn new() -> Self {
        Self::with_cache_mb(DEFAULT_CACHE_MB)
    }

    /// An analyzer whose cache uses at most `megabytes` of memory.
    pub fn with_cache_mb(megabytes: usize) -> Self {
        RustAnalyzer {
            evaluator: Box::new(TerminalEvaluator),
            cache: RefCell::new(TranspositionTable::with_size_bytes(megabytes << 20)),
            lookups: Cell::new(0),
            hits: Cell::new(0),
            generator: RefCell::new(MoveGenerator::new()),
            bearoff: None,
        }
    }

    /// An analyzer using `evaluator`, with a cache of `megabytes`.
    pub fn with_evaluator(evaluator: Box<dyn Evaluator<P>>, megabytes: usize) -> Self {
        let mut analyzer = Self::with_cache_mb(megabytes);
        analyzer.evaluator = evaluator;
        analyzer
    }

    /// Replace the evaluator. Clears the cache, whose values came from the old one.
    pub fn set_evaluator(&mut self, evaluator: Box<dyn Evaluator<P>>) {
        self.evaluator = evaluator;
        self.clear_cache();
    }

    pub fn evaluator(&self) -> &dyn Evaluator<P> {
        self.evaluator.as_ref()
    }

    /// Use the bearoff database for positions it covers, or stop using one (None). Such positions
    /// get their exact value at any depth: the search doesn't look past them. Clears the cache.
    pub fn set_bearoff(&mut self, bearoff: Option<Arc<BearoffDb>>) {
        self.bearoff = bearoff;
        self.clear_cache();
    }

    pub fn bearoff(&self) -> Option<&BearoffDb> {
        self.bearoff.as_deref()
    }

    /// The exact value from the bearoff database when it covers the position, else the
    /// evaluator's. (Finished games are covered too, with the same values.)
    #[inline]
    fn static_equity(&self, position: &P) -> f64 {
        match &self.bearoff {
            Some(db) => db.equity(position).unwrap_or_else(|| self.evaluator.evaluate(position)),
            None => self.evaluator.evaluate(position),
        }
    }

    /// Replace the cache with an empty one of at most `megabytes`, keeping the evaluator.
    pub fn set_cache_mb(&mut self, megabytes: usize) {
        // Free the old table before allocating the new one, so both never exist at once.
        *self.cache.borrow_mut() = TranspositionTable::with_size_bytes(1 << 20);
        *self.cache.borrow_mut() = TranspositionTable::with_size_bytes(megabytes.max(1) << 20);
        self.lookups.set(0);
        self.hits.set(0);
    }

    pub fn clear_cache(&self) {
        self.cache.borrow_mut().clear();
        self.lookups.set(0);
        self.hits.set(0);
    }

    const DICE_ROLLS: [(i32, i32); 21] = [
        (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
        (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
        (3, 3), (3, 4), (3, 5), (3, 6),
        (4, 4), (4, 5), (4, 6),
        (5, 5), (5, 6),
        (6, 6)
    ];

    pub fn dice_rolls() -> &'static [(i32, i32)] {
        &Self::DICE_ROLLS
    }

    /// Number of cached positions (one per position and depth). Scans the whole cache.
    pub fn cache_len(&self) -> usize {
        self.cache.borrow().len()
    }

    /// Maximum number of cached positions.
    pub fn cache_capacity(&self) -> usize {
        self.cache.borrow().capacity()
    }

    /// Memory reserved for the cache, in bytes.
    pub fn cache_size_bytes(&self) -> usize {
        self.cache.borrow().size_bytes()
    }

    /// (lookups, hits) since the last `clear_cache`.
    pub fn cache_stats(&self) -> (u64, u64) {
        (self.lookups.get(), self.hits.get())
    }

    /// Equity of the position for `player`, before the side to move has rolled.
    pub fn equity(&self, position: &P, player: Player, max_depth: u32) -> f64 {
        let equity = self.equity_for_turn(position, max_depth);
        if player == position.get_turn() { equity } else { -equity }
    }

    /// Best move for the side to move with the given dice.
    /// Returns (position after the move, equity for the player who moved).
    pub fn best_move(&self, position: &P, die1: i32, die2: i32, max_depth: u32) -> Option<(P, f64)> {
        Some(self.best_move_with_depth(position, die1, die2, max_depth))
    }

    fn equity_for_turn(&self, position: &P, depth: u32) -> f64 {
        if depth == 0 || evaluator::is_game_over(position) {
            return self.static_equity(position);
        }
        if let Some(equity) = self.bearoff.as_ref().and_then(|db| db.equity(position)) {
            return equity;
        }

        let key = position.cache_key(depth as u16);
        self.lookups.set(self.lookups.get() + 1);
        if let Some(equity) = self.cache.borrow().get(&key) {
            self.hits.set(self.hits.get() + 1);
            return equity;
        }

        let mut total = 0.0;
        for &(die1, die2) in Self::dice_rolls() {
            let (_, equity) = self.best_move_with_depth(position, die1, die2, depth - 1);
            let multiplier = if die1 != die2 { 2.0 } else { 1.0 };
            total += multiplier * equity;
        }

        let equity = total / 36.0;
        self.cache.borrow_mut().store(key, depth.min(u8::MAX as u32) as u8, equity);
        equity
    }

    fn best_move_with_depth(&self, position: &P, die1: i32, die2: i32, depth: u32) -> (P, f64) {
        let mut best: Option<(P, f64)> = None;

        if depth == 0 {
            // Last level: children are evaluated statically, and only the best one matters, so
            // score each resulting position as it is generated instead of collecting them.
            // (A child has the opponent to move, so negate its evaluation.)
            best = self.generator.borrow_mut()
                .best_position(position, die1, die2, |child| -self.static_equity(child));
        } else {
            // Each distinct resulting position once (many move sequences lead to the same
            // position), in a deterministic order, so ties are always broken the same way.
            // The generator is borrowed only for this statement: the recursion below uses it too.
            let children = self.generator.borrow_mut().legal_positions(position, die1, die2);
            for child in children {
                // The child has the opponent to move, so negate its equity.
                let equity = -self.equity_for_turn(&child, depth);
                match &best {
                    Some((_, best_equity)) if equity <= *best_equity => {}
                    _ => best = Some((child, equity)),
                }
            }
        }

        best.unwrap_or_else(|| {
            let mut after_pass = position.clone();
            after_pass.switch_turn();
            let equity = -self.equity_for_turn(&after_pass, depth);
            (after_pass, equity)
        })
    }
}

impl<P: Position + Clone> Default for RustAnalyzer<P> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::Player;

    fn pos(me: &[(usize, u8)], opp: &[(usize, u8)], turn: u8) -> impl Position + Clone {
        let mut me_vec = vec![0u8; 26];
        let mut opp_vec = vec![0u8; 26];
        for &(i, count) in me  { me_vec[i] = count; }
        for &(i, count) in opp { opp_vec[i] = count; }
        crate::build_position(me_vec, opp_vec, turn, 1, None).unwrap()
    }

    #[test]
    fn test_bearoff_database_gives_exact_values_at_any_depth() {
        let checkers = 5;
        let path = std::env::temp_dir().join(format!("bearoff_analyzer_{}.bd", std::process::id()));
        crate::bearoff::write(&path, checkers, &crate::bearoff::generate(checkers, 2, |_, _| {})).unwrap();
        let db = Arc::new(BearoffDb::open(&path).unwrap());
        std::fs::remove_file(&path).unwrap();
        // Bearoff: Me 3 on the 6-point and 1 on the 5-point (11 off), Opponent 2 on the 4-point.
        let p = pos(&[(0, 11), (6, 3), (5, 1)], &[(0, 13), (4, 2)], 1);
        let exact = db.equity(&p).unwrap();
        // Without the database a deep search with the placeholder evaluator approaches the exact
        // value; with it every depth gives exactly the stored value.
        let mut analyzer = RustAnalyzer::with_cache_mb(16);
        analyzer.set_bearoff(Some(db.clone()));
        for depth in 0..3 {
            assert_eq!(analyzer.equity(&p, Player::Me, depth), exact);
            assert_eq!(analyzer.equity(&p, Player::Opponent, depth), -exact);
        }
        // At depth 0 a move is chosen by the database values of the children.
        let (child, equity) = analyzer.best_move(&p, 6, 5, 0).unwrap();
        assert_eq!(equity, -db.equity(&child).unwrap());
        // Outside the bearoff (a checker on the 7-point) the evaluator is used: 0 for the placeholder.
        let contact = pos(&[(0, 10), (7, 1), (6, 3), (5, 1)], &[(0, 13), (4, 2)], 1);
        assert_eq!(analyzer.equity(&contact, Player::Me, 0), 0.0);
        analyzer.set_bearoff(None);
        assert_eq!(analyzer.equity(&p, Player::Me, 0), 0.0);
    }

    #[test]
    fn test_dice_rolls() {
        let rolls = RustAnalyzer::<crate::cache_friendly_position::CacheFriendlyPosition>::dice_rolls();
        assert_eq!(rolls.len(), 21);
        assert!(rolls.contains(&(1, 1)));
        assert!(rolls.contains(&(1, 6)));
        assert!(rolls.contains(&(6, 6)));
        assert!(!rolls.contains(&(2, 1)));
    }

    #[test]
    fn test_equity_known_win() {
        let position = pos(&[(0, 15)], &[(1, 15)], 1);
        assert_eq!(RustAnalyzer::with_cache_mb(4).equity(&position, Player::Me, 3), 1.0);
        assert_eq!(RustAnalyzer::with_cache_mb(4).equity(&position, Player::Opponent, 3), -1.0);
    }

    #[test]
    fn test_equity_known_loss() {
        let position = pos(&[(1, 15)], &[(0, 15)], 1);
        assert_eq!(RustAnalyzer::with_cache_mb(4).equity(&position, Player::Me, 3), -1.0);
    }

    #[test]
    fn test_equity_unknown_at_depth_zero() {
        let starting = &[(6, 5), (8, 3), (13, 5), (24, 2)];
        let position = pos(starting, starting, 1);
        assert_eq!(RustAnalyzer::with_cache_mb(4).equity(&position, Player::Me, 0), 0.0);
    }

    #[test]
    fn test_equity() {
        let analyzer = RustAnalyzer::with_cache_mb(4);

        let position = pos(&[(1, 1), (0, 14)], &[(0, 15)], 1);
        assert_eq!(analyzer.equity(&position, Player::Opponent, 3), 1.0);
        assert_eq!(analyzer.equity(&position, Player::Me, 3), -1.0);

        // Me (to move) has 2 checkers on the 2-point: both dice >= 2 (25 rolls) or 1-1 bears both
        // off, so 26 of 36 rolls win immediately; otherwise Opponent bears off and wins.
        let position = pos(&[(2, 2), (0, 13)], &[(1, 1), (0, 14)], 1);
        assert_eq!(analyzer.equity(&position, Player::Me, 3), 26.0 / 36.0 - 10.0 / 36.0);
        assert_eq!(analyzer.equity(&position, Player::Opponent, 3), 10.0 / 36.0 - 26.0 / 36.0);
    }

    #[test]
    fn test_best_move() {
        let analyzer = RustAnalyzer::with_cache_mb(4);
        let mut position = pos(&[(1, 1), (0, 14)], &[(1, 1), (0, 14)], 1);

        let (best_pos, equity) = analyzer.best_move(&position, 2, 1, 3).unwrap();
        assert_eq!(equity, 1.0);
        assert_eq!(best_pos.get_checkers(Player::Me, 0).unwrap(), 15);

        position.set_checkers(Player::Me, 2, 1).unwrap();
        position.set_checkers(Player::Me, 0, 13).unwrap();
        let (best_pos, equity) = analyzer.best_move(&position, 2, 1, 3).unwrap();
        assert_eq!(equity, 1.0);
        assert_eq!(best_pos.get_checkers(Player::Me, 0).unwrap(), 15);

        position.set_checkers(Player::Me, 3, 1).unwrap();
        position.set_checkers(Player::Me, 2, 0).unwrap();
        let (_, equity) = analyzer.best_move(&position, 2, 1, 3).unwrap();
        assert_eq!(equity, -1.0);

        let (best_pos, equity) = analyzer.best_move(&position, 1, 1, 3).unwrap();
        assert_eq!(equity, 1.0);
        assert_eq!(best_pos.get_checkers(Player::Me, 0).unwrap(), 15);
    }

    #[test]
    fn test_best_move_no_legal_moves() {
        // Me is on the bar against a closed board (Opponent holds its points 1-6).
        let position = pos(&[(25, 1), (0, 14)], &[(1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (0, 3)], 1);

        let (new_pos, _) = RustAnalyzer::with_cache_mb(4).best_move(&position, 6, 5, 0).unwrap();
        assert_eq!(new_pos.get_turn(), Player::Opponent);
        assert_eq!(new_pos.get_checkers(Player::Me, 25).unwrap(), 1);
    }

    #[test]
    fn test_cache_is_keyed_by_depth() {
        let analyzer = RustAnalyzer::with_cache_mb(4);
        // Me wins 26/36 immediately; at depth 1 the misses evaluate to 0, at depth 3 to a loss.
        let position = pos(&[(2, 2), (0, 13)], &[(1, 1), (0, 14)], 1);

        let shallow = analyzer.equity(&position, Player::Me, 1);
        let deep = analyzer.equity(&position, Player::Me, 3);
        assert!(analyzer.cache.borrow().get(&position.cache_key(1)).is_some());
        assert!(analyzer.cache.borrow().get(&position.cache_key(3)).is_some());
        // A fresh analyzer must give the same deep value: the cached shallow value was not reused.
        assert_eq!(deep, RustAnalyzer::with_cache_mb(4).equity(&position, Player::Me, 3));
        assert_eq!(shallow, 26.0 / 36.0);
        assert_eq!(deep, 16.0 / 36.0);
    }
}
