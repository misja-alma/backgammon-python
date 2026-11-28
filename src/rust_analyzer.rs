use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use crate::bearoff::BearoffDb;
use crate::transposition_table::SharedTable;
use crate::evaluator::{self, Evaluator, Scoring, TerminalEvaluator};
use crate::position::{Position, Player};
use crate::move_generator::MoveGenerator;
use crate::game::{Dice, MAX_TURNS};

/// Expectimax search over dice rolls (port of the Python `PythonAnalyzer`).
///
/// Equity is computed for the side to move and negated when viewed from the other side.
/// `depth` is the number of remaining roll levels: at depth 0 the evaluator gives the answer.
///
/// With `set_threads(n)`, searches of at least one roll level run on n threads: `best_move` and
/// `analyze_moves` hand out the moves, `equity` and `outcome_for_turn` the 21 rolls, to worker
/// analyzers (each with its own copy of the evaluator, see `Evaluator::fork`, and move generator)
/// that share the cache and the bearoff database. The results are exactly those of one thread:
/// every task computes what the serial search would, and they are combined in the same order.
pub struct RustAnalyzer<P: Position + Clone> {
    /// Static evaluation where the search stops. Pluggable: the placeholder `TerminalEvaluator`
    /// by default, or e.g. a `NeuralNetEvaluator` (see `set_evaluator`).
    evaluator: Box<dyn Evaluator<P>>,
    /// (position, remaining depth) → value (equity and probabilities) for the side to move. Keyed by the position itself
    /// (in compact form, see `Position::cache_key`), so different positions never share an entry.
    /// Fixed size: when full, old entries are replaced (see `TranspositionTable`), so memory stays
    /// bounded however long the analyzer is used. Shared with the worker analyzers.
    cache: Arc<SharedTable<P::CacheKey, Value>>,
    /// Reused for every roll, so its hash tables aren't rebuilt from scratch each time.
    generator: RefCell<MoveGenerator<P>>,
    /// Exact values for bearoff positions (see `set_bearoff`), or None.
    bearoff: Option<Arc<BearoffDb>>,
    /// What the game is played for (see `set_scoring`).
    scoring: Scoring,
    /// Threads for searches of at least one roll level (see `set_threads`).
    threads: usize,
    /// Which moves are searched deeper (see `set_move_filter`); None: all.
    filter: Option<MoveFilter>,
    /// The other threads' analyzers (threads - 1 of them, or none when the evaluator can't be
    /// forked).
    workers: RefCell<Vec<Worker<P>>>,
}

/// A position's value for the side to move, as the search computes it: the equity under the
/// analyzer's scoring and the 5 outcome probabilities (see `Evaluator::outcome`; NaN where the
/// evaluator doesn't know them). The probabilities are kept as f32, all the display needs, which
/// keeps cache entries small; both come from the same search.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Value {
    pub equity: f64,
    pub outcome: [f32; evaluator::MONEY_OUTPUTS],
}

impl Value {
    /// The same value from the other side's point of view.
    #[inline]
    pub fn flip(self) -> Value {
        Value { equity: -self.equity, outcome: evaluator::flip_outcome(&self.outcome) }
    }
}

/// Which moves the search looks at more deeply (gnubg's "move filter"): at every move decision
/// searched at least one roll level deep, all moves are first scored by their static (1-ply)
/// equity for the mover, and only those within `margin` of the best are searched deeper, at most
/// the best `max_moves` of them. The others are never chosen. Equity units are those of the
/// scoring (money: points per game; double match point: 2 × winning chance − 1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoveFilter {
    pub margin: f64,
    pub max_moves: usize,
}

/// A worker analyzer of the parallel search. Its evaluator comes from `Evaluator::fork`, which
/// gives an evaluator that may move to another thread, and every other part of an analyzer may
/// too (positions are plain data, the cache and the bearoff database are thread-safe): so a
/// worker may be sent to the thread that uses it.
struct Worker<P: Position + Clone>(RustAnalyzer<P>);

unsafe impl<P: Position + Clone + Send> Send for Worker<P> where P::CacheKey: Send + Sync {}

/// Default size of the analyzer's cache. The app overrides it from its settings
/// (`Settings.rust_cache_mb` in Python, applied through `set_cache_size`).
pub const DEFAULT_CACHE_MB: usize = 1024;

impl<P: Position + Clone + Send + Sync + 'static> RustAnalyzer<P> where P::CacheKey: Send + Sync {
    /// An analyzer with a cache of `DEFAULT_CACHE_MB`.
    pub fn new() -> Self {
        Self::with_cache_mb(DEFAULT_CACHE_MB)
    }

    /// An analyzer whose cache uses at most `megabytes` of memory.
    pub fn with_cache_mb(megabytes: usize) -> Self {
        RustAnalyzer {
            evaluator: Box::new(TerminalEvaluator),
            cache: Arc::new(SharedTable::with_size_bytes(megabytes << 20)),
            generator: RefCell::new(MoveGenerator::new()),
            bearoff: None,
            scoring: Scoring::Dmp,
            threads: 1,
            filter: None,
            workers: RefCell::new(Vec::new()),
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
        self.make_workers();
    }

    pub fn evaluator(&self) -> &dyn Evaluator<P> {
        self.evaluator.as_ref()
    }

    /// Use the bearoff database for positions it covers, or stop using one (None). Such positions
    /// get their exact value at any depth: the search doesn't look past them. Clears the cache.
    pub fn set_bearoff(&mut self, bearoff: Option<Arc<BearoffDb>>) {
        self.bearoff = bearoff;
        self.clear_cache();
        self.make_workers();
    }

    pub fn bearoff(&self) -> Option<&BearoffDb> {
        self.bearoff.as_deref()
    }

    /// Play for double match point (every win counts 1, the default) or cubeless money (gammons
    /// count 2, backgammons 3). Clears the cache.
    pub fn set_scoring(&mut self, scoring: Scoring) {
        self.scoring = scoring;
        for worker in self.workers.get_mut() {
            worker.0.scoring = scoring;
        }
        self.clear_cache();
    }

    pub fn scoring(&self) -> Scoring {
        self.scoring
    }

    /// Search on `threads` threads (1: only the calling one, the default). Needs an evaluator
    /// that can be forked (`Evaluator::fork`); with one that can't, searches stay on one thread.
    pub fn set_threads(&mut self, threads: usize) {
        self.threads = threads.max(1);
        self.make_workers();
    }

    /// The threads searches use: 1 when the evaluator can't be forked.
    pub fn threads(&self) -> usize {
        1 + self.workers.borrow().len()
    }

    /// (Re)make the worker analyzers for the current threads, evaluator, cache and bearoff
    /// database.
    fn make_workers(&mut self) {
        let mut workers = Vec::new();
        for _ in 1..self.threads {
            let Some(evaluator) = self.evaluator.fork() else {
                workers.clear();
                break;
            };
            workers.push(Worker(RustAnalyzer {
                evaluator,
                cache: self.cache.clone(),
                generator: RefCell::new(MoveGenerator::new()),
                bearoff: self.bearoff.clone(),
                scoring: self.scoring,
                threads: 1,
                filter: self.filter,
                workers: RefCell::new(Vec::new()),
            }));
        }
        *self.workers.get_mut() = workers;
    }

    /// Runs task 0..tasks on the calling thread and the workers (each task once, taken in order
    /// by whichever thread is free), and returns the results in task order.
    fn run_parallel<T: Send>(&self, tasks: usize, task: &(dyn Fn(&RustAnalyzer<P>, usize) -> T + Sync)) -> Vec<T> {
        let mut workers = self.workers.borrow_mut();
        let next = AtomicUsize::new(0);
        let results: Vec<Mutex<Option<T>>> = (0..tasks).map(|_| Mutex::new(None)).collect();
        let work = |analyzer: &RustAnalyzer<P>| loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= tasks {
                break;
            }
            let result = task(analyzer, i);
            *results[i].lock().unwrap() = Some(result);
        };
        std::thread::scope(|scope| {
            for worker in workers.iter_mut().take(tasks.saturating_sub(1)) {
                let work = &work;
                scope.spawn(move || work(&worker.0));
            }
            work(self);
        });
        results.into_iter().map(|r| r.into_inner().unwrap().expect("every task ran")).collect()
    }

    /// Whether a search of `depth` roll levels runs on several threads.
    fn parallel(&self, depth: u32) -> bool {
        depth >= 1 && !self.workers.borrow().is_empty()
    }

    /// The exact value from the bearoff database when it covers the position (for money only once
    /// no gammon is possible: it knows only winning chances), None otherwise.
    #[inline]
    fn bearoff_equity(&self, position: &P) -> Option<f64> {
        let db = self.bearoff.as_ref()?;
        if self.scoring == Scoring::Money && evaluator::gammon_possible(position) {
            return None;
        }
        db.equity(position)
    }

    /// The static equity (`static_value`'s) of a position after a move from `parent` (evaluated incrementally from the
    /// parent by evaluators that can, see `Evaluator::evaluate_child_scored`).
    fn static_child_equity(&self, parent: &P, child: &P) -> f64 {
        self.bearoff_equity(child).unwrap_or_else(|| self.evaluator.evaluate_child_scored(parent, child, self.scoring))
    }

    /// Replace the cache with an empty one of at most `megabytes`, keeping the evaluator.
    pub fn set_cache_mb(&mut self, megabytes: usize) {
        // Free the old table before allocating the new one, so both never exist at once.
        self.workers.get_mut().clear();
        self.cache = Arc::new(SharedTable::with_size_bytes(1 << 20));
        self.cache = Arc::new(SharedTable::with_size_bytes(megabytes.max(1) << 20));
        self.make_workers();
    }

    pub fn clear_cache(&self) {
        self.cache.clear();
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
        self.cache.len()
    }

    /// Maximum number of cached positions.
    pub fn cache_capacity(&self) -> usize {
        self.cache.capacity()
    }

    /// Memory reserved for the cache, in bytes.
    pub fn cache_size_bytes(&self) -> usize {
        self.cache.size_bytes()
    }

    /// (lookups, hits) since the last `clear_cache`.
    pub fn cache_stats(&self) -> (u64, u64) {
        self.cache.stats()
    }

    /// Equity of the position for `player`, before the side to move has rolled.
    pub fn equity(&self, position: &P, player: Player, max_depth: u32) -> f64 {
        let equity = self.value(position, max_depth).equity;
        if player == position.get_turn() { equity } else { -equity }
    }

    /// The outcome probabilities for the side to move, before it rolls, searching `depth` roll
    /// levels the same way as `equity` (every roll's move is the one the search chooses; its
    /// probabilities are averaged).
    pub fn outcome_for_turn(&self, position: &P, depth: u32) -> [f64; evaluator::MONEY_OUTPUTS] {
        self.value(position, depth).outcome.map(|x| x as f64)
    }

    /// The side to move's equity and outcome probabilities before it rolls, searching `depth`
    /// roll levels (on the threads from depth 1).
    pub fn value(&self, position: &P, depth: u32) -> Value {
        if self.parallel(depth) {
            self.value_parallel(position, depth)
        } else {
            self.value_for_turn(position, depth)
        }
    }

    /// The values of the rolls, weighted (in roll order: the same sums on any number of threads).
    fn average(rolls: &[(i32, i32)], values: impl IntoIterator<Item = Value>) -> Value {
        let mut equity = 0.0;
        let mut outcome = [0.0f64; evaluator::MONEY_OUTPUTS];
        for (&(die1, die2), v) in rolls.iter().zip(values) {
            let weight = if die1 != die2 { 2.0 } else { 1.0 };
            equity += weight * v.equity;
            for k in 0..evaluator::MONEY_OUTPUTS {
                outcome[k] += weight * v.outcome[k] as f64;
            }
        }
        Value { equity: equity / 36.0, outcome: outcome.map(|o| (o / 36.0) as f32) }
    }

    /// The value where the search stops: exact from the bearoff database when it covers the
    /// position (finished games too, see `static_outcome` for the probabilities), else the
    /// evaluator's (one evaluation for both, see `Evaluator::evaluate_full`).
    fn static_value(&self, position: &P) -> Value {
        if let Some(equity) = self.bearoff_equity(position) {
            return Value { equity, outcome: self.static_outcome(position).map(|x| x as f32) };
        }
        let (equity, outcome) = self.evaluator.evaluate_full(position, self.scoring);
        Value { equity, outcome: outcome.map(|x| x as f32) }
    }

    /// `value_for_turn` with the 21 rolls on the threads.
    fn value_parallel(&self, position: &P, depth: u32) -> Value {
        if evaluator::is_game_over(position) || self.bearoff_equity(position).is_some() {
            return self.static_value(position);
        }
        let key = position.cache_key(depth as u16);
        if let Some(value) = self.cache.get(&key) {
            return value;
        }
        let rolls = Self::dice_rolls();
        let values = self.run_parallel(rolls.len(), &|a, i| a.best_move_with_depth(position, rolls[i].0, rolls[i].1, depth - 1).1);
        let value = Self::average(rolls, values);
        self.cache.store(key, depth.min(u8::MAX as u32) as u8, value);
        value
    }

    /// `value_for_turn(position, depth)` of each position (depth >= 1), with every (position,
    /// roll) a task of its own, so that the threads stay busy to the end (a move has only ~18
    /// candidates). The same values as one thread: the rolls are added in the same order, and
    /// the results cached the same way.
    fn values_parallel(&self, positions: &[P], depth: u32) -> Vec<Value> {
        // Known without searching: finished games, bearoff positions, cached results.
        let mut known: Vec<Option<Value>> = positions.iter().map(|p| {
            if evaluator::is_game_over(p) || self.bearoff_equity(p).is_some() {
                Some(self.static_value(p))
            } else {
                self.cache.get(&p.cache_key(depth as u16))
            }
        }).collect();
        let open: Vec<usize> = (0..positions.len()).filter(|&i| known[i].is_none()).collect();
        let rolls = Self::dice_rolls();
        let n = rolls.len();
        let values = self.run_parallel(open.len() * n, &|a, t| {
            let (die1, die2) = rolls[t % n];
            a.best_move_with_depth(&positions[open[t / n]], die1, die2, depth - 1).1
        });
        for (k, &i) in open.iter().enumerate() {
            let value = Self::average(rolls, values[k * n..(k + 1) * n].iter().copied());
            self.cache.store(positions[i].cache_key(depth as u16), depth.min(u8::MAX as u32) as u8, value);
            known[i] = Some(value);
        }
        known.into_iter().map(|v| v.unwrap()).collect()
    }

    /// Search only some moves deeper (see `MoveFilter`), or every move (None, the default).
    /// Clears the cache, whose values depend on it.
    pub fn set_move_filter(&mut self, filter: Option<MoveFilter>) {
        self.filter = filter;
        for worker in self.workers.get_mut() {
            worker.0.filter = filter;
        }
        self.clear_cache();
    }

    pub fn move_filter(&self) -> Option<MoveFilter> {
        self.filter
    }

    /// Which of the moves from `position` (to `children`) are searched deeper: those that pass
    /// the move filter by their 1-ply equity for the mover (evaluated incrementally), or all
    /// without a filter. In the children's order.
    fn candidate_mask(&self, position: &P, children: &[P]) -> Vec<bool> {
        let Some(filter) = self.filter else {
            return vec![true; children.len()];
        };
        if children.len() <= 1 {
            return vec![true; children.len()];
        }
        let scores: Vec<f64> = children.iter().map(|c| -self.static_child_equity(position, c)).collect();
        let best = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut kept: Vec<usize> = (0..children.len()).filter(|&i| scores[i] >= best - filter.margin).collect();
        if kept.len() > filter.max_moves {
            // The best max_moves of them (a stable sort: equal scores keep the children's order).
            kept.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
            kept.truncate(filter.max_moves.max(1));
        }
        let mut mask = vec![false; children.len()];
        for i in kept {
            mask[i] = true;
        }
        mask
    }

    /// The moves `candidate_mask` keeps, in their order.
    fn candidates(&self, position: &P, children: Vec<P>) -> Vec<P> {
        if self.filter.is_none() {
            return children;
        }
        let mask = self.candidate_mask(position, &children);
        children.into_iter().zip(mask).filter(|(_, keep)| *keep).map(|(c, _)| c).collect()
    }

    /// Best move for the side to move with the given dice.
    /// Returns (position after the move, equity for the player who moved).
    pub fn best_move(&self, position: &P, die1: i32, die2: i32, max_depth: u32) -> Option<(P, f64)> {
        if self.parallel(max_depth) {
            let children = self.generator.borrow_mut().legal_positions(position, die1, die2);
            let children = self.candidates(position, children);
            if !children.is_empty() {
                let equities: Vec<f64> = self.values_parallel(&children, max_depth).into_iter().map(|v| -v.equity).collect();
                // The first of the best, as `best_move_with_depth` chooses.
                let mut best = 0;
                for (i, &e) in equities.iter().enumerate() {
                    if e > equities[best] {
                        best = i;
                    }
                }
                return Some((children[best].clone(), equities[best]));
            }
        }
        let (child, value) = self.best_move_with_depth(position, die1, die2, max_depth);
        Some((child, value.equity))
    }

    /// The 5 outcome probabilities for the side to move (see `Evaluator::outcome`; NaN where the
    /// evaluator doesn't know them), where the search stops: exact for finished games and
    /// bearoff positions (gammons there only once none is possible; at double match point the
    /// database's winning chance is used anyway, like `static_equity`).
    fn static_outcome(&self, position: &P) -> [f64; evaluator::MONEY_OUTPUTS] {
        if let Some(o) = evaluator::terminal_outcome(position) {
            return o.map(|x| x as f64);
        }
        if let Some(db) = &self.bearoff {
            let no_gammon = !evaluator::gammon_possible(position);
            if no_gammon || self.scoring == Scoring::Dmp {
                if let Some(win) = db.win_probability(position) {
                    let rest = if no_gammon { 0.0 } else { f64::NAN };
                    return [win, rest, rest, rest, rest];
                }
            }
        }
        self.evaluator.outcome(position)
    }

    /// Points won (±1, ±2 or ±3: cubeless, gammons and backgammons count) by the side to move at
    /// `start` in game number `game` of a rollout, both sides playing the move this analyzer
    /// chooses at `depth` roll levels (the scoring's move choice). The first two rolls rotate
    /// through all 36 × 36 ordered combinations as `game` counts up (games 0..1296 play each
    /// once); later rolls are random, from dice seeded with `seed` and `game`, so a game's result
    /// doesn't depend on which thread plays it.
    pub fn rollout_game(&self, start: &P, game: usize, seed: u64, depth: u32) -> i32 {
        let side = start.get_turn();
        let mut position = start.clone();
        let mut dice = Dice::new(seed ^ (game as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        for turn in 0..MAX_TURNS {
            if let Some(points) = evaluator::terminal_money_equity(&position) {
                let points = points as i32;
                return if position.get_turn() == side { points } else { -points };
            }
            let rotated = |n: usize| ((n % 36 / 6) as i32 + 1, (n % 6) as i32 + 1);
            let (die1, die2) = match turn {
                0 => rotated(game),
                1 => rotated(game / 36),
                _ => (dice.roll(), dice.roll()),
            };
            // Serial on purpose: the games themselves run on the threads (see `rollout`).
            position = self.best_move_with_depth(&position, die1, die2, depth).0;
        }
        0   // never ends (cannot happen in a real game): counted as a draw
    }

    /// `rollout_game` for games `first..first + games`, on the threads; the results in game order.
    pub fn rollout(&self, start: &P, first: usize, games: usize, seed: u64, depth: u32) -> Vec<i32> {
        if self.workers.borrow().is_empty() {
            return (first..first + games).map(|g| self.rollout_game(start, g, seed, depth)).collect();
        }
        self.run_parallel(games, &|a, i| a.rollout_game(start, first + i, seed, depth))
    }

    /// Every distinct move with the given dice (the passed position when none is legal), best
    /// first: (position after the move, equity for the mover, outcome probabilities for the
    /// mover, roll levels searched). The moves the move filter keeps (all without one) are
    /// searched `depth` roll levels as in `best_move` and come first (equal equities keep the
    /// order in which `best_move` meets them, so the first is the move it chooses); the others
    /// follow with their static values (0 roll levels, 1-ply).
    pub fn analyze_moves(&self, position: &P, die1: i32, die2: i32, depth: u32)
                         -> Vec<(P, f64, [f64; evaluator::MONEY_OUTPUTS], u32)> {
        let mut children = self.generator.borrow_mut().legal_positions(position, die1, die2);
        if children.is_empty() {
            let mut passed = position.clone();
            passed.switch_turn();
            children.push(passed);
        }
        let mask = if depth == 0 { vec![true; children.len()] } else { self.candidate_mask(position, &children) };
        let searched: Vec<P> = children.iter().zip(&mask).filter(|(_, &m)| m).map(|(c, _)| c.clone()).collect();
        let searched_values: Vec<Value> = if depth == 0 {
            searched.iter().map(|c| self.static_value(c)).collect()
        } else if self.parallel(depth) {
            self.values_parallel(&searched, depth)
        } else {
            searched.iter().map(|c| self.value_for_turn(c, depth)).collect()
        };
        let mut searched_values = searched_values.into_iter();
        let mut moves: Vec<(P, f64, [f64; evaluator::MONEY_OUTPUTS], u32)> = children.into_iter().zip(mask)
            .map(|(child, m)| {
                let (value, levels) = if m {
                    (searched_values.next().unwrap(), depth)
                } else {
                    (self.static_value(&child), 0)
                };
                let value = value.flip();
                (child, value.equity, value.outcome.map(|x| x as f64), levels)
            })
            .collect();
        moves.sort_by(|a, b| b.3.cmp(&a.3).then(b.1.total_cmp(&a.1)));
        moves
    }

    /// The side to move's value before it rolls, searching `depth` roll levels on this thread.
    fn value_for_turn(&self, position: &P, depth: u32) -> Value {
        if depth == 0 || evaluator::is_game_over(position) || self.bearoff_equity(position).is_some() {
            return self.static_value(position);
        }

        let key = position.cache_key(depth as u16);
        if let Some(value) = self.cache.get(&key) {
            return value;
        }

        let rolls = Self::dice_rolls();
        let value = Self::average(rolls, rolls.iter().map(|&(die1, die2)| {
            self.best_move_with_depth(position, die1, die2, depth - 1).1
        }));
        self.cache.store(key, depth.min(u8::MAX as u32) as u8, value);
        value
    }

    /// The move the search chooses with these dice: (position after it, its value for the player
    /// who moved).
    fn best_move_with_depth(&self, position: &P, die1: i32, die2: i32, depth: u32) -> (P, Value) {
        let mut best: Option<(P, Value)> = None;

        if depth == 0 {
            // Last level: children are evaluated statically, and only the best one matters, so
            // score each resulting position as it is generated instead of collecting them.
            // (A child has the opponent to move, so negate its evaluation.)
            // The children are compared incrementally from the parent (faster, see
            // `Evaluator::evaluate_child_scored`); the chosen one is then evaluated in full (its
            // equity and probabilities), so that the search's values don't depend on that
            // shortcut.
            best = self.generator.borrow_mut()
                .best_position(position, die1, die2, |child| -self.static_child_equity(position, child))
                .map(|(child, _)| {
                    let value = self.static_value(&child).flip();
                    (child, value)
                });
        } else {
            // Each distinct resulting position once (many move sequences lead to the same
            // position), in a deterministic order, so ties are always broken the same way; only
            // those the move filter keeps (all without one).
            // The generator is borrowed only for this statement: the recursion below uses it too.
            let children = self.generator.borrow_mut().legal_positions(position, die1, die2);
            for child in self.candidates(position, children) {
                // The child has the opponent to move, so flip its value.
                let value = self.value_for_turn(&child, depth).flip();
                match &best {
                    Some((_, b)) if value.equity <= b.equity => {}
                    _ => best = Some((child, value)),
                }
            }
        }

        best.unwrap_or_else(|| {
            let mut after_pass = position.clone();
            after_pass.switch_turn();
            let value = self.value_for_turn(&after_pass, depth).flip();
            (after_pass, value)
        })
    }
}

impl<P: Position + Clone + Send + Sync + 'static> Default for RustAnalyzer<P> where P::CacheKey: Send + Sync {
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
    fn test_money_scoring_counts_gammons() {
        // Me bears off its last checker; Opponent has borne off nothing: a gammon (2 points), and
        // with a checker on its 20-point (Me's home board) a backgammon (3).
        let gammon = pos(&[(1, 1), (0, 14)], &[(1, 15)], 1);
        let backgammon = pos(&[(1, 1), (0, 14)], &[(1, 14), (20, 1)], 1);
        let mut analyzer = RustAnalyzer::with_cache_mb(4);
        assert_eq!(analyzer.best_move(&gammon, 2, 1, 0).unwrap().1, 1.0);
        analyzer.set_scoring(Scoring::Money);
        assert_eq!(analyzer.best_move(&gammon, 2, 1, 0).unwrap().1, 2.0);
        assert_eq!(analyzer.best_move(&backgammon, 2, 1, 1).unwrap().1, 3.0);
        assert_eq!(analyzer.equity(&gammon, Player::Opponent, 2), -2.0);
    }

    #[test]
    fn test_parallel_search_gives_the_serial_results() {
        use crate::encoding::Encoding;
        use crate::neural_net::{NeuralNet, NeuralNetEvaluator, Objective};
        let net = NeuralNet::new_random_for(Encoding::TdNack, &[32, 16], Objective::Money, 4);
        let make = |threads: usize| {
            let mut a = RustAnalyzer::with_evaluator(Box::new(NeuralNetEvaluator { net: net.clone() }), 16);
            a.set_scoring(Scoring::Money);
            a.set_threads(threads);
            a
        };
        let (serial, parallel) = (make(1), make(4));
        assert_eq!((serial.threads(), parallel.threads()), (1, 4));
        let mut position = crate::EnginePosition::new();
        position.setup_starting_position();
        position.set_turn(Player::Me);
        for depth in 0..3 {
            for (d1, d2) in [(3, 1), (6, 6)] {
                serial.clear_cache();
                parallel.clear_cache();
                let a = serial.best_move(&position, d1, d2, depth).unwrap();
                let b = parallel.best_move(&position, d1, d2, depth).unwrap();
                assert_eq!(a.0.cache_key(0), b.0.cache_key(0));
                assert_eq!(a.1, b.1);
                let ma = serial.analyze_moves(&position, d1, d2, depth.min(1));
                let mb = parallel.analyze_moves(&position, d1, d2, depth.min(1));
                assert_eq!(ma.len(), mb.len());
                for (x, y) in ma.iter().zip(&mb) {
                    assert_eq!(x.0.cache_key(0), y.0.cache_key(0));
                    assert_eq!(x.1, y.1);
                    assert_eq!(x.2.map(f64::to_bits), y.2.map(f64::to_bits));
                }
            }
            if depth <= 2 {
                assert_eq!(serial.equity(&position, Player::Me, depth), parallel.equity(&position, Player::Me, depth));
                let (oa, ob) = (serial.outcome_for_turn(&position, depth), parallel.outcome_for_turn(&position, depth));
                assert_eq!(oa.map(f64::to_bits), ob.map(f64::to_bits));
            }
        }
    }

    #[test]
    fn test_rollout_is_the_same_on_threads_and_counts_points() {
        use crate::encoding::Encoding;
        use crate::neural_net::{NeuralNet, NeuralNetEvaluator, Objective};
        let net = NeuralNet::new_random_for(Encoding::TdNack, &[32, 16], Objective::Money, 4);
        let make = |threads: usize| {
            let mut a = RustAnalyzer::with_evaluator(Box::new(NeuralNetEvaluator { net: net.clone() }), 16);
            a.set_scoring(Scoring::Money);
            a.set_threads(threads);
            a
        };
        let (serial, parallel) = (make(1), make(4));
        let mut position = crate::EnginePosition::new();
        position.setup_starting_position();
        position.set_turn(Player::Me);
        let a = serial.rollout(&position, 30, 50, 7, 0);
        let b = parallel.rollout(&position, 30, 50, 7, 0);
        assert_eq!(a, b);
        assert!(a.iter().all(|&p| p != 0 && (-3..=3).contains(&p)));
        // Games 1296 apart have the same first two rolls but (usually) different later ones.
        let c = serial.rollout(&position, 30 + 1296, 50, 7, 0);
        assert_ne!(a, c);
        // A finished game: the side to move has borne off everything.
        let (mut me, mut opp) = (vec![0u8; 26], vec![0u8; 26]);
        me[0] = 15;
        opp[6] = 15;
        let won = crate::build_position(me, opp, 1, 1, None).unwrap();
        assert_eq!(serial.rollout(&won, 0, 3, 1, 0), vec![2, 2, 2]);
    }

    /// A random money net analyzer on `threads` threads with a move filter.
    fn filtered_analyzer(filter: Option<MoveFilter>, threads: usize) -> RustAnalyzer<crate::EnginePosition> {
        use crate::encoding::Encoding;
        use crate::neural_net::{NeuralNet, NeuralNetEvaluator, Objective};
        let net = NeuralNet::new_random_for(Encoding::TdNack, &[32, 16], Objective::Money, 4);
        let mut a = RustAnalyzer::with_evaluator(Box::new(NeuralNetEvaluator { net }), 16);
        a.set_scoring(Scoring::Money);
        a.set_threads(threads);
        a.set_move_filter(filter);
        a
    }

    fn start() -> crate::EnginePosition {
        let mut position = crate::EnginePosition::new();
        position.setup_starting_position();
        position.set_turn(Player::Me);
        position
    }

    #[test]
    fn test_a_move_filter_that_keeps_everything_changes_nothing() {
        let wide = Some(MoveFilter { margin: f64::INFINITY, max_moves: usize::MAX });
        let (none, all) = (filtered_analyzer(None, 1), filtered_analyzer(wide, 1));
        let position = start();
        for depth in 0..3 {
            let (a, b) = (none.best_move(&position, 3, 1, depth).unwrap(), all.best_move(&position, 3, 1, depth).unwrap());
            assert_eq!(a.0.cache_key(0), b.0.cache_key(0));
            assert_eq!(a.1, b.1);
            assert_eq!(none.value(&position, depth), all.value(&position, depth));
        }
    }

    #[test]
    fn test_move_filter_keeps_the_margin_and_at_most_max_moves() {
        let position = start();
        let a = filtered_analyzer(Some(MoveFilter { margin: 0.1, max_moves: 3 }), 1);
        let children = a.generator.borrow_mut().legal_positions(&position, 3, 1);
        let scores: Vec<f64> = children.iter().map(|c| -a.static_child_equity(&position, c)).collect();
        let best = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mask = a.candidate_mask(&position, &children);
        let kept: Vec<usize> = (0..children.len()).filter(|&i| mask[i]).collect();
        assert!(!kept.is_empty() && kept.len() <= 3);
        assert!(kept.iter().all(|&i| scores[i] >= best - 0.1));
        // The kept ones are the best: no left-out move within the margin scores higher.
        let worst_kept = kept.iter().map(|&i| scores[i]).fold(f64::INFINITY, f64::min);
        assert!((0..children.len()).filter(|&i| !mask[i]).all(|i| scores[i] <= worst_kept || scores[i] < best - 0.1));
        // A zero margin keeps only the best (and moves tied with it).
        let tight = filtered_analyzer(Some(MoveFilter { margin: 0.0, max_moves: 5 }), 1);
        let mask = tight.candidate_mask(&position, &children);
        assert!(mask.iter().filter(|&&m| m).count() >= 1);
    }

    #[test]
    fn test_filtered_analysis_lists_the_left_out_moves_last_at_one_ply() {
        let a = filtered_analyzer(Some(MoveFilter { margin: 0.1, max_moves: 2 }), 1);
        let position = start();
        let moves = a.analyze_moves(&position, 3, 1, 1);
        let searched = moves.iter().take_while(|m| m.3 == 1).count();
        assert!((1..=2).contains(&searched));
        assert!(moves[searched..].iter().all(|m| m.3 == 0));
        assert!(moves.len() > searched);
        let (_, best) = a.best_move(&position, 3, 1, 1).unwrap();
        assert_eq!(moves[0].1, best);
    }

    #[test]
    fn test_filtered_search_is_the_same_on_threads() {
        let filter = Some(MoveFilter { margin: 0.1, max_moves: 3 });
        let (serial, parallel) = (filtered_analyzer(filter, 1), filtered_analyzer(filter, 4));
        let position = start();
        for depth in 1..3 {
            let (a, b) = (serial.best_move(&position, 6, 4, depth).unwrap(), parallel.best_move(&position, 6, 4, depth).unwrap());
            assert_eq!(a.0.cache_key(0), b.0.cache_key(0));
            assert_eq!(a.1, b.1);
            let (ma, mb) = (serial.analyze_moves(&position, 6, 4, depth), parallel.analyze_moves(&position, 6, 4, depth));
            assert_eq!(ma.iter().map(|m| (m.1, m.3)).collect::<Vec<_>>(), mb.iter().map(|m| (m.1, m.3)).collect::<Vec<_>>());
            assert_eq!(serial.value(&position, depth), parallel.value(&position, depth));
        }
    }

    #[test]
    fn test_analyze_moves_ranks_like_best_move_with_matching_probabilities() {
        // Me (to move) has checkers on its 3- and 2-points, Opponent one checker on its 1-point.
        let position = pos(&[(3, 1), (2, 1), (0, 13)], &[(1, 1), (0, 14)], 1);
        let analyzer = RustAnalyzer::with_cache_mb(4);
        for depth in 0..3 {
            let moves = analyzer.analyze_moves(&position, 2, 1, depth);
            assert!(moves.len() > 1);
            let (_, best) = analyzer.best_move(&position, 2, 1, depth).unwrap();
            assert_eq!(moves[0].1, best);
            for (_, equity, outcome, levels) in &moves {
                // At double match point the equity is 2 * winning chance - 1 (the probabilities
                // are kept as f32).
                assert!((equity - (2.0 * outcome[0] - 1.0)).abs() < 1e-6, "depth {}: {} vs {:?}", depth, equity, outcome);
                assert_eq!(*levels, depth);   // no move filter: every move searched
            }
            assert!(moves.windows(2).all(|w| w[0].1 >= w[1].1));
        }
        // No legal move: the passed position.
        let blocked = pos(&[(25, 1), (0, 14)], &[(1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (0, 3)], 1);
        let moves = analyzer.analyze_moves(&blocked, 6, 5, 1);
        assert_eq!(moves.len(), 1);
        assert_eq!(moves[0].0.get_turn(), Player::Opponent);
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
        assert!(analyzer.cache.get(&position.cache_key(1)).is_some());
        assert!(analyzer.cache.get(&position.cache_key(3)).is_some());
        // A fresh analyzer must give the same deep value: the cached shallow value was not reused.
        assert_eq!(deep, RustAnalyzer::with_cache_mb(4).equity(&position, Player::Me, 3));
        assert_eq!(shallow, 26.0 / 36.0);
        assert_eq!(deep, 16.0 / 36.0);
    }
}
