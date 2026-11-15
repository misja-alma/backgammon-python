# Neural net evaluator — progress tracker

Tracks the work of replacing the placeholder `Evaluator` with a neural net, and the search changes
that come with it (caching leaf evaluations, pruning). Update the checkboxes, the status table and
the log as work is done. Background on the current engine: [architecture.md](architecture.md).

**Status:** not started (planning) · **Last updated:** 2026-11-15

| Workstream | Status |
|---|---|
| 1. Evaluator interface | not started |
| 2. Net design | not started |
| 3. Training | not started |
| 4. Inference in Rust | not started |
| 5. Search: leaf-level caching and duplicates | not started |
| 6. Search: pruning | not started |
| 7. Integration (Python, UI, settings) | not started |
| 8. Measuring strength and speed | not started |

---

## Starting point (2026-11-15)

What the current engine does and costs, as a baseline to compare against.

- **Evaluator:** `Evaluator::evaluate` (Python `backgammon/game/evaluator.py`, Rust `src/evaluator.rs`)
  returns equity for the side to move: +1 won, −1 lost, 0 for every unfinished position. No gammons.
  So the engines currently play almost randomly until the end of the game.
- **Search:** expectimax over the 21 rolls, `depth` = roll levels still to search; depth 0 = the
  evaluator alone. At the last level, `MoveGenerator::best_position` scores each resulting position
  as soon as it is generated, without a duplicate check and without the cache — on purpose, because
  scoring a leaf (a few ns) is cheaper than a cache lookup (~100 ns: hashing a 60-byte key plus a
  random read in a 1 GB table).
- **Cache:** `TranspositionTable`, fixed size (1 GB default, `Settings.rust_cache_mb`), keyed by
  (position, remaining depth), only for positions with depth ≥ 1. Two-slot buckets, deeper entries
  kept preferentially.
- **Speed (Rust engine, `SimplePosition`, this machine):**

  | Workload | Time |
  |---|---|
  | Depth-3 evaluation, middle-game position | 3.8 s |
  | Depth-3 move, opening position (6-5) | 8.4 s |
  | Full Rust-vs-Rust game, depth 1 | ~0.1 s |
  | Full Rust-vs-Rust game, depth 2 | 0.5 s |
  | Leaves evaluated in a depth-3 evaluation | ~700 million before the duplicate fixes (to re-measure) |

- **Size of the tree:** each search level multiplies the work by about 21 rolls × ~20 distinct
  moves ≈ 400. With an evaluator costing even 1 µs, an unpruned depth-3 search would take minutes;
  hence workstreams 5 and 6.

---

## 1. Evaluator interface

Make the evaluator pluggable so the search doesn't care whether it is the placeholder or a net.

- [ ] Rust `Evaluator` trait (or generic parameter on `RustAnalyzer`), with the current one as
      `TerminalEvaluator` (or similar) so existing tests keep working.
- [ ] Decide the output: a single equity, or probabilities (win, gammon win, backgammon win,
      gammon loss, backgammon loss) from which equity is computed. Probabilities are standard for
      backgammon nets and needed once gammons count.
- [ ] Gammons/backgammons in the rules and the equity (currently ±1 only). Affects
      `Evaluator`, `Game` (winner reporting), the Analyze display (winning chance can no longer be
      derived from equity alone) and the Python–Rust agreement tests.
- [ ] Batch evaluation API (evaluate many positions at once), if inference benefits from it (see 4).
- [ ] Python side: decide whether the Python engine gets the net too (slow), stays on the
      placeholder, or is retired as a reference implementation.

## 2. Net design

- [ ] Input encoding. Candidates: TD-Gammon style (per point and player, 4 units per point:
      1, 2, 3, ≥4 checkers; plus bar, borne off, side to move — 198 inputs), or a gnubg-style
      encoding with extra hand-crafted features.
- [ ] Architecture: MLP, number and size of hidden layers, activation. Start small (e.g. one
      hidden layer of 64–128 units) to keep inference cheap.
- [ ] Separate nets per phase? (contact, race, bear-off). A bear-off database is an option for
      the end of the game.
- [ ] Weight file format and versioning (which net is loaded, compatibility with the encoding).

## 3. Training

- [ ] Approach: TD(λ) self-play (TD-Gammon), supervised learning on rollout results, or both
      (bootstrap with TD, refine with rollouts).
- [ ] Tooling: train in Python (e.g. PyTorch), run inference in Rust; or train in Rust. The
      Rust-driven game (`play_game`) can generate self-play games quickly.
- [ ] Training data/game generation pipeline, reproducible from seeds.
- [ ] Checkpoints and a record of which net version is in use.

## 4. Inference in Rust

- [ ] Forward pass in Rust (f32, dense layers). Consider SIMD and batching.
- [ ] Incremental input updates: a half-move changes only a few inputs, so the first layer's
      activations could be updated instead of recomputed (matters a lot for speed, since leaves
      differ from their parent by one move).
- [ ] Target and measured cost per evaluation (e.g. aim for well under 1 µs).
- [ ] Encoding the position directly from `SimplePosition`'s arrays in `explore`, without copying.

## 5. Search: leaf-level caching and duplicates

With an expensive evaluator the trade-offs measured so far flip: a leaf evaluation will cost more
than a cache lookup or a hash.

- [ ] Check (and store) leaf evaluations in a cache. Probably a separate, smaller table for depth-0
      evaluations, so they don't push out the expensive deeper entries in the main table.
      Measure the leaf hit rate first.
- [ ] Skip duplicate leaves within a roll before evaluating (the `legal_positions` mode already
      deduplicates; `best_position` does not). After the canonical move ordering of 2026-11-15
      duplicates are already much rarer — measure how many remain.
- [ ] Re-measure where the time goes once the net is in (profiling as in architecture.md,
      "Search performance").

## 6. Search: pruning

Needed to make depth 2–3 affordable with a real evaluator.

- [ ] Move filtering: evaluate all candidate moves cheaply (depth 0), search only the best few
      (top N, or within a threshold of the best) deeper. Standard in backgammon engines.
- [ ] Apply filtering at every level of the search, not only at the root.
- [ ] Optionally: iterative deepening, and reduced roll sets at deep levels.
- [ ] Measure the cost of pruning in playing strength (see 8) against the speed gained; make the
      filter settings configurable.

## 7. Integration (Python, UI, settings)

- [ ] Settings: evaluator choice (placeholder / net), net file to load, pruning settings.
- [ ] Load the net once in the Rust engine (thread-local analyzer), with an error shown in the UI if
      the file is missing or incompatible.
- [ ] Analyze menu: show win/gammon probabilities as well as equity.
- [ ] Engine players use the net in games; check game speed at depths 1–3.
- [ ] Tests: evaluator interface, encoding, a fixed net giving known outputs, Python–Rust parity
      where it still applies.

## 8. Measuring strength and speed

- [ ] Benchmark positions and timings (extend `position_benchmark` / a new benchmark) with the net.
- [ ] Strength: matches between engine versions (net vs placeholder, net vs previous net, depth
      vs depth), run Rust-vs-Rust with seeds; record win rates with confidence intervals.
- [ ] Optional: compare move choices against a reference engine on a set of positions.

---

## Decisions

Record decisions here as they are made (date, decision, reason).

| Date | Decision | Reason |
|---|---|---|
| | | |

## Open questions

- Single equity output or win/gammon probabilities from the start?
- Train in Python or in Rust?
- Keep the Python engine as a reference, or retire it once the net is in?
- How much strength may pruning cost for a given speed-up?

## Log

- **2026-11-15** — Tracker created. Baseline recorded above. Search already prepared: fixed-size
  transposition table, position-based move generation with canonical move ordering, leaf scoring
  without copies.
