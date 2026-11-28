# Neural net evaluator — progress tracker

Tracks the work of replacing the placeholder `Evaluator` with a neural net, and the search changes
that come with it (caching leaf evaluations, pruning). Update the checkboxes, the status table and
the log as work is done. Background on the current engine: [architecture.md](architecture.md).

**Status:** first net trained (TD(λ) self-play) · **Last updated:** 2026-11-19

| Workstream | Status |
|---|---|
| 1. Evaluator interface | done (single equity output, no gammons yet) |
| 2. Net design | 54-128-128-128-1 (default since 2026-11-19, was 54-128-64-1) |
| 3. Training | TD(λ) self-play in Rust, optionally multi-threaded (`--threads N`, ~8,000 games/s with 12) |
| 4. Inference in Rust | straightforward forward pass, zero inputs skipped |
| 5. Search: leaf-level caching and duplicates | not started |
| 6. Search: pruning | not started |
| 7. Integration (Python, UI, settings) | net loaded from `Settings.rust_network` |
| 8. Measuring strength and speed | benchmark vs bgsage (`make strength`) |

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

- [x] Rust `Evaluator<P>` trait (`src/evaluator.rs`); `RustAnalyzer` holds a
      `Box<dyn Evaluator<P>>` (`set_evaluator`, `with_evaluator`). Implementations:
      `TerminalEvaluator` (the old placeholder, still the default), `PipCountEvaluator` (yardstick
      for training) and `NeuralNetEvaluator`.
- [x] Decide the output: a single equity, or probabilities (win, gammon win, backgammon win,
      gammon loss, backgammon loss) from which equity is computed. Probabilities are standard for
      backgammon nets and needed once gammons count.
- [ ] Gammons/backgammons in the rules and the equity (currently ±1 only). Affects
      `Evaluator`, `Game` (winner reporting), the Analyze display (winning chance can no longer be
      derived from equity alone) and the Python–Rust agreement tests.
- [ ] Batch evaluation API (evaluate many positions at once), if inference benefits from it (see 4).
- [x] Python side: decide whether the Python engine gets the net too (slow), stays on the
      placeholder, or is retired as a reference implementation.

## 2. Net design

- [x] Input encoding (`src/encoding.rs`): the net stores its encoding's name and encodes its own
      inputs, so nets with different encodings can be swapped in. Per player, side to move first:
      - `counts-pips-v1` (54 inputs): the checker count of every point incl. borne off and bar
        (÷15) and the pip count (÷167).
      - `td-counts-v1` (254 inputs, 2026-11-20): TD-Gammon style plus the real count. For each of
        points 1–24 and the bar, 5 units for n checkers: n≥1, n≥2, n≥3, max(n−3,0)/2, n/15; then
        borne off ÷15 and the pip count ÷167. (`train --encoding td`.)
      - `td-flags-v1` (257 inputs, 2026-11-20): `td-counts-v1` plus 3 flags: race (no contact
        left), and per player (on roll first) a nackgammon flag: not a race and ≥4 checkers on
        own points 19–24. A td-counts net can be extended to it with zero weights for the flags
        (`train --resume X --encoding flags`).
      A gnubg-style encoding with extra hand-crafted features is still a candidate.
- [x] Architecture (`src/neural_net.rs`): 54 → 128 ReLU → 128 ReLU → 128 ReLU → 1 sigmoid =
      winning chance of the side to move (first version: 54 → 128 → 64 → 1). Layer sizes come from the weight file. Was: Start small (e.g. one
      hidden layer of 64–128 units) to keep inference cheap.
- [ ] Separate nets per phase? (contact, race, bear-off). A bear-off database is an option for
      the end of the game.
- [x] Weight file format and versioning (`BGNET 1`: text header with encoding, layer sizes,
      activations and games trained, then little-endian f32). Loading rejects unknown encodings. (which net is loaded, compatibility with the encoding).

## 3. Training

- [x] Approach: TD(λ) self-play (`src/td_training.rs`, `cargo run --release --bin train`).
      Was: TD(λ) self-play (TD-Gammon), supervised learning on rollout results, or both
      (bootstrap with TD, refine with rollouts).
- [x] Tooling: trained in Rust, no dependencies. Was: train in Python (e.g. PyTorch), run inference in Rust; or train in Rust. The
      Rust-driven game (`play_game`) can generate self-play games quickly.
- [ ] Training data/game generation pipeline, reproducible from seeds.
- [ ] Checkpoints and a record of which net version is in use.

## 4. Inference in Rust

- [x] Forward pass in Rust (f32, dense layers, `[input][output]` weight layout so zero inputs
      are skipped). Profiled with `cargo run --release --bin forward_profile`, then optimized
      (2026-11-19): hidden layers in chunks of 32 outputs kept in vector registers with fused
      multiply-adds over the active inputs only, the output layer as a dot product, scratch buffers
      reused per thread. 615 → 265 ns per evaluation; depth-2 searches twice as fast; same moves
      (strength benchmark identical). Remaining: ~85 ns collecting active inputs/overhead,
      104 ns layer 2, 55 ns layer 1.
- [ ] Incremental input updates: a half-move changes only a few inputs, so the first layer's
      activations could be updated instead of recomputed (matters a lot for speed, since leaves
      differ from their parent by one move).
- [x] Target and measured cost per evaluation (e.g. aim for well under 1 µs): 265 ns.
- [ ] Encoding the position directly from `SimplePosition`'s arrays in `explore`, without copying.

## 5. Search: leaf-level caching and duplicates

With an expensive evaluator the trade-offs measured so far flip: a leaf evaluation will cost more
than a cache lookup or a hash.

- [x] **Measured 2026-11-19, not worth it** (`cargo run --release --bin leaf_cache_benchmark`):
      a net evaluation costs 530 ns and a lookup 40 ns, but only ~10% of the leaves in a depth-2
      search repeat a position. Caching leaves in the main table made depth-2 searches 15-17%
      slower (warm table). Evaluation is ~96% of search time, so speed must come from a faster
      forward pass (section 4) or fewer leaves (section 6). Original item:
      Check (and store) leaf evaluations in a cache. Probably a separate, smaller table for depth-0
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
      Net file: `Settings.rust_network` (no dialog yet).
- [x] Load the net once in the Rust engine (thread-local analyzer), with an error shown in the UI if
      the file is missing or incompatible (`load_network`, logged in the text area at startup).
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
| 2026-11-19 | Single output: winning chance of the side to move | No gammons in the rules yet |
| 2026-11-19 | Inputs: raw checker counts per point + pip count, side to move first | User's design |
| 2026-11-19 | Hidden layers 128 and 64, ReLU | User's design (sizes); ReLU is cheaper to evaluate than sigmoid |
| 2026-11-19 | TD(λ) self-play, trained in Rust | User's choice; self-play games already run in Rust |
| 2026-11-19 | Net only in the Rust engine | The Python engine stays the placeholder reference |
| 2026-11-19 | Strength measured against bgsage 1-ply | `benchmarks/strength.py`, paired games |
| 2026-11-19 | Hidden layers 128-128-128 | Size experiment: ~1.2 points stronger than 128-64 at 3M games (3 seeds each) |
| 2026-11-19 | End training with weight averaging (`--average-last`) | +1.5-3 points over the last raw snapshot |

## Open questions

- Single equity output or win/gammon probabilities from the start?
- Train in Python or in Rust?
- Keep the Python engine as a reference, or retire it once the net is in?
- How much strength may pruning cost for a given speed-up?

## Log

- **2026-11-20** — Lower α at the plateau (branches from the 33M σ 0.2 net, 3M games, 12 threads,
  averaged over the last 500k; nets in `nets/alpha/`): α 0.01 (control) 43.45%, 0.005 43.10%,
  0.002 42.89%, 0.01→0.001 decay 43.08% (±0.9 each). No gain from a lower learning rate; not
  continued. Together with the perturbation test: the ~43% plateau is a limit of this net size,
  input encoding and training method.
- **2026-11-20** — Shrink-and-perturb test at the plateau (nets in `nets/perturb/`): branches from
  the raw 30M net (41.7%), 3M games each with 12 threads, averaged over the last 500k. Right after
  perturbing: σ 0.05 37.2%, σ 0.1 21.0%, σ 0.2 2.5%. After 3M games: control-a 43.16%, control-b
  43.14%, σ 0.05 42.93%, σ 0.1 42.38%, σ 0.2 43.68% (±0.9 each). No breakthrough: all branches
  within noise of the controls, even σ 0.2, which nearly destroyed the net, recovered to the same
  level in 3M games. The plateau looks like a limit of the net/inputs/training method, not a local
  optimum.
  New default `nets/td.bgnet` = σ 0.2 branch averaged (43.68%); previous default
  `nets/parallel/h128-128-128-t12-10M-avg.bgnet`.
- **2026-11-20** — The 10M parallel run continued to 30M games (12 threads, ~40 min). Raw
  checkpoints: 14M 38.7%, 18M 41.0%, 22M 40.9%, 26M 42.1%, 30M 41.7%; averaged over the last
  500k: **42.93%** (`nets/parallel/h128-128-128-t12-30M-avg.bgnet`). The curve flattens from ~18M
  (~0.25 points per million games after that). New `train --perturb/--shrink` for branching.
- **2026-11-20** — Parallel training (`train --threads N`): every thread plays whole games on its
  own copy of the net and adds its weight changes to the shared net after each game. 128-128-128:
  858 games/s on 1 thread, 5,900 on 10, ~8,000 on 12 in real runs. Per game it learns less well
  (stale weights): 3M games with 12 threads gave 29.9 / 29.9% averaged (serial: 32.7-33.6%);
  α 0.005 didn't help (26.5%). But it is ~9.5× faster: 10M games in ~20 minutes gave 28.2% (3M),
  32.8% (6M), 36.5% (10M, raw) and **37.16% averaged over the last 500k** — the strongest net so
  far (`nets/parallel/h128-128-128-t12-10M-avg.bgnet`). Serial runs stay reproducible.
  Now the default `nets/td.bgnet` (previous default: `nets/long/h128-128-128-s2-avg.bgnet`).
- **2026-11-20** — Dead ReLU units (zero in every position of 40 self-play games, now shown by
  `forward_profile`): 128-128-128 nets have 2-9% dead units in layer 1 and 0-5% in layers 2-3,
  at 250k as well as at 10M games (10M net: 11 / 4 / 2 of 128). Not a major loss of capacity.
- **2026-11-20** — Training profiled (`cargo run --release --bin train_profile`). Per move
  (128-128-128): gradient 7.3 µs, trace 3.2, weights 3.8, move choice 10.5 (15 evaluations).
  New `NeuralNet::forward_keep` + `td_update` do back-propagation, trace and weight update in one
  pass: training 666 → 862 games/s (128-128-128), ~1570 → 2286 games/s (128-64). Same results
  as before (test against the old path). Adam still uses the old path.
- **2026-11-19** — Hidden-layer sizes compared, each trained from scratch (α 0.01, λ 0.7, seed 1;
  nets in `nets/size/`). Win rate vs bgsage 1-ply at 250k / 500k / 1M games:
  64-32 3.6 / 7.1 / 9.9%; 128-64 6.7 / 10.9 / 17.8%; 256-128 7.2 / 9.1 / 19.4%;
  256-128-64 8.2 / 16.4 / 21.2%; 512-256 13.1 / 17.8% / – (only 500k trained).
  Bigger nets learn more per game, but cost time: training 1,381 / 479 / 432 / 105 games/s for
  128-64 / 256-128 / 256-128-64 / 512-256, evaluation 280 ns (128-64), 780 ns (256-128),
  4.6 µs (512-256). Per hour of training 128-64 is still far ahead (3.3M games = 31%).
  Second round, same settings: 64-128 5.8 / 8.8 / 10.3%; 128-128 8.4 / 12.8 / 17.0%;
  128-256 6.6 / 9.2 / 17.9%; 128-128-128 7.7 / 15.4 / 24.8% (best at 1M of all sizes).
  Depth matters more than width: both 3-hidden-layer nets beat every 2-layer net of similar or
  larger size. Evaluation: 64-128 260 ns, 128-128 390 ns, 128-256 690 ns.
  128-64-128-64 (4 hidden layers): 7.7 / 15.4 / 16.3% (750k: 15.8%), 718 games/s. Fast start,
  then stalls: the 64-unit bottlenecks seem to limit it. Not better than 128-128-128.
  128-64-128: 7.6 / 14.5 / 16.3 (750k) / 17.8%; 162-108-128: 8.0 / 13.5 / 16.6 (750k) / 17.8%.
  The larger 162-108-128 ending far below 128-128-128 suggests the 24.8% may be partly luck of
  the training run: run-to-run variance (other seeds) has not been measured yet.
- **2026-11-19** — Run-to-run variance (seeds 1/2/3, 1M games, nets in `nets/seeds/`):
  128-64 17.8 / 14.9 / 17.4% (mean 16.7); 128-128-128 24.8 / 18.7 / 12.7% (mean 18.7).
  At 500k: 128-64 10.9 / 9.0 / 9.0%, 128-128-128 15.4 / 8.8 / 9.2%. Seed 1 of 128-128-128 was
  a lucky run; the spread between runs (up to 12 points) is far larger than the benchmark error.
  Conclusion: single 1M-game runs can't rank net shapes; the size experiment is inconclusive
  apart from 64-wide first layers being too small. Default stays 128-64.
- **2026-11-19** — Stable endings: `train --average-last N` (mean of the weights over the last N
  games). Every 1M net continued 100k games (α 0.01, checkpoints every 25k), nets in
  `nets/stable/`. Snapshot noise: 128-64 s1 17.8/18.9/19.0/19.4/19.1% (1.0M…1.1M, steady),
  128-64 s2 14.9/15.1/11.9/14.9/16.3% (one 3-point dip), 128-128-128 s1 24.8/24.9/27.0/27.2/25.7%.
  The average usually matches or beats every snapshot (128-128-128 s1: 28.2%, 128-64 s2: 16.7%).
  Averaged nets: 64-32 11.4, 64-128 11.5, 128-128 18.3, 128-64-128 18.6, 128-64 18.8/16.7/18.4
  (s1/s2/s3), 128-256 19.4, 162-108-128 19.4, 256-128 19.7, 128-128-128 28.2/19.6/15.1,
  256-128-64 27.0%. Two-layer nets with ≥128 first-layer units all land at 17-20%; three-layer
  runs are bimodal: 2 of 6 reach 27-28%, the rest 15-20%. Run-to-run spread is real (not
  snapshot noise): averaging didn't shrink the gap between 128-128-128 seeds.
- **2026-11-19** — Nine runs continued to 3M games (α 0.01, average of the last 200k games),
  nets in `nets/long/`. Averaged / raw 3M / raw 2.05M:
  128-64 s1 32.6/30.0/29.3, s2 31.1/30.2/28.3, s3 32.0/28.8/20.8;
  128-128-128 s1 32.7/31.4/28.5, s2 33.6/31.3/29.8, s3 33.0/30.9/30.5;
  128-64-128 31.9/29.8/29.3; 162-108-128 32.6/31.7/30.5; 256-128-64 32.8/32.1/29.8.
  At 3M the run-to-run spread has collapsed (the 1M gaps were differences in learning speed, not
  in final quality). Means of the averaged nets: 128-64 31.9%, 128-128-128 33.1%: three layers
  help ~1 point at 3M. Averaging adds 1.5-3 points over the raw snapshot: always use it.
  New default `nets/td.bgnet` = 128-128-128 s2 averaged (33.56%); old default kept as
  `nets/td-const.bgnet`. `DEFAULT_HIDDEN` is now [128, 128, 128].
- **2026-11-20** — Adam tested (`train --optimizer adam`, TD update δ·e fed to Adam as the
  gradient; nets in `nets/adam/`). Sweep at 300k games: α 3e-5/1e-4/3e-4/1e-3 gave 16.7/15.5/
  14.5/13.3% (128-128) and 15.6/15.9/15.0/13.1% (128-128-128), vs ~8% for SGD at 250k. Long runs
  with α 3e-5, 3 seeds each, averaged / raw 3M / raw 1M:
  Adam 128-128-128 17.3/16.3/16.6, 16.9/15.2/16.2, 18.9/17.1/16.5;
  Adam 128-128 18.3/17.0/18.7, 20.2/18.3/17.3, 18.1/17.4/17.3;
  SGD 128-128 31.2/30.4/17.0, 31.1/29.9/16.3, 30.7/29.3/17.3.
  Adam's head start is gone by 1M and it then stalls at 16-20%; SGD reaches 31-33%. Likely cause:
  Adam normalizes every weight's step to ~α however small the TD errors get, so the net keeps
  jittering at a fixed noise level, while SGD's steps shrink with the errors. Rejected; SGD stays.
  128-128 with SGD (mean 31.0%) is no better than 128-64 (31.9%); 128-128-128 (33.1%) stays.
- **2026-11-19** — Learning-rate decay tested: from the 2.3M net (28.63% vs bgsage), 1M more games
  with α 0.01→0.001 (exponential) gave 30.79%, the control with constant α 0.01 31.06% (±0.88%
  each): no measurable benefit from decay. The extra games themselves gave ~+2.3 points.
  Default net `nets/td.bgnet` = the 3.3M constant-α net.
- **2026-11-19** — Forward pass 2.3× faster (615 → 265 ns), see section 4.
- **2026-11-19** — Leaf caching measured and rejected (see section 5). First trained net
  (300k TD(λ) games) wins 8.37% vs bgsage 1-ply.
- **2026-11-19** — Baseline vs bgsage 1-ply (placeholder evaluator, depth 0): 0.31% of 10 000
  games. Evaluator trait, net, encoding, TD(λ) trainer, weight files and Python loading added.
- **2026-11-15** — Tracker created. Baseline recorded above. Search already prepared: fixed-size
  transposition table, position-based move generation with canonical move ordering, leaf scoring
  without copies.

**2026-11-20 — Continued the best net (p0.2, 33M) by 7M games** (1M segments, 12 threads, α 0.01, averaged over each segment's last 500k; nets in `nets/to40m/`). Against bgsage: 34M 43.39, 35M 43.03, 36M 43.13, 37M 43.64, 38M 43.77, 39M 43.48, 40M 43.56 (±0.9 each). Flat: mean 43.43 vs 43.68 for the 33M default. No further improvement from more training; default unchanged.

**2026-11-20 — Richer inputs (`td-counts-v1`) on 128-128.** Two 128-128 nets, 12 threads, α 0.01, in 2M segments, each segment's last 500k averaged (`nets/enc/`). Against bgsage (±0.9):

| games | td-counts-v1 | counts-pips-v1 (control) |
|---|---|---|
| 2M | 45.44 | 16.67 |
| 4M | 46.19 | 29.95 |
| 6M | 46.37 | 31.72 |
| 8M | 46.73 | 32.52 |
| 10M | 46.96 | 34.51 |

The new inputs win by a wide margin: after 2M games the 128-128 net already beats the best old net (128-128-128, 33M games, 43.68%), and it is still creeping up (~0.4 points per 2M). Training speed ~9,400 games/s (control 14,500, a smaller first layer).

**2026-11-20 — td-counts 128-128 continued to 30M** (same settings, 2M segments). Against bgsage: 12M 47.58, 14M 47.29, 16M 48.13, 18M 47.68, 20M 47.98, 22M 48.63, 24M 48.44, 26M 48.72, 28M 48.22, 30M 48.41 (±0.9). Plateau from ~22M at ~48.5% (rise mostly done by 16M). Default `nets/td.bgnet` = `nets/enc/td-26M-avg.bgnet` (48.72%); λ = 0.7 throughout, untested so far.

**2026-11-20 — λ at the td-counts plateau.** Branches from `nets/enc/td-26M.bgnet` (raw), 3M games each, 12 threads, α 0.01, averaged over the last 500k (`nets/lambda/`). Against bgsage (±0.9): λ 0.7 (control) 48.65, 0.6 48.40, 0.65 47.74, 0.75 49.12, 0.8 48.70. All within noise of the control (a difference of two runs has ±1.3); λ is flat over 0.6–0.8. Default unchanged.

**2026-11-20 — λ check, more runs.** Two more independent 3M branches each for λ 0.7 / 0.75 / 0.8 (seeds 82, 83), all nine benchmarked with 30,000 games on the same new dice (seed 2, ±0.53): λ 0.7: 47.71, 47.73, 47.79 (mean 47.74); λ 0.75: 47.53, 47.46, 47.14 (47.38); λ 0.8: 47.30, 47.45, 47.32 (47.36). Every 0.7 run beats every 0.75 and 0.8 run on the same dice; the 49.12 of the first 0.75 test was benchmark luck. λ stays 0.7. (Absolute levels are ~1 point below the seed-1 10k benchmarks: one dice sample vs another, σ of the difference ≈ 0.58.)

**2026-11-20 — Nackgammon training.** `train --variant nackgammon` and `benchmarks.strength --variant nackgammon` added. Branch from the backgammon plateau net (`nets/enc/td-26M.bgnet`), nackgammon starts only, 2M segments, averaged (`nets/nack/`). Against bgsage, 10k games, seed 1 (±0.95):

| net | on nackgammon | on backgammon |
|---|---|---|
| default (backgammon-trained) | 47.13 | 48.72 |
| +2M nackgammon | 46.91 | 46.85 |
| +4M nackgammon | 48.46 | 47.67 |
| +6M nackgammon | 48.05 | 47.63 |

The backgammon net already plays nackgammon well (transfer); nackgammon training adds ~1 point there and costs ~1 point on backgammon, which then stabilizes (a new balance, not continued forgetting). Candidate next: mixed training (both starts) or a separate net per variant.

**2026-11-20 — Race + nackgammon flags (`td-flags-v1`).** (`nets/flags/`)
- A: the td-counts plateau net (26M raw) extended with zero-weight flag inputs, 3 branches of 3M (seeds 81–83), 30k games seed 2: 47.79 / 47.72 / 47.71 (mean 47.74) — identical to the λ 0.7 controls without flags (47.71 / 47.73 / 47.79, mean 47.74). No gain at the plateau on backgammon.
- B: fresh 128-128 flags net, 2M segments, 10k seed 1: 2M 45.43, 4M 45.80, 6M 47.32, 8M 48.26, 10M 47.77 vs td-counts without flags 45.44 / 46.19 / 46.37 / 46.73 / 46.96. From 6M on ~1 point ahead (mean 47.78 vs 46.69): possibly faster learning, but one training run each (run-to-run variance is large early in training); the plateau of the td-counts net was ~48.5.
- C: fresh 128-128 flags net, mixed training (`--variant mixed`, alternating backgammon / nackgammon starts), 2M segments, 10k seed 1 (±0.95):

| games | C on backgammon | C on nackgammon | B (bg only) on backgammon |
|---|---|---|---|
| 2M | 45.57 | 44.82 | 45.43 |
| 4M | 47.04 | 46.25 | 45.80 |
| 6M | 47.10 | 46.36 | 47.32 |
| 8M | 47.34 | 46.55 | 48.26 |
| 10M | 46.94 | 47.86 | 47.77 |

  B's 10M net on nackgammon: 46.36. Mixed costs ~0.7 on backgammon (6–10M mean 47.13 vs 47.78) and gains ~1 on nackgammon (8–10M mean 47.2 vs 46.36). Neither run has plateaued; the specialists at their plateau: 48.72 (backgammon) and 48.46 (nackgammon, `nets/nack/nack-4M-avg`).

**2026-11-20 — C continued to 30M.** Same settings, 2M segments from `nets/flags/mixed-10M`, seeds 66–75, 10k seed 1 (±0.95):

| games | 12M | 14M | 16M | 18M | 20M | 22M | 24M | 26M | 28M | 30M |
|---|---|---|---|---|---|---|---|---|---|---|
| backgammon | 46.94 | 47.10 | 47.10 | 47.53 | 48.17 | 47.52 | 48.36 | 47.95 | 47.83 | 48.19 |
| nackgammon | 47.21 | 47.13 | 48.38 | 48.14 | 47.53 | 46.90 | 47.73 | 47.44 | 48.01 | 47.80 |

Plateau from ~20M (backgammon) and ~16M (nackgammon). 20–30M means: backgammon 48.00 vs 48.48 for the backgammon specialist at its plateau (td-counts 22–30M); nackgammon 47.57 vs ~48.25 for the nackgammon specialist (4M/6M) and 47.13 for the backgammon net. The mixed net is ~0.5 point behind each specialist, but clearly better on nackgammon than the backgammon-only default. Best per variant: separate nets (default for backgammon, `nets/nack/nack-4M-avg` for nackgammon); one mixed net (`nets/flags/mixed-24M-avg` or `-30M-avg`) is the simpler alternative at ~0.5 cost.

**2026-11-21 — Net per variant; architecture sweep started.** The app now switches the Rust engine's net with the game type: `Settings.rust_network` (`nets/td.bgnet`) for backgammon, `Settings.rust_nackgammon_network` (`nets/nack.bgnet` = `nets/nack/nack-4M-avg`) for nackgammon. New encoding `td-nack-v1` (256 inputs: td-counts + the two nackgammon flags, no race flag; `--encoding nack`). Sweep (`nets/arch/run.py`, results in `nets/arch/results.txt`): 256-256, 256-128, 256-256-256, 512-128, mixed training, 4M segments, α 0.01, λ 0.7, averaged over the last 500k; each average benchmarked with 40k games on both variants (seed 1, ±0.46); a net stops after 3 segments without beating its best mean by 0.3 (max 48M). Training speed (12 threads): 128-128 7.7k games/s, 256-128 3.8k, 256-256 2.5k, 256-256-256 and 512-128 1.6k. Baseline: the 128-128 mixed flags net C at 30M, same benchmark.

**2026-11-22 — Default net 256-256; two-sided bearoff database.** Default (both variants): `nets/td.bgnet` = `nets/arch/256-256-44M-avg` (40k: bg 49.56, nack 50.67). Bearoff database (`src/bearoff.rs`, `make bearoff`, `data/bearoff-6x15.bd`): every position with both sides in their home boards (6 points × up to 15 checkers, 54,264 positions per side, 2.94 billion pairs), the exact cubeless winning chance of the side to move with both sides maximizing it, stored as u16 (5.9 GB, memory-mapped). Generated in 2 minutes (4 threads, 12 GB peak) via precomputed one-sided move tables and layers of equal total pip count. Checks: equal to a full-depth search on small positions; one-step recursion with bgsage's move generator on 3000 random positions within 0.000012; vs bgsage's one-sided-based values mean |diff| 0.0003, max 0.010 (bgsage picks moves by minimal expected rolls, not maximal winning chance). gnubg: same recursion but two-sided only up to 11 checkers (ships 6 × 6), one-sided distributions beyond. The engine uses it at every search node (exact value, no search below) when `Settings.rust_bearoff_db` exists; the benchmark only with `--bearoff`.

**2026-11-22 — Bearoff database in training and benchmark.** `train --bearoff data/bearoff-6x15.bd`: a self-play game stops at the first bearoff position; the exact winning chance is the final TD target (like a game result), and moves into bearoff positions are chosen by the database. Games get ~7% shorter (63.5 → 59.2 turns), speed about the same. The benchmark (`benchmarks/strength.py`) uses the database by default from now on (`--bearoff none` for the old behaviour), so scores from here on are not directly comparable with earlier ones. The sweep continues in this mode (`nets/arch/run.py`: `train-db-bin`, re-benchmark of the unfinished nets' last average with the database as their new reference).

**2026-11-22 — Re-benchmark with the bearoff database; 756 stopped.** Same nets, 40k per variant (±0.46), database in the benchmark only (nets trained without it):

| net | bg before | bg with db | nack before | nack with db | mean before | mean with db |
|---|---|---|---|---|---|---|
| 128-128 C 30M | 47.93 | 47.99 | 48.27 | 48.35 | 48.10 | 48.17 |
| 256-128 24M | | 49.65 | | 49.76 | 49.61 | 49.70 |
| 512-128 24M | | 49.65 | | 50.42 | 49.94 | 50.03 |
| 256-256-256 36M | 49.81 | 49.86 | 50.51 | 50.54 | 50.16 | 50.20 |
| 256-256 44M (default) | 49.56 | 49.60 | 50.67 | 50.72 | 50.12 | 50.16 |

A consistent +0.04 to +0.09: the nets already play bearoffs nearly perfectly at 1-ply, and the paired dice make the shift visible despite the noise. 756 (single layer) stopped at 20M: 47.55 (+0.26 per 4M, still >2 behind every two-layer net). 512-256 continues with the database in training.

**2026-11-22 — Half-precision evaluation net.** `src/eval_net.rs`: an inference-only copy of a trained net per hidden layer in f32, f16 (weights, biases and sums in half precision, 8 lanes per NEON register, 128 outputs per pass, non-zero units handed on via a branch-free table compaction), f16 weights with f32 sums, or int8 (`sdot`, groups of 4 inputs). Measured with `cargo run --release --bin eval_precision` on the default 256-256 (101,708 candidate positions from 100 self-play games; training running alongside):

| variant | ns/eval | other move than f32 | equity loss per decision |
|---|---|---|---|
| f32 | ~1650 | – | – |
| f16 weights, f32 sums | ~1290 | 0.06% | 0.000000 |
| f16 | ~550 | 0.5% | 0.000002 |
| int8 | ~810 | 6.9% | 0.00016 |
| f32 compute, 8 / 6 / 5 / 4-bit weights | – | 3.9 / 12 / 22 / 51% | 0.00006 / 0.0007 / 0.003 / 0.03 |

int8 is both less accurate (per-output-unit weight scales) and slower than f16 (active inputs are scattered over the groups of 4). Depth-1 search: 18.0 → 6.3 ms per move (2.84×), other move in 2 of 989; depth 2: 12.7 → 3.3 s per move (3.89×), other move in 0 of 300 (fresh 256 MB cache per move, training running alongside). vs bgsage (40k, bearoff db): f16 bg 49.50, nack 50.62 vs f32 49.60, 50.72: same strength within noise. The app uses `nets/td-eval.bgnet` (`make eval-net`, f16 copy of `nets/td.bgnet`); `load_network` reads both kinds of file.

**2026-11-22 — Quantization-aware training (queued after the sweep).** `train --quantize f16|i8`: the weights stay f32 (TD steps of ~1e-5 would vanish in f16 or int8 weights), but every game is played and judged by the net's f16 or int8 evaluation copy, rebuilt at the start of each game: move choice, estimate and target of every TD error (offline TD(λ), straight-through gradient of the f32 net). int8 copies use input scale 1/30 for layer 1 (exact) and a fixed 0.05 for layer 2. Per turn ~1.6x faster than f32 self-play (2 threads: 20.6k vs 33.7k turns/s), since the moves are chosen by the fast kernels. `nets/qat/run.py`: fresh 256-256 nets, same schedule as the sweep's f32 256-256 and like it without bearoff database, each segment's average benchmarked as its evaluation copy (40k, `--bearoff none`); reference: the int8 copy of the f32-trained 256-256 44M. Eval-net files now also hold int8 layers (stored f32, quantized on loading with the scales from the header); `make_eval_net --precision i8`.

**2026-11-23 — Quantization-aware training: result.** Mean of bg and nack vs bgsage (40k each, no bearoff db), each net benchmarked as its own evaluation copy:

| 256-256 | 8M | 16M | 24M | 32M | 40M | 44M | 48M |
|---|---|---|---|---|---|---|---|
| f32 (sweep) | 48.45 | 48.75 | 49.19 | 49.70 | 50.03 | 50.12 | 49.88 |
| trained in f16 | 47.90 | 48.69 | 49.02 | 49.05 | 49.37 | 49.46 | 49.62 (plateau) |
| trained in int8 | 48.11 | 48.83 | 49.04 | 49.39 (plateau) | | | |

Converting after training is better: the int8 copy of the f32 44M scores 49.77 and its f16 copy matches f32 (see above), while the QAT nets end 0.3–0.6 lower. Also tried 576-192 in f16: it collapsed (0.15% for 12M games, 42% at 16M), and 128-128 in int8 reached 47.44 at 20M; both stopped. From 28M on, the benchmark used bgsage 2.1.20261122: identical play (same 1000-game result), 2.5× faster benchmarks. Next: 576-192 and 128-128 trained in f32 (`run.py f32-576-192 f32-128-128`), converted at the plateau to f16 and int8 respectively.

**2026-11-23 — The 128-128 mixed net converted.** `nets/flags/mixed-30M-avg` (td-flags-v1, 128-128, mixed, 30M) vs bgsage (40k each, no bearoff db): f32 bg 47.93 nack 48.27 (mean 48.10); int8 copy 47.68 / 48.43 (48.06); f16 copy 47.98 / 48.39 (48.19). Both copies are as strong as the f32 net within the noise, even int8 (8% other moves, but cheap ones). Speed (training alongside): f32 518 ns, f16 221 ns, int8 397 ns per evaluation, against 550 ns for the default 256-256 in f16, which is ~2 points stronger (50.1). Files in `nets/convert/`. A new f32 128-128 run was dropped.

**2026-11-23 — Decision benchmark.** A fast, noise-free progress measure: checker play against bgsage's reference decisions (`make decisions NETS="…" [SET=pasko]`; `benchmarks/extract_decisions.py` + `src/bin/decision_benchmark.rs`). Source: bgsage's money benchmark (500 games of bgsage 3-ply against itself; reference 3-ply for clear decisions, truncated rollout within 0.05, full rollout of 1296 games within 0.02) and its Paskogammon counterpart. Kept: checker decisions that are not bearoffs, and of each only the candidates evaluated at 3-ply or better (≥ 2): 13,889 decisions (55,144 candidates) in five groups by bgsage's game plan: attacking 4103, priming 2434, anchoring 3193, racing 3041, purerace 1118; Pasko 2189. Scored for double match point: the reference best is the candidate with the highest rolled-out winning chance (in 15.7% of the decisions not bgsage's best money move). Per group: `best` (% picks of the reference best), `loss` (winning chance lost per decision, points), `error` (mean |net − reference| winning chance over the candidates), `other` (net's favourite among all legal moves is none of the candidates). 0.4 s per f16 net, 1 s per f32 256-256.

Validation on 84 checkpoints with 40k-game scores (architecture sweep, QAT, 576-192, 128-128): `loss` vs mean win rate Pearson −0.98 (Spearman −0.97), `best` +0.98; among the nets ≥ 49.5 still −0.72 / +0.80, with a residual sd of 0.16–0.18 around the fit: about the noise of the game benchmark itself (sd ~0.17 for the mean of 2×40k games). The groups correlate about equally (−0.96), except purerace (−0.75: few, easy decisions). Default 256-256 (f16 = f32): best 74.1%, loss 0.162; best checkpoints 512-256 32M (75.35%, 0.143) and 28M (75.08%, 0.149), as in the games. Pasko (unseen start): loss 0.27 for the default, 0.25 for 512-256 28M, 0.35 for the 128-128.

**2026-11-23 — Decision benchmark without pure races.** Pure races (bgsage's "purerace" game plan) are left out: once contact is over, the bearoff database and the search take over. Now 12,771 decisions (49,143 candidates) in four groups; Pasko 2110. Validation (85 checkpoints): `loss` vs mean win rate −0.98, `best` +0.97; nets ≥ 49.5: −0.73 / +0.78, residual sd 0.17–0.19. Default 256-256: best 74.42%, loss 0.172; 512-256 32M: 75.75%, 0.152; bgsage 1-ply (scored the same way, `post_move` win probability at a 1-point match): 76.67%, 0.137, flattered by the benchmark being bgsage's own (its games, candidates pre-selected by its 1-ply, references from its deeper searches): in games our default plays it about even.

**2026-11-23 — Training curves on the decision benchmark; plateau rule switched.** Loss per segment (decision benchmark, no pure races) of the earlier runs: 512-256 4M–32M 0.239 / 0.207 / 0.192 / 0.183 / 0.170 / 0.167 / 0.158 / 0.152 (still improving when stopped; the game dip at 32M, 50.46 → 50.30, is noise); 256-256 4M–48M 0.253 / 0.229 / 0.209 / 0.200 / 0.196 / 0.188 / 0.180 / 0.178 / 0.177 / 0.176 / 0.171 / 0.170 (near its ceiling; the game dips at 16M and 48M are noise); 576-192 4M–28M 0.234 / 0.197 / 0.186 / 0.176 / 0.168 / 0.165 / 0.160 (= 512-256). From its 32M segment on, the 576-192 run stops on the decision loss instead of the games: 3 segments without lowering the best loss by 0.005 (~0.15 points of win rate), max 48M; games against bgsage only every 12M and for the final (lowest-loss) net, then its f16 copy.

## 576-192 (f32): final result

The plateau call was shortened to 2 segments without lowering the decision loss by 0.005, with
no games for intermediate segments (the decision benchmark compares segments more reliably).
Decision loss 32M 0.160, 36M 0.158 (best 75.26%), against 0.160 at 28M: plateau, 28M stays the
best net.

| net | bg | nack | mean |
|---|---|---|---|
| 576-192 28M (f32) | 49.98 | 50.63 | 50.30 |
| 576-192 28M (f16 copy) | 50.03 | 50.64 | 50.34 |

The f16 copy plays as strong as the f32 net. Next: continued training from 28M with alpha 0.005,
then 0.0025 (nets/qat/lowalpha.py).

## 576-192: continued with a lower alpha

From the raw 28M net (decision loss 0.160), 4M-game segments scored on the decision benchmark
(nets/qat/lowalpha.py): alpha 0.005 for 12M (32M 0.153, 36M 0.152, 40M 0.151), then alpha 0.0025
from 40M for 16M (44M 0.147, 48M 0.148, 52M 0.144, 56M 0.146). For comparison, alpha 0.01 gained
0.002 over 28M-36M.

| net | decision loss | best | pasko loss | bg | nack | mean |
|---|---|---|---|---|---|---|
| 576-192 28M (alpha 0.01) | 0.160 | 74.86% | 0.290 | 49.98 | 50.63 | 50.30 |
| 576-192 a0025-52M | 0.144 | 76.19% | 0.258 | 50.26 | 50.96 | 50.61 |
| 576-192 a0025-52M (f16 copy) | | | | 50.28 | 51.12 | 50.70 |

Lowering alpha after the plateau gains about 0.3-0.4 points of win rate against bgsage 1-ply.

## 512-256 and 256-256 continued (alpha 0.01), and the candidates for a new default

Continued from where the sweep stopped them (nets/arch/continue.py; bearoff database in training
and benchmark, plateau after 2 segments without lowering the decision loss by 0.005, final games
for the lowest-loss net). 512-256: 32M 0.152, 36M 0.149, 40M 0.151. 256-256: 48M 0.170,
52M 0.166, 56M 0.169.

| net | decision loss | bg | nack | mean | f16 copy mean | f16 ns/eval |
|---|---|---|---|---|---|---|
| 256-256 52M | 0.166 | 49.57 | 50.37 | 49.97 | 49.92 | 449 |
| 512-256 36M | 0.149 | 49.90 | 50.97 | 50.44 | 50.44 | 1227 |
| 576-192 a0025-52M (low alpha) | 0.144 | 50.26 | 50.96 | 50.61 | 50.70 | 1191 |

(ns/eval: eval_precision, 30 games, the 256-256 row measured with nets/td.bgnet.) More training
at alpha 0.01 hardly changes the game results; the lower alpha does.

## Net files are marked with their objective

Header line `objective dmp` (src/neural_net.rs `Objective`): the net's single output is the
winning chance at double match point, gammons don't count. All 700 existing nets under nets/ got
the line; files without it still load as DMP nets. Gammon-aware nets will get their own objective.

## Money nets (gammons): first steps

Cubeless money, no cube. A money net (`objective money`) has 5 sigmoid outputs for the player on
roll, cumulative like GNU Backgammon's: win, win gammon, win backgammon, lose gammon, lose
backgammon; equity = 2·win - 1 + wg - lg + wbg - lbg (clipped to consistent values).

- Training (td_training.rs): TD(λ) with a trace per output, seen from a fixed side (the outputs
  swap meaning when the side to move changes). Moves by money equity; a finished game's target is
  its result; games stop at bearoff positions only when no gammon is possible any more (both
  sides have borne off a checker). The 5 traces made training 12× slower than DMP at first; a
  rewritten update (const number of traces, vectorized) brought it to ~1000 games/s on 12
  threads (DMP ~2300).
- Engine: `Scoring::Money` in the analyzer (finished games ±1/2/3, bearoff database only without
  gammon chances); a DMP net in money play uses 2·win - 1.
- Decision benchmark `--scoring money`: bgsage's reference probabilities are cumulative and give
  its cubeless equity exactly; reference best = highest cubeless equity (it differs from
  bgsage's cubeful best in 6% of the money set's decisions). Loss in millipoints per decision.
  DMP nets under money scoring: 256-256 at 4M 8.90, 16M 7.84, 48M 7.30; the default 576-192 7.07.
- Game benchmark `strength --money`: bgsage plays cubeless money too (its money setting, the
  move with the highest cubeless equity). A first 400-game check with the DMP default net: win
  rate 50.5%, but -0.41 points per game: once it can't win, every move is equal to it, and it
  loses many backgammons (41 of 200 games in a tally).

First money net: 256-256 from scratch (nets/money/run.py), compared per segment with the DMP
256-256 at the same number of games.
- Reference under money scoring: bgsage 1-ply 77.75% best, loss 3.69 millipoints (flattered, as
  in DMP: its own benchmark). The DMP nets lose about twice that, mostly through gammons.

## The standard training procedure

benchmarks/train_schedule.py (make train-schedule): 4M-game segments; alpha 0.01, then 0.005,
then 0.0025, each next phase from the best net so far once the decision loss stops improving
(2 segments without lowering it by 0.005 points (dmp) / 0.15 millipoints (money), 60% of that in
later phases; the first phase at most 80M games, later ones 16M); the decision benchmark per
segment; 40k games against bgsage every 16M games and for the final best net and its f16 copy.
The money 256-256 run switched to it after its 8M segment.

Money 256-256 at 16M: decision loss 5.66 millipoints (DMP 256-256 at 16M: 7.84), Paskogammon
9.50; 40k money games per variant against bgsage 1-ply (cubeless money): backgammon -0.046,
nackgammon -0.039, mean -0.043 points per game (the DMP default in a 400-game check: -0.41). Made
the default money-game net: nets/td-money.bgnet, f16 copy nets/td-money-eval.bgnet (decision loss
5.66 too).

## Money 256-256: the full run (train_schedule)

Decision loss (millipoints, money scoring) per segment, with the game types:

| segment | alpha | all | attacking | priming | anchoring | racing | pasko | games (ppg vs bgsage 1-ply) |
|---|---|---|---|---|---|---|---|---|
| 4M | 0.01 | 6.85 | 6.58 | 7.84 | 5.88 | 7.46 | 11.09 | |
| 8M | 0.01 | 6.10 | 5.85 | 6.94 | 5.52 | 6.37 | 10.35 | |
| 12M | 0.01 | 5.78 | 5.74 | 6.57 | 5.06 | 5.95 | 9.55 | |
| 16M | 0.01 | 5.66 | 5.49 | 6.34 | 4.97 | 6.08 | 9.50 | -0.043 |
| 20M | 0.01 | 5.53 | 5.28 | 6.23 | 5.17 | 5.69 | 10.09 | |
| 24M | 0.005 | 5.33 | 5.05 | 5.93 | 4.90 | 5.66 | 9.77 | |
| 28M | 0.005 | 5.21 | 4.90 | 5.84 | 4.88 | 5.47 | 9.52 | |
| 32M | 0.005 | 5.19 | 4.89 | 5.66 | 4.94 | 5.50 | 9.56 | -0.025 |
| 36M | 0.005 | 5.06 | 4.78 | 5.47 | 4.72 | 5.48 | 9.50 | |
| 40M | 0.0025 | 5.07 | 4.72 | 5.52 | 4.84 | 5.42 | 8.98 | |
| 44M | 0.0025 | 4.91 | 4.63 | 5.33 | 4.69 | 5.17 | 9.35 | |
| 48M | 0.0025 | 4.89 | 4.58 | 5.26 | 4.60 | 5.35 | 9.14 | -0.019 |
| 52M | 0.0025 | 4.83 | 4.63 | 5.37 | 4.48 | 5.03 | 8.67 | -0.024 (f16 copy -0.024) |

The DMP 256-256 at the same games: 8.90 at 4M, 7.14 at 52M. Lowering alpha helped again: 0.01
gained ~0.13 per segment at the end, 0.005 0.20 in its first. 48M and 52M in games are within the
noise of each other (40k games per variant).

### Lower alpha for the 512-256 and 256-256 DMP nets (train_schedule --plateaued)

Both had plateaued at alpha 0.01 (512-256 at 36M, 256-256 at 52M); continued with alpha 0.005, then
0.0025 (margin 0.003, 2 stale segments, at most 16M per phase). Decision loss in winning-chance points;
games: win % vs bgsage 1-ply, 40k games per variant.

| Net | Loss | Best | Pasko | Attacking | Priming | Anchoring | Racing | bg | nack | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 512-256 36M (alpha 0.01) | 0.149 | 75.75% | 0.258 | 0.128 | 0.142 | 0.163 | 0.166 | 49.90 | 50.97 | 50.44 |
| 512-256 a005-48M | 0.138 | 76.69% | 0.251 | 0.115 | 0.133 | 0.152 | 0.160 | 50.39 | 51.54 | 50.97 |
| **512-256 a005-52M** | **0.135** | 76.74% | 0.239 | 0.110 | 0.139 | 0.142 | 0.157 | 50.68 | 51.25 | 50.97 |
| 512-256 a005-52M (f16 copy) | | | | | | | | 50.66 | 51.19 | 50.92 |
| 256-256 52M (alpha 0.01) | 0.166 | 74.90% | 0.297 | | | | | 49.57 | 50.37 | 49.97 |
| **256-256 a0025-64M** | **0.161** | 75.09% | 0.270 | 0.142 | 0.155 | 0.175 | 0.176 | 49.79 | 50.42 | 50.11 |
| 256-256 a0025-64M (f16 copy) | | | | | | | | 49.87 | 50.45 | 50.16 |
| 576-192 a0025-52M (default) | 0.144 | 76.19% | 0.258 | | | | | 50.26 | 50.96 | 50.61 |

The 512-256 net (0.135) is now the best DMP net on the decision benchmark and in games.

**2026-11-26 — Mixture of experts (built, queued).** From the activation analysis of the 512-256 default (layer-2 neuron groups, `analysis/activations.py`), 11 game types (`analysis/export_types.py` → `nets/moe/types.txt`, each the top 8% of contact positions by its groups' mean z-scored activation; a position can have several types; races and untyped positions are generic): holding-ahead, holding-behind, blitz-attacker, blitz-defender, prime-trapper, prime-trapped, contact-bearoff-ahead, contact-bearoff-behind, back-game, escaped-opponent, late-hit. Reference (router) net frozen as `nets/moe/reference.bgnet`. Staged training (`benchmarks/moe_schedule.py`, `src/moe.rs`): 1. collect 1M positions per expert from self-play of the reference net (`moe_collect`); 2. each expert alone (`moe_expert`): self-play games starting from its own positions, its type judged by the expert, the rest by the frozen reference, only the expert learns, only on its type; alphas 0.005 → 0.0025, 2M-game segments, measured on a separate measuring set (`nets/moe/measure`, own seed, excluded from training): 1000 decisions per type judged by bgsage 3-ply DMP (`benchmarks/moe_measure.py` → `data/decisions/moe-types.txt`; positions without a real decision skipped, e.g. 88% of blitz-defender positions are dances or forced); 3. gateway (`moe_gate`, one hidden layer of 512, 12 sigmoid outputs = exp(-(error/0.03)²) against a 1-ply lookahead), experts frozen; 4. evaluator (`.bgmoe`): experts rated < 0.25 skipped, remaining ratings normalized, weighted average. Reference (512-256 default) on the measuring set: all 0.151; holding-ahead 0.172, holding-behind 0.159, blitz-attacker 0.175, blitz-defender 0.186, prime-trapper 0.154, prime-trapped 0.160, contact-bearoff-ahead 0.072, contact-bearoff-behind 0.152, back-game 0.130, escaped-opponent 0.190, late-hit 0.116, generic 0.140. (bgsage lists bearing off the last checker with one die as legal when both dice could be played; our generator requires both dice: such no-decision positions are skipped.) Starts after the 384-384 and money 512-256 runs.

**2026-11-26 — Decision benchmark leaves out decisions without a difference.** `decision_benchmark` now drops decisions where every candidate is worth the same under the scoring (every net scores 0 there): money set 52 under dmp scoring (moves differing only in gammons) and 4 under money scoring, Paskogammon 9 and 2. Losses are ~0.4% higher than before (fewer decisions in the denominator), best % slightly lower. The running 384-384 and money 512-256 schedules keep their own (old) copy of the binary until they finish. Re-measured (dmp: money set / pasko; money nets: millipoints):

| Net | money set | attacking | priming | anchoring | racing | pasko |
|---|---|---|---|---|---|---|
| 512-256 a005-52M (default DMP) | 0.135 | 0.111 | 0.139 | 0.143 | 0.157 | 0.240 |
| 576-192 a0025-52M (previous default) | 0.145 | 0.125 | 0.144 | 0.158 | 0.158 | 0.259 |
| 256-256 a0025-64M | 0.162 | 0.143 | 0.155 | 0.176 | 0.176 | 0.271 |
| 384-384 32M (alpha 0.01, running) | 0.164 | 0.137 | 0.161 | 0.177 | 0.188 | 0.275 |
| money 256-256 a0025-52M (default money), money scoring | 4.831 | 4.635 | 5.366 | 4.483 | 5.033 | 8.674 |
| money 512-256 12M (running), money scoring | 5.413 | 4.967 | 5.790 | 5.167 | 5.970 | 9.311 |
| 512-256 DMP default, money scoring | 6.464 | 6.117 | 6.402 | 6.752 | 6.678 | 9.730 |

## Money 512-256 (train_schedule): the new default money net

Decision loss (millipoints, money scoring): alpha 0.01 to 24M (4.75), 0.005 to 36M (4.29), 0.0025
to 52M. Best: 52M avg, loss 4.047 (attacking 4.05, priming 4.42, anchoring 3.51, racing 4.31,
Paskogammon 8.09); the 256-256 money default had 4.83. 40k money games per variant against
bgsage: backgammon +0.0046, nackgammon +0.0093, mean +0.0069 ppg (f16 copy +0.0066; 256-256
default -0.0238). Made the default money net (2026-11-27): nets/td-money.bgnet, f16 copy
nets/td-money-eval.bgnet; the previous one is nets/td-money-256-256-52M(-eval).bgnet.

## Training speed: offline TD(λ) (2026-11-27)

Profile of money training (512-256 net, one thread, `train_profile_money`): 79% of the time went
to the dense eligibility traces (264k parameters x 5 outputs decayed and added after every move,
223 µs per move), 19% to the move choice (18.4 evaluations of ~2.9 µs). Now offline TD(λ): the
weights stay fixed during a game and change at its end by what the traces would add up to (one
sparse backward pass per position of its λ-return error; zero inputs and inactive ReLUs skipped),
and the chosen position is no longer evaluated again (the next turn's estimate is its target).
Training (12 threads): 471 -> 2497 games/s (5.3x); a 4M segment takes 26 min instead of ~2.5 h.

Quality: 4M games at alpha 0.0025 from the 512-256 money run's 44M and 48M raw nets, against the
old run's next segments (decision loss in millipoints, money / Paskogammon):

    from 44M:  old 4.157 / 8.055   offline 4.015 / 7.905   offline + f16 move choice 4.075 / 8.233
    from 48M:  old 4.048 / 8.099   offline 3.955 / 7.895   offline + f16 move choice 4.015 / 8.089

Offline TD(λ) is at least as good: the standard from now on (SGD; Adam stays online). Choosing the
moves with the f16 copy (`train --search f16`, money nets) doubles the speed again (4964 games/s)
but trails by 0.06 both times: not used. The f32 forward pass is limited by its weight loads (the
f16 copy is 2.5x faster per evaluation); the move choice is now ~80% of the training time.

## Incremental first layer (2026-11-27)

The positions after the legal moves of one position differ from that position (seen by the
opponent, who is on roll in all of them) in ~10.4 of the 256 inputs, against ~43 non-zero inputs.
So the first layer's sums of the position before the move are computed once, and each candidate
adds only the rows of its differing inputs, times the difference (`NeuralNet::layer1_base`,
`forward_all_incremental`; for the f16 evaluation copy `EvalNet::forward_all_incremental`).
Equal to a full evaluation up to rounding (unit tests).

- Training (move choice, f32): 512-256 money, one thread 290 -> 413 games/s; 12 threads 2497 ->
  3811 games/s (8.1x the 471 of the start of the day); DMP 512-256 1999 -> 2709. Also in the
  MoE expert trainer.
- Engine (the search's last level, `Evaluator::evaluate_child_scored`, f16 nets): best move at 1
  roll level 7.9 -> 5.6 ms (1.39x), at 2 levels 1.46x (`search_benchmark`). The incremental values
  only choose the move at the last level; the chosen position is then evaluated in full, so the
  search's values are those of full evaluations (only near-ties can go the other way: 0 of 200
  moves at 1 level, largest equity difference 0.00008).

## Further inference speed: tried, not kept (2026-11-27)

Engine search profile (`search_benchmark`, money f16 net, best move at 1 roll level): 7244
evaluations per move, 98% of the time in the net (~740 ns per incremental evaluation), the rest of
the search 16 ns per evaluation.

- Batched evaluation of sibling candidates (layer 2 for groups of 2-8 children together, each
  weight row loaded once for the union of their active units; bit-identical): siblings do share
  most units (85 active per child, 115 for 4 together, 129 for all ~20), but it was slower: 863
  ns per evaluation for groups of 4 (8: 1489, registers spill; 2: 984) against 738. The f16 kernel
  is limited by load/FMA throughput, not by memory (one 8-lane load feeds one 8-lane FMA), so
  saving loads doesn't help while the union costs ~35% multiply-adds by zero. The same arithmetic
  holds for f32. Collecting the children as a list also tripled the rest of the search.
- int8 layers (existing `Kind::I8`, `eval_precision`, money 512-256): [f16 i8] 1523 ns per
  evaluation against 987 for [f16 f16], and 5.4% other moves than f32 (loss 0.126 millipoints per
  decision; f16: 0.55%, no measurable loss). Not worth it without quantization-aware training.
- NEON intrinsics for the f32 sparse layer (16 accumulators, no bounds checks): no change (2741 vs
  2724 ns per evaluation; training 405 vs 413 games/s): the compiler's code was already the same
  loads and FMAs, limited by load throughput.

## f16 evaluation: memory, prefetching, output layer, encoding (2026-11-27)

Stage profile of an incremental evaluation as the search makes it (`incremental_profile`, money
512-256 f16, 24892 candidates of 1382 positions), ns per candidate, before -> after:

    whole evaluator call            659 -> 557
    layer 2                         287 (273 with its weight rows already in L1; 323 with prefetching)
    layer 1 (incremental)           129
    output layer + sigmoids         135 -> 29   (rows of 8 per input, two vector FMAs per active unit)
    parent's base per candidate      63 -> 22   (16 registers instead of 1)
    encoding + differing inputs      68 -> 45   (from the points whose checker counts differ:
                                                `Encoding::encode_diff`, exact, reused buffers)

Layer 2 waits for memory only 5% of its time, and prefetching the next rows makes it slower: the
kernels are at the instruction-throughput limit, so assembly wouldn't help either. Packing the
non-zero outputs (push_nonzero): a table for the bit counts helped a little (576 -> 568); reducing
the lane masks of 4 registers together was slower (layer 1 128 -> 142), not kept.

Search (`search_benchmark`, best move at 1 roll level): money 7.75 ms (full evaluations, before
today) -> 4.59 ms (1.7x); 2 roll levels 3.28 s -> 1.95 s; DMP 7.6 -> 5.2 ms. The results are
unchanged (the incremental values only choose the move at the last level; 0 other moves).

## Multi-threaded engine search (2026-11-27)

`RustAnalyzer::set_threads(n)`: searches of at least one roll level run on n threads. Worker
analyzers (own copy of the evaluator via `Evaluator::fork`, own move generator) share the cache (now
a `SharedTable`: 64 shards behind their own locks) and the bearoff database. `best_move` hands out
(candidate, roll) pairs, `equity`/`outcome_for_turn` the 21 rolls, `analyze_moves` the candidates;
the results are combined in the serial order, so they are exactly those of one thread (unit test,
and `search_benchmark` checks every move). The app uses `Settings.rust_threads` (0: one per core);
the strength benchmark's worker processes stay at one thread each.

Best move, money 512-256 f16 (`search_benchmark ... 1,4,8,10,12,14`, M4 Pro, 10 performance + 4
efficiency cores):

    threads          1      4      8     10     14
    1 roll level  4.63   1.31   0.77   0.67   0.65 ms   (6.9x at 10)
    2 roll levels 1948      -      -    229    218 ms   (8.5x at 10, 8.9x at 14)

Through the Python binding (analyze_moves at 2 levels + analyze_position at 2 levels, opening
position, 3-1): 2.81 s -> 0.39 s with 10 threads, identical results.

## MoE experts: measuring by the candidates' game type (2026-11-27)

The first money expert (holding-ahead, 256-128) seemed to get worse on its own type (9.46 ->
11.48, then 10.07), but it was measured on the wrong decisions: the measuring set groups decisions
by the position before the move, while the net evaluates the positions after the moves, with the
opponent on roll, which are mostly of the mirror type (holding-behind). On the holding-behind group
the expert had improved (5.73 -> 5.12). `decision_benchmark --candidate-types` now counts a decision
for a type when most of its candidates are of that type, and moe_schedule measures the experts that
way: holding-ahead base 6.10 -> 4.76 after one 2M segment at alpha 0.005 (value net 512-256: 4.55).
