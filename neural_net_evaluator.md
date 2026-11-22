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
