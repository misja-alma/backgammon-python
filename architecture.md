# Architecture

This project is a desktop backgammon board visualizer with position analysis. The user sets up
a position by clicking on the board, and the application computes winning chances by searching
the game tree. The UI and game model are written in Python; the performance-critical analysis
engine also has a Rust implementation that is compiled into a native Python extension.

## High-level components

```
┌──────────────────────────────────────────────────────────────────────┐
│ Python process                                                       │
│                                                                      │
│  main.py ── event loop                                               │
│     │                                                                │
│     ▼                                                                │
│  ┌──────────────────────┐      ┌──────────────────────────────────┐  │
│  │ UI  (backgammon/ui)  │─────▶│ Game model (backgammon/game)     │  │
│  │ pygame               │      │ Position, Player, Utils,         │  │
│  │ Board, BoardMap,     │      │ Settings, Game, MoveBuilder,     │  │
│  │ TextArea, dialogs,   │      │ Evaluator                        │  │
│  │ GameController       │      └──────────────────────────────────┘  │
│  └──────────┬───────────┘                     ▲                      │
│             │ analyze() / engine moves        │                      │
│             ▼                                 │                      │
│  ┌──────────────────────────────────────┐     │                      │
│  │ AnalysisWorker (background thread)   │     │                      │
│  └──────────────────┬───────────────────┘     │                      │
│                     ▼                         │                      │
│  ┌──────────────────────────────────────┐     │                      │
│  │ AnalyzerProxy (backgammon/game)      │─────┘                      │
│  │ engine/depth per call or Settings    │                            │
│  └───────┬──────────────────────┬───────┘                            │
│          │                      │                                    │
│          ▼                      ▼  PyO3 FFI (plain lists / ints)     │
│  ┌────────────────┐   ┌─────────────────────────────────────────┐    │
│  │ PythonAnalyzer │   │ backgammon_rust  (native extension .so) │    │
│  │ + Evaluator    │   │ Rust engine: RustAnalyzer, Evaluator,   │    │
│  └────────────────┘   │ CacheFriendlyPosition, Utils                     │    │
│                       └─────────────────────────────────────────┘    │
│                                                                      │
│  Logger ──▶ TextArea (cross-cutting status output)                   │
└──────────────────────────────────────────────────────────────────────┘
```

| Component | Location | Language | Responsibility |
|---|---|---|---|
| Entry point / event loop | `main.py` | Python | Creates the board, dispatches pygame events, redraws at 60 fps |
| UI | `backgammon/ui/` | Python (pygame) | Rendering the board, menus, dialogs, text output; translating clicks into position edits or game moves; driving a game (`GameController`) |
| Game model | `backgammon/game/position.py`, `utils.py`, `settings.py`, `game.py` | Python | Board state, rules and move generation, settings, playing a game |
| Evaluator | `backgammon/game/evaluator.py`, `src/evaluator.rs` | Python + Rust | Static equity of a position, used where the search stops. In Rust a pluggable `Evaluator` trait: placeholder (default), pip count, or neural net |
| Neural net | `src/neural_net.rs`, `src/encoding.rs` | Rust | 54-128-64-1 net estimating the winning chance of the side to move; weight files in `nets/`, loaded from `Settings.rust_network` |
| Training | `src/td_training.rs`, `src/bin/train.rs` | Rust | TD(λ) self-play training of the net (`make train`) |
| Strength benchmark | `benchmarks/strength.py` | Python | Paired games of our engine against bgsage (`make strength`), results in `benchmarks/strength_results.csv` |
| Analysis worker | `backgammon/game/analysis_worker.py` | Python | Runs analysis and engine moves on one dedicated background thread so the UI stays responsive |
| Analyzer proxy | `backgammon/game/analyzer_proxy.py` | Python | Synchronous facade over the engines; dispatches to the Python, Rust or bgsage engine |
| bgsage engine | `backgammon/game/bgsage_analyzer.py`, `bgsage_server.py` | Python | Client for the external bgsage engine, which runs as a subprocess in `.venv-bgsage` (it needs Python ≥ 3.10) and is spoken to with JSON lines over a pipe |
| Python engine | `backgammon/game/python_analyzer.py` | Python | Reference implementation of the search |
| Rust engine | `src/` | Rust | Fast implementation of the same search, exposed to Python as module `backgammon_rust` |
| Logger | `backgammon/logger.py` | Python | Routes status messages to the on-screen text area (or stdout) |
| Benchmarks | `src/bin/benchmark.rs`, `src/bin/position_benchmark.rs` | Rust | Standalone binaries timing the Rust engine, and comparing the two position types |
| Tests | `tests/` (Python), `#[cfg(test)]` modules in `src/` (Rust) | Both | Unit tests per side |

### How the pieces are linked

- **Python ↔ Rust** — The Rust crate (`Cargo.toml`, crate name `backgammon_rust`, type
  `cdylib` + `rlib`) uses [PyO3](https://pyo3.rs) to export `equity`, `best_move`, `play_game`,
  `clear_cache`, `set_cache_size` and `cache_size_mb`. [maturin](https://www.maturin.rs)
  (`pyproject.toml`) builds the crate with the `extension-module` feature and installs it into
  the virtualenv (`make build` → `maturin develop --release`; without `--release` the engine is
  an unoptimized debug build, about 8× slower). Python imports it with
  `import backgammon_rust`.
- **Data crossing the boundary** is deliberately primitive: two `list[int]` of length 26 (checker
  counts for ME and OPPONENT), the side to move as `0/1`, the cube value and cube owner
  (`None` = centered, otherwise `0/1`), the player of interest as `0/1`, dice, and the search
  depth. Rust returns a `float`, or for `best_move` a tuple `(me_checkers, opp_checkers, turn, equity)`
  that the proxy turns back into a Python `Position` (copying the cube from the input, since a
  move doesn't change it). The Python `Position` object itself never crosses the boundary.
- **Threading across the boundary** — The Rust functions release the GIL (`py.allow_threads`)
  while searching, so the pygame main thread keeps running during a Rust analysis. All engine
  calls are made from the single `AnalysisWorker` thread (see below).
- **Rust without Python** — Because all PyO3 code is behind `#[cfg(feature = "extension-module")]`
  and the crate also builds as an `rlib`, the engine can be used from plain Rust: `cargo test`
  and `cargo run --bin benchmark --release` do not need Python.
- **UI ↔ model** — `Board` owns the current `Position` and, when no game is running, mutates it
  directly in response to clicks. During a game the `GameController` owns a `Game`, and the
  board shows the controller's `display_position()`. There is no event bus or observer;
  the main loop simply redraws the whole board from the position every frame.
- **Build/run orchestration** — `Makefile`: `make build`, `make run`, `make test`
  (Rust tests → rebuild extension → Python tests). Rust changes are only visible to Python after
  a rebuild.

### Shared conventions

- **Point indexing** (both languages): each player has an array of 26 slots indexed from their
  *own* perspective. `0` = borne off, `1–24` = board points, `25` = bar. Both players move from
  high to low. Point `p` for one player is point `25 - p` for the other.
- **Players**: `Player.ME` (=1, drawn white, bottom-right home) and `Player.OPPONENT` (=0, drawn
  red). Checker arrays are named after the player, not a colour: `me_checkers` /
  `opponent_checkers` in Python (and in Rust's `SimplePosition`).
- **Equity** is a float in `[-1, 1]` (gammons are not scored yet): +1 is a certain win.
  Internally the engines work with equity *for the side to move* and negate it for the other
  player. While gammons are not scored, winning chance = (equity + 1) / 2, which the Analyze menu
  shows next to the equity.
- **Moves** are tuples of half-moves `(start, target)` in the mover's numbering, e.g.
  `((8, 5), (6, 5))`, written as `8/5 6/5` (`bar` = 25, `off` = 0). The empty tuple means
  "no legal move".

---

## Component: Entry point (`main.py`)

A classic single-threaded pygame loop:

1. Construct `Board`, which initialises pygame and the window and holds a `Position` set to the
   starting position.
2. Each iteration, poll events and dispatch them:
   - `QUIT` / `Esc` → exit.
   - `WINDOWRESIZED` → `Board.handle_resize`.
   - Other keys → `Board.handle_key_press` (F10 / Alt+M toggles the menu, Ctrl+A analyzes).
   - Mouse wheel / button-up / motion → `TextArea.handle_event` (scrolling, scrollbar drag).
   - Mouse button-down → a *chain of responsibility*: text area → menu → then either
     `Board.handle_game_click` (during a game) or, when editing, cube → turn indicator → checker
     placement. Each handler returns `True` if it consumed the click. When editing, left click
     edits ME's checkers and right click (or Ctrl+click, for the Mac Magic Mouse) edits OPPONENT's.
3. `board.update()` (advance the game, show queued log messages and any finished analysis),
   then `board.draw()` and `clock.tick(60)`.

Searches never run inside this loop: analyses and engine moves are handed off to the
`AnalysisWorker` thread and their results are picked up by `board.update()` on a later frame, so
the window stays responsive.

## Component: UI (`backgammon/ui/`, Python + pygame)

### `BoardMap` — layout and geometry
A pure geometry/layout object created from the window size. All pixel dimensions (margins,
point width, checker radius, font sizes, menu metrics, cube and arrow locations, text-area
rect) are derived from a 1024×768 reference design scaled by `sx`, `sy` and `s = min(sx, sy)`.
On resize a new `BoardMap` is simply constructed.

The board keeps fixed proportions (14 point widths wide, 20 checker radii tall). Its
`point_width` is the largest value that fits **both** the available width (between the side
margins) and the available height (between the top margin and the point labels above the text
area). When height is the limit, the board is centered horizontally. Everything on the board is
positioned from its origin `(board_x, board_y)`. The cube and turn arrow are positioned
relative to the board's left and right edges, and `board_margin` is only the spacing around it.

It also owns:
- `get_point_coordinates(point, player)` — maps a logical point to screen coordinates, taking
  the player's perspective into account. `point_at(x, y, player)` is the reverse, used for
  entering moves: it returns the point under the mouse in that player's numbering (25 for the
  bar, 0 for the bear-off column). `dice_center(player)` places the dice on the mover's half.
- Precomputed hit-test tables `me_possible_positions` / `opponent_possible_positions`:
  `(point, checker_num, (x, y))` for every visible checker slot, used to map a click to
  "set this point to N checkers".
- Position-editing helpers invoked from clicks: `adjust_checkers`, `increment_checkers`,
  `decrement_checkers`, `handle_cube_click`. Cube edits go through `Position.set_cube`, and the cube is
  capped at 64 (`MAX_CUBE_VALUE`). Checkers are conserved: adding to a point takes
  from the player's borne-off pile, removing returns them there, and placing on a point occupied
  by the other player sends that player's checkers off.

### `Board` — rendering and interaction controller
Owns the pygame screen, the `BoardMap`, the `TextArea`, the current `Position`, the
`AnalysisWorker`, the `GameController`, and menu state.
- `draw()` repaints everything every frame: points, bar, bear-off strip, checkers
  (max 5 drawn per point, with a count label beyond that), point labels, doubling cube, turn
  arrow, the dice (during a displayed game), text area, and finally the menu bar and any open
  submenu (drawn last so it overlays). During a game the checker the human has selected is
  highlighted.
- Menus are defined as data (`self.menus`): `Edit` (Clear Board, Starting Position), `Play`
  (Start…, Stop) and `Analyze` (Show equity, Settings). `is_menu_item_enabled()` decides which
  items can be used: Start… only without a game, Stop only during one, and Edit is disabled during
  a game. Disabled items are drawn grey and ignore clicks. Menus are drawn and hit-tested
  manually; no widget toolkit is used.
- `analyze()` submits `equity(position, Player.ME)` to the worker and stores the returned
  future; a second request while one is running is refused with a message. `update()`, called
  every frame, advances the `GameController`, takes over its display position, flushes `Logger`
  and, once an analysis is done, prints the equity and winning chance (or the error).
- Registers its `TextArea` with `Logger` so that log output from anywhere ends up on screen.

### `TextArea` — scrollable output panel
A self-contained widget at the bottom of the window: keeps a list of lines, shows the last
three by default, supports mouse-wheel scrolling, clicking the track to page, and dragging the
scrollbar thumb. Exposes `show_text`, `clear`, `handle_event` and `draw`.

### Dialogs (`dialog.py`, `SettingsDialog`, `PlaySetupDialog`)
`Dialog` is the shared base: a *nested event loop* that draws a dimmed snapshot of the board plus
the dialog panel each frame and runs until OK/Enter (returns `result()`) or Cancel/Esc/close
(returns `None`). It provides drawing helpers for text, radio buttons, checkboxes, buttons and
−/+ steppers, each with a greyed-out disabled state.
- `SettingsDialog` — engine (Python or Rust) and search depth for the Analyze menu, and the Rust
  engine's cache size (−/+ halve and double it, 64 MB–16 GB), written to `Settings` on OK. A
  changed cache size is applied on the worker thread, after any running search; the cache then
  starts empty.
- `PlaySetupDialog` — for each player, Human / Python engine / Rust engine, plus a search
  depth (0–4, enabled for engines). Also a **Show game** checkbox, enabled only when both players
  are engines. Returns a `GameConfig` and remembers the last choice.

### `GameController` — driving a game without blocking
Created by `Board`. `update()` runs every frame:
- **Engine to move:** after `ENGINE_MOVE_DELAY` (0.6s, so a displayed game can be followed) it
  submits `Game.engine_move` to the worker. When the future is done it plays the move, which
  logs it.
- **Human to move:** it creates a `MoveBuilder` for the legal moves. `handle_click(point)`
  selects a checker and then its destination (several dice in one go if needed). Right click
  cancels the selection or undoes a step, and Backspace undoes. When all dice are used the move
  is played. Without a legal move the player passes automatically after `PASS_DELAY`.
  `display_position()` shows the half-entered move on the board.
- **Hidden engine game** (both engines, Show game off): runs as a single worker task. The
  board keeps showing the starting position, and only the result and number of turns are logged;
  the final position is then shown. There are two variants:
  - **Rust engine vs Rust engine:** the whole game, opening roll included, is played inside Rust
    (`Game.play_in_rust` → `backgammon_rust.play_game`) with a seed taken from Python's random
    generator. This can't be interrupted, so `can_stop` is false and **Stop** is disabled until it
    finishes.
  - **Any other combination:** `Game.play_to_end` in Python, which checks between moves whether
    it was stopped.
- **Stop / end:** each game (except one played inside Rust) can be stopped. Each game has a generation number. Results that arrive after Stop (a running
  search can't be interrupted) belong to an old generation and are thrown away, and a hidden game
  checks the generation between moves to stop early. When a game ends, the winner is logged and
  the final position stays on the board for editing.

## Component: Game model (`backgammon/game/`, Python)

### `Position` and `Player`
- `Player` enum with `other_player()`.
- `Position` holds two 26-element lists, the side to move, and the doubling cube
  (`cube_value`, `cube_owner`). Mutators validate input and invalidate a lazily computed
  `hash_code()` used as the analyzer cache key.
- Helpers: `setup_starting_position`, `clear` (all checkers borne off), `switch_turn`,
  `set_cube`, `double_cube`. All state changes should go through these methods so the cached
  hash is invalidated.

### `Utils` — rules and move generation
Static functions: `pipcount`, `can_bear_off`, `can_move`, `apply_half_move` (moving onto a
single opponent checker hits it, sending it to the opponent's bar), `apply_move`
(applies a sequence of half-moves and switches turn), `possible_moves` (recursive search over
each distinct dice ordering: one for doubles, two otherwise), and `valid_possible_moves` which
enforces the backgammon rules that you must use as many dice as possible and, if only one die can
be used, the higher one (a move "uses the higher die" if it is legal with that die alone in the
position). It returns move *sequences*, which the UI needs (human input, move logging); many
sequences lead to the same position, so the analyzer skips children it has already evaluated.

### `Settings`
Global, class-level configuration: `analyzer` (`Analyzer.PYTHON` / `Analyzer.RUST`, default Rust)
and `analysis_depth` (default 3) for the Analyze menu, and `rust_cache_mb` (default 1024), the
memory for the Rust engine's cache. `Board` applies `rust_cache_mb` at startup through the
`AnalysisWorker` (the cache belongs to the thread that runs the engine). Game players have their own
engine and depth in `GameConfig`.

### `Game`, `GameConfig`, `MoveBuilder` (`game.py`)
UI-independent game logic (no doubling cube, no gammons):
- `PlayerType` (`HUMAN`, `PYTHON_ENGINE`, `RUST_ENGINE`), `PlayerConfig(type, depth)`,
  `GameConfig(me, opponent, show_engine_game)`.
- `Game.start()` — the opening roll: each player rolls one die, ties are rerolled, and the higher
  die moves first, playing both dice.
- `Game.play(move)` checks the move against `legal_moves()`, applies it, logs it, detects the
  winner (all 15 checkers off) and otherwise rolls for the next player. `engine_move()` asks the
  configured engine for its choice (slow, so it runs on the worker). `find_move()` turns the
  engine's resulting position back into a move. `play_to_end()` plays out an engine-only game.
  `play_in_rust(seed)` hands a hidden Rust-vs-Rust game (`runs_in_rust`) to Rust in one call.
- Dice come from an injectable `random.Random`, and messages go to an injectable `log`
  callback, so games are fully testable.
- `MoveBuilder` collects a human's half-moves and only allows steps that can still be completed
  to a legal move. `try_move(start, target)` finds the shortest chain of dice for one checker
  from start to target.

### `AnalyzerProxy`
The single entry point to the engines: `equity(position, player)` and
`best_move(position, die1, die2)`. Each call takes an optional `engine` and `depth` (used by game
players) and falls back to `Settings` (used by the Analyze menu). It either calls
`PythonAnalyzer` directly or marshals the `Position` into lists/ints for `backgammon_rust`.
`clear_caches()` clears both engines. Checker layout, turn and cube state are all passed to
Rust. The proxy is synchronous and thread-agnostic; the UI goes through `AnalysisWorker`, while
tests call the proxy directly.

### `AnalysisWorker`
Owns one long-lived daemon thread that takes tasks from a queue and runs them in order, and
returns a `concurrent.futures.Future` for each task:
- `equity(position, player)` deep-copies the position first, so the user can keep
  editing the board while the search runs.
- `submit(fn)` runs any function, which the `GameController` uses for engine moves and hidden
  games.
- `clear_caches()` is queued behind any running task.

There is one thread, and only one, because the Rust analyzer and its cache are
**thread-local**. Making every engine call from the same thread keeps the Rust cache alive between
analyses and makes `clear_cache` act on the cache that is actually in use. The thread is a
daemon, so closing the window never waits for a running search.

### `Evaluator`
`evaluate(position)` gives the equity for the side to move without searching: +1 if the side to
move has borne off all checkers, −1 if the other side has, and 0 (no opinion) otherwise. This is
the place where a real evaluation function will go. `is_game_over()` separates finished
positions from the 0 "unknown" value. The Rust `Evaluator` (`src/evaluator.rs`) is identical.

### `PythonAnalyzer` — reference engine
An expectimax search over dice rolls, working in equity for the side to move (`depth` = number
of roll levels still to search):

```
equity(pos, depth):
    if depth == 0 or game is over      → Evaluator.evaluate(pos)
    if (pos, depth) in cache           → cached value
    for each of the 21 distinct rolls (non-doubles weighted ×2, total /36):
        best_move(pos, roll, depth-1)  → max over legal moves of −equity(child, depth-1)
    cache[(pos, depth)] = average
```

If no legal move exists, the turn passes. `best_move(pos, roll, depth)` at the top level
evaluates each child with `depth` roll levels. Depth 0 therefore picks moves using the evaluator
alone, and each extra level multiplies the work by about 21 rolls × ~20 distinct moves. The cache key
includes the remaining depth, so values from different depths (e.g. two engine players with
different settings) never mix. The cache is class-level and persists for the whole session.

## Component: Rust engine (`src/`, Rust)

The Rust engine mirrors the Python model and analyzer one-to-one, but is generic over a
`Position` trait so that different board representations can be swapped in.

| File | Contents |
|---|---|
| `position.rs` | `Player` enum and the `Position` trait (get/set checkers, turn, cube, setup, and `cache_key`) |
| `simple_position.rs` | `SimplePosition`: straightforward `[u8; 26]` arrays per player (60 bytes). **Used by the engine**, because it is the fastest to search with |
| `cache_friendly_position.rs` | `CacheFriendlyPosition`: bit-packed representation (24 bytes, 24-byte cache key), for when memory matters more than speed |
| `utils.rs` | `Utils`: rules and move generation, generic over `P: Position + Clone`. `valid_possible_moves` (move sequences, port of the Python version) and `legal_positions` (distinct resulting positions, used by the analyzer) |
| `move_generator.rs` | `MoveGenerator`: position-based move generation, reusing its hash tables between calls. `legal_positions` (each distinct resulting position once) and `best_position` (scores positions as they are found and keeps the best, for the last search level) |
| `fast_hash.rs` | `FxHasher`, a fast non-cryptographic hasher, and `FastHashMap`/`FastHashSet` for the engine's tables |
| `evaluator.rs` | `Evaluator`: static equity (±1 for finished games, 0 otherwise), identical to the Python one |
| `rust_analyzer.rs` | `RustAnalyzer<P>`: the expectimax search, generic over the position type, with a fixed-size cache (`TranspositionTable`, `DEFAULT_CACHE_MB` = 1 GB, `with_cache_mb` to change) keyed by (position, remaining depth) |
| `transposition_table.rs` | `TranspositionTable`: fixed-size cache with two-slot buckets (depth-preferred + always-replace), so memory stays bounded however long a game runs |
| `game.rs` | `play_game`: a complete Rust-engine-vs-Rust-engine game with the same rules as the Python `Game` (opening roll, alternating turns, pass, win), seeded `Dice` (SplitMix64), and a `MAX_TURNS` guard |
| `lib.rs` | Module wiring, `EnginePosition` (the position type the engine uses, with the guidance on choosing), `build_position` (lists → `EnginePosition`, rejecting impossible positions), and the PyO3 bindings |
| `bin/benchmark.rs` | Times `equity` on a fixed position, clearing the cache before each run |
| `bin/position_benchmark.rs` | Runs identical searches with `SimplePosition` and `CacheFriendlyPosition` and compares time, position size and cache-key size |

### `CacheFriendlyPosition` layout and the cache key
Designed to be small, both to make the copy per candidate move cheap and to keep cache keys
compact:
- `low: u64`, `high: u64` — 24 physical points × 5 bits (4 bits count, 1 bit owner), points 1–12
  in `low` and 13–24 in `high`, so that reading a point is one 64-bit shift. ME's logical point
  `p` maps to physical `p`, OPPONENT's to `25 - p`, so a point is owned by at most one player.
- `bearoff: u8`, `bar: u8` — one nibble per player, since both can have checkers there at once.
- `state: u8` — turn bit, 2-bit cube owner, 3-bit cube exponent.

`set_checkers` refuses to put checkers on a point the other player occupies (a hit must first
move the blot to the bar) and counts above 15, instead of silently corrupting the position;
`build_position` passes that on to Python as a `ValueError`. The whole position is 24 bytes. Its `cache_key(depth)` is the packed fields plus the depth, also
24 bytes. Because the key *is* the position, different positions can never collide in the cache
(the previous key was a 64-bit hash). `SimplePosition` uses the same scheme but its key is 60
bytes, so `CacheFriendlyPosition` makes each cache entry about 2.5× smaller.

### Search performance
What made the Rust search fast, in order of impact (all measured with `position_benchmark` and
macOS `sample`):
1. **Position-based move generation** (`MoveGenerator`, also reachable as
   `Utils::legal_positions`) generates the distinct positions a roll can lead to, instead of
   move sequences. On average 4–5 sequences lead to the same position, and the old generator also
   repeated the full search for each of the 24 identical orderings of a double's dice. The new one
   explores each (intermediate position, dice left) combination once, records each resulting
   position once, and allocates nothing per sequence. Together that made searches 7–60× faster. A
   test compares it with `valid_possible_moves` on more than 10,000 positions and rolls from
   seeded games.
2. **Release build** of the Python extension (`maturin develop --release`): about 8×.
3. **Inlinable checker accessors**: their error path lives in a separate `#[cold]` function
   (`invalid_point_error`), which made all searches about 30% faster.
4. **`FxHasher`** instead of Rust's default SipHash for the cache and the duplicate checks
   (SipHash took about 30% of the time): 12–40% faster.
5. **Scoring the last level while generating it.** At depth 3, 99.8% of move generations happen
   one roll level above the leaves, producing on the order of 100 million leaf positions that are
   only passed to the Evaluator. `MoveGenerator::best_position` scores each one as it is found and
   keeps the best, with no duplicate check (harmless for a maximum) and no list of children. The
   generator is also kept in the analyzer, so its hash tables are reused instead of rebuilt for
   every roll, and duplicate tracking is skipped for non-doubles, where it can't find anything.
   Together with inlining `can_move`/`apply_half_move`: depth 3 went from 35.6 s to 18.6 s
   (`CacheFriendlyPosition`) and from 41.8 s to 12.0 s (`SimplePosition`) on a middle-game position. A test
   checks over more than 10,000 positions and rolls that both modes choose the same move, ties
   included.

6. **Changing the position in place, with bitmasks.** The move generator copies the position
   once per roll and applies each half-move in place (`make`), undoing it after the recursion
   (`unmake`), instead of copying the position for every half-move (980 million copies per
   depth-3 search). A bitmask of the mover's occupied points replaces the scan of all 25 points
   and makes "on the bar", "highest point" and "can bear off" single instructions; a bitmask of
   blocked points, computed once per roll (it can't change during a move), replaces `can_move`.
   Depth 3 went from 12.2 s to 8.8 s; the generated moves, and so the games played, are unchanged.

7. **Generating each move once.** Moves used to be generated in every order and each resulting
   position reached about 4 times; at the last search level, where duplicates aren't checked
   (hashing a leaf costs more than scoring it), that meant 4× the work. Now:
   - doubles are played in descending order of start point only, which loses no moves (any legal
     double move can be played highest checker first) and replaces the visited set;
   - for two different dice, the second order skips two-move sequences the first order is sure to
     generate as well (two different checkers, or one checker whose other route is open and
     doesn't hit), keeping anything involving the bar or bearing off.
   A move must still only count as finished when *no* half-move is legal, not merely no
   descending one. Checked against the sequence-based generator on about a million
   position/roll combinations from 500 seeded games. Depth 3 went from 8.9 s to 3.8 s (a depth-3
   opening move from 19.7 s to 8.4 s); a full depth-2 game from 1.0 s to 0.5 s. (Reading checker
   counts through `count`/`set_count`, without range checks or a `Result`, made no measurable
   difference: the compiler had already optimised the checked accessors.)

Earlier analysis (before step 6): almost all of it is checker reads and writes in move
generation (`explore`, `can_move`, `apply_half_move`); hash tables are about 4%. That is exactly
where packed bits cost more than an array, so for deep searches `SimplePosition` is about
1.5× faster than `CacheFriendlyPosition`, while `CacheFriendlyPosition` has 2.5× smaller cache entries. The engine
therefore uses `SimplePosition`, through the `EnginePosition` alias in `lib.rs`, which explains when
to choose which; switching is that one line. Copying a position doesn't show up in profiles.

### The cache: a transposition table
The analyzer's cache is a `TranspositionTable`, the standard cache of game engines, instead of a
growing hash map. It is an array of buckets allocated once (256 MB by default), indexed by the
key's hash, with two slots per bucket. Its size comes from `Settings.rust_cache_mb` (1 GB by
default, changeable in the Settings dialog), passed to Rust with `set_cache_size`. Any bucket
count is allowed (the hash is scaled onto the bucket count with a multiply and shift), so the
whole budget is used:
- slot 0 is depth-preferred: only an entry searched at least as deep may take it, since deep
  results are the most expensive to recompute; the entry it held moves to slot 1;
- slot 1 always takes the newest entry.

Each slot stores the full key, so a lookup can miss but never return another position's value.
Why: a depth-3 move adds 1–2 million entries and few are reused between moves, so the old hash
map reached 1.2 GB after three moves of a depth-3 game and would have grown to about 10 GB over a
whole game (plus the hash map's overhead: it fills to 7/8, doubles when full, and briefly holds
the old and new table during a resize). Measured on three depth-3 moves, with an earlier version that
could only use power-of-two bucket counts (and so about 60% of the requested memory): a 150 MB
table was 4% slower than the unbounded map, a 600 MB table as fast as the map, and a 38 MB table
17% slower. The 1 GB default is therefore comfortably large enough for depth 3. Because deep entries are kept preferentially, the search never has to redo a deep
subtree when the table is full; only cheap last-level positions are evicted.
`RustAnalyzer::cache_stats` reports lookups and hits for measuring this.

### Analyzer and Python bindings
`RustAnalyzer<P>` implements the same algorithm as `PythonAnalyzer` (same 21 rolls, same
Evaluator at the cutoff, same equity-for-side-to-move convention, and a cache per position and
depth). It is generic over the position type, so the search is monomorphised for
`EnginePosition` (currently `SimplePosition`).
It searches the distinct child positions from `Utils::legal_positions`, which come in a
deterministic order, so ties between equally good moves are always broken the same way and games
are reproducible from a seed.

Driving a game from Rust saves the Python overhead per move (Python's own move generation and
bookkeeping, several milliseconds per move). That overhead dominates when the search is cheap:
a Rust-driven game takes about 0.004 s at depth 1 (0.08 ms per move) and about 1.2 s at depth 2
(24 ms per move).

In `lib.rs` a `thread_local!` `RefCell<RustAnalyzer<EnginePosition>>` holds one analyzer per thread for the
life of the process, so its (fixed-size) cache is reused across calls from Python just like the Python
engine's. (`thread_local` is used because `RefCell` is not `Send`.) The exported functions
convert their arguments with `build_position` (checkers, turn and cube), move the position
into a `py.allow_threads` closure so the search runs without holding the GIL, and convert the
result back to plain lists.

## Cross-cutting: `Logger`
A tiny static logger. `Board` registers the current `TextArea` (again after every resize).
`Logger.log` writes to stdout when no UI exists (e.g. in tests). On the main thread it writes
straight to the text area. On any other thread, such as the analysis worker, it queues the
message, and `Logger.flush()` (called from `Board.update()`) moves the queue into the text area,
so pygame state is only touched from the main thread. Displayed games log their moves and
result through it.

## Testing
- **Python** (`tests/`, `unittest`):
  - `test_utils.py` — move generation and hitting.
  - `test_analyzer.py` — analyzer behaviour through `AnalyzerProxy`, including depth semantics,
    cube marshalling, explicit engine/depth, and Python–Rust agreement at depths 0–3.
  - `test_analysis_worker.py` — the background worker, thread-safe logging and `Position.set_cube`.
  - `test_game.py` — opening roll, legal/illegal moves, passing, winning, full engine games, and
    `MoveBuilder`, all with scripted dice.
  - `test_game_controller.py` — displayed and hidden engine games, delays (fake clock), Stop
    discarding a late result, human click input and undo, and automatic passing.
  - `test_board_map.py` — layout at different window sizes.
- **Rust** (inline `#[cfg(test)]` modules): position representations, move generation and the
  analyzer, runnable with `cargo test` independently of Python.
- `make test` runs Rust tests, rebuilds the extension, then runs the Python tests.

## Planned: neural net evaluator
Replacing the placeholder `Evaluator` with a neural net, together with caching leaf evaluations
and pruning the search, is planned and tracked in [neural_net_evaluator.md](neural_net_evaluator.md).
