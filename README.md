# Backgammon

A backgammon board visualizer with analysis features, built with Python and pygame.

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The Rust analyzer must be compiled and installed into the venv using [maturin](https://github.com/PyO3/maturin):

```bash
pip install maturin
maturin develop --release
```

Always use `--release` (as `make build` does): without it maturin builds the Rust code
unoptimized, which makes the engine about 8× slower.

### PyCharm

Configure PyCharm to use the venv interpreter so the Rust extension is available:

1. Open **Settings → Project → Python Interpreter**
2. Click the gear icon → **Add → Existing environment**
3. Select `.venv/bin/python3` in the project directory

## Running

**From PyCharm** (no Rust changes): run `main.py` directly. Make sure the project interpreter is set to `.venv/bin/python3` (see Setup above).

**From the terminal** (no Rust changes):
```bash
.venv/bin/python main.py
```

**After changing Rust source files**, rebuild the extension first:
```bash
make run
```
This is equivalent to `make build && .venv/bin/python main.py`.

## Running tests

Use `make test` to run the full test suite. This runs Rust tests, rebuilds the Python extension, then runs Python tests — in that order:

```bash
make test
```

You can also run each step individually:

```bash
make test-rust    # Rust unit tests only (cargo test)
make build        # Rebuild the Python extension (maturin develop --release)
make test-python  # Python tests only
```

> **Important:** always run `make build` (or `make test`) after changing any Rust source file, otherwise the installed Python extension will be stale and changes won't take effect.

## Benchmarking the Rust analyzer

A benchmark binary measures the performance of the Rust analyzer in isolation. It sets up a position with 2 checkers each on the 6-point for both players, clears the cache before every run, and prints the winning-chance calculation time for each run followed by the average.

Run it with:

```bash
cargo run --bin benchmark --release
```

Always use `--release`; the debug build is significantly slower and not representative of production performance.

To compare the two Rust position representations (`SimplePosition` and `CacheFriendlyPosition`) on identical
searches, including their memory use per cache entry:

```bash
cargo run --bin position_benchmark --release        # optional argument: cache size in MB (default 1024)
```

To adjust the scenario, edit `src/bin/benchmark.rs` — the `max_depth` and `runs` constants control search depth and the number of iterations.

## Controls

Editing a position (when no game is running):

- **Left click** — place a checker for ME
- **Right click** — place a checker for OPPONENT
- **Ctrl+A** — show the equity of the position for ME
- **Escape** — quit

Playing a game (**Play → Start…**, choose a human or an engine with its search depth for each
player; **Play → Stop** ends the game):

- **Left click** on one of your checkers, then on its destination point (or the bear-off tray on
  the right). A checker can use both dice in one go. The move is played as soon as all dice are used.
- **Right click** — cancel the selected checker, or undo the last step
- **Backspace** — undo the last step

**Analyze → Settings** also sets the memory for the Rust engine's cache (1 GB by default). The
cache never grows beyond that; less memory means more recomputation in deep searches.

A Rust engine vs Rust engine game with **Show game** unchecked is played entirely inside Rust for
speed. It can't be stopped once started; the log shows the winner when it's done.

Engine search depth: depth 0 uses only the evaluator, and each extra level multiplies the work
by roughly 400 on a full board. For the Rust engine, depth 1 takes well under a millisecond per
move and depth 2 about 25 ms (a whole Rust-vs-Rust game takes about 1 s at depth 2). Depth 3 takes
roughly 30 s to several minutes per move (doubles are the slowest), so a game takes about an
hour; depth 4 is impractical. The Python engine is several
thousand times slower and only practical at depth 0 or in the endgame.
