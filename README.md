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
maturin develop
```

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
make build        # Rebuild the Python extension (maturin develop)
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

To adjust the scenario, edit `src/bin/benchmark.rs` — the `max_depth` and `runs` constants control search depth and the number of iterations.

## Controls

- **Left click** — place a checker for ME
- **Right click** — place a checker for OPPONENT
- **Escape** — quit
