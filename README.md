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

### bgsage engine (optional)

[bgsage](https://github.com/markbgsage/bgsage) is a strong neural-network engine, available as
"bgsage" in the Settings and Start game dialogs. It needs Python 3.10 or later, so it is installed
in its own virtualenv `.venv-bgsage`, and the app runs it as a subprocess
(`backgammon/game/bgsage_server.py`). Install it from a local checkout with:

```bash
make bgsage                       # BGSAGE_SRC=~/workspace/bgsage, BGSAGE_PYTHON=python3.14 by default
```

bgsage always plays a 1-point match (no cube, no gammons). Depth 0 is bgsage's 1-ply, the raw
network evaluation; depth d is (d+1)-ply, up to depth 3 (4-ply, roughly a second per move).

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

To measure a net's checker play without playing games, score it against bgsage's reference
decisions (about a second per net; needs the bgsage checkout for the first extraction):

```bash
make decisions NETS="nets/td-eval.bgnet nets/arch/512-256-28M-avg.bgnet"   # SET=pasko for Paskogammon
```

It reports per game plan (attacking, priming, anchoring, racing) how often the net picks the move
with the highest rolled-out winning chance and how much winning chance it loses per decision.
Pure races are left out: those are the bearoff database's and the search's domain. On 85 trained
nets this ranks them like the 40k-game benchmark against bgsage. With `SCORING=money` it scores
cubeless money play instead: the reference best is the move with the highest cubeless money equity
(gammons count), and the loss is in millipoints of equity per decision.

To adjust the scenario, edit `src/bin/benchmark.rs` — the `max_depth` and `runs` constants control search depth and the number of iterations.

## Training a net

Nets are trained with TD(λ) self-play (`make train`, see `src/bin/train.rs`). The standard
procedure is `benchmarks/train_schedule.py`:

```bash
make train-schedule NAME=256-256 DIR=nets/money ARGS="--objective money --hidden 256,256"
```

It trains in 4M-game segments at learning rate 0.01, scores every segment on the decision
benchmark, and when the net stops improving there (2 segments in a row), continues from the best
net so far with half the learning rate: 0.005, then 0.0025. The 40k-game benchmark against bgsage
is only played every 16M games and for the final best net and its f16 copy. Lowering the learning
rate on plateaus gained the 576-192 net about 0.3 points of win rate and the 512-256 net (now the default) about 0.5.

### Mixture of experts (experimental)

`src/moe.rs`: one expert net per game type (11 types such as blitz, holding game, back game, found
in the hidden activations of the default net; `nets/moe/types.txt`), a generic expert for all other
positions, and a gateway net that rates how well each expert handles a position. The evaluator
skips experts rated below 0.25 and averages the rest, weighted by their normalized ratings. The
experts are cubeless money nets (256-128; DMP experts work too). Train it with
`.venv/bin/python -m benchmarks.moe_schedule --dir nets/moe` (a 256-128 money base net with the
standard schedule, collect positions per type, train each expert on its own type starting from the
base net, with positions of other types judged by the best 512-256 money net, then the gateway;
progress against bgsage 3-ply money on `data/decisions/moe-types-money.txt`); the result `nets/moe/experts/moe.bgmoe` can
be used anywhere a net file can (`decision_benchmark`, `strength --net`, the settings).
`decision_benchmark --types nets/moe/types.txt` adds scores per game type. The activation analysis
behind the types: `analysis/activations.py` (needs `.venv-analysis` with numpy, scipy and
scikit-learn; the dump comes from `activation_dump`).

## Controls

Editing a position (when no game is running):

- **Left click** — place a checker for ME
- **Right click** — place a checker for OPPONENT
- **Ctrl+A** — analyze the position (equity and winning, gammon and backgammon chances of the
  side on roll) in the analysis panel on the left
- **Escape** — quit

Playing a game (**Play → Start…**, choose a human or an engine with its search depth for each
player, the game: **Backgammon** or **Nackgammon**, and **Double match point**; **Play → Stop**
ends the game). With double match point (the default) every win counts 1 point. Without it, a
gammon (the loser has borne off no checkers) wins 2 points and a backgammon (a gammon where the
loser still has a checker on the bar or in the winner's home board) 3; the log shows the result.
Without double match point the Rust engine plays for cubeless money: finished games count 1, 2 or 3
points and the bearoff database is only used once no gammon is possible. Money games use their
own net (`rust_money_network` and `rust_nackgammon_money_network`): by default
`nets/td-money-eval.bgnet`, the f16 copy of `nets/td-money.bgnet`, a 512-256 money net (5 outputs:
win, win gammon, win backgammon, lose gammon, lose backgammon; `train --objective money`) that
plays for gammons too. A double-match-point net only knows winning chances: it plays to win the
game and doesn't save gammons once it can't win. bgsage plays for cubeless money as well; the Python engine always plays to win.
Nackgammon has the same rules with a different starting position: each player moves one checker
from the 13 point and one from the 6 point to the 23 point (2 on the 24, 2 on the 23, 4 on the 13,
3 on the 8, 4 on the 6). The Rust engine can use a separate net for each game type
(`rust_network` and `rust_nackgammon_network` in `backgammon/game/settings.py`); by default both
are `nets/td-eval.bgnet`: the half-precision (f16) copy of `nets/td.bgnet`, a 512-256 net trained
on both variants (52M games, the last 16M with a lower learning rate), which evaluates about 2.4× faster
at the same strength (regenerate it with `make eval-net` after changing `nets/td.bgnet`). The
previous default, a 576-192 net, is `nets/td-576-192.bgnet` (f16 copy `nets/td-576-192-eval.bgnet`); an older, about 2.7× faster but weaker 256-256 net is `nets/td-256-256-44M.bgnet`. Once both sides are bearing off, the
engine uses exact winning chances from a two-sided bearoff database when it exists: build it once
with `make bearoff` (about 2 minutes, writes `data/bearoff-6x15.bd`, 5.9 GB).

To play:

- The dice are shown high left, low right. **Left click** on one of your checkers moves it by the
  left die (bearing off when it goes past your 1-point), or by the right die when the left one
  can't be played with that checker. After one die is played, the next click uses the other one.
  With doubles every click moves a checker by the die, four times. Played dice are greyed out.
- **Right click**, or a click on the dice — swap the dice
- When all dice are played: **click the dice** to finish the move, or **right click** to take the
  whole move back
- **Backspace** — undo the last step
- **Ctrl+A** — list all moves with the dice in the analysis panel, best first, like eXtreme
  Gammon's move list: per move its equity (the best move's own, the others' difference to it) and
  the probabilities of the player (P: win, gammon, backgammon) and the opponent (O:). Click a move
  to see it on the board, click it again to play it. The search depth is the Analyze → Settings
  depth (2 by default, XG's 3-ply). A double match point net knows only winning chances, so its
  gammon columns show "-".

**Analyze → Settings** also sets the memory for the Rust engine's cache (1 GB by default). The
cache never grows beyond that; less memory means more recomputation in deep searches.

A Rust engine vs Rust engine game with **Show game** unchecked is played entirely inside Rust for
speed. It can't be stopped once started; the log shows the winner when it's done.

Engine search depth: depth 0 uses only the evaluator, and each extra level multiplies the work
by roughly 400 on a full board. For the Rust engine with the previous 256-256 net, depth 1 takes
well under a millisecond per move and depth 2 about 25 ms (a whole Rust-vs-Rust game takes about
1 s at depth 2). Depth 3 takes roughly 30 s to several minutes per move (doubles are the slowest),
so a game takes about an hour; depth 4 is impractical. The default 512-256 net evaluates about
2.7× slower than that net, so expect searches to take up to that much longer. The Python engine is several
thousand times slower and only practical at depth 0 or in the endgame.
