"""Extract checker-play decisions from bgsage's reference benchmarks into a compact text file
for `cargo run --release --bin decision_benchmark`.

    .venv/bin/python -m benchmarks.extract_decisions [BGSAGE_DATA_DIR]   (default ~/workspace/bgsage/data)

Sources (built by bgsage from games of bgsage 3-ply against itself, see its
scripts/benchmark_money.py): `money_benchmark` (backgammon) and `pasko_money_benchmark`
(Paskogammon). The reference precision of a decision depends on how close it is: 3-ply for
clear decisions ("3P"), a truncated rollout when best and second best are within 0.05 ("3T"),
a full rollout of 1296 games within 0.02 ("rollout"). Every candidate has rolled-out or
evaluated probabilities: win, win gammon, win backgammon, lose gammon, lose backgammon, for
the player who moved.

Kept: checker decisions that are not pure races (bgsage's game plan "purerace": once contact
is over the bearoff database and the search take over; bearoffs are dropped as well), and of
each decision only the candidates evaluated at 3-ply or better ("3-ply" and "Rollout", which includes the truncated rollouts); decisions with fewer
than 2 of those are dropped. Each decision keeps bgsage's game plan as its group: attacking,
priming, anchoring, racing (a race with contact left).

Output, data/decisions/<name>.txt, one decision per block:

    D <group> <tier> <die1> <die2> <board: 26 ints> <number of candidates>
    C <board: 26 ints> <win> <win gammon> <win bg> <lose gammon> <lose bg> <eval level>

Boards are bgsage's: the player on roll (who moves) positive, index 1-24 = that player's points
1-24 (they bear off from 1-6), 25 = their bar; the opponent negative, opponent's bar at 0.
Candidate boards are after the move, from the same side.
"""
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "decisions"
SOURCES = {"money": "money_benchmark/benchmark.json.gz", "pasko": "pasko_money_benchmark/benchmark.json.gz"}
GOOD_LEVELS = ("3-ply", "Rollout")


def is_bearoff(board):
    return all(v <= 0 for v in board[7:26]) and all(v >= 0 for v in board[0:19])


def extract(source, target):
    decisions = json.load(gzip.open(source))["decisions"]
    counts, groups, candidates = Counter(), Counter(), 0
    with open(target, "w") as out:
        out.write(f"# from {source}: see benchmarks/extract_decisions.py\n")
        for d in decisions:
            if d["kind"] != "checker":
                continue
            if d["game_plan"] == "purerace" or is_bearoff(d["board"]):
                counts["pure race or bearoff"] += 1
                continue
            good = [m for m in d["moves"] if m["eval_level"] in GOOD_LEVELS]
            if len(good) < 2:
                counts["fewer than 2 candidates at 3-ply or better"] += 1
                continue
            counts["kept"] += 1
            groups[d["game_plan"]] += 1
            candidates += len(good)
            board = " ".join(map(str, d["board"]))
            out.write(f"D {d['game_plan']} {d['tier']} {d['dice'][0]} {d['dice'][1]} {board} {len(good)}\n")
            for m in good:
                probs = " ".join(f"{p:.6f}" for p in m["probs"])
                out.write(f"C {' '.join(map(str, m['board']))} {probs} {m['eval_level']}\n")
    print(f"{target.relative_to(ROOT)}: {dict(counts)}, {candidates} candidates")
    print(f"  groups: {dict(groups)}")


def main():
    data = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "workspace/bgsage/data"
    OUT.mkdir(parents=True, exist_ok=True)
    for name, path in SOURCES.items():
        extract(data / path, OUT / f"{name}.txt")


if __name__ == "__main__":
    main()
