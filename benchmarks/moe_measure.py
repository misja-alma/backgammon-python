"""Turn a measuring set of the mixture of experts into a decision benchmark judged by bgsage.

    .venv-bgsage/bin/python benchmarks/moe_measure.py nets/moe/measure data/decisions/moe-types.txt
    .venv-bgsage/bin/python benchmarks/moe_measure.py --money nets/moe/measure data/decisions/moe-types-money.txt

Input: DIR/<game type>.pos from moe_collect (a separate collection, kept out of the training sets;
52 bytes per position: the player on roll's checkers, points 0-25 in its own numbering, then the
other player's). For each position a roll is drawn (seeded per type and index), and bgsage
evaluates the moves at 3-ply for double match point (a 1-point match), or with --money for money
(centered cube, gammons count: no Jacoby rule). Kept: the candidates bgsage evaluated at 3-ply;
positions with fewer than 2 of those (forced moves, dancing, ...), or where they all have the same
value (nothing to decide: the winning chance, with --money the cubeless money equity), are skipped;
the positions of a type are judged in order until it has --per-type decisions (or none are left). Output in the format of data/decisions (see
benchmarks/extract_decisions.py) with the game type as the group:

    D <game type> 3P <die1> <die2> <board: 26 ints> <number of candidates>
    C <board after the move: 26 ints> <win> <win gammon> <win bg> <lose gammon> <lose bg> 3-ply

Score nets on it with `decision_benchmark --set moe-types NET` (one line per game type; the money
set with `--set moe-types-money --scoring money`).
Runs in the bgsage environment (Python 3.10+): only the standard library and bgsage.
"""
import argparse
import hashlib
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

_analyzer = None
_money = False


def board_of(c):
    """bgsage board of a 52-byte position: the player on roll positive (points 1-24, bar 25), the
    other player negative in the mover's numbering, the other's bar at 0."""
    own, other = c[:26], c[26:]
    board = [0] * 26
    for point in range(1, 25):
        board[point] = own[point] - other[25 - point]
    board[25] = own[25]
    board[0] = other[25]
    return board


def init(ply, money):
    global _analyzer, _money
    _money = money
    from bgsage import BgBotAnalyzer
    _analyzer = BgBotAnalyzer(eval_level=f"{ply}ply")


def judge(task):
    name, index, c = task
    rng = random.Random(hashlib.sha256(f"{name}:{index}".encode()).digest())
    d1, d2 = rng.randint(1, 6), rng.randint(1, 6)
    board = board_of(c)
    if _money:
        result = _analyzer.checker_play(board, d1, d2, jacoby=False)
    else:
        result = _analyzer.checker_play(board, d1, d2, away1=1, away2=1)
    moves = [m for m in result.moves if m.eval_level == "3-ply"]
    # No decision to make: fewer than 2 candidates, or all equally good (e.g. a won bearoff, where
    # bgsage also lists bearing off with one die when both dice can be played).
    def value(p):   # from the probabilities as written to the file
        r = [round(x, 6) for x in (p.win, p.gammon_win, p.backgammon_win, p.gammon_loss, p.backgammon_loss)]
        return round(2 * r[0] - 1 + r[1] + r[2] - r[3] - r[4], 6) if _money else r[0]
    values = [value(m.probs) for m in moves]
    if len(moves) < 2 or max(values) == min(values):
        return name, index, None
    lines = [f"D {name} 3P {d1} {d2} {' '.join(map(str, board))} {len(moves)}"]
    for m in moves:
        p = m.probs
        lines.append(f"C {' '.join(map(str, m.board))} {p.win:.6f} {p.gammon_win:.6f} {p.backgammon_win:.6f} "
                     f"{p.gammon_loss:.6f} {p.backgammon_loss:.6f} 3-ply")
    return name, index, "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("out")
    ap.add_argument("--per-type", type=int, default=1000)
    ap.add_argument("--ply", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--money", action="store_true", help="money instead of double match point")
    a = ap.parse_args()

    names = [line.split()[0] for line in (Path(a.dir) / "counts.txt").read_text().splitlines()[1:]]
    data = {name: (Path(a.dir) / f"{name}.pos").read_bytes() for name in names}
    started = time.time()
    results = {name: [] for name in names}
    skipped = {name: 0 for name in names}
    done = {name: 0 for name in names}   # positions of the type judged so far
    with Pool(a.workers, initializer=init, initargs=(a.ply, a.money)) as pool:
        # In rounds: each type that still needs decisions gets the positions it probably needs, by
        # its share of usable positions so far (many are forced moves in some types, e.g. dancing).
        while True:
            tasks = []
            for name in names:
                need = a.per_type - len(results[name])
                available = len(data[name]) // 52 - done[name]
                if need <= 0 or available <= 0:
                    continue
                rate = len(results[name]) / done[name] if done[name] else 1.0
                k = min(available, max(40, int(1.2 * need / max(rate, 0.05))))
                tasks += [(name, i, list(data[name][i * 52:(i + 1) * 52])) for i in range(done[name], done[name] + k)]
                done[name] += k
            if not tasks:
                break
            for name, index, text in pool.imap_unordered(judge, tasks, chunksize=8):
                if text is None:
                    skipped[name] += 1
                else:
                    results[name].append((index, text))
            print(f"{sum(min(len(r), a.per_type) for r in results.values())} decisions, {time.time() - started:.0f} s",
                  file=sys.stderr, flush=True)
    with open(a.out, "w") as f:
        f.write(f"# measuring set of the mixture of experts from {a.dir}: bgsage {a.ply}-ply, {'money' if a.money else 'double match point'}; "
                f"see benchmarks/moe_measure.py\n")
        for name in names:
            for _, text in sorted(results[name])[:a.per_type]:
                f.write(text + "\n")
    for name in names:
        print(f"{name:24s} {min(len(results[name]), a.per_type):5d} decisions ({skipped[name]} of {done[name]} positions skipped)")
    print(f"wrote {a.out} in {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
