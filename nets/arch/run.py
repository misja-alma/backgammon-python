"""Architecture sweep: td-nack-v1 nets of several sizes, mixed training (alternating backgammon /
nackgammon starts), each trained in 4M-game segments until it plateaus. Every segment's average
is benchmarked against bgsage on both variants with 40k games (seed 1).

Round-robin: each round trains every unfinished net one segment, so the curves grow side by side.
Plateau: the mean of the two variants has not beaten the net's best by PLATEAU_MARGIN for
PLATEAU_SEGMENTS segments in a row (or MAX_GAMES reached). State is kept in state.json, so the
script can be restarted and continues where it stopped."""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "nets/arch"
STATE = DIR / "state.json"
RESULTS = DIR / "results.txt"
NETS = ["256,256", "256,128", "256,256,256", "512,128", "512,256", "756"]
SEGMENT = 4_000_000
MAX_GAMES = 48_000_000
PLATEAU_SEGMENTS = 3
PLATEAU_MARGIN = 0.3
GAMES = 40000
VARIANTS = ["backgammon", "nackgammon"]
# Since 2026-11-22: training stops games at the first bearoff position (exact value as the
# target) and the benchmark looks bearoff positions up. When this phase starts, every unfinished
# net's last average is benchmarked again with the database and becomes its new best (stale 0),
# since the earlier scores were measured without it.
BEAROFF = "data/bearoff-6x15.bd"
DROP = ["756"]   # stopped by hand: no use continuing


def log(line):
    print(line, flush=True)
    with open(RESULTS, "a") as f:
        f.write(line + "\n")


def bench(net, note, variant):
    out = subprocess.run([sys.executable, "-m", "benchmarks.strength", "--net", str(net), "--variant", variant,
                          "--games", str(GAMES), "--bearoff", BEAROFF, "--note", note],
                         cwd=ROOT, capture_output=True, text=True)
    match = re.search(r"Win rate ([\d.]+)%", out.stdout + out.stderr)
    if not match:
        raise RuntimeError(out.stdout + out.stderr)
    return float(match.group(1))


def name(hidden):
    return hidden.replace(",", "-")


def main():
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    if "baseline" not in state:
        net = ROOT / "nets/flags/mixed-30M-avg.bgnet"
        scores = [bench(net, "arch baseline: 128-128 td-flags mixed 30M", v) for v in VARIANTS]
        state["baseline"] = scores
        STATE.write_text(json.dumps(state, indent=1))
        log(f"baseline 128-128 td-flags mixed 30M: bg {scores[0]:.2f}  nack {scores[1]:.2f}  "
            f"mean {sum(scores) / 2:.2f}")
    nets = state.setdefault("nets", {})
    for h in NETS:   # nets added to NETS later join with a fresh entry
        nets.setdefault(h, {"games": 0, "scores": [], "best": None, "stale": 0, "done": False})
    for h in DROP:
        if not nets[h]["done"]:
            nets[h]["done"] = True
            log(f"{name(h):12s} stopped at {nets[h]['games'] // 1_000_000}M (no use continuing)")
    if not state.get("bearoff_phase"):
        for hidden, n in nets.items():
            if n["done"] or n["games"] == 0:
                continue
            avg = DIR / f"{name(hidden)}-{n['games'] // 1_000_000}M-avg.bgnet"
            scores = [bench(avg, f"arch {name(hidden)} {n['games'] // 1_000_000}M, with bearoff db", v)
                      for v in VARIANTS]
            mean = sum(scores) / 2
            n["scores"].append([n["games"], *scores, "bearoff"])
            n["best"], n["stale"] = [n["games"], mean], 0
            log(f"{name(hidden):12s} {n['games'] // 1_000_000:3d}M: bg {scores[0]:.2f}  nack {scores[1]:.2f}  "
                f"mean {mean:.2f}   (again, with the bearoff database; training uses it from here on)")
        state["bearoff_phase"] = True
        STATE.write_text(json.dumps(state, indent=1))
    round_ = 0
    while not all(n["done"] for n in nets.values()):
        round_ += 1
        for hidden, n in nets.items():
            if n["done"]:
                continue
            start, games = n["games"], n["games"] + SEGMENT
            out = DIR / f"{name(hidden)}-{games // 1_000_000}M.bgnet"
            seed = 1000 + 100 * NETS.index(hidden) + games // SEGMENT
            cmd = [str(DIR / "train-db-bin"), "--bearoff", BEAROFF, "--variant", "mixed", "--threads", "12", "--games", str(SEGMENT),
                   "--report", "1000000", "--average-last", "500000", "--alpha", "0.01", "--seed", str(seed),
                   "--out", str(out)]
            if start == 0:
                cmd += ["--encoding", "nack", "--hidden", hidden]
            else:
                cmd += ["--resume", str(DIR / f"{name(hidden)}-{start // 1_000_000}M.bgnet")]
            with open(out.with_suffix(".log"), "w") as f:
                subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
            avg = out.with_name(out.stem + "-avg.bgnet")
            scores = [bench(avg, f"arch {name(hidden)} td-nack mixed {games // 1_000_000}M", v) for v in VARIANTS]
            mean = sum(scores) / 2
            n["games"] = games
            n["scores"].append([games, *scores])
            if n["best"] is None or mean >= n["best"][1] + PLATEAU_MARGIN:
                n["best"], n["stale"] = [games, mean], 0
            else:
                n["stale"] += 1
            n["done"] = n["stale"] >= PLATEAU_SEGMENTS or games >= MAX_GAMES
            STATE.write_text(json.dumps(state, indent=1))
            log(f"{name(hidden):12s} {games // 1_000_000:3d}M: bg {scores[0]:.2f}  nack {scores[1]:.2f}  "
                f"mean {mean:.2f}" + ("   -> plateau" if n["done"] else ""))


if __name__ == "__main__":
    main()
