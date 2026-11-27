"""Continue the 512-256 and 256-256 nets of the architecture sweep, which were still improving on
the decision benchmark when the sweep stopped them on the games; starts after the low-alpha run
(nets/qat/lowalpha.py) is completely finished.

Same training as the sweep (mixed, alpha 0.01, lambda 0.7, 4M-game segments, average of the last
500k, bearoff database in training and benchmark). Each segment's average is scored on the
decision benchmark; a net stops after STALE segments without lowering its best loss by MARGIN
(counting from where the sweep stopped, so its last segment is the baseline), or at MAX_GAMES.
Then its lowest-loss net plays the game benchmark against bgsage (40k per variant, seed 1, with
the bearoff database), and so does its f16 copy. 512-256
first, then 256-256. Results in nets/arch/results.txt, state in nets/arch/continue.json."""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

DIR = Path(__file__).resolve().parent
ROOT = DIR.parents[1]
sys.path.insert(0, str(ROOT / "nets/qat"))
import run as qat  # noqa: E402  (decisions, eval_copy)

STATE = DIR / "continue.json"
RESULTS = DIR / "results.txt"
NETS = ["512,256", "256,256"]
SWEEP_ORDER = ["256,256", "256,128", "256,256,256", "512,128", "512,256"]  # seeds as in nets/arch/run.py
BEAROFF = "data/bearoff-6x15.bd"
SEGMENT = 4_000_000
STALE = 2
MARGIN = 0.005
MAX_GAMES = 80_000_000
GAMES = 40000
VARIANTS = ["backgammon", "nackgammon"]


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


def save(state):
    STATE.write_text(json.dumps(state, indent=1))


def main():
    lowalpha = ROOT / "nets/qat/lowalpha.json"
    while not (lowalpha.exists() and "final" in json.loads(lowalpha.read_text())):
        time.sleep(120)
    sweep = json.loads((DIR / "state.json").read_text())["nets"]
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    for hidden in NETS:
        name = hidden.replace(",", "-")
        n = state.setdefault(hidden, {})
        if "start" not in n:
            games = sweep[hidden]["games"]
            b, loss, pasko, groups = qat.decisions(DIR / f"{name}-{games // 1_000_000}M-avg.bgnet")
            n.update(start=games, games=games, best=[games, loss], stale=0, done=False,
                     decisions=[[games, b, loss, pasko, groups]])
            save(state)
            log(f"--- continuing {name} from {games // 1_000_000}M (decision loss {loss:.3f}): plateau after "
                f"{STALE} segments without lowering it by {MARGIN}")
        while not n["done"]:
            games = n["games"] + SEGMENT
            m = games // 1_000_000
            out = DIR / f"{name}-{m}M.bgnet"
            seed = 1000 + 100 * SWEEP_ORDER.index(hidden) + games // SEGMENT
            cmd = [str(DIR / "train-db-bin"), "--bearoff", BEAROFF, "--variant", "mixed", "--threads", "12",
                   "--games", str(SEGMENT), "--report", "1000000", "--average-last", "500000", "--alpha", "0.01",
                   "--seed", str(seed), "--resume", str(DIR / f"{name}-{n['games'] // 1_000_000}M.bgnet"),
                   "--out", str(out)]
            with open(out.with_suffix(".log"), "w") as f:
                subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
            b, loss, pasko, groups = qat.decisions(out.with_name(out.stem + "-avg.bgnet"))
            n["decisions"].append([games, b, loss, pasko, groups])
            n["games"] = games
            if loss <= n["best"][1] - MARGIN:
                n["best"], n["stale"] = [games, loss], 0
            else:
                n["stale"] += 1
            n["done"] = n["stale"] >= STALE or games >= MAX_GAMES
            save(state)
            log(f"{name:11s} {m:3d}M: decisions best {b:.2f}%  loss {loss:.3f}  pasko {pasko:.3f}  "
                + " ".join(f"{g[:4]} {v:.3f}" for g, v in groups.items())
                + ("   -> plateau" if n["done"] else ""))
        if "final" not in n:
            # The lowest loss, not the last net that lowered it by MARGIN.
            g, _, loss, *_ = min(n["decisions"], key=lambda row: row[2])
            net = DIR / f"{name}-{g // 1_000_000}M-avg.bgnet"
            scores = [bench(net, f"arch {name} {g // 1_000_000}M (lowest decision loss)", v) for v in VARIANTS]
            copy = qat.eval_copy(net, "f16")
            half = [bench(copy, f"arch {name} {g // 1_000_000}M (f16 copy)", v) for v in VARIANTS]
            n["final"] = {"net": str(net), "loss": loss, "f32": scores, "f16": half}
            save(state)
            log(f"--- {name} best: {g // 1_000_000}M (decision loss {loss:.3f}): bg {scores[0]:.2f}  "
                f"nack {scores[1]:.2f}  mean {sum(scores) / 2:.2f}; f16 copy: bg {half[0]:.2f}  nack {half[1]:.2f}  "
                f"mean {sum(half) / 2:.2f}")


if __name__ == "__main__":
    main()
