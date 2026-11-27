"""The first money net (cubeless, gammons count): a 256-256 td-nack-v1 net with 5 outputs, trained
from scratch with TD(lambda) self-play (src/td_training.rs), mixed backgammon/nackgammon starts,
alpha 0.01, lambda 0.7, in 4M-game segments (average of the last 500k), the bearoff database
in training (games stop there once no gammon is possible).

Every segment's average is scored on the decision benchmark (src/bin/decision_benchmark.rs) with
money scoring (equity lost per decision, millipoints, on the money and Paskogammon sets) and with
dmp scoring (its winning-chance output alone). For comparison, the DMP 256-256 net of the
architecture sweep at the same number of games (nets/arch/256-256-*M-avg) under money scoring.
Stops after STALE segments without lowering the best money loss by MARGIN millipoints, or at
MAX_GAMES. Results in nets/money/results.txt, state in nets/money/state.json.

Uses copies of the binaries (train-bin, decisions-bin), so rebuilding doesn't disturb it."""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

DIR = Path(__file__).resolve().parent
ROOT = DIR.parents[1]
STATE = DIR / "state.json"
RESULTS = DIR / "results.txt"
NAME = "256-256"
SEGMENT = 4_000_000
STALE = 2
MARGIN = 0.15   # millipoints of equity per decision
MAX_GAMES = 60_000_000
BEAROFF = "data/bearoff-6x15.bd"


def log(line):
    print(line, flush=True)
    with open(RESULTS, "a") as f:
        f.write(line + "\n")


def decisions(net, scoring, dataset="money"):
    """(best %, loss, per-group loss) on the decision benchmark."""
    out = subprocess.run([str(DIR / "decisions-bin"), "--set", dataset, "--scoring", scoring, str(net)],
                         cwd=ROOT, check=True, capture_output=True, text=True).stdout
    best, loss = re.search(r"^all\s+\d+\s+([\d.]+)%\s+([\d.]+)", out, re.M).groups()
    groups = dict(re.findall(r"^(attacking|priming|anchoring|racing)\s+\d+\s+[\d.]+%\s+([\d.]+)", out, re.M))
    return float(best), float(loss), {g: float(v) for g, v in groups.items()}


def score(net):
    best, loss, groups = decisions(net, "money")
    _, pasko, _ = decisions(net, "money", "pasko")
    dmp_best, dmp_loss, _ = decisions(net, "dmp")
    return {"best": best, "loss": loss, "pasko": pasko, "groups": groups, "dmp_best": dmp_best, "dmp_loss": dmp_loss}


def train(games, previous):
    m = games // 1_000_000
    out = DIR / f"{NAME}-{m}M.bgnet"
    avg = out.with_name(out.stem + "-avg.bgnet")
    log_file = out.with_suffix(".log")
    if avg.exists() and "average of the last" in log_file.read_text():
        return avg
    cmd = [str(DIR / "train-bin"), "--variant", "mixed", "--threads", "12", "--games", str(SEGMENT),
           "--report", "1000000", "--average-last", "500000", "--alpha", "0.01", "--lambda", "0.7",
           "--seed", str(9000 + games // SEGMENT), "--bearoff", BEAROFF, "--out", str(out)]
    cmd += ["--resume", str(previous)] if previous else ["--objective", "money", "--encoding", "nack",
                                                         "--hidden", "256,256"]
    with open(log_file, "w") as f:
        subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
    return avg


def main():
    # The first segment was started by hand: wait for it.
    first = DIR / f"{NAME}-4M.log"
    while first.exists() and "average of the last" not in first.read_text():
        time.sleep(60)
    state = json.loads(STATE.read_text()) if STATE.exists() else {"segments": [], "best": None, "stale": 0}
    if not state["segments"]:
        log(f"--- money net {NAME} from scratch; decision benchmark: money loss in millipoints per decision "
            f"(money set; pasko), dmp loss in winning-chance points; [DMP 256-256 at the same games: money loss]")
    while not state.get("done"):
        games = (len(state["segments"]) + 1) * SEGMENT
        previous = DIR / f"{NAME}-{games // 1_000_000 - 4}M.bgnet" if state["segments"] else None
        avg = train(games, previous)
        s = score(avg)
        dmp_net = ROOT / f"nets/arch/256-256-{games // 1_000_000}M-avg.bgnet"
        s["dmp_net_loss"] = decisions(dmp_net, "money")[1] if dmp_net.exists() else None
        s["games"] = games
        state["segments"].append(s)
        if state["best"] is None or s["loss"] <= state["best"][1] - MARGIN:
            state["best"], state["stale"] = [games, s["loss"]], 0
        else:
            state["stale"] += 1
        state["done"] = state["stale"] >= STALE or games >= MAX_GAMES
        STATE.write_text(json.dumps(state, indent=1))
        compare = f"   [DMP net: {s['dmp_net_loss']:.3f}]" if s["dmp_net_loss"] is not None else ""
        log(f"{NAME} {games // 1_000_000:3d}M: money best {s['best']:.2f}%  loss {s['loss']:.3f}  pasko {s['pasko']:.3f}  "
            + " ".join(f"{g[:4]} {v:.3f}" for g, v in s["groups"].items())
            + f" | dmp best {s['dmp_best']:.2f}%  loss {s['dmp_loss']:.3f}" + compare
            + ("   -> plateau" if state["done"] else ""))


if __name__ == "__main__":
    sys.exit(main())
