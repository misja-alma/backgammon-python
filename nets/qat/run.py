"""Quantization-aware training from scratch: fresh td-nack-v1 nets (256-256 in f16
and int8, then 128-128 in int8 and 576-192 in f16), mixed training,
every game played and judged by the net's f16 or int8 evaluation copy (`train --quantize`).
Same schedule as the architecture sweep's 256-256 (4M-game segments, alpha 0.01, lambda 0.7,
average of the last 500k games) and, like that run, no bearoff database in training or benchmark,
so the curves compare directly with the f32 256-256 in nets/arch/state.json.

Each segment's average is converted to its evaluation copy in the run's own precision and that
copy is benchmarked against bgsage (40k games per variant, seed 1, --bearoff none). Round-robin,
plateau as in the sweep. First, as reference: the int8 copy of the f32-trained default net
(256-256 44M), quantized after training."""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "nets/qat"
STATE = DIR / "state.json"
RESULTS = DIR / "results.txt"
# name: (precision, hidden sizes). The order fixes the seeds; only append. Run a subset with
# `run.py NAME...` (default: the first two, the original 256-256 runs).
RUNS = {
    "f16": ("f16", "256,256"),
    "i8": ("i8", "256,256"),
    "i8-128-128": ("i8", "128,128"),
    "f16-576-192": ("f16", "576,192"),
    # Trained in f32 (no --quantize, benchmarked as f32 like the architecture sweep); at the
    # plateau the best segment is converted to the precision in CONVERT and benchmarked again.
    "f32-576-192": ("f32", "576,192"),
    "f32-128-128": ("f32", "128,128"),
}
CONVERT = {"f32-576-192": "f16", "f32-128-128": "i8"}
SEGMENT = 4_000_000
MAX_GAMES = 48_000_000
PLATEAU_SEGMENTS = 3
PLATEAU_MARGIN = 0.3
# Runs with DECISION_PLATEAU stop on the decision benchmark (src/bin/decision_benchmark.rs:
# winning chance lost per decision against bgsage's reference checker plays, noise-free) instead
# of the games: after DECISION_SEGMENTS segments without lowering the best loss by DECISION_MARGIN
# (~0.15 points of win rate). They play the games against bgsage only for the final (lowest-loss)
# net: the decision benchmark compares segments more reliably than the games.
DECISION_PLATEAU = {"f32-576-192"}
DECISION_MARGIN = 0.005
DECISION_SEGMENTS = 2
GAMES = 40000
VARIANTS = ["backgammon", "nackgammon"]


def log(line):
    print(line, flush=True)
    with open(RESULTS, "a") as f:
        f.write(line + "\n")


def bench(net, note, variant):
    out = subprocess.run([sys.executable, "-m", "benchmarks.strength", "--net", str(net), "--variant", variant,
                          "--games", str(GAMES), "--bearoff", "none", "--note", note],
                         cwd=ROOT, capture_output=True, text=True)
    match = re.search(r"Win rate ([\d.]+)%", out.stdout + out.stderr)
    if not match:
        raise RuntimeError(out.stdout + out.stderr)
    return float(match.group(1))


def eval_copy(net, precision):
    out = net.with_name(net.stem + f"-{precision}.bgnet")
    subprocess.run([str(DIR / "make-eval-bin"), "--precision", precision, str(net), str(out)],
                   cwd=ROOT, check=True, capture_output=True)
    return out


def decisions(net):
    """(best %, loss, Pasko loss, per-group loss) on the decision benchmark."""
    def run(*extra):
        return subprocess.run([str(ROOT / "target/release/decision_benchmark"), *extra, str(net)],
                              cwd=ROOT, check=True, capture_output=True, text=True).stdout
    money, pasko = run(), run("--set", "pasko")
    best, loss = re.search(r"^all\s+\d+\s+([\d.]+)%\s+([\d.]+)", money, re.M).groups()
    pasko_loss = re.search(r"^all\s+\d+\s+[\d.]+%\s+([\d.]+)", pasko, re.M)[1]
    groups = dict(re.findall(r"^(attacking|priming|anchoring|racing)\s+\d+\s+[\d.]+%\s+([\d.]+)", money, re.M))
    return float(best), float(loss), float(pasko_loss), {g: float(v) for g, v in groups.items()}


def decision_plateau(n):
    """Best (games, loss) and the number of segments since, from n["decisions"]."""
    best, stale = None, 0
    for games, _, loss, *_ in n["decisions"]:
        if best is None or loss <= best[1] - DECISION_MARGIN:
            best, stale = [games, loss], 0
        else:
            stale += 1
    return best, stale


def f32_curve(hidden):
    arch = json.loads((ROOT / "nets/arch/state.json").read_text())["nets"].get(hidden)
    return {s[0]: (s[1] + s[2]) / 2 for s in arch["scores"] if len(s) == 3} if arch else {}


def main():
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    state_ref[0] = state
    selected = sys.argv[1:] or ["f16", "i8"]
    if "ptq_i8" not in state:
        copy = eval_copy(ROOT / "nets/arch/256-256-44M-avg.bgnet", "i8")
        scores = [bench(copy, "qat reference: i8 copy of f32 256-256 44M", v) for v in VARIANTS]
        state["ptq_i8"] = scores
        STATE.write_text(json.dumps(state, indent=1))
        log(f"reference: int8 copy of the f32-trained 256-256 44M: bg {scores[0]:.2f}  nack {scores[1]:.2f}  "
            f"mean {sum(scores) / 2:.2f}   (f32: bg 49.56  nack 50.67  mean 50.12)")
    runs = state.setdefault("runs", {})
    for r in selected:
        runs.setdefault(r, {"games": 0, "scores": [], "best": None, "stale": 0, "done": False})
    while not all(runs[r]["done"] for r in selected):
        for name in selected:
            n = runs[name]
            precision, hidden = RUNS[name]
            label = hidden.replace(",", "-")
            curve = f32_curve(hidden)
            if n["done"]:
                continue
            start, games = n["games"], n["games"] + SEGMENT
            out = DIR / f"{name}-{games // 1_000_000}M.bgnet"
            seed = 5000 + 100 * list(RUNS).index(name) + games // SEGMENT
            cmd = [str(DIR / "train-bin"), *(["--quantize", precision] if precision != "f32" else []), "--variant", "mixed", "--threads", "12",
                   "--games", str(SEGMENT), "--report", "1000000", "--average-last", "500000", "--alpha", "0.01",
                   "--seed", str(seed), "--out", str(out)]
            if start == 0:
                cmd += ["--encoding", "nack", "--hidden", hidden]
            else:
                cmd += ["--resume", str(DIR / f"{name}-{start // 1_000_000}M.bgnet")]
            avg = out.with_name(out.stem + "-avg.bgnet")
            log_file = out.with_suffix(".log")
            finished = avg.exists() and log_file.exists() and "average of the last" in log_file.read_text()
            if not finished:  # (a restarted runner reuses a segment whose training had finished)
                with open(log_file, "w") as f:
                    subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
            if name in DECISION_PLATEAU:
                decision_segment(name, n, games, avg)
                continue
            copy = avg if precision == "f32" else eval_copy(avg, precision)
            scores = [bench(copy, f"qat {precision} {label} {games // 1_000_000}M ({precision} copy)", v)
                      for v in VARIANTS]
            mean = sum(scores) / 2
            n["games"] = games
            n["scores"].append([games, *scores])
            if n["best"] is None or mean >= n["best"][1] + PLATEAU_MARGIN:
                n["best"], n["stale"] = [games, mean], 0
            else:
                n["stale"] += 1
            n["done"] = n["stale"] >= PLATEAU_SEGMENTS or games >= MAX_GAMES
            STATE.write_text(json.dumps(state, indent=1))
            ref = curve.get(games)
            log(f"{name:11s} {games // 1_000_000:3d}M: bg {scores[0]:.2f}  nack {scores[1]:.2f}  mean {mean:.2f}"
                + (f"   (f32 {label} at {games // 1_000_000}M: {ref:.2f})" if ref is not None else "")
                + ("   -> plateau" if n["done"] else ""))
            if n["done"] and name in CONVERT:
                best = n["best"][0] // 1_000_000
                target = CONVERT[name]
                copy = eval_copy(DIR / f"{name}-{best}M-avg.bgnet", target)
                scores = [bench(copy, f"{name} {best}M ({target} copy)", v) for v in VARIANTS]
                n["converted"] = [best * 1_000_000, target, *scores]
                STATE.write_text(json.dumps(state, indent=1))
                log(f"{name:11s} {best:3d}M {target} copy: bg {scores[0]:.2f}  nack {scores[1]:.2f}  "
                    f"mean {sum(scores) / 2:.2f}")


def decision_segment(name, n, games, avg):
    """Bookkeeping of a segment of a DECISION_PLATEAU run; at the end the games and conversion."""
    m = games // 1_000_000
    d = n.setdefault("decisions", [])
    # Segments benchmarked before the switch to this rule: score them first.
    for g in range(SEGMENT, games + 1, SEGMENT):
        if not any(row[0] == g for row in d):
            d.append([g, *decisions(DIR / f"{name}-{g // 1_000_000}M-avg.bgnet")])
    d.sort(key=lambda row: row[0])
    n["games"] = games
    best, stale = decision_plateau(n)
    n["stale"] = stale
    n["done"] = stale >= DECISION_SEGMENTS or games >= MAX_GAMES
    row = next(row for row in d if row[0] == games)
    line = (f"{name:11s} {m:3d}M: decisions best {row[1]:.2f}%  loss {row[2]:.3f}  pasko {row[3]:.3f}  "
            + " ".join(f"{g[:4]} {v:.3f}" for g, v in row[4].items()))
    STATE.write_text(json.dumps(state_ref[0], indent=1))
    log(line + ("   -> plateau" if n["done"] else ""))
    if not n["done"]:
        return
    b = best[0] // 1_000_000
    net = DIR / f"{name}-{b}M-avg.bgnet"
    if best[0] not in {s[0] for s in n["scores"]}:
        scores = [bench(net, f"{name} {b}M (lowest decision loss)", v) for v in VARIANTS]
        n["scores"].append([best[0], *scores])
        log(f"{name:11s} {b:3d}M (lowest decision loss {best[1]:.3f}): bg {scores[0]:.2f}  nack {scores[1]:.2f}  "
            f"mean {sum(scores) / 2:.2f}")
    n["best"] = [best[0], best[1]]
    STATE.write_text(json.dumps(state_ref[0], indent=1))
    if name in CONVERT:
        target = CONVERT[name]
        copy = eval_copy(net, target)
        scores = [bench(copy, f"{name} {b}M ({target} copy)", v) for v in VARIANTS]
        n["converted"] = [best[0], target, *scores]
        STATE.write_text(json.dumps(state_ref[0], indent=1))
        log(f"{name:11s} {b:3d}M {target} copy: bg {scores[0]:.2f}  nack {scores[1]:.2f}  mean {sum(scores) / 2:.2f}")


state_ref = [None]


if __name__ == "__main__":
    main()
