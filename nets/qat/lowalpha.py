"""Continued training of the 576-192 net with a lower learning rate, after its run (run.py
f32-576-192) is completely finished.

From the raw net of that run's best segment (lowest decision loss): phases with alpha 0.005, then
0.0025, each phase from the best net of the one before. 4M-game segments (mixed, lambda 0.7,
average of the last 500k), each scored on the decision benchmark; a phase stops after
STALE segments without lowering its best loss by MARGIN, or after MAX_PHASE games. Then the
overall best net (lowest decision loss) plays the game benchmark against bgsage (40k per
variant, --bearoff none), and so does its f16 copy. Results in nets/qat/results.txt, state in
nets/qat/lowalpha.json."""
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run  # noqa: E402  (decisions, bench, eval_copy, log)

DIR = run.DIR
ROOT = run.ROOT
STATE = DIR / "lowalpha.json"
PARENT = "f32-576-192"
PHASES = [("a005", 0.005), ("a0025", 0.0025)]
SEGMENT = 4_000_000
STALE = 2
MARGIN = 0.003
MAX_PHASE = 16_000_000


def save(state):
    STATE.write_text(json.dumps(state, indent=1))


def main():
    # Wait for the parent run: plateau, games for its final net and the f16 copy.
    while True:
        parent = json.loads((DIR / "state.json").read_text())["runs"].get(PARENT, {})
        if parent.get("done") and "converted" in parent:
            break
        time.sleep(120)
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    if "start" not in state:
        games = parent["best"][0]
        row = next(r for r in parent["decisions"] if r[0] == games)
        state["start"] = {"net": str(DIR / f"{PARENT}-{games // 1_000_000}M.bgnet"), "games": games, "loss": row[2]}
        state["phases"] = {}
        save(state)
        run.log(f"--- low alpha: continuing {PARENT} from {games // 1_000_000}M (decision loss {row[2]:.3f})")
    start = state["start"]
    best = {"net": start["net"], "avg": str(DIR / f"{PARENT}-{start['games'] // 1_000_000}M-avg.bgnet"),
            "games": start["games"], "loss": start["loss"]}
    for k, (tag, alpha) in enumerate(PHASES):
        p = state["phases"].setdefault(tag, {"from": dict(best), "segments": [], "done": False})
        origin = p["from"]
        while not p["done"]:
            done_games = len(p["segments"]) * SEGMENT
            total = origin["games"] + done_games + SEGMENT
            name = f"{PARENT}-{tag}-{total // 1_000_000}M"
            out = DIR / f"{name}.bgnet"
            previous = origin["net"] if not p["segments"] else p["segments"][-1]["net"]
            cmd = [str(DIR / "train-bin"), "--variant", "mixed", "--threads", "12", "--games", str(SEGMENT),
                   "--report", "1000000", "--average-last", "500000", "--alpha", str(alpha),
                   "--seed", str(7000 + 100 * k + len(p["segments"])), "--resume", previous, "--out", str(out)]
            with open(out.with_suffix(".log"), "w") as f:
                subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
            avg = out.with_name(out.stem + "-avg.bgnet")
            b, loss, pasko, groups = run.decisions(avg)
            p["segments"].append({"net": str(out), "avg": str(avg), "games": total, "best": b, "loss": loss,
                                  "pasko": pasko, "groups": groups})
            phase_best = min([origin["loss"]] + [s["loss"] for s in p["segments"][:-1]])
            p["stale"] = 0 if loss <= phase_best - MARGIN else p.get("stale", 0) + 1
            p["done"] = p["stale"] >= STALE or done_games + SEGMENT >= MAX_PHASE
            save(state)
            run.log(f"{PARENT} alpha {alpha} {total // 1_000_000:3d}M: decisions best {b:.2f}%  loss {loss:.3f}  "
                    f"pasko {pasko:.3f}  " + " ".join(f"{g[:4]} {v:.3f}" for g, v in groups.items())
                    + ("   -> phase done" if p["done"] else ""))
        for s in p["segments"]:
            if s["loss"] < best["loss"]:
                best = {"net": s["net"], "avg": s["avg"], "games": s["games"], "loss": s["loss"]}
        save(state)
    if "final" not in state:
        avg = Path(best["avg"])
        label = avg.stem
        scores = [run.bench(avg, f"low alpha: {label}", v) for v in run.VARIANTS]
        copy = run.eval_copy(avg, "f16")
        half = [run.bench(copy, f"low alpha: {label} (f16 copy)", v) for v in run.VARIANTS]
        state["final"] = {"net": str(avg), "loss": best["loss"], "f32": scores, "f16": half}
        save(state)
        run.log(f"--- low alpha best: {label} (decision loss {best['loss']:.3f}, start {start['loss']:.3f}): "
                f"bg {scores[0]:.2f}  nack {scores[1]:.2f}  mean {sum(scores) / 2:.2f}; f16 copy: bg {half[0]:.2f}  "
                f"nack {half[1]:.2f}  mean {sum(half) / 2:.2f}")


if __name__ == "__main__":
    main()
