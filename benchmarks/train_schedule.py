"""The standard way to train a net: TD(lambda) self-play in segments, with the learning rate lowered
each time the net plateaus, progress measured on the decision benchmark, games only now and then.

    .venv/bin/python -m benchmarks.train_schedule NAME --dir nets/money --objective money --hidden 256,256
    .venv/bin/python -m benchmarks.train_schedule NAME --dir DIR --resume NET.bgnet   (continue a trained net)
    ... --resume NET.bgnet --plateaued --alphas 0.005,0.0025   (a net that plateaued at alpha 0.01)

Phases with alpha 0.01, 0.005 and 0.0025 (`--alphas`). Every phase trains 4M-game segments
(mixed backgammon/nackgammon starts, lambda 0.7, 12 threads, the average of the last 500k games
is the segment's result) and scores each segment's average on the decision benchmark
(src/bin/decision_benchmark.rs; money scoring for money nets: millipoints per decision, else
dmp scoring: winning-chance points; also the Paskogammon set). A phase ends after STALE
segment(s) without lowering the best loss by its margin (STALE_LAST in the last phase, before the
run stops), or after its maximum number of games; the
next phase starts from the raw net of the best segment so far (lowest loss). The first phase may
run up to --max-games in total, later phases 16M games each.

The 40k-game benchmark against bgsage (benchmarks.strength, 40k games per variant, seed 1; money
games with --money, scored in points per game) is played only every --game-every games and, at
the end, for the overall best net and its f16 evaluation copy.

Files in --dir: NAME-<M>M.bgnet (first phase) and NAME-a<alpha>-<M>M.bgnet (later phases), each
with -avg and .log; NAME-state.json; results in results.txt. Rerunning resumes: finished segments
(and their scores) are reused. The binaries are copied to --dir/bin first, so rebuilding the
project during a run doesn't affect it (but don't rebuild the Python extension while the games
are played).
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEGMENT = 4_000_000
STALE = 1        # stale segments that end a phase
STALE_LAST = 2   # ... the last phase (after it the run stops)
LATER_PHASE_GAMES = 16_000_000
GAMES = 40000
VARIANTS = ["backgammon", "nackgammon"]
# Margin of the first phase per objective (loss units of its scoring); later phases use 60% of it.
MARGIN = {"dmp": 0.005, "money": 0.15}
BINARIES = {"train": "train", "decision_benchmark": "decisions", "make_eval_net": "make-eval"}


class Run:
    def __init__(self, args):
        self.a = args
        self.dir = Path(args.dir).resolve()
        self.bin = self.dir / "bin"
        self.state_file = self.dir / f"{args.name}-state.json"
        self.state = json.loads(self.state_file.read_text()) if self.state_file.exists() else {}

    # --- helpers ------------------------------------------------------------------------------

    def log(self, line):
        print(line, flush=True)
        with open(self.dir / "results.txt", "a") as f:
            f.write(line + "\n")

    def save(self):
        self.state_file.write_text(json.dumps(self.state, indent=1))

    def copy_binaries(self):
        self.bin.mkdir(parents=True, exist_ok=True)
        subprocess.run(["cargo", "build", "--release", "--bins"], cwd=ROOT, check=True, capture_output=True)
        for source, target in BINARIES.items():
            if not (self.bin / target).exists():
                shutil.copy2(ROOT / "target/release" / source, self.bin / target)

    def scoring(self):
        return self.state["objective"]

    def decisions(self, net, dataset="money"):
        """(best %, loss, per-group loss) on the decision benchmark with the net's own scoring."""
        out = subprocess.run([str(self.bin / "decisions"), "--set", dataset, "--scoring", self.scoring(), str(net)],
                             cwd=ROOT, check=True, capture_output=True, text=True).stdout
        best, loss = re.search(r"^all\s+\d+\s+([\d.]+)%\s+([\d.]+)", out, re.M).groups()
        groups = dict(re.findall(r"^(attacking|priming|anchoring|racing)\s+\d+\s+[\d.]+%\s+([\d.]+)", out, re.M))
        return float(best), float(loss), {g: float(v) for g, v in groups.items()}

    def games(self, net, note):
        """Per variant: win rate (dmp) or points per game (money) against bgsage 1-ply."""
        money = self.scoring() == "money"
        bearoff = self.a.bearoff if self.a.bearoff != "none" else "none"
        scores = []
        for variant in VARIANTS:
            cmd = [sys.executable, "-m", "benchmarks.strength", "--net", str(net), "--variant", variant,
                   "--games", str(GAMES), "--bearoff", bearoff, "--note", note] + (["--money"] if money else [])
            out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            text = out.stdout + out.stderr
            pattern = r"Points per game ([+-][\d.]+)" if money else r"Win rate ([\d.]+)%"
            match = re.search(pattern, text)
            if not match:
                raise RuntimeError(text)
            scores.append(float(match.group(1)))
        return scores

    def format_games(self, scores):
        if self.scoring() == "money":
            return f"bg {scores[0]:+.4f}  nack {scores[1]:+.4f}  mean {sum(scores) / 2:+.4f} ppg"
        return f"bg {scores[0]:.2f}  nack {scores[1]:.2f}  mean {sum(scores) / 2:.2f}"

    # --- training -----------------------------------------------------------------------------

    def segment_name(self, phase, total):
        prefix = self.a.name if self.first(phase) else f"{self.a.name}-a{str(self.state['alphas'][phase])[2:]}"
        return f"{prefix}-{total // 1_000_000}M"

    def first(self, phase):
        """Whether this is a first phase (full margin, up to --max-games): not when the resumed net
        already plateaued at a higher alpha (--plateaued)."""
        return phase == 0 and not self.state.get("plateaued")

    def train(self, phase, total, previous, seed):
        out = self.dir / f"{self.segment_name(phase, total)}.bgnet"
        avg = out.with_name(out.stem + "-avg.bgnet")
        log_file = out.with_suffix(".log")
        if avg.exists() and log_file.exists() and "average of the last" in log_file.read_text():
            return out, avg
        cmd = [str(self.bin / "train"), "--variant", self.a.variant, "--threads", "12", "--games", str(SEGMENT),
               "--report", "1000000", "--average-last", "500000", "--alpha", str(self.state["alphas"][phase]),
               "--lambda", "0.7", "--seed", str(seed), "--out", str(out)]
        if self.a.bearoff != "none":
            cmd += ["--bearoff", self.a.bearoff]
        if previous:
            cmd += ["--resume", str(previous)]
        else:
            cmd += ["--objective", self.a.objective, "--encoding", self.a.encoding, "--hidden", self.a.hidden]
        with open(log_file, "w") as f:
            subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
        return out, avg

    def run(self):
        a = self.a
        self.dir.mkdir(parents=True, exist_ok=True)
        self.copy_binaries()
        st = self.state
        if "phases" not in st:
            start_games = 0
            if a.resume:
                head = Path(a.resume).read_bytes()[:400].decode("latin-1")
                start_games = int(re.search(r"^games (\d+)", head, re.M).group(1))
                objective = re.search(r"^objective (\w+)", head, re.M)
                a.objective = objective.group(1) if objective else "dmp"
            st.update(objective=a.objective, alphas=[float(x) for x in a.alphas.split(",")], phases=[],
                      start={"net": a.resume, "games": start_games}, games_played=[], done=False,
                      plateaued=bool(a.plateaued))
            self.save()
            what = f"from {a.resume}" if a.resume else f"{a.objective} {a.hidden} from scratch"
            unit = "millipoints" if a.objective == "money" else "winning-chance points"
            self.log(f"--- {a.name}: {what}; alphas {a.alphas}, plateau after {STALE} segment(s) "
                     f"({STALE_LAST} in the last phase) without lowering the decision loss ({unit}, {a.objective} scoring) by {MARGIN[a.objective]} "
                     f"(later phases 60%); games every {a.game_every // 1_000_000}M and for the final net")
        best = st.get("best")   # overall best segment: {raw, avg, games, loss}
        if best is None and a.resume:
            avg = Path(a.resume).with_name(Path(a.resume).stem + "-avg.bgnet")   # its average, if there is one
            avg = avg if avg.exists() else Path(a.resume)
            _, loss, _ = self.decisions(avg)
            best = st["best"] = {"raw": a.resume, "avg": str(avg), "games": st["start"]["games"], "loss": loss}
            self.save()
        for phase, alpha in enumerate(st["alphas"]):
            if phase < len(st["phases"]) and st["phases"][phase]["done"]:
                continue
            if phase == len(st["phases"]):
                origin = dict(best) if best else None
                st["phases"].append({"alpha": alpha, "from": origin, "segments": [], "stale": 0, "done": False})
                self.save()
                if not self.first(phase):
                    self.log(f"--- {a.name}: alpha {alpha} from {Path(origin['raw']).name} "
                             f"(decision loss {origin['loss']:.3f})")
            p = st["phases"][phase]
            margin = MARGIN[st["objective"]] * (1.0 if self.first(phase) else 0.6)
            limit = a.max_games if self.first(phase) else (p["from"]["games"] if p["from"] else 0) + LATER_PHASE_GAMES
            stale_limit = STALE_LAST if phase == len(st["alphas"]) - 1 else STALE
            if p["segments"] and p["stale"] >= stale_limit:   # resumed under a stricter plateau rule
                p["done"] = True
                self.save()
                self.log(f"--- {a.name}: alpha {alpha} ends after {p['stale']} stale segment(s)")
                continue
            while not p["done"]:
                origin_games = p["from"]["games"] if p["from"] else 0
                total = origin_games + (len(p["segments"]) + 1) * SEGMENT
                previous = (p["segments"][-1]["raw"] if p["segments"] else (p["from"]["raw"] if p["from"] else None))
                raw, avg = self.train(phase, total, previous, a.seed + 100 * phase + len(p["segments"]))
                b, loss, groups = self.decisions(avg)
                _, pasko, _ = self.decisions(avg, "pasko")
                seg = {"raw": str(raw), "avg": str(avg), "games": total, "best": b, "loss": loss, "pasko": pasko,
                       "groups": groups}
                p["segments"].append(seg)
                phase_best = min([p["from"]["loss"] if p["from"] else float("inf")]
                                 + [s["loss"] for s in p["segments"][:-1]])
                p["stale"] = 0 if loss <= phase_best - margin else p["stale"] + 1
                p["done"] = p["stale"] >= stale_limit or total >= limit
                if best is None or loss < best["loss"]:
                    best = st["best"] = {"raw": seg["raw"], "avg": seg["avg"], "games": total, "loss": loss}
                line = (f"{a.name} alpha {alpha} {total // 1_000_000:3d}M: decisions best {b:.2f}%  loss {loss:.3f}  "
                        f"pasko {pasko:.3f}  " + " ".join(f"{g[:4]} {v:.3f}" for g, v in groups.items()))
                if a.compare:
                    other = ROOT / a.compare.format(m=total // 1_000_000)
                    if other.exists():
                        line += f"   [compare: {self.decisions(other)[1]:.3f}]"
                if total % a.game_every == 0 and total not in st["games_played"]:
                    self.save()
                    scores = self.games(avg, f"{a.name} alpha {alpha} {total // 1_000_000}M")
                    st["games_played"].append(total)
                    seg["games_result"] = scores
                    line += "   | " + self.format_games(scores)
                self.save()
                self.log(line + ("   -> phase done" if p["done"] else ""))
        if "final" not in st:
            avg = Path(best["avg"])
            scores = self.games(avg, f"{a.name}: final {avg.stem}")
            copy = avg.with_name(avg.stem + "-f16.bgnet")
            subprocess.run([str(self.bin / "make-eval"), "--precision", "f16", str(avg), str(copy)],
                           cwd=ROOT, check=True, capture_output=True)
            half = self.games(copy, f"{a.name}: final {avg.stem} (f16 copy)")
            st["final"] = {"net": str(avg), "loss": best["loss"], "f32": scores, "f16": half}
            st["done"] = True
            self.save()
            self.log(f"--- {a.name} best: {avg.stem} (decision loss {best['loss']:.3f}): {self.format_games(scores)}; "
                     f"f16 copy: {self.format_games(half)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", help="name of the run, prefix of its files")
    parser.add_argument("--dir", required=True, help="directory for the nets, logs and results")
    parser.add_argument("--objective", choices=["dmp", "money"], default="dmp", help="for a new net (dmp)")
    parser.add_argument("--hidden", default="256,256", help="hidden layers of a new net (256,256)")
    parser.add_argument("--encoding", default="nack", help="encoding of a new net (nack)")
    parser.add_argument("--resume", help="continue this trained net (raw net, not -avg) instead of a new one")
    parser.add_argument("--plateaued", action="store_true",
                        help="the resumed net already plateaued at a higher alpha: every phase is a later phase "
                             "(e.g. with --alphas 0.005,0.0025)")
    parser.add_argument("--variant", default="mixed", help="training starts (mixed)")
    parser.add_argument("--bearoff", default="data/bearoff-6x15.bd",
                        help="bearoff database for training and games, or 'none' (data/bearoff-6x15.bd)")
    parser.add_argument("--alphas", default="0.01,0.005,0.0025", help="learning rate per phase (0.01,0.005,0.0025)")
    parser.add_argument("--max-games", type=int, default=80_000_000, help="end of the first phase (80M)")
    parser.add_argument("--game-every", type=int, default=16_000_000, help="40k games every N games (16M)")
    parser.add_argument("--seed", type=int, default=9001, help="seed of the first segment (9001)")
    parser.add_argument("--compare", help="another net per segment, e.g. nets/arch/256-256-{m}M-avg.bgnet")
    Run(parser.parse_args()).run()


if __name__ == "__main__":
    main()
