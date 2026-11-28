"""The training procedure of the mixture of experts (src/moe.rs), in stages:

The experts are cubeless money nets of --hidden (256-128). The game types (DIR/types.txt) still
classify with their own reference net (the DMP 512-256 they were found in).

0. Base net: a money net of --hidden trained with the standard schedule (benchmarks.train_schedule,
   alphas 0.01, 0.005, 0.0025) in DIR/base; its best segment (raw net) starts every expert. The
   value net, which judges positions off an expert's type, is the best net of the finished money
   run (--value-from, its state file), copied to DIR/value.bgnet.
1. Collect: self-play of the reference net (the current default), every position classified by game
   type and stored in a set per expert (moe_collect, --per-type positions each): DIR/positions/.
2. Experts, one at a time, only on their own game type (moe_expert: self-play games starting from
   the expert's set; positions of its type judged by the expert, the rest by the frozen value net;
   only the expert learns, only from its own positions). The standard schedule per expert: the
   experts start from the base net, which already plateaued at alpha 0.01, so alphas 0.005 then
   0.0025; --segment games per segment; a phase ends after 1 segment (2 in the last phase) without
   lowering the expert's best loss by 0.09 millipoints (no limit on the number of segments since
   2026-11-27; it was 4, which stopped experts that were still improving); the next phase starts
   from the best segment. Progress is measured against bgsage on the measuring set: a separate
   collection (DIR/measure, seed 4242, never trained on: the training collection excludes it) of
   1000 decisions per game type judged by bgsage at 3-ply for money
   (benchmarks/moe_measure.py --money -> data/decisions/moe-types-money.txt; money scoring,
   millipoints); the expert's loss on the decisions of its type against the base net's. A
   decision counts for a type when most of its candidates (the positions after the moves, with the
   opponent on roll) are of that type: those are what an expert of the type evaluates (types come
   in mirror pairs: the candidates of a holding-ahead decision are mostly holding-behind positions).
   Until 2026-11-27 decisions were grouped by the position before the move, which measured each
   expert on its mirror type.
3. Gateway (moe_gate), with the experts frozen (their best segments): a single hidden layer of 512,
   trained on self-play positions to rate how well each expert judges them. Only the gateway learns.
4. The evaluator (DIR/experts/moe.bgmoe): experts rated below 0.25 skipped, the remaining ratings
   normalized, the weighted average of those experts. Scored on the decision benchmark (overall
   and per game type) and in 40k cubeless money games per variant against bgsage (points per game).

    .venv/bin/python -m benchmarks.moe_schedule --dir nets/moe \
        --wait-for nets/money/512-256-state.json --value-from nets/money/512-256-state.json

Waits until every --wait-for state file says done, then builds the project (also the Python
extension, for the game benchmark) and copies the binaries to DIR/bin. Results in
DIR/results.txt, state in DIR/moe-state.json; rerunning resumes.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STALE = 1        # stale segments that end a phase
STALE_LAST = 2   # ... the last phase
MARGIN = 0.09    # millipoints (money scoring): the later-phase margin of train_schedule
VARIANTS = ["backgammon", "nackgammon"]
BINARIES = ["moe_collect", "moe_expert", "moe_gate", "decision_benchmark"]
MEASURE = "moe-types-money"   # the measuring set (data/decisions), money


class Run:
    def __init__(self, a):
        self.a = a
        self.dir = Path(a.dir).resolve()
        self.bin = self.dir / "bin"
        self.experts = self.dir / "experts"
        self.state_file = self.dir / "moe-state.json"
        self.state = json.loads(self.state_file.read_text()) if self.state_file.exists() else {}

    def log(self, line):
        print(line, flush=True)
        with open(self.dir / "results.txt", "a") as f:
            f.write(line + "\n")

    def save(self):
        self.state_file.write_text(json.dumps(self.state, indent=1))

    def names(self):
        names = [line.split()[1] for line in (ROOT / self.a.types).read_text().splitlines() if line.startswith("type ")]
        return names + ["generic"]

    def decisions(self, model, dataset="money"):
        """Overall (best %, loss) and per group {name: (n, loss)}, money scoring (millipoints):
        bgsage's game plans for the money set; for the measuring set the game types of the
        candidates (`--candidate-types`: a decision counts for a type when most of the positions
        after its moves are of that type, the positions an expert of the type evaluates)."""
        extra = ["--candidate-types", self.a.types] if dataset == MEASURE else []
        out = subprocess.run([str(self.bin / "decision_benchmark"), "--set", dataset, "--scoring", "money"] + extra + [str(model)],
                             cwd=ROOT, check=True, capture_output=True, text=True).stdout
        best, loss = re.search(r"^all\s+\d+\s+([\d.]+)%\s+([\d.]+)", out, re.M).groups()
        if dataset == MEASURE:
            groups = {g: (int(n), float(v)) for g, n, v in re.findall(r"^type (\S+)\s+(\d+)\s+[\d.]+%\s+([\d.]+)", out, re.M)}
        else:
            groups = {g: (int(n), float(v)) for g, n, v in re.findall(r"^(\S+)\s+(\d+)\s+[\d.]+%\s+([\d.]+)", out, re.M)
                      if g not in ("all", "group")}
        return float(best), float(loss), groups

    def games(self, model, note):
        scores = []
        for variant in VARIANTS:
            out = subprocess.run([sys.executable, "-m", "benchmarks.strength", "--net", str(model), "--variant", variant,
                                  "--games", str(self.a.games), "--bearoff", self.a.bearoff, "--note", note, "--money"],
                                 cwd=ROOT, capture_output=True, text=True)
            match = re.search(r"Points per game ([+-][\d.]+)", out.stdout + out.stderr)
            if not match:
                raise RuntimeError(out.stdout + out.stderr)
            scores.append(float(match.group(1)))
        return scores

    @staticmethod
    def groups_text(groups):
        return " ".join(f"{g[:4]} {v:.3f}" for g, (n, v) in groups.items())

    @staticmethod
    def types_text(types):
        return ", ".join(f"{t} {v:.3f}" for t, (n, v) in types.items())

    # --- stages -------------------------------------------------------------------------------

    def base(self):
        """Stage 0: the base money net (standard schedule), and the value net."""
        a, st = self.a, self.state
        base_dir = self.dir / "base"
        name = a.hidden.replace(",", "-")
        state_file = base_dir / f"{name}-state.json"
        if not (state_file.exists() and json.loads(state_file.read_text()).get("done")):
            self.log(f"--- stage 0: base money net {name} (standard schedule) in {base_dir}")
            with open(self.dir / "base.out", "a") as f:
                subprocess.run([sys.executable, "-m", "benchmarks.train_schedule", name, "--dir", str(base_dir),
                                "--objective", "money", "--hidden", a.hidden, "--seed", str(a.seed + 50)],
                               cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
        best = json.loads(state_file.read_text())["best"]
        shutil.copy2(best["raw"], self.dir / "base.bgnet")
        value = json.loads((ROOT / a.value_from).read_text())["best"]["avg"]
        shutil.copy2(value, self.dir / "value.bgnet")
        st["base"] = {"raw": best["raw"], "loss": best["loss"], "value": value}
        self.save()
        self.log(f"--- base net: {Path(best['raw']).name} (decision loss of its average {best['loss']:.3f}), copied to "
                 f"{self.dir / 'base.bgnet'}; value net {Path(value).name}, copied to {self.dir / 'value.bgnet'}")

    def setup(self):
        a, st = self.a, self.state
        if not a.no_build:
            subprocess.run(["make", "build"], cwd=ROOT, check=True, capture_output=True)
        subprocess.run(["cargo", "build", "--release", "--bins"], cwd=ROOT, check=True, capture_output=True)
        self.bin.mkdir(parents=True, exist_ok=True)
        for b in BINARIES:
            shutil.copy2(ROOT / "target/release" / b, self.bin / b)
        if not (ROOT / f"data/decisions/{MEASURE}.txt").exists():
            raise SystemExit("no measuring set: run moe_collect into DIR/measure and benchmarks/moe_measure.py --money first")
        init = self.dir / "base.bgnet"
        _, loss, groups = self.decisions(init)
        _, all_types, types = self.decisions(init, MEASURE)
        _, value_loss, _ = self.decisions(self.dir / "value.bgnet")
        _, value_all, value_types = self.decisions(self.dir / "value.bgnet", MEASURE)
        st.update(reference={"loss": loss, "groups": groups, "types": types, "types_all": all_types},
                  value={"loss": value_loss, "types": value_types, "types_all": value_all}, experts={}, done=False)
        self.save()
        self.log(f"--- mixture of money experts from {init.name} (decision loss {loss:.3f} millipoints: {self.groups_text(groups)}); "
                 f"{len(self.names())} experts; base net on the measuring set (bgsage 3-ply money, loss {all_types:.3f}): " +
                 ", ".join(f"{t} {v:.3f} (n={n})" for t, (n, v) in types.items()))
        self.log(f"    value net: decision loss {value_loss:.3f}, measuring set {value_all:.3f}: {self.types_text(value_types)}")

    def collect(self):
        a, st = self.a, self.state
        out = self.dir / "positions"
        subprocess.run([str(self.bin / "moe_collect"), "--types", a.types, "--out", str(out), "--per-type",
                        str(a.per_type), "--threads", str(a.threads), "--bearoff", a.bearoff,
                        "--exclude", str(self.dir / "measure")],
                       cwd=ROOT, check=True, capture_output=True)
        counts = (out / "counts.txt").read_text().strip().splitlines()
        st["collected"] = counts
        self.save()
        self.log(f"--- stage 1, collected: {counts[0]}; " + ", ".join(counts[1:]))

    def train_expert(self, name):
        a, st = self.a, self.state
        e = st["experts"].setdefault(name, {})
        ref_n, ref_loss = st["reference"]["types"][name]
        if "phases" not in e:
            e.update(phases=[], best={"raw": None, "avg": None, "games": 0, "loss": ref_loss}, done=False)
            self.save()
            self.log(f"--- stage 2, expert {name}: {ref_n} decisions of its type in the measuring set, "
                     f"base net loss {ref_loss:.3f}")
        segments = self.experts / "segments"
        segments.mkdir(parents=True, exist_ok=True)
        for phase, alpha in enumerate(a.alphas):
            if phase < len(e["phases"]) and e["phases"][phase]["done"]:
                continue
            if phase == len(e["phases"]):
                e["phases"].append({"alpha": alpha, "from": dict(e["best"]), "segments": [], "stale": 0, "done": False})
                self.save()
            p = e["phases"][phase]
            stale_limit = STALE_LAST if phase == len(a.alphas) - 1 else STALE
            while not p["done"]:
                games = p["from"]["games"] + (len(p["segments"]) + 1) * a.segment
                stem = segments / f"{name}-a{str(alpha)[2:]}-{games / 1e6:g}M"
                raw, avg = Path(f"{stem}.bgnet"), Path(f"{stem}-avg.bgnet")
                if not avg.exists():
                    work = self.experts / f"{name}.bgnet"
                    previous = p["segments"][-1]["raw"] if p["segments"] else p["from"]["raw"]
                    if previous is None:
                        work.unlink(missing_ok=True)     # starts from --init
                    else:
                        shutil.copy2(previous, work)
                    seed = a.seed + 1000 * self.names().index(name) + 100 * phase + len(p["segments"])
                    with open(f"{stem}.log", "w") as f:
                        subprocess.run([str(self.bin / "moe_expert"), "--types", a.types, "--expert", name,
                                        "--starts", str(self.dir / "positions" / f"{name}.pos"), "--dir", str(self.experts),
                                        "--init", str(self.dir / "base.bgnet"),
                                        "--value-net", str(self.dir / "value.bgnet"), "--games", str(a.segment), "--alpha", str(alpha),
                                        "--lambda", "0.7", "--threads", str(a.threads), "--seed", str(seed),
                                        "--report", str(min(500_000, a.segment)), "--average-last", str(a.segment // 8),
                                        "--bearoff", a.bearoff], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
                    shutil.copy2(work, raw)
                    shutil.copy2(self.experts / f"{name}-avg.bgnet", avg)
                _, overall, groups = self.decisions(avg)
                _, _, types = self.decisions(avg, MEASURE)
                n, loss = types[name]
                p["segments"].append({"raw": str(raw), "avg": str(avg), "games": games, "loss": loss, "overall": overall})
                phase_best = min([p["from"]["loss"]] + [s["loss"] for s in p["segments"][:-1]])
                p["stale"] = 0 if loss <= phase_best - MARGIN else p["stale"] + 1
                p["done"] = p["stale"] >= stale_limit
                if loss < e["best"]["loss"]:
                    e["best"] = {"raw": str(raw), "avg": str(avg), "games": games, "loss": loss}
                self.save()
                share = re.findall(r"(\d+)% of the moves on its own type", Path(f"{stem}.log").read_text())
                self.log(f"{name} alpha {alpha} {games / 1e6:g}M: own type loss {loss:.3f} (n={n}, base {ref_loss:.3f}, "
                         f"best {e['best']['loss']:.3f})   all decisions {overall:.3f} ({self.groups_text(groups)})" +
                         (f"   {share[-1]}% of moves on its type" if share else "") + ("   -> phase done" if p["done"] else ""))
        best = e["best"]["avg"] or str(self.dir / "base.bgnet")
        shutil.copy2(best, self.experts / f"{name}-best.bgnet")
        e["done"] = True
        self.save()
        self.log(f"--- expert {name} done: {Path(best).name}, own type loss {e['best']['loss']:.3f} (base {ref_loss:.3f})")

    def gateway(self):
        a, st = self.a, self.state
        out = subprocess.run([str(self.bin / "moe_gate"), "--types", a.types, "--dir", str(self.experts), "--suffix", "-best",
                              "--value-net", str(self.dir / "value.bgnet"), "--positions", str(a.gate_positions), "--threads", str(a.threads), "--hidden", "512",
                              "--threshold", str(a.threshold), "--bearoff", a.bearoff],
                             cwd=ROOT, check=True, capture_output=True, text=True).stdout
        (self.experts / "gate.log").write_text(out)
        self.log("--- stage 3, gateway (512 hidden), experts frozen:")
        for line in out.splitlines():
            self.log("    " + line)
        for kind in ["router", "moe"]:
            b, loss, groups = self.decisions(self.experts / f"{kind}.bgmoe")
            _, all_types, types = self.decisions(self.experts / f"{kind}.bgmoe", MEASURE)
            label = "game types pick the experts" if kind == "router" else f"gateway, threshold {a.threshold}"
            self.log(f"mixture ({label}): decisions best {b:.2f}%  loss {loss:.3f}  {self.groups_text(groups)}   "
                     f"[base {st['reference']['loss']:.3f}, value net {st['value']['loss']:.3f}]; measuring set "
                     f"{all_types:.3f} [base {st['reference']['types_all']:.3f}, value net {st['value']['types_all']:.3f}]: "
                     f"{self.types_text(types)}")
            st[kind] = {"loss": loss, "groups": groups, "types": types, "types_all": all_types}
        self.save()

    def run(self):
        a = self.a
        self.dir.mkdir(parents=True, exist_ok=True)
        while not all(Path(p).exists() and json.loads(Path(p).read_text()).get("done") for p in a.wait_for):
            time.sleep(300)
        st = self.state
        if "base" not in st:
            self.base()
        if "reference" not in st:
            self.setup()
        if "collected" not in st:
            self.collect()
        for name in self.names():
            if not st["experts"].get(name, {}).get("done"):
                self.train_expert(name)
        if "moe" not in st:
            self.gateway()
        if "final" not in st:
            model = self.experts / "moe.bgmoe"
            scores = self.games(model, "mixture of money experts (gateway)") if a.games else [float("nan")] * 2
            st["final"] = {"games": scores}
            st["done"] = True
            self.save()
            self.log(f"--- mixture of money experts with gateway: decision loss {st['moe']['loss']:.3f} "
                     f"(base {st['reference']['loss']:.3f}, value net {st['value']['loss']:.3f}); games (ppg) "
                     f"bg {scores[0]:+.4f}  nack {scores[1]:+.4f}  mean {sum(scores) / 2:+.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="nets/moe")
    ap.add_argument("--types", default="nets/moe/types.txt")
    ap.add_argument("--hidden", default="256,128", help="hidden layers of the experts (the base net)")
    ap.add_argument("--value-from", default="nets/money/512-256-state.json",
                    help="state file of the money run whose best net judges positions off an expert's type")
    ap.add_argument("--per-type", type=int, default=1_000_000, help="positions collected per expert (1M)")
    ap.add_argument("--alphas", type=lambda s: [float(x) for x in s.split(",")], default=[0.005, 0.0025])
    ap.add_argument("--segment", type=int, default=2_000_000, help="games per expert segment (2M)")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--bearoff", default="data/bearoff-6x15.bd")
    ap.add_argument("--gate-positions", type=int, default=200000)
    ap.add_argument("--threshold", type=float, default=0.25, help="gateway rating below which an expert is skipped")
    ap.add_argument("--games", type=int, default=40000, help="games per variant in the final benchmark (0: none)")
    ap.add_argument("--seed", type=int, default=9601)
    ap.add_argument("--wait-for", nargs="*", default=[], help="state files of runs to finish first")
    ap.add_argument("--no-build", action="store_true", help="don't rebuild the Python extension (for tests)")
    Run(ap.parse_args()).run()


if __name__ == "__main__":
    main()
