"""Strength benchmark: one of our engines against bgsage.

    .venv/bin/python -m benchmarks.strength                # 10000 games, Rust depth 0 vs bgsage 1-ply
    .venv/bin/python -m benchmarks.strength --games 2000 --depth 1
    .venv/bin/python -m benchmarks.strength --variant nackgammon

Games are played in pairs with the same seed, our engine taking ME in one game and OPPONENT in the
other, so both sides get the same dice until the moves diverge. That cancels part of the luck and
makes the result more precise than the same number of unpaired games. With the same arguments the
benchmark is fully reproducible: compare evaluator changes using the same --seed.

Every run is appended to benchmarks/strength_results.csv (with the git commit) to track progress.
"""
import argparse
import csv
import datetime
import math
import multiprocessing
import os
import random
import subprocess
import sys
import time
from pathlib import Path

from backgammon.game.game import Game, GameConfig, PlayerConfig, PlayerType
from backgammon.game.position import GameVariant, Player
from backgammon.game.bgsage_analyzer import BgsageAnalyzer
from backgammon.game.analyzer_proxy import AnalyzerProxy
from backgammon.game.settings import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_FILE = PROJECT_ROOT / "benchmarks" / "strength_results.csv"
ENGINES = {"rust": PlayerType.RUST_ENGINE, "python": PlayerType.PYTHON_ENGINE}


VARIANTS = {variant.name.lower(): variant for variant in GameVariant}


def play_game(index: int, engine: PlayerType, depth: int, bgsage_depth: int, seed: int,
              variant: GameVariant = GameVariant.BACKGAMMON) -> tuple:
    """Game index: pair index // 2 decides the dice, index % 2 which side our engine plays.
    Returns (index, whether our engine won, number of turns)."""
    ours, theirs = PlayerConfig(engine, depth), PlayerConfig(PlayerType.BGSAGE_ENGINE, bgsage_depth)
    our_side = Player.ME if index % 2 == 0 else Player.OPPONENT
    config = GameConfig(ours, theirs) if our_side == Player.ME else GameConfig(theirs, ours)
    config.variant = variant
    game = Game(config, random.Random(f"{seed}-{index // 2}"))
    game.start()
    game.play_to_end()
    return index, game.winner == our_side, game.turns


def init_worker(network, bearoff=None) -> None:
    """Pool initializer: load the Rust engine's evaluator (and bearoff database) in this worker
    process (the games run on the worker's main thread, where the engine's per-thread state lives)."""
    AnalyzerProxy.set_rust_network(network)
    AnalyzerProxy.set_rust_bearoff(bearoff)


def _play_game_args(args: tuple) -> tuple:
    return play_game(*args)


def git_commit() -> str:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=PROJECT_ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def summarize(results: list) -> dict:
    """Win rate with its standard error, computed over game pairs (the pairs are independent,
    the two games within a pair are not)."""
    wins = [won for _, won, _ in sorted(results)]
    pair_scores = [(wins[i] + wins[i + 1]) / 2 for i in range(0, len(wins) - 1, 2)]
    n = len(pair_scores)
    mean = sum(pair_scores) / n
    variance = sum((s - mean) ** 2 for s in pair_scores) / (n - 1) if n > 1 else 0.0
    return {"games": 2 * n, "wins": sum(wins[:2 * n]), "win_rate": mean, "std_error": math.sqrt(variance / n),
            "avg_turns": sum(t for _, _, t in results) / len(results)}


RESULT_COLUMNS = ["date", "commit", "engine", "depth", "bgsage_ply", "games", "seed",
                  "win_rate", "std_error", "seconds", "note", "evaluator", "variant"]


def add_variant_column() -> None:
    """Results files from before the variant column: add it, 'backgammon' for the old rows."""
    if not RESULTS_FILE.exists():
        return
    with RESULTS_FILE.open(newline="") as f:
        rows = list(csv.reader(f))
    if not rows or "variant" in rows[0]:
        return
    rows = [rows[0] + ["variant"]] + [row + ["backgammon"] for row in rows[1:]]
    with RESULTS_FILE.open("w", newline="") as f:
        csv.writer(f).writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=int, default=10000, help="number of games, rounded up to even (10000)")
    parser.add_argument("--engine", choices=ENGINES, default="rust", help="our engine (rust)")
    parser.add_argument("--depth", type=int, default=0, help="our engine's depth (0)")
    parser.add_argument("--bgsage-depth", type=int, default=0, help="bgsage's depth; 0 is 1-ply (0)")
    parser.add_argument("--variant", choices=VARIANTS, default="backgammon",
                        help="starting position (backgammon)")
    parser.add_argument("--seed", type=int, default=1, help="dice seed (1)")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="parallel processes (all cores)")
    parser.add_argument("--net", default=Settings.rust_network,
                        help=f"Rust engine's net file, or 'none' for the placeholder ({Settings.rust_network})")
    default_bearoff = (Settings.rust_bearoff_db
                       if Settings.rust_bearoff_db and (PROJECT_ROOT / Settings.rust_bearoff_db).exists() else None)
    parser.add_argument("--bearoff", default=default_bearoff,
                        help=f"Rust engine's bearoff database, or 'none' ({default_bearoff}; since 2026-11-22, "
                             "earlier results are without; bgsage always uses its own)")
    parser.add_argument("--note", default="", help="free text stored with the result, e.g. what changed")
    parser.add_argument("--no-save", action="store_true", help=f"don't append to {RESULTS_FILE.name}")
    args = parser.parse_args()

    if not BgsageAnalyzer.is_available():
        sys.exit("bgsage is not installed; run 'make bgsage' first")
    network = None if args.net in (None, "none") else args.net
    evaluator = AnalyzerProxy.set_rust_network(network)   # fails early for a missing or bad file
    bearoff = None if args.bearoff in (None, "none") else args.bearoff
    bearoff_description = AnalyzerProxy.set_rust_bearoff(bearoff)
    games = args.games + args.games % 2
    engine = ENGINES[args.engine]
    variant = VARIANTS[args.variant]
    tasks = [(i, engine, args.depth, args.bgsage_depth, args.seed, variant) for i in range(games)]
    print(f"{args.engine} engine depth {args.depth} vs bgsage {args.bgsage_depth + 1}-ply, {args.variant}: "
          f"{games} games on {args.workers} processes, seed {args.seed}")
    if args.engine == "rust":
        print(f"Evaluator: {evaluator}; {bearoff_description}")

    start = time.time()
    results = []
    # spawn: each worker starts its own bgsage subprocess and Rust engine state.
    with multiprocessing.get_context("spawn").Pool(args.workers, init_worker, (network, bearoff)) as pool:
        for result in pool.imap_unordered(_play_game_args, tasks, chunksize=20):
            results.append(result)
            if len(results) % 500 == 0 or len(results) == games:
                wins = sum(won for _, won, _ in results)
                print(f"  {len(results):6d}/{games}  win rate {wins / len(results):6.2%}  "
                      f"{time.time() - start:6.0f}s", flush=True)
    seconds = time.time() - start

    summary = summarize(results)
    rate, se = summary["win_rate"], summary["std_error"]
    print(f"\nWin rate {rate:.2%} ± {1.96 * se:.2%} (95% CI), {summary['wins']}/{summary['games']} games, "
          f"{summary['avg_turns']:.1f} turns per game, {seconds:.0f}s")

    if not args.no_save:
        add_variant_column()
        is_new = not RESULTS_FILE.exists()
        with RESULTS_FILE.open("a", newline="") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(RESULT_COLUMNS)
            writer.writerow([datetime.datetime.now().isoformat(timespec="seconds"), git_commit(), args.engine,
                             args.depth, args.bgsage_depth + 1, summary["games"], args.seed,
                             f"{rate:.4f}", f"{se:.4f}", round(seconds), args.note,
                             evaluator if args.engine == "rust" else "", args.variant])
        print(f"Saved to {RESULTS_FILE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
