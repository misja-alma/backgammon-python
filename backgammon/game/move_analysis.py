"""Analysis of a position or of a roll's moves, as shown in the analysis panel (like eXtreme
Gammon's move list): per move its equity and outcome probabilities, best first.

Probabilities are for the player in question: win, win gammon, win backgammon, lose gammon,
lose backgammon (cumulative: a gammon includes the backgammons). A double match point net only
knows the winning chance; the others are then NaN ("unknown").
"""
import copy
import math
import os
import random
from dataclasses import dataclass, field
from typing import Optional

from backgammon.game.analyzer_proxy import AnalyzerProxy
from backgammon.game.game import Move, format_move
from backgammon.game.position import Position, Player
from backgammon.game.utils import Utils


@dataclass
class CandidateMove:
    move: Optional[Move]          # None when no legal move of ours leads to the position
    position: Position            # after the move
    equity: float                 # for the mover
    probs: list[float]            # for the mover, see the module docstring
    error: Optional[float] = None  # rollout: standard error of the equity
    before: Optional[Position] = None   # the position before the move (for marking hits)
    levels: Optional[int] = None   # roll levels searched, when fewer than the analysis's (outside
                                   # the search space: 1-ply)

    @property
    def notation(self) -> str:
        """Standard notation, highest starting point first, hits marked: e.g. '24/18 13/9*'."""
        if self.move is None:
            return "?"
        return format_move(self.move, self.before, highest_first=True)


@dataclass
class Analysis:
    position: Position            # analyzed position (a copy), before the move
    dice: Optional[tuple[int, int]]   # None: the position itself, before the roll
    depth: int
    money: bool                   # cubeless money (gammons count) or double match point
    equity: Optional[float] = None          # position analysis: for the side on roll
    probs: Optional[list[float]] = None     # position analysis: for the side on roll
    moves: list[CandidateMove] = field(default_factory=list)   # move analysis, best first
    rollout_games: Optional[int] = None     # a rollout of this many games (per move), else a search
    error: Optional[float] = None           # position rollout: standard error of the equity

    @property
    def player(self) -> Player:
        return self.position.get_turn()

    def matches(self, position: Position, dice: Optional[tuple[int, int]]) -> bool:
        """Whether this analysis is of this position (and dice)."""
        return (dice == self.dice and position.get_turn() == self.position.get_turn() and
                position.me_checkers == self.position.me_checkers and
                position.opponent_checkers == self.position.opponent_checkers)


def _moves_by_result(position: Position, die1: int, die2: int) -> dict:
    """Legal moves keyed by the checkers they lead to."""
    result = {}
    for move in sorted(Utils.valid_possible_moves(position, die1, die2)):
        after = Utils.apply_move(position, move)
        result.setdefault((tuple(after.me_checkers), tuple(after.opponent_checkers)), move)
    return result


def analyze(position: Position, dice: Optional[tuple[int, int]], depth: int, money: bool) -> Analysis:
    """Analyze with the Rust engine (its current net): the moves with these dice, or the position
    itself when dice is None. Runs the search: call it off the UI thread."""
    snapshot = copy.deepcopy(position)
    analysis = Analysis(snapshot, dice, depth, money)
    if dice is None:
        analysis.equity, analysis.probs = AnalyzerProxy.analyze_position(snapshot, depth, money)
        return analysis
    moves = _moves_by_result(snapshot, *dice)
    for after, equity, probs, levels in AnalyzerProxy.analyze_moves(snapshot, *dice, depth, money):
        key = (tuple(after.me_checkers), tuple(after.opponent_checkers))
        move = moves.get(key, () if not moves else None)
        analysis.moves.append(CandidateMove(move, after, equity, probs, before=snapshot,
                                            levels=levels if levels < depth else None))
    return analysis


@dataclass
class RolloutProgress:
    """How far a rollout is; written by the worker thread, read by the UI, which can ask it to
    stop."""
    moves: int = 1         # moves rolled out (1 for a position)
    game: int = 0          # games finished, for every move
    games: int = 0         # games asked for, per move
    stop: bool = False     # set by the UI: finish after the engine call in progress


ROLLOUT_BATCH = 60   # games per engine call at 1-ply, between progress updates and stop checks


def _rollout_batch(depth: int) -> int:
    """Games per engine call: at 2-ply and more a game takes long, so one per engine thread,
    to keep the progress moving and Stop quick."""
    if depth == 0:
        return ROLLOUT_BATCH
    from backgammon.game.settings import Settings
    return Settings.rust_threads or os.cpu_count() or 1


def _rollout_stats(points: list[int], money: bool) -> tuple[float, list[float], float]:
    """(equity, probabilities, standard error of the equity) of rollout games won by these
    points, for the side they're counted for."""
    n = len(points)
    probs = [sum(p > 0 for p in points) / n, sum(p >= 2 for p in points) / n,
             sum(p == 3 for p in points) / n, sum(p <= -2 for p in points) / n,
             sum(p == -3 for p in points) / n]
    if not money:
        # Playing for double match point, a side that can't win any more doesn't care how badly
        # it loses, so the rollout's gammons say nothing: unknown, as in a search.
        probs[1:] = [math.nan] * 4
    values = [p if money else (1 if p > 0 else -1) for p in points]
    equity = sum(values) / n
    variance = sum((v - equity) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
    return equity, probs, math.sqrt(variance / n)


def position_key(position: Position) -> tuple:
    """Identifies a position after a move (the checkers), e.g. to name candidates for a rollout."""
    return tuple(position.me_checkers), tuple(position.opponent_checkers)


def rollout(position: Position, dice: Optional[tuple[int, int]], games: int, depth: int, money: bool,
            progress: RolloutProgress, candidates: Optional[int] = None,
            preferred: Optional[list[tuple]] = None) -> Optional[Analysis]:
    """Roll out, with the Rust engine playing at depth roll levels, the position itself (dice
    None) or the moves with the dice, games games each, cubeless. The moves: candidates of them
    (all when None), taken first from preferred (position_key of positions after moves, in
    order: e.g. the moves selected in the analysis, then the others by the analysis's ranking)
    and then by their 1-ply equity. The first two rolls after the
    position (or move) rotate through the 1296 combinations; every move gets the same dice (the
    same seed), so their differences are measured more precisely.

    The moves advance together, a batch of games at a time, so that when progress.stop is set
    every move has the same number of games: the result then covers the games played so far
    (its rollout_games), or is None when not a single batch finished. Updates progress as it
    goes; call it off the UI thread."""
    snapshot = copy.deepcopy(position)
    seed = random.getrandbits(63)
    if dice is None:
        analysis = Analysis(snapshot, None, depth, money)
        starts = [snapshot]
    else:
        analysis = analyze(snapshot, dice, 0, money)   # every move, best first at 1-ply
        analysis.depth = depth
        if preferred:
            rank = {key: i for i, key in enumerate(preferred)}
            analysis.moves.sort(key=lambda m: rank.get(position_key(m.position), len(rank)))  # stable
        if candidates is not None:
            analysis.moves = analysis.moves[:max(1, candidates)]
        starts = [m.position for m in analysis.moves]
    progress.moves, progress.game, progress.games = len(starts), 0, games

    points = [[] for _ in starts]
    done = 0
    while done < games and not progress.stop:
        batch = min(_rollout_batch(depth), games - done)
        results = []
        for start in starts:
            if progress.stop:
                break
            results.append(AnalyzerProxy.rollout(start, done, batch, seed, depth, money))
        if len(results) < len(starts):
            break   # stopped within the batch: it isn't counted
        for move_points, result in zip(points, results):
            move_points += result
        done += batch
        progress.game = done
    if done == 0:
        return None

    analysis.rollout_games = done
    if dice is None:
        analysis.equity, analysis.probs, analysis.error = _rollout_stats(points[0], money)
        return analysis
    for candidate, move_points in zip(analysis.moves, points):
        # After the move the opponent is on roll: its points are the mover's losses.
        candidate.equity, candidate.probs, candidate.error = _rollout_stats([-p for p in move_points], money)
    analysis.moves.sort(key=lambda m: -m.equity)
    return analysis


def format_probs(probs: list[float]) -> str:
    """'50.4 16.3 1.8' for win, gammon, backgammon (percentages; '-' when unknown)."""
    return " ".join("-" if math.isnan(p) else f"{100 * p:.1f}" for p in probs)
