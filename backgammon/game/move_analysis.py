"""Analysis of a position or of a roll's moves, as shown in the analysis panel (like eXtreme
Gammon's move list): per move its equity and outcome probabilities, best first.

Probabilities are for the player in question: win, win gammon, win backgammon, lose gammon,
lose backgammon (cumulative: a gammon includes the backgammons). A double match point net only
knows the winning chance; the others are then NaN ("unknown").
"""
import copy
import math
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

    @property
    def notation(self) -> str:
        """Standard notation, highest starting point first, e.g. '24/18 13/9'."""
        if self.move is None:
            return "?"
        return format_move(tuple(sorted(self.move, key=lambda half: (-half[0], -half[1]))))


@dataclass
class Analysis:
    position: Position            # analyzed position (a copy), before the move
    dice: Optional[tuple[int, int]]   # None: the position itself, before the roll
    depth: int
    money: bool                   # cubeless money (gammons count) or double match point
    equity: Optional[float] = None          # position analysis: for the side on roll
    probs: Optional[list[float]] = None     # position analysis: for the side on roll
    moves: list[CandidateMove] = field(default_factory=list)   # move analysis, best first

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
    for after, equity, probs in AnalyzerProxy.analyze_moves(snapshot, *dice, depth, money):
        key = (tuple(after.me_checkers), tuple(after.opponent_checkers))
        move = moves.get(key, () if not moves else None)
        analysis.moves.append(CandidateMove(move, after, equity, probs))
    return analysis


def format_probs(probs: list[float]) -> str:
    """'50.4 16.3 1.8' for win, gammon, backgammon (percentages; '-' when unknown)."""
    return " ".join("-" if math.isnan(p) else f"{100 * p:.1f}" for p in probs)
