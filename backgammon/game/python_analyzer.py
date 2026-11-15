import copy

from backgammon.game.evaluator import Evaluator
from backgammon.game.position import Position, Player
from backgammon.game.utils import Utils


class PythonAnalyzer:
    """Expectimax search over dice rolls. Reference implementation of the Rust engine.

    Equity is computed for the side to move and negated when viewed from the other side.
    `depth` is the number of remaining roll levels: at depth 0 the Evaluator gives the answer.
    """

    dice_rolls = []
    for d1 in range(1, 7):
        for d2 in range(d1, 7):
            dice_rolls.append((d1, d2))

    # (position hash, remaining depth) → equity for the side to move.
    # Persists across calls so the cache accumulates over a whole session.
    _cache: dict[tuple[int, int], float] = {}

    @classmethod
    def clear_cache(cls) -> None:
        cls._cache.clear()

    @staticmethod
    def equity(position: Position, player: Player, max_depth: int = 1) -> float:
        """Equity of the position for player, before the side to move has rolled."""
        equity = PythonAnalyzer._equity_for_turn(position, max_depth)
        return equity if player == position.get_turn() else -equity

    @staticmethod
    def best_move(position: Position, die1: int, die2: int, max_depth: int = 1):
        """Best move for the side to move with the given dice.

        Returns (position after the move, equity for the player who moved).
        """
        return PythonAnalyzer._best_move(position, die1, die2, max_depth)

    @staticmethod
    def _equity_for_turn(position: Position, depth: int) -> float:
        if depth <= 0 or Evaluator.is_game_over(position):
            return Evaluator.evaluate(position)

        key = (position.hash_code(), depth)
        cached = PythonAnalyzer._cache.get(key)
        if cached is not None:
            return cached

        total = 0.0
        for die1, die2 in PythonAnalyzer.dice_rolls:
            _, equity = PythonAnalyzer._best_move(position, die1, die2, depth - 1)
            total += (2 if die1 != die2 else 1) * equity

        equity = total / 36
        PythonAnalyzer._cache[key] = equity
        return equity

    @staticmethod
    def _best_move(position: Position, die1: int, die2: int, depth: int):
        best = None
        seen = set()
        for half_moves in Utils.valid_possible_moves(position, die1, die2):
            child = Utils.apply_move(position, half_moves)
            # Many move sequences lead to the same position: evaluate each position only once.
            key = (tuple(child.me_checkers), tuple(child.opponent_checkers))
            if key in seen:
                continue
            seen.add(key)
            # The child has the opponent to move, so negate its equity.
            equity = -PythonAnalyzer._equity_for_turn(child, depth)
            if best is None or equity > best[1]:
                best = child, equity

        if best is None:
            after_pass = copy.deepcopy(position)
            after_pass.switch_turn()
            return after_pass, -PythonAnalyzer._equity_for_turn(after_pass, depth)

        return best
