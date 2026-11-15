from backgammon.game.position import Position

CHECKERS_PER_PLAYER = 15


class Evaluator:
    """Static evaluation of a position, used by the engines where the search stops.

    Equity is always from the point of view of the side to move: +1 is a certain win, -1 a certain
    loss. Gammons and backgammons are not scored yet, so equity is in [-1, 1].

    For positions that are not finished this currently returns 0 (no opinion); a real evaluation
    will replace that later.
    """

    WIN = 1.0
    LOSS = -1.0
    UNKNOWN = 0.0

    @staticmethod
    def is_game_over(position: Position) -> bool:
        turn = position.get_turn()
        return (position.get_checkers(turn, 0) == CHECKERS_PER_PLAYER or
                position.get_checkers(turn.other_player(), 0) == CHECKERS_PER_PLAYER)

    @staticmethod
    def evaluate(position: Position) -> float:
        turn = position.get_turn()
        if position.get_checkers(turn, 0) == CHECKERS_PER_PLAYER:
            return Evaluator.WIN
        if position.get_checkers(turn.other_player(), 0) == CHECKERS_PER_PLAYER:
            return Evaluator.LOSS
        return Evaluator.UNKNOWN

    @staticmethod
    def equity_to_winning_chance(equity: float) -> float:
        """Convert equity to winning probability. Only valid while gammons are not scored."""
        return (equity + 1.0) / 2.0
