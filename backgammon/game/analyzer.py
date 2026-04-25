from backgammon.game.position import Position, Player
from backgammon.game.python_analyzer import PythonAnalyzer
import backgammon_rust


class Analyzer:
    """Proxy that dispatches to PythonAnalyzer or RustAnalyzer based on use_rust."""

    use_rust: bool = True

    @classmethod
    def winning_chances(cls, position: Position, player: Player) -> float:
        if cls.use_rust:
            return backgammon_rust.winning_chances(
                position.black_checkers,
                position.white_checkers,
                1 if position.get_turn() == Player.ME else 0,
                1 if player == Player.ME else 0,
            )
        return PythonAnalyzer.winning_chances(position, player)

    @classmethod
    def best_move(cls, position: Position, die1: int, die2: int):
        if cls.use_rust:
            result = backgammon_rust.best_move(
                position.black_checkers,
                position.white_checkers,
                1 if position.get_turn() == Player.ME else 0,
                die1,
                die2,
            )
            if result is None:
                return None
            me_checkers, opp_checkers, turn, pw = result
            new_pos = Position()
            new_pos.black_checkers = list(me_checkers)
            new_pos.white_checkers = list(opp_checkers)
            new_pos.set_turn(Player.ME if turn == 1 else Player.OPPONENT)
            return new_pos, pw
        return PythonAnalyzer.best_move(position, die1, die2)

    @classmethod
    def find_known_pw(cls, position: Position, player: Player):
        if cls.use_rust:
            pw = cls.winning_chances(position, player)
            if pw in (0.0, 1.0):
                return pw
            return None
        return PythonAnalyzer.find_known_pw(position, player)
