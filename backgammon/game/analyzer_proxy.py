from typing import Optional

from backgammon.game.position import Position, Player
from backgammon.game.python_analyzer import PythonAnalyzer
from backgammon.game.settings import Settings, Analyzer
import backgammon_rust


class AnalyzerProxy:
    """Synchronous facade over the Python and Rust engines.

    Each call can name the engine and search depth explicitly (as game players do); when they
    are omitted, the global Settings are used (as the Analyze menu does).
    """

    @staticmethod
    def _player_to_int(player: Player) -> int:
        return 1 if player == Player.ME else 0

    @classmethod
    def _rust_position_args(cls, position: Position) -> tuple:
        """Marshal a Position into the primitive arguments the Rust engine expects."""
        cube_owner = None if position.cube_owner is None else cls._player_to_int(position.cube_owner)
        return (
            position.me_checkers,
            position.opponent_checkers,
            cls._player_to_int(position.get_turn()),
            position.cube_value,
            cube_owner,
        )

    @staticmethod
    def _resolve(engine: Optional[Analyzer], depth: Optional[int]) -> tuple[Analyzer, int]:
        return (engine if engine is not None else Settings.analyzer,
                depth if depth is not None else Settings.analysis_depth)

    @classmethod
    def equity(cls, position: Position, player: Player,
               engine: Optional[Analyzer] = None, depth: Optional[int] = None) -> float:
        """Equity of the position for player, in [-1, 1]."""
        engine, depth = cls._resolve(engine, depth)
        if engine == Analyzer.RUST:
            return backgammon_rust.equity(
                *cls._rust_position_args(position),
                cls._player_to_int(player),
                depth,
            )
        return PythonAnalyzer.equity(position, player, depth)

    @classmethod
    def best_move(cls, position: Position, die1: int, die2: int,
                  engine: Optional[Analyzer] = None, depth: Optional[int] = None):
        """Best move for the side to move: (position after the move, equity for the mover)."""
        engine, depth = cls._resolve(engine, depth)
        if engine == Analyzer.RUST:
            result = backgammon_rust.best_move(
                *cls._rust_position_args(position),
                die1,
                die2,
                depth,
            )
            if result is None:
                return None
            me_checkers, opp_checkers, turn, equity = result
            new_pos = Position()
            new_pos.me_checkers = list(me_checkers)
            new_pos.opponent_checkers = list(opp_checkers)
            new_pos.set_turn(Player.ME if turn == 1 else Player.OPPONENT)
            new_pos.set_cube(position.cube_value, position.cube_owner)
            return new_pos, equity
        return PythonAnalyzer.best_move(position, die1, die2, depth)

    @staticmethod
    def play_rust_game(me_depth: int, opponent_depth: int, seed: int) -> tuple[Player, Position, int]:
        """Play a complete Rust-engine-vs-Rust-engine game inside Rust.

        Cannot be interrupted. Returns (winner, final position, number of turns).
        """
        winner, me_checkers, opp_checkers, turns = backgammon_rust.play_game(me_depth, opponent_depth, seed)
        position = Position()
        position.me_checkers = list(me_checkers)
        position.opponent_checkers = list(opp_checkers)
        winner = Player.ME if winner == 1 else Player.OPPONENT
        position.set_turn(winner.other_player())  # the turn passed after the winning move
        return winner, position, turns

    @staticmethod
    def set_rust_cache_size(megabytes: int) -> None:
        """Replace the Rust engine's cache with an empty one of this size. The cache is per
        thread, so call this on the thread that runs the engine (the AnalysisWorker)."""
        backgammon_rust.set_cache_size(megabytes)

    @staticmethod
    def rust_cache_size_mb() -> int:
        """Memory reserved for the Rust engine's cache on the calling thread, in MB."""
        return backgammon_rust.cache_size_mb()

    @classmethod
    def clear_caches(cls) -> None:
        PythonAnalyzer.clear_cache()
        backgammon_rust.clear_cache()
