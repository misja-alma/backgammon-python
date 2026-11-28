from pathlib import Path
from typing import Optional

from backgammon.game.position import Position, Player
from backgammon.game.bgsage_analyzer import BgsageAnalyzer
from backgammon.game.python_analyzer import PythonAnalyzer
from backgammon.game.settings import Settings, Analyzer
import backgammon_rust

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class AnalyzerProxy:
    """Synchronous facade over the Python, Rust and bgsage engines.

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
               engine: Optional[Analyzer] = None, depth: Optional[int] = None, money: bool = False) -> float:
        """Equity of the position for player: in [-1, 1] at double match point, in [-3, 3] with
        money=True (cubeless money, gammons and backgammons count; Rust engine and bgsage)."""
        engine, depth = cls._resolve(engine, depth)
        if engine == Analyzer.RUST:
            backgammon_rust.set_money_scoring(money)
            return backgammon_rust.equity(
                *cls._rust_position_args(position),
                cls._player_to_int(player),
                depth,
            )
        if engine == Analyzer.BGSAGE:
            return BgsageAnalyzer.equity(position, player, depth, money)
        return PythonAnalyzer.equity(position, player, depth)

    @classmethod
    def best_move(cls, position: Position, die1: int, die2: int,
                  engine: Optional[Analyzer] = None, depth: Optional[int] = None, money: bool = False):
        """Best move for the side to move: (position after the move, equity for the mover). With
        money=True the Rust engine and bgsage play for cubeless money (gammons count); the Python
        engine always plays for double match point."""
        engine, depth = cls._resolve(engine, depth)
        if engine == Analyzer.RUST:
            backgammon_rust.set_money_scoring(money)
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
        if engine == Analyzer.BGSAGE:
            return BgsageAnalyzer.best_move(position, die1, die2, depth, money)
        return PythonAnalyzer.best_move(position, die1, die2, depth)

    @classmethod
    def analyze_moves(cls, position: Position, die1: int, die2: int, depth: int, money: bool = False) -> list:
        """Every move with the dice, best first, by the Rust engine searching depth roll levels:
        (position after the move, equity for the mover, [win, win gammon, win backgammon,
        lose gammon, lose backgammon] for the mover, NaN where the net doesn't know them, roll
        levels searched). Moves outside the search space (see set_rust_move_filter) come last,
        with their 1-ply values (0 roll levels)."""
        backgammon_rust.set_money_scoring(money)
        result = []
        for me_checkers, opp_checkers, turn, equity, probs, levels in backgammon_rust.analyze_moves(
                *cls._rust_position_args(position), die1, die2, depth):
            after = Position()
            after.me_checkers = list(me_checkers)
            after.opponent_checkers = list(opp_checkers)
            after.set_turn(Player.ME if turn == 1 else Player.OPPONENT)
            after.set_cube(position.cube_value, position.cube_owner)
            result.append((after, equity, list(probs), levels))
        return result

    @classmethod
    def analyze_position(cls, position: Position, depth: int, money: bool = False) -> tuple[float, list]:
        """The side on roll's equity and outcome probabilities (as in analyze_moves) before it
        rolls, by the Rust engine searching depth roll levels."""
        backgammon_rust.set_money_scoring(money)
        equity, probs = backgammon_rust.analyze_position(*cls._rust_position_args(position), depth)
        return equity, list(probs)

    @classmethod
    def rollout(cls, position: Position, first: int, games: int, seed: int, depth: int,
                money: bool = False) -> list[int]:
        """Games first..first + games of a cubeless rollout of the position by the Rust engine
        (both sides playing its move at depth roll levels, for money or double match point): the
        points the side on roll won in each game (±1, ±2, ±3). The first two rolls rotate through
        the 1296 combinations; later ones are random from seed (the same seed and game number give
        the same game)."""
        backgammon_rust.set_money_scoring(money)
        return backgammon_rust.rollout(*cls._rust_position_args(position), first, games, seed, depth)

    @staticmethod
    def play_rust_game(me_depth: int, opponent_depth: int, seed: int,
                       nackgammon: bool = False, money: bool = False) -> tuple[Player, Position, int]:
        """Play a complete Rust-engine-vs-Rust-engine game inside Rust, from the backgammon or
        (nackgammon=True) the nackgammon starting position, both engines playing for cubeless
        money with money=True, else for double match point.

        Cannot be interrupted. Returns (winner, final position, number of turns).
        """
        backgammon_rust.set_money_scoring(money)
        winner, me_checkers, opp_checkers, turns = backgammon_rust.play_game(me_depth, opponent_depth, seed, nackgammon)
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
    def set_rust_threads(threads: int) -> int:
        """Let the Rust engine search on this many threads (searches of at least one roll level;
        the results are the same as with one). Per thread like the cache: call this on the thread
        that runs the engine. Returns the number of threads in use."""
        return backgammon_rust.set_threads(threads)

    @staticmethod
    def set_rust_move_filter(margin: Optional[float], max_moves: Optional[int]) -> None:
        """The Rust engine's search space: at every move decision of a search of at least one roll
        level, only the moves within margin equity of the best by their 1-ply equity are searched
        deeper, at most max_moves of them (None: no limit; both None: every move). Per thread like
        the cache: call it on the engine's thread. Clears the cache when it changes."""
        backgammon_rust.set_move_filter(margin, max_moves)

    @staticmethod
    def rust_cache_size_mb() -> int:
        """Memory reserved for the Rust engine's cache on the calling thread, in MB."""
        return backgammon_rust.cache_size_mb()

    @staticmethod
    def set_rust_network(path: Optional[str]) -> str:
        """Use the neural net in the weight file at path (relative paths are taken from the
        project root) as the Rust engine's evaluator, or the placeholder evaluator for None.
        Like the cache, this is per thread: call it on the thread that runs the engine.
        Returns a description of the evaluator; raises ValueError for a missing or bad file."""
        if path is None:
            backgammon_rust.use_placeholder_evaluator()
        else:
            full_path = Path(path) if Path(path).is_absolute() else PROJECT_ROOT / path
            if not full_path.exists():
                raise ValueError(f"Network file {full_path} not found")
            backgammon_rust.load_network(str(full_path))
        return backgammon_rust.evaluator_name()

    @staticmethod
    def set_rust_bearoff(path: Optional[str]) -> str:
        """Use the bearoff database file at path (relative paths are taken from the project root)
        for the Rust engine's bearoff positions, or none for None. Per thread, like the net.
        Returns a description; raises ValueError for a missing or bad file."""
        if path is None:
            return backgammon_rust.load_bearoff(None)
        full_path = Path(path) if Path(path).is_absolute() else PROJECT_ROOT / path
        if not full_path.exists():
            raise ValueError(f"Bearoff database {full_path} not found (build it with 'make bearoff')")
        return backgammon_rust.load_bearoff(str(full_path))

    @classmethod
    def clear_caches(cls) -> None:
        PythonAnalyzer.clear_cache()
        backgammon_rust.clear_cache()
