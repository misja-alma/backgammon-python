import copy
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from backgammon.game.analyzer_proxy import AnalyzerProxy
from backgammon.game.evaluator import CHECKERS_PER_PLAYER
from backgammon.game.position import Position, Player
from backgammon.game.settings import Analyzer
from backgammon.game.utils import Utils

HalfMove = tuple[int, int]   # (start point, target point) from the mover's perspective
Move = tuple[HalfMove, ...]  # a full move; the empty tuple means "no legal move, pass"


class PlayerType(Enum):
    HUMAN = "Human"
    PYTHON_ENGINE = "Python engine"
    RUST_ENGINE = "Rust engine"

    @property
    def engine(self) -> Optional[Analyzer]:
        return {PlayerType.PYTHON_ENGINE: Analyzer.PYTHON,
                PlayerType.RUST_ENGINE: Analyzer.RUST}.get(self)

    @property
    def is_engine(self) -> bool:
        return self.engine is not None


@dataclass
class PlayerConfig:
    type: PlayerType = PlayerType.HUMAN
    depth: int = 1  # search depth, only used for engines


@dataclass
class GameConfig:
    me: PlayerConfig = field(default_factory=PlayerConfig)
    opponent: PlayerConfig = field(default_factory=lambda: PlayerConfig(PlayerType.RUST_ENGINE))
    show_engine_game: bool = True  # only relevant when both players are engines

    def player(self, player: Player) -> PlayerConfig:
        return self.me if player == Player.ME else self.opponent

    @property
    def all_engines(self) -> bool:
        return self.me.type.is_engine and self.opponent.type.is_engine


def player_name(player: Player) -> str:
    return "ME" if player == Player.ME else "OPPONENT"


def format_move(move: Move) -> str:
    """Format a move in standard notation, e.g. '8/5 6/5', 'bar/22', '3/off'."""
    if not move:
        return "no legal move"

    def point(p: int) -> str:
        return "bar" if p == 25 else "off" if p == 0 else str(p)

    return " ".join(f"{point(start)}/{point(target)}" for start, target in move)


class Game:
    """A single game of backgammon (no doubling cube, no gammons). Contains no UI code.

    Flow: start() does the opening roll; then, as long as the game is not over, the side to move
    plays a move with the current dice via play() (or play_position() for engines), after which
    the dice for the next turn are rolled automatically.
    """

    def __init__(self, config: GameConfig, rng: Optional[random.Random] = None,
                 log: Callable[[str], None] = lambda message: None):
        self.config = config
        self.rng = rng or random.Random()
        self.log = log
        self.position = Position()
        self.position.setup_starting_position()
        self.dice: Optional[tuple[int, int]] = None
        self.winner: Optional[Player] = None
        self.turns = 0
        self._legal_moves: Optional[set[Move]] = None

    # --- state ---

    @property
    def is_over(self) -> bool:
        return self.winner is not None

    @property
    def runs_in_rust(self) -> bool:
        """A hidden game between two Rust engines is played entirely inside Rust (fast, but it
        can't be stopped once started)."""
        config = self.config
        return (not config.show_engine_game and
                config.me.type == PlayerType.RUST_ENGINE and
                config.opponent.type == PlayerType.RUST_ENGINE)

    @property
    def current_player(self) -> Player:
        return self.position.get_turn()

    @property
    def current_player_config(self) -> PlayerConfig:
        return self.config.player(self.current_player)

    def legal_moves(self) -> set[Move]:
        """All legal moves with the current dice; empty when the player must pass."""
        if self._legal_moves is None:
            self._legal_moves = Utils.valid_possible_moves(self.position, *self.dice)
        return self._legal_moves

    # --- flow ---

    def _roll_die(self) -> int:
        return self.rng.randint(1, 6)

    def start(self) -> None:
        """Opening roll: each player rolls one die until they differ; the higher die moves first
        and plays both dice."""
        while True:
            me_die, opponent_die = self._roll_die(), self._roll_die()
            if me_die != opponent_die:
                break
            self.log(f"Opening roll: both players roll {me_die}, rolling again")

        starter = Player.ME if me_die > opponent_die else Player.OPPONENT
        self.position.set_turn(starter)
        self._set_dice((me_die, opponent_die) if starter == Player.ME else (opponent_die, me_die))
        self.log(f"Opening roll: ME rolls {me_die}, OPPONENT rolls {opponent_die}. "
                 f"{player_name(starter)} starts with {self.dice[0]}-{self.dice[1]}")

    def _set_dice(self, dice: tuple[int, int]) -> None:
        self.dice = dice
        self._legal_moves = None

    def play(self, move: Move) -> None:
        """Play a full move for the side to move. Pass the empty tuple when no move is legal."""
        if self.is_over:
            raise ValueError("The game is over")
        legal = self.legal_moves()
        if (move or legal) and move not in legal:
            raise ValueError(f"Illegal move {format_move(move)} with {self.dice}")

        mover = self.current_player
        self.turns += 1
        if move:
            self.position = Utils.apply_move(self.position, move)
        else:
            self.position = copy.deepcopy(self.position)
            self.position.switch_turn()
        self.log(f"{player_name(mover)} rolls {self.dice[0]}-{self.dice[1]}: {format_move(move)}")

        if self.position.get_checkers(mover, 0) == CHECKERS_PER_PLAYER:
            self.winner = mover
            self.dice = None
            self._legal_moves = None
            self.log(f"{player_name(mover)} wins")
        else:
            self._set_dice((self._roll_die(), self._roll_die()))

    def find_move(self, position_after: Position) -> Move:
        """The legal move that leads to position_after (e.g. as chosen by an engine)."""
        legal = self.legal_moves()
        if not legal:
            return ()
        for move in legal:
            result = Utils.apply_move(self.position, move)
            if (result.me_checkers == position_after.me_checkers and
                    result.opponent_checkers == position_after.opponent_checkers):
                return move
        raise ValueError("The given position is not reachable with a legal move")

    def engine_move(self) -> Move:
        """Let the engine configured for the side to move choose a move. Can be slow: call it
        off the UI thread. Does not change the game; pass the result to play()."""
        config = self.current_player_config
        if not config.type.is_engine:
            raise ValueError("The side to move is not an engine")
        if not self.legal_moves():
            return ()
        position_after, _ = AnalyzerProxy.best_move(self.position, *self.dice,
                                                    engine=config.type.engine, depth=config.depth)
        return self.find_move(position_after)

    def play_in_rust(self, seed: int) -> None:
        """Play the whole game inside Rust, including the opening roll (see runs_in_rust).
        Cannot be interrupted. Call on a fresh game (without start()), off the UI thread."""
        if not self.runs_in_rust:
            raise ValueError("Only a hidden Rust engine vs Rust engine game can run in Rust")
        winner, position, turns = AnalyzerProxy.play_rust_game(
            self.config.me.depth, self.config.opponent.depth, seed)
        self.position = position
        self.turns = turns
        self.dice = None
        self.winner = winner

    def play_to_end(self, is_cancelled: Callable[[], bool] = lambda: False) -> None:
        """Play the rest of the game with engines only (used when the game is not displayed)."""
        while not self.is_over and not is_cancelled():
            self.play(self.engine_move())


class MoveBuilder:
    """Builds a human's move one half-move at a time, only allowing steps that can still be
    completed to a legal move."""

    def __init__(self, legal_moves: set[Move]):
        self.legal_moves = legal_moves
        self.half_moves: list[HalfMove] = []
        self._full_length = max((len(m) for m in legal_moves), default=0)

    def _continuations(self) -> list[Move]:
        """Remainders of the legal moves that start with the half-moves entered so far."""
        prefix = tuple(self.half_moves)
        return [m[len(prefix):] for m in self.legal_moves if m[:len(prefix)] == prefix]

    def try_move(self, start: int, target: int) -> bool:
        """Move a checker from start to target, using one die or, if needed, several dice in a
        row with the same checker. Returns False if that can't lead to a legal move."""
        best: Optional[Move] = None
        for rest in self._continuations():
            position = start
            for i, (s, t) in enumerate(rest):
                if s != position:
                    break
                position = t
                if position == target:
                    if best is None or i + 1 < len(best):
                        best = rest[:i + 1]
                    break
        if best is None:
            return False
        self.half_moves.extend(best)
        return True

    def undo(self) -> None:
        if self.half_moves:
            self.half_moves.pop()

    def can_move_from(self, start: int) -> bool:
        return any(rest and rest[0][0] == start for rest in self._continuations())

    @property
    def is_complete(self) -> bool:
        return len(self.half_moves) == self._full_length

    @property
    def move(self) -> Move:
        return tuple(self.half_moves)
