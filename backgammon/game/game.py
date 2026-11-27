import copy
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from backgammon.game.analyzer_proxy import AnalyzerProxy
from backgammon.game.evaluator import CHECKERS_PER_PLAYER
from backgammon.game.position import GameVariant, Position, Player
from backgammon.game.settings import Analyzer
from backgammon.game.utils import Utils

HalfMove = tuple[int, int]   # (start point, target point) from the mover's perspective
Move = tuple[HalfMove, ...]  # a full move; the empty tuple means "no legal move, pass"


class PlayerType(Enum):
    HUMAN = "Human"
    PYTHON_ENGINE = "Python engine"
    RUST_ENGINE = "Rust engine"
    BGSAGE_ENGINE = "bgsage engine"

    @property
    def engine(self) -> Optional[Analyzer]:
        return {PlayerType.PYTHON_ENGINE: Analyzer.PYTHON,
                PlayerType.RUST_ENGINE: Analyzer.RUST,
                PlayerType.BGSAGE_ENGINE: Analyzer.BGSAGE}.get(self)

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
    variant: GameVariant = GameVariant.BACKGAMMON
    # Double match point: every win counts 1 point (no gammons or backgammons).
    dmp: bool = True

    def player(self, player: Player) -> PlayerConfig:
        return self.me if player == Player.ME else self.opponent

    @property
    def all_engines(self) -> bool:
        return self.me.type.is_engine and self.opponent.type.is_engine


class WinType(Enum):
    """How a game was won; the value is the number of points it is worth (without dmp)."""
    SINGLE = 1
    GAMMON = 2       # the loser has borne off no checkers
    BACKGAMMON = 3   # a gammon, and the loser still has a checker on the bar or in the winner's home board

    @staticmethod
    def of(position: Position, winner: Player) -> "WinType":
        """The win type of a finished game, from its final position."""
        loser = winner.other_player()
        if position.get_checkers(loser, 0) > 0:
            return WinType.SINGLE
        # The winner's home board is the loser's points 19-24 (in the loser's numbering); 25 is the bar.
        if any(position.get_checkers(loser, point) > 0 for point in range(19, 26)):
            return WinType.BACKGAMMON
        return WinType.GAMMON


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
    """A single game of backgammon or nackgammon, without doubling cube. With `config.dmp`
    (double match point) every win is worth 1 point; otherwise a gammon is worth 2 and a
    backgammon 3 (see WinType). Contains no UI code.

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
        self.position.setup_starting_position(config.variant)
        self.dice: Optional[tuple[int, int]] = None
        self.winner: Optional[Player] = None
        self.win_type: Optional[WinType] = None   # set when the game is over, also with dmp
        self.turns = 0
        self._legal_moves: Optional[set[Move]] = None

    # --- state ---

    @property
    def is_over(self) -> bool:
        return self.winner is not None

    @property
    def points(self) -> Optional[int]:
        """Points won by the winner (None while the game is going on): always 1 with dmp."""
        if self.win_type is None:
            return None
        return 1 if self.config.dmp else self.win_type.value

    def result_text(self) -> str:
        """E.g. 'ME wins', or without dmp 'OPPONENT wins a gammon (2 points)'."""
        name = player_name(self.winner)
        if self.config.dmp:
            return f"{name} wins"
        if self.win_type == WinType.SINGLE:
            return f"{name} wins (1 point)"
        return f"{name} wins a {self.win_type.name.lower()} ({self.points} points)"

    def _set_winner(self, winner: Player) -> None:
        self.winner = winner
        self.win_type = WinType.of(self.position, winner)
        self.dice = None
        self._legal_moves = None

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
            self._set_winner(mover)
            self.log(self.result_text())
        else:
            self._set_dice((self._roll_die(), self._roll_die()))

    def find_move(self, position_after: Position) -> Move:
        """The legal move that leads to position_after (e.g. as chosen by an engine).

        When position_after bears off the mover's last checker, any legal move that does the same
        is returned: all winning moves are equal (the win type depends only on the loser's
        checkers). Engines differ on whether the
        last checker may come off with one die when both dice could be played (e.g. bgsage plays
        2/off with 4-1, where our move generator requires 2/1/off)."""
        legal = self.legal_moves()
        if not legal:
            return ()
        mover = self.current_player
        wins = position_after.get_checkers(mover, 0) == CHECKERS_PER_PLAYER
        winning_move = None
        for move in legal:
            result = Utils.apply_move(self.position, move)
            if (result.me_checkers == position_after.me_checkers and
                    result.opponent_checkers == position_after.opponent_checkers):
                return move
            if wins and winning_move is None and result.get_checkers(mover, 0) == CHECKERS_PER_PLAYER:
                winning_move = move
        if winning_move is not None:
            return winning_move
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
                                                    engine=config.type.engine, depth=config.depth,
                                                    money=not self.config.dmp)
        return self.find_move(position_after)

    def play_in_rust(self, seed: int) -> None:
        """Play the whole game inside Rust, including the opening roll (see runs_in_rust).
        Cannot be interrupted. Call on a fresh game (without start()), off the UI thread."""
        if not self.runs_in_rust:
            raise ValueError("Only a hidden Rust engine vs Rust engine game can run in Rust")
        winner, position, turns = AnalyzerProxy.play_rust_game(
            self.config.me.depth, self.config.opponent.depth, seed,
            nackgammon=self.config.variant == GameVariant.NACKGAMMON, money=not self.config.dmp)
        self.position = position
        self.turns = turns
        self._set_winner(winner)

    def play_to_end(self, is_cancelled: Callable[[], bool] = lambda: False) -> None:
        """Play the rest of the game with engines only (used when the game is not displayed)."""
        while not self.is_over and not is_cancelled():
            self.play(self.engine_move())


class MoveBuilder:
    """Builds a human's move one half-move at a time, only allowing steps that can still be
    completed to a legal move."""

    def __init__(self, legal_moves: set[Move], dice: Optional[tuple[int, int]] = None):
        self.legal_moves = legal_moves
        self.half_moves: list[HalfMove] = []
        # The die each half-move used (try_step), None for half-moves entered by target.
        self.dice_used: list[Optional[int]] = []
        self.dice = dice
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
        self.dice_used.extend([None] * len(best))
        return True

    def remaining_dice(self) -> list[int]:
        """The dice not used yet (four for doubles), with the dice given at construction."""
        if self.dice is None:
            return []
        dice = [self.dice[0]] * 4 if self.dice[0] == self.dice[1] else list(self.dice)
        for used in self.dice_used:
            if used in dice:
                dice.remove(used)
        return dice[:max(0, len(dice) - self.dice_used.count(None))]

    @staticmethod
    def _fits(half: HalfMove, die: int) -> bool:
        """Whether the half-move can be played with this die (bearing off may use a higher die;
        the legal moves already say when that is allowed)."""
        start, target = half
        return start - target == die or (target == 0 and die > start)

    def _assignable(self, rest: Move, dice: list[int]) -> bool:
        """Whether the half-moves can be played with these dice, in some order of the dice."""
        if not rest:
            return True
        for i, die in enumerate(dice):
            if (i == 0 or die != dice[i - 1]) and self._fits(rest[0], die) and \
                    self._assignable(rest[1:], dice[:i] + dice[i + 1:]):
                return True
        return False

    def try_step(self, start: int, die: int) -> bool:
        """Move the checker on start by one die (bearing off when it goes past the bearoff
        line), if that can still be completed to a legal move with the other dice."""
        remaining = self.remaining_dice()
        if die not in remaining:
            return False
        half = (start, max(start - die, 0))
        rest_dice = list(remaining)
        rest_dice.remove(die)
        for rest in self._continuations():
            if rest and rest[0] == half and self._fits(half, die) and self._assignable(rest[1:], sorted(rest_dice)):
                self.half_moves.append(half)
                self.dice_used.append(die)
                return True
        return False

    def undo(self) -> None:
        if self.half_moves:
            self.half_moves.pop()
            self.dice_used.pop()

    def reset(self) -> None:
        self.half_moves.clear()
        self.dice_used.clear()

    def can_move_from(self, start: int) -> bool:
        return any(rest and rest[0][0] == start for rest in self._continuations())

    @property
    def is_complete(self) -> bool:
        return len(self.half_moves) == self._full_length

    @property
    def move(self) -> Move:
        return tuple(self.half_moves)
