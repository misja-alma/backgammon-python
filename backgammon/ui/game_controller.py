import copy
import time
from typing import Callable, Optional

from backgammon.game.analysis_worker import AnalysisWorker
from backgammon.game.game import Game, GameConfig, MoveBuilder, player_name
from backgammon.game.position import Position, Player
from backgammon.game.settings import Settings
from backgammon.game.utils import Utils
from backgammon.logger import Logger


class GameController:
    """Drives a Game from the pygame main loop without blocking it.

    update() is called every frame. Engine moves are computed on the AnalysisWorker thread;
    human moves are entered by clicking: a click on a checker moves it by the left die (or, when
    that can't be played, by the other one); after one die is played, the next click uses the
    other. Doubles: every click moves a checker by the die. A right click, or a click on the dice,
    swaps the dice. When all dice are played, a click on the dice confirms the move and a right
    click takes it back completely. A game between
    two engines that is not displayed runs completely on the worker thread.

    Each game gets a generation number. Results from the worker that arrive after the game was
    stopped (or replaced) carry an old number and are ignored, since a running search can't be
    interrupted.
    """

    ENGINE_MOVE_DELAY = 0.6   # seconds between displayed engine moves
    PASS_DELAY = 1.0          # seconds before a human without legal moves passes automatically

    def __init__(self, worker: AnalysisWorker, clock: Callable[[], float] = time.monotonic):
        self.worker = worker
        self.clock = clock
        self.game: Optional[Game] = None
        self._generation = 0
        self._pending = None            # (generation, Future) of the running worker task
        self._next_action_at = 0.0
        self.move_builder: Optional[MoveBuilder] = None
        self.selected_point: Optional[int] = None   # (not used: checkers move by die)
        self.dice_order: Optional[list[int]] = None   # the human's dice as shown, the left one first
        self._unstoppable = False       # the running game is played inside Rust
        self._final_position: Optional[Position] = None

    # --- state queries for the UI ---

    @property
    def is_running(self) -> bool:
        return self.game is not None

    @property
    def can_stop(self) -> bool:
        """A game played inside Rust can't be interrupted, so Stop is unavailable for it."""
        return self.is_running and not self._unstoppable

    @property
    def is_hidden(self) -> bool:
        return self.game is not None and not self.game.config.show_engine_game

    @property
    def is_human_turn(self) -> bool:
        return (self.game is not None and not self.game.is_over and
                not self.game.current_player_config.type.is_engine)

    def display_position(self) -> Optional[Position]:
        """The position to draw: the game position plus the human's half-entered move.

        After a game ends, returns its final position once (so the board can show it), then None.
        """
        if self.game is None:
            position, self._final_position = self._final_position, None
            return position
        if self.is_hidden and not self.game.is_over:
            # The worker thread is playing the game; show where it started until it's done.
            return self._start_position
        position = self.game.position
        if self.move_builder is not None:
            for start, target in self.move_builder.half_moves:
                position = Utils.apply_half_move(position, start, target, self.game.current_player)
        return position

    # --- start / stop ---

    def start(self, config: GameConfig, rng=None) -> None:
        if self.is_running and not self.can_stop:
            raise RuntimeError("A game played in Rust is still running")
        self.stop(silent=True)
        self._generation += 1
        hidden = not config.show_engine_game
        # A hidden game logs only its outcome, not hundreds of moves.
        self.game = Game(config, rng=rng, log=(lambda message: None) if hidden else Logger.log)
        self._start_position = copy.deepcopy(self.game.position)
        # Each variant and scoring (double match point or money) has its own net; the switch is
        # queued before any engine move.
        self.worker.use_rust_network(Settings.rust_network_for(config.variant, config.dmp)).add_done_callback(
            self._log_network)

        if self.game.runs_in_rust:
            # Rust plays the whole game, opening roll included; Python only waits for the result.
            Logger.log("Playing Rust engine vs Rust engine game in Rust (it can't be stopped)...")
            self._unstoppable = True
            game, seed = self.game, self.game.rng.getrandbits(64)
            self._pending = (self._generation, self.worker.submit(lambda: game.play_in_rust(seed)))
            return

        self.game.start()
        self._start_position = copy.deepcopy(self.game.position)
        if hidden:
            Logger.log("Playing engine vs engine game...")
            generation, game = self._generation, self.game
            future = self.worker.submit(
                lambda: game.play_to_end(is_cancelled=lambda: generation != self._generation))
            self._pending = (generation, future)
        else:
            self._next_action_at = self.clock() + self.ENGINE_MOVE_DELAY

    @staticmethod
    def _log_network(future) -> None:
        """Report a net that failed to load (runs on the worker thread)."""
        error = future.exception()
        if error is not None:
            Logger.log(f"Rust engine evaluator: placeholder ({error})")

    def stop(self, silent: bool = False) -> None:
        if self.is_running and not self.can_stop:
            return   # a game played in Rust can't be interrupted
        if self.game is not None and not silent:
            Logger.log("Game stopped")
        self._generation += 1
        self.game = None
        self._pending = None
        self.move_builder = None
        self.selected_point = None

    # --- per-frame driving ---

    def update(self) -> None:
        game = self.game
        if game is None:
            return
        if self.is_hidden:
            # The whole game runs as one worker task; finish when it is done.
            if self._collect_worker_result() and self.game is not None:
                self._finish()
            return
        if self._collect_worker_result():
            return
        if game.is_over:
            self._finish()
            return

        now = self.clock()
        if game.current_player_config.type.is_engine:
            if self._pending is None and now >= self._next_action_at:
                self._pending = (self._generation, self.worker.submit(game.engine_move))
        elif not game.legal_moves():
            if now >= self._next_action_at:
                self._play(())
        elif self.move_builder is None:
            self.move_builder = MoveBuilder(game.legal_moves(), game.dice)
            self.dice_order = sorted(game.dice, reverse=True)   # high left, low right

    def _collect_worker_result(self) -> bool:
        """Handle a finished worker task. Returns True if one was handled this frame."""
        if self._pending is None:
            return False
        generation, future = self._pending
        if not future.done():
            return False
        self._pending = None
        if generation != self._generation:
            return True   # result of a stopped game
        if future.exception() is not None:
            Logger.log(f"Engine failed: {future.exception()}")
            self._unstoppable = False
            self.stop(silent=True)
            return True
        if not self.is_hidden:
            self._play(future.result())
        return True

    def _play(self, move) -> None:
        self.game.play(move)
        self.move_builder = None
        self.dice_order = None
        self.selected_point = None
        if not self.game.is_over:
            delay = self.ENGINE_MOVE_DELAY if self.game.current_player_config.type.is_engine else self.PASS_DELAY
            self._next_action_at = self.clock() + delay

    def _finish(self) -> None:
        """The game is over: report the winner (displayed games already logged it) and stop."""
        if self.is_hidden:
            if self.game.is_over:
                Logger.log(f"{self.game.result_text()} after {self.game.turns} turns")
        self._final_position = self.game.position
        self._unstoppable = False
        self.game = None
        self.move_builder = None
        self.selected_point = None

    # --- human input ---

    def handle_click(self, point: Optional[int], is_left_click: bool) -> None:
        """A click on the board (point from the current player's perspective, None outside the
        points) during a human turn. Left click on a checker: move it by the left die, or by the
        other die when the left one can't be played with it. Right click: swap the dice, or when
        all dice are played, take the whole move back."""
        if not self.is_human_turn or self.move_builder is None:
            return
        builder = self.move_builder
        if not is_left_click:
            if builder.is_complete and builder.half_moves:
                builder.reset()
            else:
                self.swap_dice()
            return
        if point is None or builder.is_complete:
            return
        remaining = builder.remaining_dice()
        for die in sorted(set(remaining), key=lambda d: self.dice_order.index(d)):
            if builder.try_step(point, die):
                return

    def click_dice(self) -> None:
        """A click on the dice during a human turn: confirm the move when all dice are played,
        otherwise swap the dice."""
        if not self.is_human_turn or self.move_builder is None:
            return
        if self.move_builder.is_complete:
            self._play(self.move_builder.move)
        else:
            self.swap_dice()

    def swap_dice(self) -> None:
        if self.dice_order is not None and not self.move_builder.is_complete:
            self.dice_order.reverse()

    def display_dice(self) -> list[tuple[int, bool]]:
        """The dice to show, left to right, each with whether it has been played. High left and
        low right, unless the human swapped them; for doubles the right die is played after the
        first two steps and the left one after four."""
        game = self.game
        if game is None or game.dice is None:
            return []
        order = self.dice_order if self.dice_order is not None else sorted(game.dice, reverse=True)
        builder = self.move_builder
        if builder is None or not self.is_human_turn:
            return [(d, False) for d in order]
        used = len(builder.half_moves)
        if order[0] == order[1]:
            return [(order[0], used >= 4), (order[1], used >= 2)]
        left = builder.remaining_dice()
        shown = []
        for d in order:
            if d in left:
                left.remove(d)
                shown.append((d, False))
            else:
                shown.append((d, True))
        return shown

    def play_move(self, move) -> bool:
        """Play a complete legal move for the human on turn (e.g. chosen in the analysis panel)."""
        if not self.is_human_turn or (move not in self.game.legal_moves() and (move or self.game.legal_moves())):
            return False
        self._play(move)
        return True

    def undo(self) -> None:
        if self.is_human_turn and self.move_builder is not None:
            self.selected_point = None
            self.move_builder.undo()
