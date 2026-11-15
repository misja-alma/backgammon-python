import os
import random
import threading
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from backgammon.game.analysis_worker import AnalysisWorker  # noqa: E402
from backgammon.game.game import GameConfig, PlayerConfig, PlayerType  # noqa: E402
from backgammon.game.position import Player  # noqa: E402
from backgammon.logger import Logger  # noqa: E402
from backgammon.ui.game_controller import GameController  # noqa: E402
from tests.test_game import ScriptedDice  # noqa: E402


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class CapturingTextArea:
    def __init__(self):
        self.lines = []

    def show_text(self, text):
        self.lines.append(text)


def config(me, opponent, show=True, depth=0):
    return GameConfig(PlayerConfig(me, depth), PlayerConfig(opponent, depth), show_engine_game=show)


class GameControllerTest(unittest.TestCase):

    def setUp(self):
        self.clock = FakeClock()
        self.controller = GameController(AnalysisWorker(), clock=self.clock)
        self.log = CapturingTextArea()
        Logger.set_text_area(self.log)

    def tearDown(self):
        Logger.set_text_area(None)

    def run_until(self, condition, max_frames=20000):
        """Call update() like the main loop does, advancing the fake clock each frame."""
        for _ in range(max_frames):
            if condition():
                return
            self.clock.now += 1.0
            self.controller.update()
            Logger.flush()
            time.sleep(0.0005)
        self.fail("condition not reached")

    def test_displayed_engine_game_runs_to_the_end(self):
        self.controller.start(config(PlayerType.RUST_ENGINE, PlayerType.PYTHON_ENGINE), rng=random.Random(3))
        self.run_until(lambda: not self.controller.is_running)
        self.assertRegex(self.log.lines[-1], r"^(ME|OPPONENT) wins$")
        self.assertTrue(any(" rolls " in line and "/" in line for line in self.log.lines))
        self.assertTrue(self.controller.can_stop is False)

    def test_engine_waits_between_moves(self):
        self.controller.start(config(PlayerType.RUST_ENGINE, PlayerType.RUST_ENGINE), rng=ScriptedDice([3, 1]))
        self.controller.update()  # clock not advanced: the delay has not passed yet
        self.assertIsNone(self.controller._pending)

    def test_hidden_engine_game_logs_only_the_result(self):
        self.controller.start(config(PlayerType.RUST_ENGINE, PlayerType.PYTHON_ENGINE, show=False),
                              rng=random.Random(5))
        self.assertTrue(self.controller.can_stop)
        start = self.controller.display_position()
        self.run_until(lambda: not self.controller.is_running)
        self.assertEqual(["Playing engine vs engine game...", self.log.lines[-1]], self.log.lines)
        self.assertRegex(self.log.lines[-1], r"^(ME|OPPONENT) wins after \d+ turns$")
        self.assertEqual(15, start.get_checkers(Player.ME, 13) + start.get_checkers(Player.ME, 6) +
                         start.get_checkers(Player.ME, 8) + start.get_checkers(Player.ME, 24))

    def test_hidden_rust_game_is_played_inside_rust(self):
        def python_game_code_must_not_run(*args, **kwargs):
            raise AssertionError("Python game code used")

        with patch('backgammon.game.game.Game.play', python_game_code_must_not_run), \
             patch('backgammon.game.game.Game.engine_move', python_game_code_must_not_run):
            self.controller.start(config(PlayerType.RUST_ENGINE, PlayerType.RUST_ENGINE, show=False),
                                  rng=random.Random(5))
            self.run_until(lambda: not self.controller.is_running)

        self.assertEqual("Playing Rust engine vs Rust engine game in Rust (it can't be stopped)...",
                         self.log.lines[0])
        self.assertRegex(self.log.lines[-1], r"^(ME|OPPONENT) wins after \d+ turns$")
        final = self.controller.display_position()
        self.assertIsNotNone(final)
        winner = Player.ME if self.log.lines[-1].startswith("ME") else Player.OPPONENT
        self.assertEqual(15, final.get_checkers(winner, 0))
        self.assertIsNone(self.controller.display_position())  # handed out only once

    def test_rust_game_cannot_be_stopped(self):
        gate = threading.Event()
        self.controller.worker.submit(lambda: gate.wait(timeout=5))  # keep the game queued
        self.controller.start(config(PlayerType.RUST_ENGINE, PlayerType.RUST_ENGINE, show=False),
                              rng=random.Random(1))
        self.assertFalse(self.controller.can_stop)
        self.controller.stop()
        self.assertTrue(self.controller.is_running)
        with self.assertRaises(RuntimeError):
            self.controller.start(config(PlayerType.HUMAN, PlayerType.HUMAN))
        gate.set()
        self.run_until(lambda: not self.controller.is_running)
        self.assertFalse(self.controller.can_stop)

    def test_stop_discards_a_late_engine_result(self):
        gate = threading.Event()

        def slow_engine_move(game_self):
            gate.wait(timeout=5)
            return next(iter(game_self.legal_moves()))

        with patch('backgammon.game.game.Game.engine_move', slow_engine_move):
            self.controller.start(config(PlayerType.RUST_ENGINE, PlayerType.HUMAN), rng=ScriptedDice([6, 1]))
            self.clock.now += 10
            self.controller.update()
            self.assertIsNotNone(self.controller._pending)
            future = self.controller._pending[1]

            self.controller.stop()
            self.assertFalse(self.controller.is_running)
            gate.set()
            future.result(timeout=5)
            self.controller.update()  # must not apply the result to anything

        self.assertFalse(self.controller.is_running)
        self.assertEqual("Game stopped", self.log.lines[-1])

    def test_human_enters_a_move_by_clicking(self):
        self.controller.start(config(PlayerType.HUMAN, PlayerType.HUMAN), rng=ScriptedDice([3, 1, 6, 5]))
        self.controller.update()
        self.assertTrue(self.controller.is_human_turn)

        self.controller.handle_click(8, is_left_click=True)
        self.assertEqual(8, self.controller.selected_point)
        self.controller.handle_click(5, is_left_click=True)
        self.assertIsNone(self.controller.selected_point)
        # The half-entered move is shown on the board.
        self.assertEqual(1, self.controller.display_position().get_checkers(Player.ME, 5))

        self.controller.handle_click(6, is_left_click=True)
        self.controller.handle_click(5, is_left_click=True)
        self.assertEqual(Player.OPPONENT, self.controller.game.current_player)
        self.assertEqual(2, self.controller.game.position.get_checkers(Player.ME, 5))

    def test_right_click_cancels_selection_then_undoes(self):
        self.controller.start(config(PlayerType.HUMAN, PlayerType.HUMAN), rng=ScriptedDice([3, 1]))
        self.controller.update()
        self.controller.handle_click(8, is_left_click=True)
        self.controller.handle_click(5, is_left_click=True)
        self.controller.handle_click(6, is_left_click=True)

        self.controller.handle_click(None, is_left_click=False)
        self.assertIsNone(self.controller.selected_point)
        self.assertEqual(1, len(self.controller.move_builder.half_moves))
        self.controller.handle_click(None, is_left_click=False)
        self.assertEqual(0, len(self.controller.move_builder.half_moves))

    def test_clicks_are_ignored_on_an_engine_turn(self):
        self.controller.start(config(PlayerType.RUST_ENGINE, PlayerType.HUMAN), rng=ScriptedDice([3, 1]))
        self.controller.handle_click(8, is_left_click=True)
        self.assertIsNone(self.controller.selected_point)

    def test_human_without_legal_moves_passes_automatically(self):
        self.controller.start(config(PlayerType.HUMAN, PlayerType.HUMAN), rng=ScriptedDice([6, 5, 2, 3]))
        game = self.controller.game
        position = game.position
        position.clear()
        position.set_checkers(Player.ME, 25, 1)
        position.set_checkers(Player.ME, 0, 14)
        for point in range(1, 7):
            position.set_checkers(Player.OPPONENT, point, 2)
        position.set_checkers(Player.OPPONENT, 0, 3)
        game._legal_moves = None

        self.controller.update()
        self.assertEqual(Player.ME, game.current_player)  # waits PASS_DELAY first
        self.clock.now += GameController.PASS_DELAY
        self.controller.update()
        self.assertEqual(Player.OPPONENT, game.current_player)
        self.assertIn("ME rolls 6-5: no legal move", self.log.lines)


if __name__ == '__main__':
    unittest.main()
