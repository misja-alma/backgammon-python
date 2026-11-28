import math
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame  # noqa: E402

from backgammon.game import move_analysis  # noqa: E402
from backgammon.game.move_analysis import ROLLOUT_BATCH, RolloutProgress, _rollout_stats, rollout  # noqa: E402
from backgammon.game.position import Player, Position  # noqa: E402
from backgammon.ui.board_map import BoardMap  # noqa: E402
from backgammon.ui.rollout_dialog import RolloutDialog  # noqa: E402


class RolloutStatsTest(unittest.TestCase):

    def test_money_counts_gammons_and_backgammons(self):
        equity, probs, error = _rollout_stats([1, 2, 3, -1, -2, -1], money=True)
        self.assertAlmostEqual(2 / 6, equity)
        self.assertEqual([3 / 6, 2 / 6, 1 / 6, 1 / 6, 0.0], probs)
        self.assertGreater(error, 0)

    def test_double_match_point_counts_wins_only(self):
        equity, probs, _ = _rollout_stats([1, 3, -2, -1], money=False)
        self.assertEqual(0.0, equity)
        self.assertEqual(0.5, probs[0])
        self.assertTrue(all(math.isnan(p) for p in probs[1:]))


class RolloutStopTest(unittest.TestCase):
    """rollout() with the engine replaced by a stub that wins every game by one point and asks
    to stop after a number of engine calls."""

    def run_rollout(self, dice, stop_after_calls):
        progress = RolloutProgress()
        calls = []

        def fake_rollout(position, first, games, seed, depth, money):
            calls.append((first, games))
            if len(calls) == stop_after_calls:
                progress.stop = True
            return [1] * games

        position = Position()
        position.setup_starting_position()
        position.set_turn(Player.ME)
        with patch.object(move_analysis.AnalyzerProxy, "rollout", side_effect=fake_rollout):
            return rollout(position, dice, 1296, 0, True, progress), progress, calls

    def test_a_stopped_position_rollout_keeps_the_finished_batches(self):
        analysis, progress, calls = self.run_rollout(None, stop_after_calls=3)
        self.assertEqual(3 * ROLLOUT_BATCH, analysis.rollout_games)
        self.assertEqual(3 * ROLLOUT_BATCH, progress.game)
        self.assertEqual(1.0, analysis.equity)

    def test_stopped_moves_all_have_the_same_games(self):
        analysis, progress, calls = self.run_rollout((3, 1), stop_after_calls=16 + 5)
        # 16 moves: one full batch, the second batch stopped after 5 moves and not counted.
        self.assertEqual(16, len(analysis.moves))
        self.assertEqual(ROLLOUT_BATCH, analysis.rollout_games)
        self.assertEqual(16 + 5, len(calls))
        self.assertTrue(all(m.equity == -1.0 for m in analysis.moves))   # the opponent won them all

    def test_stopped_before_a_batch_finished_gives_none(self):
        analysis, _, _ = self.run_rollout((3, 1), stop_after_calls=2)
        self.assertIsNone(analysis)

    def test_unstopped_rollout_plays_every_game_once(self):
        analysis, _, calls = self.run_rollout(None, stop_after_calls=0)
        self.assertEqual(1296, analysis.rollout_games)
        self.assertEqual(list(range(0, 1296, ROLLOUT_BATCH)), [first for first, _ in calls])


class RolloutCandidatesTest(unittest.TestCase):
    """Which moves rollout() rolls out: candidates of them, preferred ones first."""

    def run_rollout(self, candidates=None, preferred=None):
        position = Position()
        position.setup_starting_position()
        position.set_turn(Player.ME)
        with patch.object(move_analysis.AnalyzerProxy, "rollout", side_effect=lambda p, first, n, *a: [1] * n):
            return rollout(position, (3, 1), 36, 0, True, RolloutProgress(), candidates, preferred), position

    def test_the_best_at_one_ply_by_default(self):
        everything, position = self.run_rollout()
        self.assertEqual(16, len(everything.moves))
        best3, _ = self.run_rollout(candidates=3)
        ranking = [m.notation for m in move_analysis.analyze(position, (3, 1), 0, True).moves]
        self.assertEqual(sorted(ranking[:3]), sorted(m.notation for m in best3.moves))

    def test_preferred_moves_come_first(self):
        _, position = self.run_rollout(candidates=1)
        moves = move_analysis.analyze(position, (3, 1), 0, True).moves
        worst, second_worst = moves[-1], moves[-2]
        preferred = [move_analysis.position_key(worst.position), move_analysis.position_key(second_worst.position)]
        two, _ = self.run_rollout(candidates=2, preferred=preferred)
        self.assertEqual({worst.notation, second_worst.notation}, {m.notation for m in two.moves})
        three, _ = self.run_rollout(candidates=3, preferred=preferred)     # filled up with the best
        self.assertEqual({worst.notation, second_worst.notation, moves[0].notation}, {m.notation for m in three.moves})


class RolloutDialogTest(unittest.TestCase):

    def setUp(self):
        pygame.init()
        self.screen = pygame.display.set_mode((1024, 768))
        RolloutDialog._last_games, RolloutDialog._last_ply = 1296, 1

    def tearDown(self):
        RolloutDialog._last_games, RolloutDialog._last_ply = 1296, 1
        pygame.quit()

    @staticmethod
    def key(key, text=""):
        return SimpleNamespace(key=key, unicode=text)

    def test_defaults_to_1296_games_at_one_ply(self):
        self.assertEqual((1296, 0, 3), RolloutDialog(self.screen, BoardMap(1024, 768)).result())

    def test_typing_the_number_of_games(self):
        dialog = RolloutDialog(self.screen, BoardMap(1024, 768))
        for _ in range(4):
            dialog.handle_key(self.key(pygame.K_BACKSPACE))
        dialog.handle_key(self.key(pygame.K_0, "0"))   # no leading zeros
        dialog.handle_key(self.key(pygame.K_a, "a"))   # digits only
        for digit in "324":
            dialog.handle_key(self.key(pygame.K_0, digit))
        self.assertEqual((324, 0, 3), dialog.result())

    def test_the_strength_stepper_goes_from_one_to_three_ply(self):
        dialog = RolloutDialog(self.screen, BoardMap(1024, 768))
        dialog.handle_click(dialog.btn_plus.center)
        self.assertEqual((1296, 1, 3), dialog.result())   # 2-ply is depth 1
        for _ in range(5):
            dialog.handle_click(dialog.btn_plus.center)
        self.assertEqual(3, dialog.ply)
        for _ in range(5):
            dialog.handle_click(dialog.btn_minus.center)
        self.assertEqual((1296, 0, 3), dialog.result())

    def test_controls_are_right_of_the_labels_at_any_window_size(self):
        for width, height in ((800, 600), (1340, 768), (2560, 1440)):
            board_map = BoardMap(width, height)
            dialog = RolloutDialog(self.screen, board_map)
            label_right = dialog.x + 20 + max(dialog.small_font.size(text)[0]
                                              for text in (dialog.GAMES_LABEL, dialog.PLY_LABEL))
            self.assertLess(label_right, dialog.field_games.left)
            self.assertLess(label_right, dialog.btn_minus.left)
            self.assertLessEqual(dialog.btn_plus.right, dialog.x + dialog.width - 10)
            self.assertLess(dialog.btn_plus.bottom, dialog.btn_ok.top)

    def test_candidates_are_typed_after_tab_and_disabled_for_a_position(self):
        dialog = RolloutDialog(self.screen, BoardMap(1024, 768), candidates=2, note="2 moves selected")
        self.assertEqual((1296, 0, 2), dialog.result())
        dialog.handle_key(self.key(pygame.K_TAB))
        dialog.handle_key(self.key(pygame.K_BACKSPACE))
        dialog.handle_key(self.key(pygame.K_5, "5"))
        self.assertEqual((1296, 0, 5), dialog.result())
        dialog.handle_key(self.key(pygame.K_BACKSPACE))
        self.assertFalse(dialog.result())                       # no candidates: refused
        position = RolloutDialog(self.screen, BoardMap(1024, 768), candidates=None)
        position.handle_key(self.key(pygame.K_TAB))             # stays on the games field
        position.handle_key(self.key(pygame.K_BACKSPACE))
        self.assertEqual((129, 0, None), position.result())

    def test_no_games_is_refused(self):
        dialog = RolloutDialog(self.screen, BoardMap(1024, 768))
        for _ in range(4):
            dialog.handle_key(self.key(pygame.K_BACKSPACE))
        self.assertFalse(dialog.result())
