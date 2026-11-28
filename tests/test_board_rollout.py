"""Board.rollout: the candidate moves come from the selection in the analysis panel."""
import os
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame  # noqa: E402

from backgammon.game.move_analysis import analyze, position_key  # noqa: E402
from backgammon.game.position import Player  # noqa: E402
from backgammon.ui.board import Board  # noqa: E402


class BoardRolloutCandidatesTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.board = Board()
        cls.board.analysis_worker.submit(lambda: None).result()   # the engine is set up

    @classmethod
    def tearDownClass(cls):
        pygame.quit()

    def start_rollout(self, dialog_answer):
        board = self.board
        dialogs = []

        class FakeDialog:
            def __init__(self, screen, board_map, candidates=3, note=""):
                dialogs.append((candidates, note))

            def run(self, snapshot):
                return dialog_answer

        board._pending_analysis = None
        with patch("backgammon.ui.board.RolloutDialog", FakeDialog), \
                patch.object(board.analysis_worker, "rollout", return_value=MagicMock(done=lambda: False)) as run:
            board.rollout()
        board._pending_analysis, board._rollout_progress = None, None
        return dialogs[0], run.call_args.args

    def setUp(self):
        board = self.board
        board.restore_starting_position()
        board.position.set_turn(Player.ME)
        board.edit_dice = (3, 1)
        self.analysis = analyze(board.position, (3, 1), 0, True)
        board.analysis_panel.set_analysis(self.analysis)

    def test_selected_moves_set_the_candidates(self):
        panel = self.board.analysis_panel
        panel.marked, panel.selected = [4, 2], 2
        (count, note), args = self.start_rollout((36, 0, 2))
        self.assertEqual(2, count)
        self.assertIn("2 moves selected", note)
        candidates, preferred = args[6], args[7]
        self.assertEqual(2, candidates)
        moves = self.analysis.moves
        self.assertEqual([position_key(moves[2].position), position_key(moves[4].position)], preferred[:2])
        self.assertEqual(len(moves), len(preferred))             # then the rest, in the analysis's order

    def test_without_a_selection_three_candidates_by_the_analysis(self):
        (count, note), args = self.start_rollout((36, 0, 3))
        self.assertEqual(3, count)
        self.assertIn("Shift", note)                              # the tip
        self.assertEqual([position_key(m.position) for m in self.analysis.moves], args[7])

    def test_a_position_rollout_has_no_candidates(self):
        self.board.edit_dice = None
        self.board.analysis_panel.clear()
        (count, _), args = self.start_rollout((36, 0, None))
        self.assertIsNone(count)
        self.assertIsNone(args[6])


if __name__ == "__main__":
    unittest.main()
