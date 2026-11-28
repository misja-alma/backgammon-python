import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame  # noqa: E402

from backgammon.game.position import Player  # noqa: E402
from backgammon.ui.board import Board  # noqa: E402


class EditDiceTest(unittest.TestCase):
    """Setting the dice of the edited position by clicking them."""

    @classmethod
    def setUpClass(cls):
        cls.board = Board()

    @classmethod
    def tearDownClass(cls):
        pygame.quit()

    def setUp(self):
        self.board.edit_dice = None
        self.board.position.setup_starting_position()
        self.board.position.set_turn(Player.ME)

    def click(self, die, left=True):
        return self.board.handle_dice_edit_click(self.board._dice_rects()[die].center, is_left_click=left)

    def test_no_dice_means_the_position_before_the_roll(self):
        self.assertEqual(2, len(self.board._dice_rects()))   # the placeholders
        self.assertIsNone(self.board._analysis_target()[1])

    def test_left_clicks_raise_a_die_and_wrap(self):
        self.assertTrue(self.click(0))
        self.assertEqual((2, 1), self.board.edit_dice)
        self.click(1)
        self.click(1)
        self.assertEqual((2, 3), self.board.edit_dice)
        for _ in range(4):
            self.click(1)
        self.assertEqual((2, 1), self.board.edit_dice)   # 3 → 4, 5, 6, 1
        self.assertEqual((2, 1), self.board._analysis_target()[1])

    def test_right_clicks_lower_and_remove_the_dice_below_one(self):
        self.board.edit_dice = (2, 4)
        self.click(0, left=False)
        self.assertEqual((1, 4), self.board.edit_dice)
        self.click(0, left=False)
        self.assertIsNone(self.board.edit_dice)

    def test_a_right_click_on_the_placeholders_sets_one_one(self):
        self.click(1, left=False)
        self.assertEqual((1, 1), self.board.edit_dice)

    def test_clicks_elsewhere_are_not_dice_clicks(self):
        self.assertFalse(self.board.handle_dice_edit_click((1, 1), is_left_click=True))
        self.assertIsNone(self.board.edit_dice)

    def test_the_dice_follow_the_side_on_roll(self):
        me_rects = self.board._dice_rects()
        self.board.position.set_turn(Player.OPPONENT)
        self.assertNotEqual(me_rects, self.board._dice_rects())

    def test_changing_the_board_removes_the_dice(self):
        self.board.edit_dice = (3, 1)
        self.board.restore_starting_position()
        self.assertIsNone(self.board.edit_dice)
        self.board.edit_dice = (3, 1)
        self.board.clear_board()
        self.assertIsNone(self.board.edit_dice)


if __name__ == "__main__":
    unittest.main()


class RolloutProgressLineTest(unittest.TestCase):
    """A rollout's progress takes one line of the text area below the board."""

    @classmethod
    def setUpClass(cls):
        cls.board = Board()

    @classmethod
    def tearDownClass(cls):
        pygame.quit()

    def test_progress_updates_one_line_and_ends_with_the_result(self):
        import time
        from backgammon.game.move_analysis import RolloutProgress
        board = self.board
        board.analysis_worker.submit(lambda: None).result()   # the engine is set up
        board.text_area.clear()
        board.text_area.show_text("earlier")
        board._rollout_progress = RolloutProgress(games=1296)
        board._rollout_start = time.monotonic()
        board._rollout_line = None
        board._pending_analysis = board.analysis_worker.rollout(board.position, None, 1296, 0, False,
                                                                board._rollout_progress)
        while board._pending_analysis is not None:
            board.update()
            time.sleep(0.01)
        lines = board.text_area._lines
        rollout_lines = [line for line in lines if "oll" in line]
        self.assertEqual(1, len(rollout_lines), lines)
        self.assertTrue(rollout_lines[0].startswith("Rollout finished: 1296 games"), lines)
        self.assertEqual("earlier", lines[0])
