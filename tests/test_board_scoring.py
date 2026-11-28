"""The Game control above the board: Unlimited (money) or DMP, for edited positions."""
import os
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame  # noqa: E402

from backgammon.game.game import GameConfig, PlayerConfig, PlayerType  # noqa: E402
from backgammon.game.settings import Settings  # noqa: E402
from backgammon.ui.board import Board  # noqa: E402

MONEY_XGID = "XGID=-b----E-C---eE---c-e----B-:0:0:1:31:0:0:0:0:10"
DMP_XGID = "XGID=-b----E-C---eE---c-e----B-:0:0:1:31:0:0:0:1:10"
MATCH_XGID = "XGID=-b----E-C---eE---c-e----B-:0:0:1:31:4:2:0:7:10"


class ScoringControlTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.board = Board()

    @classmethod
    def tearDownClass(cls):
        pygame.quit()

    def setUp(self):
        self.board.game_controller.stop(silent=True)
        self.board.edit_money = True

    def click(self, money: bool) -> bool:
        _, options = self.board._scoring_layout()
        rect = next(r for m, r, _ in options if m == money)
        return self.board.handle_scoring_click(rect.center)

    def paste(self, text: str) -> None:
        with patch("backgammon.ui.board.paste_from_clipboard", return_value=text):
            self.board.paste_position()

    def copied(self) -> str:
        out = []
        with patch("backgammon.ui.board.copy_to_clipboard", side_effect=out.append):
            self.board.copy_position()
        return out[0]

    def test_clicks_choose_the_scoring(self):
        self.assertTrue(self.board.money_scoring)
        self.assertTrue(self.click(False))
        self.assertFalse(self.board.money_scoring)
        self.click(True)
        self.assertTrue(self.board.money_scoring)
        self.assertFalse(self.board.handle_scoring_click((1, 1)))

    def test_loading_shows_the_scoring_of_the_position(self):
        self.paste(DMP_XGID)
        self.assertFalse(self.board.edit_money)
        self.paste(MONEY_XGID)
        self.assertTrue(self.board.edit_money)
        self.paste(DMP_XGID)
        self.paste(MATCH_XGID)          # a longer match: gammons count, as in a money game
        self.assertTrue(self.board.edit_money)

    def test_saving_and_copying_use_it(self):
        self.paste(MONEY_XGID)
        self.click(False)
        self.assertEqual(DMP_XGID, self.copied())
        self.assertEqual(1, self.board._saved_position().match_length)
        self.click(True)
        self.assertEqual(MONEY_XGID, self.copied())

    def test_analysis_uses_it_with_the_matching_net(self):
        board = self.board
        self.paste(DMP_XGID)
        with patch.object(board.analysis_worker, "analyze", return_value=MagicMock(done=lambda: False)) as analyze, \
                patch.object(board.analysis_worker, "use_rust_network") as network:
            board._pending_analysis = None
            board.analyze()
            self.assertFalse(analyze.call_args.args[3])                 # money = False
            network.assert_called_with(Settings.rust_network_for(None, dmp=True))
            board._pending_analysis = None
            self.click(True)
            board.analyze()
            self.assertTrue(analyze.call_args.args[3])
            network.assert_called_with(Settings.rust_network_for(None, dmp=False))
        board._pending_analysis = None

    def test_during_a_game_it_shows_the_games_scoring_and_cant_be_changed(self):
        board = self.board
        human = PlayerConfig(PlayerType.HUMAN)
        board.game_controller.start(GameConfig(human, human, dmp=False))
        board.edit_money = False
        self.assertTrue(board.money_scoring)
        self.assertTrue(self.click(False))          # on the control, but ignored
        self.assertTrue(board.money_scoring)
        self.assertFalse(board.edit_money)
        board.game_controller.stop(silent=True)
        self.assertFalse(board.money_scoring)       # back to the edited position's scoring


if __name__ == "__main__":
    unittest.main()
