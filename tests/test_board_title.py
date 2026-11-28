"""The title bar shows the file the position was last loaded from or saved to."""
import os
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame  # noqa: E402

from backgammon.game.position_formats import Format  # noqa: E402
from backgammon.ui.board import Board  # noqa: E402


class TitleBarTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.board = Board()

    @classmethod
    def tearDownClass(cls):
        pygame.quit()

    @staticmethod
    def title() -> str:
        return pygame.display.get_caption()[0]

    def test_load_and_save_show_the_file_name_and_a_new_board_clears_it(self):
        board = self.board
        board.restore_starting_position()
        self.assertEqual("Alma Gammon", self.title())
        with tempfile.TemporaryDirectory() as d:
            saved = os.path.join(d, "my position.xgp")
            with patch("backgammon.ui.board.SavePositionDialog") as dialog, \
                    patch("backgammon.ui.board.choose_file_to_save", return_value=saved):
                dialog.return_value.run.return_value = Format.XG
                board.save_position()
            self.assertEqual("Alma Gammon - my position.xgp", self.title())
            text = os.path.join(d, "other.txt")
            with open(text, "w") as f:
                f.write("XGID=-b----E-C---eE---c-e----B-:0:0:1:31:0:0:0:0:10\n")
            with patch("backgammon.ui.board.choose_file_to_open", return_value=text):
                board.load_position()
            self.assertEqual("Alma Gammon - other.txt", self.title())
            with patch("backgammon.ui.board.choose_file_to_open", return_value=None):   # cancelled
                board.load_position()
            self.assertEqual("Alma Gammon - other.txt", self.title())
        board.restore_starting_position()
        self.assertEqual("Alma Gammon", self.title())

    def test_a_pasted_position_has_no_file(self):
        board = self.board
        board.set_position_file("/tmp/x.sgf")
        with patch("backgammon.ui.board.paste_from_clipboard",
                   return_value="XGID=-b----E-C---eE---c-e----B-:0:0:1:00:0:0:0:0:10"):
            board.paste_position()
        self.assertEqual("Alma Gammon", self.title())


if __name__ == "__main__":
    unittest.main()
