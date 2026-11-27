import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame  # noqa: E402

from backgammon.ui.board_map import BoardMap  # noqa: E402

# Window sizes: default, macOS maximized (wide), very wide, tall/narrow, small.
SIZES = [(1024, 768), (1728, 1051), (2560, 1080), (900, 1200), (1500, 900), (640, 480)]


class BoardMapLayoutTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pygame.init()

    @classmethod
    def tearDownClass(cls):
        pygame.quit()

    def test_board_fits_window_and_does_not_overlap_text_area(self):
        for width, height in SIZES:
            with self.subTest(size=(width, height)):
                m = BoardMap(width, height)
                board_bottom = m.board_y + m.board_height
                self.assertGreaterEqual(m.board_x, 0)
                self.assertLessEqual(m.board_x + m.board_width, width)
                self.assertLess(board_bottom + m.label_margin, m.text_area_y)
                self.assertLessEqual(m.text_area_y + m.text_area_height, height)

    def test_board_keeps_its_proportions(self):
        # Points are always 14 per board width and the board is ~20 checker radii tall,
        # regardless of the window's aspect ratio.
        for width, height in SIZES:
            with self.subTest(size=(width, height)):
                m = BoardMap(width, height)
                self.assertLess(m.board_width - 14 * m.point_width, 14)
                self.assertEqual(20 * m.checker_radius, m.board_height)

    def test_board_is_centered_right_of_the_panel(self):
        for width, height in SIZES:
            with self.subTest(size=(width, height)):
                m = BoardMap(width, height)
                left = m.board_x - m.panel_width
                right = width - (m.board_x + m.board_width)
                self.assertGreater(left, 0)
                self.assertLessEqual(abs(left - right), 1)

    def test_default_size_layout(self):
        m = BoardMap(1024, 768)
        self.assertEqual(m.panel_width + m.board_margin, m.board_x)
        self.assertEqual(m.board_margin, m.board_y)
        self.assertEqual(1024 - m.panel_width - 2 * m.board_margin, m.board_width)


if __name__ == '__main__':
    unittest.main()
