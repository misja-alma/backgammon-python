import os
import unittest
from unittest.mock import MagicMock

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame  # noqa: E402

from backgammon.game.settings import Settings  # noqa: E402
from backgammon.ui.board_map import BoardMap  # noqa: E402
from backgammon.ui.settings_dialog import SettingsDialog, format_megabytes  # noqa: E402


class SettingsDialogTest(unittest.TestCase):

    def setUp(self):
        pygame.init()
        self.screen = pygame.display.set_mode((1024, 768))
        self.worker = MagicMock()
        self.saved = (Settings.rust_cache_mb, Settings.rust_threads, Settings.analysis_depth)
        Settings.rust_cache_mb = 1024
        Settings.rust_threads = 0

    def tearDown(self):
        Settings.rust_cache_mb, Settings.rust_threads, Settings.analysis_depth = self.saved
        pygame.quit()

    def dialog(self, result):
        dialog = SettingsDialog(self.screen, BoardMap(1024, 768), self.worker)
        dialog.run = lambda snapshot: result  # skip the interactive loop
        return dialog

    def test_cache_buttons_halve_and_double_within_limits(self):
        dialog = self.dialog(True)
        dialog.handle_click(dialog.btn_cache_plus.center)
        self.assertEqual(2048, dialog.cache_mb)
        for _ in range(10):
            dialog.handle_click(dialog.btn_cache_minus.center)
        self.assertEqual(SettingsDialog.MIN_CACHE_MB, dialog.cache_mb)
        for _ in range(20):
            dialog.handle_click(dialog.btn_cache_plus.center)
        self.assertEqual(SettingsDialog.MAX_CACHE_MB, dialog.cache_mb)

    def test_ok_applies_a_changed_cache_size(self):
        dialog = self.dialog(True)
        dialog.handle_click(dialog.btn_cache_minus.center)
        dialog.show(None)
        self.assertEqual(512, Settings.rust_cache_mb)
        self.worker.set_rust_cache_size.assert_called_once_with(512)

    def test_unchanged_size_or_cancel_does_not_resize(self):
        self.dialog(True).show(None)
        cancelled = self.dialog(None)
        cancelled.handle_click(cancelled.btn_cache_plus.center)
        cancelled.show(None)
        self.assertEqual(1024, Settings.rust_cache_mb)
        self.worker.set_rust_cache_size.assert_not_called()

    def test_ply_one_is_depth_zero(self):
        dialog = self.dialog(True)
        for _ in range(10):
            dialog.handle_click(dialog.btn_minus.center)
        dialog.show(None)
        self.assertEqual(0, Settings.analysis_depth)   # shown as 1-ply

    def test_threads_default_to_the_cores_and_ok_applies_a_change(self):
        dialog = self.dialog(True)
        self.assertEqual(SettingsDialog.MAX_THREADS, dialog.threads)
        dialog.handle_click(dialog.btn_threads_plus.center)
        self.assertEqual(SettingsDialog.MAX_THREADS, dialog.threads)
        for _ in range(SettingsDialog.MAX_THREADS + 1):
            dialog.handle_click(dialog.btn_threads_minus.center)
        self.assertEqual(1, dialog.threads)
        dialog.show(None)
        self.assertEqual(1, Settings.rust_threads)
        self.worker.set_rust_threads.assert_called_once_with(1)

    def test_unchanged_threads_are_not_reset(self):
        self.dialog(True).show(None)
        self.worker.set_rust_threads.assert_not_called()

    def test_format(self):
        self.assertEqual("1 GB", format_megabytes(1024))
        self.assertEqual("512 MB", format_megabytes(512))


if __name__ == '__main__':
    unittest.main()
