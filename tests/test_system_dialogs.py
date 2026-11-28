import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from backgammon.game import settings  # noqa: E402
from backgammon.ui import system_dialogs as sd  # noqa: E402


class FileDialogFoldersTest(unittest.TestCase):
    """Saving starts in the app's folder; loading where the last file was loaded from."""

    def setUp(self):
        self.home = tempfile.TemporaryDirectory()
        self.folder = Path(self.home.name) / "almagammon"
        self.patch = patch.object(settings, "APP_FOLDER", self.folder)
        self.patch.start()
        self.scripts = []

    def tearDown(self):
        self.patch.stop()
        self.home.cleanup()

    def native(self, answer):
        def run(save, prompt, folder, default_name=None, file_types=None):
            self.scripts.append((save, folder, default_name, file_types))
            return answer
        return patch.object(sd, "_native_panel", side_effect=run)

    def osascript(self, answer):
        def run(lines):
            self.scripts.append(" ".join(lines))
            return answer
        return patch.object(sd, "_run_osascript", side_effect=run)

    def test_save_starts_in_the_app_folder_which_is_made(self):
        with patch.object(sd, "MACOS", True), patch.object(sd, "NATIVE", False), self.osascript(str(self.folder / "p.txt")):
            self.assertEqual(str(self.folder / "p.txt"), sd.choose_file_to_save(None, None, None, "Save", "p.txt"))
        self.assertTrue(self.folder.is_dir())
        self.assertIn(f'default location (POSIX file "{self.folder}")', self.scripts[0])

    def test_load_starts_in_the_app_folder_then_where_the_last_file_was_loaded(self):
        other = Path(self.home.name) / "elsewhere"
        other.mkdir()
        with patch.object(sd, "MACOS", True), patch.object(sd, "NATIVE", False), self.osascript(str(other / "x.xgp")):
            sd.choose_file_to_open(None, None, None, "Load")
        self.assertIn(f'POSIX file "{self.folder}"', self.scripts[0])
        self.assertEqual(other, sd.load_folder())
        with patch.object(sd, "MACOS", True), patch.object(sd, "NATIVE", False), self.osascript(None):   # cancelled: nothing changes
            sd.choose_file_to_open(None, None, None, "Load")
        self.assertIn(f'POSIX file "{other}"', self.scripts[1])
        self.assertEqual(other, sd.load_folder())
        # Saving still starts in the app folder.
        with patch.object(sd, "MACOS", True), patch.object(sd, "NATIVE", False), self.osascript(None):
            sd.choose_file_to_save(None, None, None, "Save", "p.txt")
        self.assertIn(f'POSIX file "{self.folder}"', self.scripts[2])

    def test_a_removed_load_folder_falls_back_to_the_app_folder(self):
        gone = Path(self.home.name) / "gone"
        gone.mkdir()
        with patch.object(sd, "MACOS", True), patch.object(sd, "NATIVE", False), self.osascript(str(gone / "x.txt")):
            sd.choose_file_to_open(None, None, None, "Load")
        gone.rmdir()
        self.assertEqual(self.folder, sd.load_folder())


class NativePanelFoldersTest(FileDialogFoldersTest):
    """The same with the system's own panels (PyObjC)."""

    def test_native_panels_start_in_the_right_folders_and_filter_types(self):
        other = Path(self.home.name) / "elsewhere"
        other.mkdir()
        with patch.object(sd, "NATIVE", True), self.native(str(other / "x.xgp")):
            self.assertEqual(str(other / "x.xgp"), sd.choose_file_to_open(None, None, None, "Load"))
        with patch.object(sd, "NATIVE", True), self.native(None):
            sd.choose_file_to_open(None, None, None, "Load")
            sd.choose_file_to_save(None, None, None, "Save", "position.xgp")
        self.assertEqual((False, self.folder, None, sd.POSITION_FILE_TYPES), self.scripts[0])
        self.assertEqual((False, other, None, sd.POSITION_FILE_TYPES), self.scripts[1])
        self.assertEqual((True, self.folder, "position.xgp", ["xgp"]), self.scripts[2])


class SingleExtensionTest(unittest.TestCase):

    def test_a_doubled_extension_is_removed(self):
        self.assertEqual("/a/p.xgp", sd._single_extension("/a/p.xgp.xgp"))
        self.assertEqual("/a/p.xgp", sd._single_extension("/a/p.xgp"))
        self.assertEqual("/a/p.txt.xgp", sd._single_extension("/a/p.txt.xgp"))
        self.assertIsNone(sd._single_extension(None))


if __name__ == "__main__":
    unittest.main()
