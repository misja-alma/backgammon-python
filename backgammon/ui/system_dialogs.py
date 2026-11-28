"""The system's file dialogs and clipboard, used by File > Load/Save position and Edit > Copy/Paste.

Saving starts in the app's folder (~/almagammon, made when needed); loading starts in the folder a
file was last loaded from (remembered across sessions in ~/almagammon/.last-load-folder), or the
app's folder the first time.

On macOS the file dialogs are the system's own open and save panels, run inside the app through
PyObjC (pyobjc-framework-Cocoa: they open at once, double-clicking a file opens it); without PyObjC
they are shown with AppleScript (`osascript`, slower to appear). Like the app's own dialogs they
block until closed. The clipboard is `pbcopy` / `pbpaste`. Elsewhere the file dialogs fall back to
a path typed into the app's own dialog (PathDialog), and the clipboard to pygame's.
"""
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pygame

from backgammon.game import settings
from backgammon.ui.dialog import Dialog

MACOS = sys.platform == "darwin"
LAST_LOAD_FILE = ".last-load-folder"
POSITION_FILE_TYPES = ["xgp", "sgf", "txt"]   # what the load dialog shows

try:
    import AppKit
    import Foundation
    NATIVE = MACOS
except ImportError:   # not macOS, or PyObjC not installed
    NATIVE = False


def _make_panel(save: bool, prompt: str, folder: Path, default_name: Optional[str] = None,
                file_types: Optional[list[str]] = None):
    """The system's save or open panel, starting in folder. file_types: the extensions shown
    (open) or enforced (save)."""
    panel = AppKit.NSSavePanel.savePanel() if save else AppKit.NSOpenPanel.openPanel()
    panel.setTitle_(prompt)
    panel.setMessage_(prompt)
    if folder.is_dir():
        panel.setDirectoryURL_(Foundation.NSURL.fileURLWithPath_isDirectory_(str(folder), True))
    if save:
        panel.setNameFieldStringValue_(default_name or "")
        panel.setCanCreateDirectories_(True)
        panel.setExtensionHidden_(False)
    else:
        panel.setCanChooseFiles_(True)
        panel.setCanChooseDirectories_(False)
        panel.setAllowsMultipleSelection_(False)
    if file_types:
        panel.setAllowedFileTypes_(file_types)
    return panel


def _run_panel(panel) -> Optional[str]:
    """Show the panel until it closes; the chosen path, or None when cancelled."""
    AppKit.NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
    result = panel.runModal()
    # Clicks and keys meant for the panel must not reach the board.
    pygame.event.clear()
    if result != AppKit.NSModalResponseOK or panel.URL() is None:
        return None
    return str(panel.URL().path())


def _native_panel(save: bool, prompt: str, folder: Path, default_name: Optional[str] = None,
                  file_types: Optional[list[str]] = None) -> Optional[str]:
    return _single_extension(_run_panel(_make_panel(save, prompt, folder, default_name, file_types)))


def _single_extension(path: Optional[str]) -> Optional[str]:
    """The path without a doubled extension ('p.xgp.xgp' -> 'p.xgp'), which the save panel can
    give when the name already has the enforced extension."""
    if path is None:
        return None
    p = Path(path)
    if p.suffix and p.stem.endswith(p.suffix):
        return str(p.with_name(p.stem))
    return path


def _applescript_string(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _run_osascript(lines: list[str]) -> Optional[str]:
    """The script's output, or None when the user cancelled (osascript then fails)."""
    args = ["osascript"]
    for line in lines:
        args += ["-e", line]
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def app_folder() -> Path:
    """The app's folder in the home folder, made if it doesn't exist yet."""
    folder = settings.APP_FOLDER
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return folder


def save_folder() -> Path:
    return app_folder()


def load_folder() -> Path:
    """Where a file was last loaded from, when that folder still exists; else the app's folder."""
    try:
        folder = Path((app_folder() / LAST_LOAD_FILE).read_text(encoding="utf-8").strip())
        if folder.is_dir():
            return folder
    except OSError:
        pass
    return app_folder()


def _remember_load_folder(path: Optional[str]) -> Optional[str]:
    if path:
        try:
            (app_folder() / LAST_LOAD_FILE).write_text(str(Path(path).parent), encoding="utf-8")
        except OSError:
            pass
    return path


def _location_clause(folder: Path) -> str:
    return f" default location (POSIX file {_applescript_string(str(folder))})" if folder.is_dir() else ""


def choose_file_to_open(screen, board_map, board_snapshot, prompt: str) -> Optional[str]:
    """The path of a file the user picks, or None when cancelled."""
    folder = load_folder()
    if NATIVE:
        return _remember_load_folder(_native_panel(False, prompt, folder, file_types=POSITION_FILE_TYPES))
    if MACOS:
        return _remember_load_folder(_run_osascript([
            "tell me to activate",
            f"POSIX path of (choose file with prompt {_applescript_string(prompt)}{_location_clause(folder)})",
        ]))
    return _remember_load_folder(PathDialog(screen, board_map, prompt, str(folder) + "/").run(board_snapshot))


def choose_file_to_save(screen, board_map, board_snapshot, prompt: str, default_name: str) -> Optional[str]:
    """The path the user picks to save to (macOS asks before replacing a file), or None."""
    folder = save_folder()
    if NATIVE:
        extension = Path(default_name).suffix.lstrip(".")
        return _native_panel(True, prompt, folder, default_name, [extension] if extension else None)
    if MACOS:
        return _run_osascript([
            "tell me to activate",
            f"POSIX path of (choose file name with prompt {_applescript_string(prompt)} "
            f"default name {_applescript_string(default_name)}{_location_clause(folder)})",
        ])
    return PathDialog(screen, board_map, prompt, str(folder / default_name)).run(board_snapshot)


def copy_to_clipboard(text: str) -> None:
    if MACOS:
        subprocess.run(["pbcopy"], input=text, text=True, check=True)
        return
    if not pygame.scrap.get_init():
        pygame.scrap.init()
    pygame.scrap.put(pygame.SCRAP_TEXT, text.encode())


def paste_from_clipboard() -> str:
    if MACOS:
        return subprocess.run(["pbpaste"], capture_output=True, text=True, check=True).stdout
    if not pygame.scrap.get_init():
        pygame.scrap.init()
    data = pygame.scrap.get(pygame.SCRAP_TEXT)
    return data.decode(errors="replace").rstrip("\0") if data else ""


class PathDialog(Dialog):
    """A file path typed in (where there is no native file dialog). run() returns the path, or
    None when cancelled or empty."""

    def __init__(self, screen, board_map, title: str, path: str):
        super().__init__(screen, board_map, title, 560, 150)
        self.path = path
        self.field = self.rect(20, 60, 520, 28)

    def draw_content(self) -> None:
        # The end of a long path stays visible.
        text = self.path
        while text and self.small_font.size(text)[0] > self.field.width - 16:
            text = text[1:]
        self.draw_field(self.field, text, True)

    def handle_key(self, event) -> None:
        if event.key == pygame.K_BACKSPACE:
            self.path = self.path[:-1]
        elif event.unicode and event.unicode.isprintable():
            self.path += event.unicode

    def result(self):
        return self.path.strip() or None
