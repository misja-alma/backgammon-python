from backgammon.game.settings import Settings, Analyzer
from backgammon.ui.dialog import Dialog


def format_megabytes(megabytes: int) -> str:
    return f"{megabytes // 1024} GB" if megabytes >= 1024 else f"{megabytes} MB"


class SettingsDialog(Dialog):
    """Engine and depth used by the Analyze menu, and the Rust engine's cache size."""

    MIN_DEPTH, MAX_DEPTH = 1, 99
    MIN_CACHE_MB, MAX_CACHE_MB = 64, 16 * 1024   # the −/+ buttons halve and double the size

    def __init__(self, screen, board_map, analysis_worker):
        super().__init__(screen, board_map, "Settings", 360, 290)
        self.analysis_worker = analysis_worker
        self.analyzer = Settings.analyzer
        self.depth = Settings.analysis_depth
        self.cache_mb = Settings.rust_cache_mb
        self.radio_python = self.rect(30, 75, 16, 16)
        self.radio_rust = self.rect(30, 105, 16, 16)
        self.radio_bgsage = self.rect(30, 135, 16, 16)
        self.btn_minus = self.rect(195, 160, 28, 28)
        self.btn_plus = self.rect(275, 160, 28, 28)
        self.btn_cache_minus = self.rect(195, 200, 28, 28)
        self.btn_cache_plus = self.rect(275, 200, 28, 28)

    def show(self, board_snapshot) -> None:
        if not self.run(board_snapshot):
            return
        Settings.analyzer = self.analyzer
        Settings.analysis_depth = self.depth
        if self.cache_mb != Settings.rust_cache_mb:
            Settings.rust_cache_mb = self.cache_mb
            # Runs on the engine's thread, after any search in progress; the cache starts empty.
            self.analysis_worker.set_rust_cache_size(self.cache_mb)

    def draw_content(self) -> None:
        self.draw_text(20, 52, "Analyzer:")
        self.draw_radio(self.radio_python, self.analyzer == Analyzer.PYTHON, "Python")
        self.draw_radio(self.radio_rust, self.analyzer == Analyzer.RUST, "Rust")
        self.draw_radio(self.radio_bgsage, self.analyzer == Analyzer.BGSAGE, "bgsage (1-point match)")
        self.draw_text(20, 168, "Analysis depth:")
        self.draw_stepper(self.btn_minus, self.btn_plus, self.depth)
        self.draw_text(20, 208, "Rust engine cache:")
        self.draw_stepper(self.btn_cache_minus, self.btn_cache_plus, format_megabytes(self.cache_mb))

    def handle_click(self, pos) -> None:
        if self.radio_python.collidepoint(pos):
            self.analyzer = Analyzer.PYTHON
        elif self.radio_rust.collidepoint(pos):
            self.analyzer = Analyzer.RUST
        elif self.radio_bgsage.collidepoint(pos):
            self.analyzer = Analyzer.BGSAGE
            self.depth = self.analyzer.clamp_depth(self.depth)
        elif self.btn_minus.collidepoint(pos):
            self.depth = max(self.MIN_DEPTH, self.depth - 1)
        elif self.btn_plus.collidepoint(pos):
            self.depth = self.analyzer.clamp_depth(min(self.MAX_DEPTH, self.depth + 1))
        elif self.btn_cache_minus.collidepoint(pos):
            self.cache_mb = max(self.MIN_CACHE_MB, self.cache_mb // 2)
        elif self.btn_cache_plus.collidepoint(pos):
            self.cache_mb = min(self.MAX_CACHE_MB, self.cache_mb * 2)
