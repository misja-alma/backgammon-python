import os

from backgammon.game.settings import Settings, Analyzer
from backgammon.ui.dialog import Dialog


def format_megabytes(megabytes: int) -> str:
    return f"{megabytes // 1024} GB" if megabytes >= 1024 else f"{megabytes} MB"


def format_margin(margin) -> str:
    return "all" if margin is None else f"{margin:.2f}"


def format_moves(moves) -> str:
    return "all" if moves is None else str(moves)


def step(values: list, current, direction: int):
    """The value after current in values (direction 1) or before it (-1), staying at the ends; a
    current value not in the list goes to the nearest listed one in that direction."""
    if current in values:
        return values[max(0, min(len(values) - 1, values.index(current) + direction))]
    order = [v for v in values if v is not None]
    if direction > 0:
        return next((v for v in order if v > current), values[-1])
    return next((v for v in reversed(order) if v < current), values[0])


class SettingsDialog(Dialog):
    """Engine and search strength used by the Analyze menu, the Rust engine's search space,
    cache size and threads. The strength is shown in plies, as in other programs: 1-ply is the
    net's evaluation of the positions after each move (search depth 0), n-ply searches n - 1 roll
    levels deeper. The search space limits which moves a search of 2-ply or more looks at more
    deeply: those within the equity margin of the best at 1-ply, at most the maximum number."""

    MARGINS = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.16, 0.2, 0.3, 0.5, None]
    MAX_MOVES = list(range(1, 21)) + [None]

    MIN_DEPTH, MAX_DEPTH = 0, 98
    MAX_THREADS = os.cpu_count() or 1
    MIN_CACHE_MB, MAX_CACHE_MB = 64, 16 * 1024   # the −/+ buttons halve and double the size

    def __init__(self, screen, board_map, analysis_worker):
        super().__init__(screen, board_map, "Settings", 360, 430)
        self.analysis_worker = analysis_worker
        self.analyzer = Settings.analyzer
        self.depth = Settings.analysis_depth
        self.cache_mb = Settings.rust_cache_mb
        self.threads = min(Settings.rust_threads or self.MAX_THREADS, self.MAX_THREADS)
        self.margin = Settings.search_space_margin
        self.moves = Settings.search_space_moves
        self.radio_python = self.rect(30, 75, 16, 16)
        self.radio_rust = self.rect(30, 105, 16, 16)
        self.radio_bgsage = self.rect(30, 135, 16, 16)
        self.btn_minus = self.rect(195, 160, 28, 28)
        self.btn_plus = self.rect(275, 160, 28, 28)
        self.btn_margin_minus = self.rect(195, 225, 28, 28)
        self.btn_margin_plus = self.rect(275, 225, 28, 28)
        self.btn_moves_minus = self.rect(195, 260, 28, 28)
        self.btn_moves_plus = self.rect(275, 260, 28, 28)
        self.btn_cache_minus = self.rect(195, 300, 28, 28)
        self.btn_cache_plus = self.rect(275, 300, 28, 28)
        self.btn_threads_minus = self.rect(195, 340, 28, 28)
        self.btn_threads_plus = self.rect(275, 340, 28, 28)

    def show(self, board_snapshot) -> None:
        if not self.run(board_snapshot):
            return
        Settings.analyzer = self.analyzer
        Settings.analysis_depth = self.depth
        if self.cache_mb != Settings.rust_cache_mb:
            Settings.rust_cache_mb = self.cache_mb
            # Runs on the engine's thread, after any search in progress; the cache starts empty.
            self.analysis_worker.set_rust_cache_size(self.cache_mb)
        if (self.margin, self.moves) != (Settings.search_space_margin, Settings.search_space_moves):
            Settings.search_space_margin, Settings.search_space_moves = self.margin, self.moves
            self.analysis_worker.set_rust_move_filter(self.margin, self.moves)
        if self.threads != (Settings.rust_threads or self.MAX_THREADS):
            Settings.rust_threads = self.threads
            self.analysis_worker.set_rust_threads(self.threads)

    def draw_content(self) -> None:
        self.draw_text(20, 52, "Analyzer:")
        self.draw_radio(self.radio_python, self.analyzer == Analyzer.PYTHON, "Python")
        self.draw_radio(self.radio_rust, self.analyzer == Analyzer.RUST, "Rust")
        self.draw_radio(self.radio_bgsage, self.analyzer == Analyzer.BGSAGE, "bgsage (1-point match)")
        self.draw_text(20, 168, "Analysis ply:")
        self.draw_stepper(self.btn_minus, self.btn_plus, f"{self.depth + 1}-ply")
        self.draw_text(20, 202, "Search space (2-ply and more):")
        self.draw_text(35, 232, "Equity margin:")
        self.draw_stepper(self.btn_margin_minus, self.btn_margin_plus, format_margin(self.margin))
        self.draw_text(35, 267, "Max moves:")
        self.draw_stepper(self.btn_moves_minus, self.btn_moves_plus, format_moves(self.moves))
        self.draw_text(20, 308, "Rust engine cache:")
        self.draw_stepper(self.btn_cache_minus, self.btn_cache_plus, format_megabytes(self.cache_mb))
        self.draw_text(20, 348, "Evaluator threads:")
        self.draw_stepper(self.btn_threads_minus, self.btn_threads_plus, self.threads)

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
        elif self.btn_margin_minus.collidepoint(pos):
            self.margin = step(self.MARGINS, self.margin, -1)
        elif self.btn_margin_plus.collidepoint(pos):
            self.margin = step(self.MARGINS, self.margin, 1)
        elif self.btn_moves_minus.collidepoint(pos):
            self.moves = step(self.MAX_MOVES, self.moves, -1)
        elif self.btn_moves_plus.collidepoint(pos):
            self.moves = step(self.MAX_MOVES, self.moves, 1)
        elif self.btn_cache_minus.collidepoint(pos):
            self.cache_mb = max(self.MIN_CACHE_MB, self.cache_mb // 2)
        elif self.btn_cache_plus.collidepoint(pos):
            self.cache_mb = min(self.MAX_CACHE_MB, self.cache_mb * 2)
        elif self.btn_threads_minus.collidepoint(pos):
            self.threads = max(1, self.threads - 1)
        elif self.btn_threads_plus.collidepoint(pos):
            self.threads = min(self.MAX_THREADS, self.threads + 1)
