from typing import Optional

import pygame

from backgammon.ui.dialog import Dialog


class RolloutDialog(Dialog):
    """Settings for a rollout: the number of games (per move), the strength of the engine playing
    them, in plies (1-ply: each move chosen by the net's evaluation of the positions after it,
    depth 0), and when moves are rolled out (dice set) how many candidate moves. run() returns
    (games, depth, candidates), or None when cancelled.

    candidates: the starting number of candidates (the moves selected in the analysis, when
    there are any; see Board.rollout), or None when a position is rolled out (no moves: the
    field is disabled)."""

    MIN_PLY, MAX_PLY = 1, 3   # 1296 games of one position: 0.1 s at 1-ply, ~35 s at 2-ply
    MAX_GAMES = 1_000_000
    MAX_CANDIDATES = 999
    DEFAULT_CANDIDATES = 3
    GAMES_LABEL, PLY_LABEL, CANDIDATES_LABEL = "Number of games:", "Evaluator strength:", "Candidate moves:"

    # Remembered between openings of the dialog.
    _last_games = 1296
    _last_ply = 1

    def __init__(self, screen, board_map, candidates: Optional[int] = DEFAULT_CANDIDATES, note: str = ""):
        # The controls go right of the longest label, sized to the text (measured in design
        # units, see Dialog), so that nothing overlaps.
        font = self.design_font()
        labels = (self.GAMES_LABEL, self.PLY_LABEL, self.CANDIDATES_LABEL)
        label_width = max(font.size(text)[0] for text in labels)
        box = max(28, font.get_linesize() + 10)                   # button and field height
        gap = 14
        field_width = font.size("8" * len(str(self.MAX_GAMES)))[0] + 16
        value_width = font.size(f"{self.MAX_PLY}-ply")[0] + 16    # between the stepper's buttons
        stepper_width = 2 * box + value_width
        controls_x = 20 + label_width + 20
        self.note = note
        note_width = font.size(note)[0] + 40 if note else 0
        width = max(controls_x + max(field_width, stepper_width) + 20, note_width, 340)
        top = 30 + font.get_linesize() + 20                       # below the title
        rows = 3
        note_height = font.get_linesize() + 6 if note else 0
        height = top + rows * box + (rows - 1) * gap + note_height + 70   # rows, note, OK / Cancel
        super().__init__(screen, board_map, "Rollout", width, height)

        row_y = [top + i * (box + gap) for i in range(rows)]
        self.label_y = [y + (box - font.get_linesize()) // 2 for y in row_y]
        self.note_y = row_y[-1] + box + 6
        self.games_text = str(RolloutDialog._last_games)
        self.ply = RolloutDialog._last_ply
        self.candidates_enabled = candidates is not None
        self.candidates_text = str(candidates) if candidates is not None else ""
        self.field_games = self.rect(controls_x, row_y[0], field_width, box)
        self.btn_minus = self.rect(controls_x, row_y[1], box, box)
        self.btn_plus = self.rect(controls_x + box + value_width, row_y[1], box, box)
        self.field_candidates = self.rect(controls_x, row_y[2], field_width, box)
        self.focus = "games"   # the field with the keyboard focus: "games" or "candidates"

    @staticmethod
    def _number(text: str, maximum: int) -> Optional[int]:
        return int(text) if text and 0 < int(text) <= maximum else None

    def _games(self) -> Optional[int]:
        return self._number(self.games_text, self.MAX_GAMES)

    def _candidates(self) -> Optional[int]:
        return self._number(self.candidates_text, self.MAX_CANDIDATES)

    def draw_content(self) -> None:
        self.draw_text(20, self.label_y[0], self.GAMES_LABEL)
        self.draw_field(self.field_games, self.games_text, self.focus == "games")
        self.draw_text(20, self.label_y[1], self.PLY_LABEL)
        self.draw_stepper(self.btn_minus, self.btn_plus, f"{self.ply}-ply")
        enabled = self.candidates_enabled
        self.draw_text(20, self.label_y[2], self.CANDIDATES_LABEL, enabled=enabled)
        if enabled:
            self.draw_field(self.field_candidates, self.candidates_text, self.focus == "candidates")
        else:
            self.draw_text(self.field_candidates.x - self.x + 6, self.label_y[2], "- (no dice)", enabled=False)
        if self.note:
            self.draw_text(20, self.note_y, self.note, enabled=False)

    def handle_click(self, pos) -> None:
        if self.field_games.collidepoint(pos):
            self.focus = "games"
        elif self.candidates_enabled and self.field_candidates.collidepoint(pos):
            self.focus = "candidates"
        elif self.btn_minus.collidepoint(pos):
            self.ply = max(self.MIN_PLY, self.ply - 1)
        elif self.btn_plus.collidepoint(pos):
            self.ply = min(self.MAX_PLY, self.ply + 1)

    def handle_key(self, event) -> None:
        if event.key == pygame.K_TAB:
            if self.candidates_enabled:
                self.focus = "candidates" if self.focus == "games" else "games"
            return
        attr, maximum = (("games_text", self.MAX_GAMES) if self.focus == "games"
                         else ("candidates_text", self.MAX_CANDIDATES))
        text = getattr(self, attr)
        if event.key == pygame.K_BACKSPACE:
            text = text[:-1]
        elif event.unicode.isdigit() and len(text) < len(str(maximum)):
            text = (text + event.unicode).lstrip("0")
        setattr(self, attr, text)

    def run(self, board_snapshot):
        while True:
            result = super().run(board_snapshot)
            # OK (or Enter) with an invalid number: keep the dialog open.
            if result is not False:
                return result

    def result(self):
        games = self._games()
        candidates = self._candidates() if self.candidates_enabled else None
        if games is None or (self.candidates_enabled and candidates is None):
            return False
        RolloutDialog._last_games, RolloutDialog._last_ply = games, self.ply
        return games, self.ply - 1, candidates
