import math
from typing import Optional

import pygame

from backgammon.game.move_analysis import Analysis, format_probs
from backgammon.game.position import Player, Position


def _player_name(player: Player) -> str:
    return "ME" if player == Player.ME else "OPPONENT"


class AnalysisPanel:
    """The analysis column left of the board, like eXtreme Gammon's move list.

    A move analysis lists every move, best first, in two lines: the move with its equity (the
    best move's own, the others' difference to it), then the probabilities of the player (P: win,
    gammon, backgammon) and of the opponent (O:). Clicking a move selects it and shows it on the
    board; clicking the selected move again plays it, when it's that player's turn. Shift- or
    Cmd-click selects several moves (shown in light blue, the last one on the board), which
    Analyze > Rollout then rolls out. A position
    analysis shows the equity and probabilities of the side on roll. A rollout is shown the same
    way, with the standard error of each equity (±). Moves outside the search space (not searched
    deeper) follow the others, marked with the plies they were evaluated at (1-ply).
    """

    BACKGROUND = (250, 250, 250)
    HEADER = (225, 225, 230)
    SELECTED = (0, 102, 204)
    MARKED = (190, 215, 245)     # moves selected for a rollout, besides the one shown
    BEST = (0, 110, 0)
    WORSE = (170, 0, 0)
    GREY = (110, 110, 110)
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)

    def __init__(self, screen, board_map):
        self.screen = screen
        self.map = board_map
        self.analysis: Optional[Analysis] = None
        self.status: str = "Ctrl+A: analyze the position, or the moves with the dice (click the dice on the board to set them)"
        self.selected: Optional[int] = None
        # The moves selected for a rollout (Shift- or Cmd-click adds or removes one; a plain
        # click selects just the clicked move), in the order they were selected.
        self.marked: list[int] = []
        self._scroll = 0   # first visible move

    # --- state ---

    def show_status(self, text: str) -> None:
        self.analysis, self.selected, self._scroll = None, None, 0
        self.marked = []
        self.status = text

    def set_analysis(self, analysis: Analysis) -> None:
        self.analysis, self.selected, self._scroll = analysis, None, 0
        self.marked = []
        self.status = ""

    def clear(self) -> None:
        self.show_status("")

    def preview(self) -> Optional[Position]:
        """The selected move's position, to show on the board, or None."""
        if self.analysis is None or self.selected is None:
            return None
        return self.analysis.moves[self.selected].position

    def marked_moves(self) -> list:
        """The moves selected for a rollout, best first (as listed)."""
        if self.analysis is None:
            return []
        return [self.analysis.moves[i] for i in sorted(self.marked)]

    def selected_move(self):
        if self.analysis is None or self.selected is None:
            return None
        return self.analysis.moves[self.selected]

    # --- geometry ---

    def _rect(self) -> pygame.Rect:
        return pygame.Rect(self.map.panel_x, self.map.panel_y, self.map.panel_width, self.map.panel_height)

    def _fonts(self):
        size = self.map.ui_font_size
        return pygame.font.Font(None, round(size * 0.95)), pygame.font.Font(None, round(size * 0.8))

    def _row_height(self) -> int:
        big, small = self._fonts()
        return big.get_linesize() + small.get_linesize() + 6

    def _header_height(self) -> int:
        big, _ = self._fonts()
        return 2 * big.get_linesize() + 10

    def _visible_rows(self) -> int:
        return max(1, (self._rect().height - self._header_height()) // self._row_height())

    def _row_at(self, pos) -> Optional[int]:
        rect = self._rect()
        if self.analysis is None or not rect.collidepoint(pos):
            return None
        y = pos[1] - rect.y - self._header_height()
        if y < 0:
            return None
        index = self._scroll + y // self._row_height()
        return index if index < len(self.analysis.moves) else None

    # --- events ---

    def handle_wheel(self, dy: int) -> bool:
        if self.analysis is None or not self._rect().collidepoint(pygame.mouse.get_pos()):
            return False
        top = max(0, len(self.analysis.moves) - self._visible_rows())
        self._scroll = max(0, min(top, self._scroll - dy))
        return True

    def handle_click(self, pos, toggle: bool = False) -> Optional[str]:
        """A left click: None if it isn't on the panel, 'select' for a newly selected move (or a
        click elsewhere on the panel, which deselects), 'play' for a click on the selected move
        when it is the only one selected. toggle (Shift- or Cmd-click): add the move to the
        moves selected for a rollout, or remove it."""
        if not self._rect().collidepoint(pos):
            return None
        index = self._row_at(pos)
        if toggle:
            if index is None:
                return "select"
            if index in self.marked:
                self.marked.remove(index)
                self.selected = self.marked[-1] if self.marked else None
            else:
                self.marked.append(index)
                self.selected = index
            return "select"
        if index is not None and index == self.selected and self.marked == [index]:
            return "play"
        self.selected = index
        self.marked = [] if index is None else [index]
        return "select"

    # --- drawing ---

    def draw(self) -> None:
        rect = self._rect()
        big, small = self._fonts()
        pygame.draw.rect(self.screen, self.BACKGROUND, rect)
        pygame.draw.line(self.screen, self.BLACK, (rect.right - 1, rect.y), (rect.right - 1, rect.bottom), 1)
        pad = self.map.text_padding
        header = pygame.Rect(rect.x, rect.y, rect.width - 1, self._header_height())
        pygame.draw.rect(self.screen, self.HEADER, header)
        line = big.get_linesize()

        a = self.analysis
        if a is None:
            self._blit_wrapped(self.status, big, rect.x + pad, rect.y + 5, rect.width - 2 * pad)
            return
        scoring = "money" if a.money else "double match point"
        if a.dice is None:
            title = f"Position: {_player_name(a.player)} on roll"
        else:
            title = f"{_player_name(a.player)} to play {a.dice[0]}-{a.dice[1]}"
        self._blit(title, big, self.BLACK, rect.x + pad, rect.y + 5)
        if a.rollout_games is None:
            subtitle = f"{a.depth + 1}-ply, {scoring}"
        else:
            per_move = " per move" if a.dice is not None else ""
            subtitle = f"rollout {a.rollout_games} games{per_move}, {a.depth + 1}-ply, {scoring}"
        self._blit(subtitle, small, self.GREY, rect.x + pad, rect.y + 5 + line)
        y = rect.y + self._header_height() + 4

        clip = self.screen.get_clip()
        self.screen.set_clip(pygame.Rect(rect.x, y, rect.width, rect.bottom - y))
        if a.dice is None:
            error = "" if a.error is None else f"  ±{a.error:.3f}"
            self._blit(f"Equity {a.equity:+.3f}{error}", big, self.BLACK, rect.x + pad, y)
            self._draw_probs(a.probs, small, rect.x + pad, y + line, self.BLACK)
        else:
            best = a.moves[0].equity if a.moves else 0.0
            for i in range(self._scroll, min(len(a.moves), self._scroll + self._visible_rows() + 1)):
                m = a.moves[i]
                row = pygame.Rect(rect.x, y, rect.width - 1, self._row_height())
                selected = i == self.selected
                if selected:
                    pygame.draw.rect(self.screen, self.SELECTED, row)
                elif i in self.marked:
                    pygame.draw.rect(self.screen, self.MARKED, row)
                text_color = self.WHITE if selected else self.BLACK
                self._blit(f"{i + 1}. {m.notation}", big, text_color, rect.x + pad, y + 2)
                equity = f"{m.equity:+.3f}" if i == 0 else f"{m.equity - best:+.3f}"
                color = self.WHITE if selected else (self.BEST if i == 0 or m.equity >= best else self.WORSE)
                surface = big.render(equity, True, color)
                self.screen.blit(surface, (rect.right - pad - surface.get_width(), y + 2))
                self._draw_probs(m.probs, small, rect.x + pad, y + 2 + line, self.WHITE if selected else self.GREY)
                note = (f"±{m.error:.3f}" if m.error is not None else
                        f"{m.levels + 1}-ply" if m.levels is not None else None)
                if note is not None:
                    surface = small.render(note, True, self.WHITE if selected else self.GREY)
                    self.screen.blit(surface, (rect.right - pad - surface.get_width(), y + 2 + line))
                pygame.draw.line(self.screen, (220, 220, 220), (rect.x, row.bottom - 1), (rect.right - 2, row.bottom - 1))
                y += self._row_height()
        self.screen.set_clip(clip)

    def _draw_probs(self, probs, font, x, y, color) -> None:
        win = probs[0]
        mine = format_probs(probs[:3])
        theirs = format_probs([1 - win if not math.isnan(win) else win] + probs[3:])
        self._blit(f"P: {mine}    O: {theirs}", font, color, x, y)

    def _blit(self, text, font, color, x, y) -> None:
        self.screen.blit(font.render(text, True, color), (x, y))

    def _blit_wrapped(self, text, font, x, y, width) -> None:
        words, lines, current = text.split(), [], ""
        for word in words:
            trial = f"{current} {word}".strip()
            if current and font.size(trial)[0] > width:
                lines.append(current)
                current = word
            else:
                current = trial
        if current:
            lines.append(current)
        for i, text_line in enumerate(lines):
            self._blit(text_line, font, self.GREY, x, y + i * font.get_linesize())
