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
    board; clicking the selected move again plays it, when it's that player's turn. A position
    analysis shows the equity and probabilities of the side on roll.
    """

    BACKGROUND = (250, 250, 250)
    HEADER = (225, 225, 230)
    SELECTED = (0, 102, 204)
    BEST = (0, 110, 0)
    WORSE = (170, 0, 0)
    GREY = (110, 110, 110)
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)

    def __init__(self, screen, board_map):
        self.screen = screen
        self.map = board_map
        self.analysis: Optional[Analysis] = None
        self.status: str = "Ctrl+A: analyze the position, or the moves on your turn"
        self.selected: Optional[int] = None
        self._scroll = 0   # first visible move

    # --- state ---

    def show_status(self, text: str) -> None:
        self.analysis, self.selected, self._scroll = None, None, 0
        self.status = text

    def set_analysis(self, analysis: Analysis) -> None:
        self.analysis, self.selected, self._scroll = analysis, None, 0
        self.status = ""

    def clear(self) -> None:
        self.show_status("")

    def preview(self) -> Optional[Position]:
        """The selected move's position, to show on the board, or None."""
        if self.analysis is None or self.selected is None:
            return None
        return self.analysis.moves[self.selected].position

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

    def handle_click(self, pos) -> Optional[str]:
        """A left click: None if it isn't on the panel, 'select' for a newly selected move (or a
        click elsewhere on the panel, which deselects), 'play' for a click on the selected move."""
        if not self._rect().collidepoint(pos):
            return None
        index = self._row_at(pos)
        if index is not None and index == self.selected:
            return "play"
        self.selected = index
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
        self._blit(f"depth {a.depth} ({a.depth + 1}-ply), {scoring}", small, self.GREY, rect.x + pad,
                   rect.y + 5 + line)
        y = rect.y + self._header_height() + 4

        clip = self.screen.get_clip()
        self.screen.set_clip(pygame.Rect(rect.x, y, rect.width, rect.bottom - y))
        if a.dice is None:
            self._blit(f"Equity {a.equity:+.3f}", big, self.BLACK, rect.x + pad, y)
            self._draw_probs(a.probs, small, rect.x + pad, y + line, self.BLACK)
        else:
            best = a.moves[0].equity if a.moves else 0.0
            for i in range(self._scroll, min(len(a.moves), self._scroll + self._visible_rows() + 1)):
                m = a.moves[i]
                row = pygame.Rect(rect.x, y, rect.width - 1, self._row_height())
                selected = i == self.selected
                if selected:
                    pygame.draw.rect(self.screen, self.SELECTED, row)
                text_color = self.WHITE if selected else self.BLACK
                self._blit(f"{i + 1}. {m.notation}", big, text_color, rect.x + pad, y + 2)
                equity = f"{m.equity:+.3f}" if i == 0 else f"{m.equity - best:+.3f}"
                color = self.WHITE if selected else (self.BEST if i == 0 or m.equity >= best else self.WORSE)
                surface = big.render(equity, True, color)
                self.screen.blit(surface, (rect.right - pad - surface.get_width(), y + 2))
                self._draw_probs(m.probs, small, rect.x + pad, y + 2 + line, self.WHITE if selected else self.GREY)
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
