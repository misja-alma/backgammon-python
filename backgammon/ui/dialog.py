import pygame


class Dialog:
    """Base class for modal dialogs drawn over a dimmed snapshot of the board.

    A dialog runs its own event loop in run() until OK, Cancel, Esc or window close. Subclasses
    lay out their controls in __init__ (coordinates relative to the dialog's top-left corner),
    draw them in draw_content(), react to clicks in handle_click(), and build their result in
    result().

    Coordinates and sizes are in design units: pixels at the default window size (1024x768),
    where the dialog font is DESIGN_FONT_SIZE. The font grows with the window, and so does the
    layout (by `scale`), so that text and controls keep their proportions at any size.
    """

    DESIGN_FONT_SIZE = 18

    DISABLED = (150, 150, 150)
    OVERLAY = (0, 0, 0, 120)

    def __init__(self, screen, board_map, title: str, width: int, height: int):
        self.screen = screen
        self.map = board_map
        self.title = title
        small_size = self.small_font_size(board_map)
        self.scale = small_size / self.DESIGN_FONT_SIZE
        self.width, self.height = round(width * self.scale), round(height * self.scale)
        self.x = (self.map.width - self.width) // 2
        self.y = (self.map.height - self.height) // 2
        self.font = pygame.font.Font(None, self.map.ui_font_size)
        self.small_font = pygame.font.Font(None, small_size)
        self.btn_ok = self.rect(width - 170, height - 45, 70, 30)
        self.btn_cancel = self.rect(width - 90, height - 45, 80, 30)

    @staticmethod
    def small_font_size(board_map) -> int:
        """Size of the dialog font (labels and controls) for this window."""
        return max(board_map.ui_font_size - 6, Dialog.DESIGN_FONT_SIZE)

    @classmethod
    def design_font(cls):
        """The dialog font at the default window size, to measure text in design units."""
        return pygame.font.Font(None, cls.DESIGN_FONT_SIZE)

    # --- subclass hooks ---

    def draw_content(self) -> None:
        pass

    def handle_click(self, pos) -> None:
        pass

    def handle_key(self, event) -> None:
        """A key press other than Esc and Enter (which cancel and confirm)."""
        pass

    def result(self):
        """Value returned from run() when OK is pressed."""
        return True

    # --- event loop ---

    def run(self, board_snapshot):
        """Show the dialog. Returns result() on OK, None on Cancel/Esc/close."""
        while True:
            self._draw(board_snapshot)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return None
                if event.type == pygame.KEYDOWN and event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    return self.result()
                if event.type == pygame.KEYDOWN:
                    self.handle_key(event)
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.btn_ok.collidepoint(event.pos):
                        return self.result()
                    if self.btn_cancel.collidepoint(event.pos):
                        return None
                    self.handle_click(event.pos)

    def _draw(self, board_snapshot) -> None:
        self.screen.blit(board_snapshot, (0, 0))
        overlay = pygame.Surface((self.map.width, self.map.height), pygame.SRCALPHA)
        overlay.fill(self.OVERLAY)
        self.screen.blit(overlay, (0, 0))

        panel = pygame.Rect(self.x, self.y, self.width, self.height)
        pygame.draw.rect(self.screen, self.map.MENU_BG, panel)
        pygame.draw.rect(self.screen, self.map.BLACK, panel, 2)
        margin = round(20 * self.scale)
        self.screen.blit(self.font.render(self.title, True, self.map.BLACK), (self.x + margin, self.y + margin))

        self.draw_content()
        self.draw_button(self.btn_ok, "OK")
        self.draw_button(self.btn_cancel, "Cancel")
        pygame.display.flip()

    # --- layout and drawing helpers ---

    def rect(self, dx: int, dy: int, w: int, h: int) -> pygame.Rect:
        """A rect relative to the dialog's top-left corner, in design units."""
        s = self.scale
        return pygame.Rect(self.x + round(dx * s), self.y + round(dy * s), round(w * s), round(h * s))

    def _color(self, enabled: bool):
        return self.map.BLACK if enabled else self.DISABLED

    def draw_text(self, dx: int, dy: int, text: str, enabled: bool = True) -> None:
        self.screen.blit(self.small_font.render(text, True, self._color(enabled)), (self.x + round(dx * self.scale), self.y + round(dy * self.scale)))

    def _draw_label(self, rect: pygame.Rect, text: str, enabled: bool) -> None:
        label = self.small_font.render(text, True, self._color(enabled))
        self.screen.blit(label, (rect.right + 8, rect.centery - label.get_height() // 2))

    def draw_radio(self, rect: pygame.Rect, selected: bool, text: str, enabled: bool = True) -> None:
        color = self._color(enabled)
        pygame.draw.circle(self.screen, color, rect.center, round(8 * self.scale), 2)
        if selected:
            pygame.draw.circle(self.screen, color, rect.center, round(4 * self.scale))
        self._draw_label(rect, text, enabled)

    def draw_checkbox(self, rect: pygame.Rect, checked: bool, text: str, enabled: bool = True) -> None:
        color = self._color(enabled)
        pygame.draw.rect(self.screen, self.map.WHITE, rect)
        pygame.draw.rect(self.screen, color, rect, 2)
        if checked:
            pygame.draw.line(self.screen, color, (rect.x + 3, rect.centery), (rect.centerx - 1, rect.bottom - 4), 2)
            pygame.draw.line(self.screen, color, (rect.centerx - 1, rect.bottom - 4), (rect.right - 3, rect.y + 3), 2)
        self._draw_label(rect, text, enabled)

    def draw_button(self, rect: pygame.Rect, text: str, enabled: bool = True) -> None:
        pygame.draw.rect(self.screen, self.map.WHITE, rect)
        pygame.draw.rect(self.screen, self._color(enabled), rect, 2)
        label = self.small_font.render(text, True, self._color(enabled))
        self.screen.blit(label, label.get_rect(center=rect.center))

    def draw_field(self, rect: pygame.Rect, text: str, focused: bool) -> None:
        """A text field showing text, with a caret when it has the keyboard focus."""
        pygame.draw.rect(self.screen, self.map.WHITE, rect)
        pygame.draw.rect(self.screen, self.map.SELECTED if focused else self.map.BLACK, rect, 2)
        label = self.small_font.render(text, True, self.map.BLACK)
        x = rect.x + 6
        self.screen.blit(label, (x, rect.centery - label.get_height() // 2))
        if focused:
            caret_x = x + label.get_width() + 1
            pygame.draw.line(self.screen, self.map.BLACK, (caret_x, rect.y + 5), (caret_x, rect.bottom - 6), 1)

    def draw_stepper(self, minus: pygame.Rect, plus: pygame.Rect, value, enabled: bool = True) -> None:
        """A '-' button, the value (shown with str()), and a '+' button."""
        self.draw_button(minus, "-", enabled)
        self.draw_button(plus, "+", enabled)
        label = self.small_font.render(str(value), True, self._color(enabled))
        self.screen.blit(label, label.get_rect(center=((minus.right + plus.left) // 2, minus.centery)))
