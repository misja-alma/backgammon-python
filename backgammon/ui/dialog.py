import pygame


class Dialog:
    """Base class for modal dialogs drawn over a dimmed snapshot of the board.

    A dialog runs its own event loop in run() until OK, Cancel, Esc or window close. Subclasses
    lay out their controls in __init__ (coordinates relative to the dialog's top-left corner),
    draw them in draw_content(), react to clicks in handle_click(), and build their result in
    result().
    """

    DISABLED = (150, 150, 150)
    OVERLAY = (0, 0, 0, 120)

    def __init__(self, screen, board_map, title: str, width: int, height: int):
        self.screen = screen
        self.map = board_map
        self.title = title
        self.width, self.height = width, height
        self.x = (self.map.width - width) // 2
        self.y = (self.map.height - height) // 2
        self.font = pygame.font.Font(None, self.map.ui_font_size)
        self.small_font = pygame.font.Font(None, max(self.map.ui_font_size - 6, 18))
        self.btn_ok = self.rect(width - 170, height - 45, 70, 30)
        self.btn_cancel = self.rect(width - 90, height - 45, 80, 30)

    # --- subclass hooks ---

    def draw_content(self) -> None:
        pass

    def handle_click(self, pos) -> None:
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
        self.screen.blit(self.font.render(self.title, True, self.map.BLACK), (self.x + 20, self.y + 20))

        self.draw_content()
        self.draw_button(self.btn_ok, "OK")
        self.draw_button(self.btn_cancel, "Cancel")
        pygame.display.flip()

    # --- layout and drawing helpers ---

    def rect(self, dx: int, dy: int, w: int, h: int) -> pygame.Rect:
        """A rect relative to the dialog's top-left corner."""
        return pygame.Rect(self.x + dx, self.y + dy, w, h)

    def _color(self, enabled: bool):
        return self.map.BLACK if enabled else self.DISABLED

    def draw_text(self, dx: int, dy: int, text: str, enabled: bool = True) -> None:
        self.screen.blit(self.small_font.render(text, True, self._color(enabled)), (self.x + dx, self.y + dy))

    def _draw_label(self, rect: pygame.Rect, text: str, enabled: bool) -> None:
        label = self.small_font.render(text, True, self._color(enabled))
        self.screen.blit(label, (rect.right + 8, rect.centery - label.get_height() // 2))

    def draw_radio(self, rect: pygame.Rect, selected: bool, text: str, enabled: bool = True) -> None:
        color = self._color(enabled)
        pygame.draw.circle(self.screen, color, rect.center, 8, 2)
        if selected:
            pygame.draw.circle(self.screen, color, rect.center, 4)
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

    def draw_stepper(self, minus: pygame.Rect, plus: pygame.Rect, value, enabled: bool = True) -> None:
        """A '-' button, the value (shown with str()), and a '+' button."""
        self.draw_button(minus, "-", enabled)
        self.draw_button(plus, "+", enabled)
        label = self.small_font.render(str(value), True, self._color(enabled))
        self.screen.blit(label, label.get_rect(center=((minus.right + plus.left) // 2, minus.centery)))
