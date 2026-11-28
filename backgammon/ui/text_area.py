import pygame


class TextArea:
    VISIBLE_LINES = 3
    THUMB_MIN_HEIGHT = 12

    def __init__(self, screen, board_map):
        self.screen = screen
        self.map = board_map
        self._lines: list[str] = []
        self._scroll_offset = 0  # lines scrolled up from the bottom; 0 = show tail
        self._dragging = False
        self._drag_start_y = 0
        self._drag_start_offset = 0

    # --- public API ---

    def show_text(self, text: str):
        self._lines.append(text)
        self._scroll_offset = 0  # snap to bottom

    def show_or_replace(self, index, text: str) -> int:
        """Replace line index with text, so that a changing status (e.g. a rollout's progress)
        takes one line, or add text as a new line when index is None or no longer exists (the area
        was cleared). Returns the line's index, for the next update."""
        if index is None or index >= len(self._lines):
            self.show_text(text)
            return len(self._lines) - 1
        self._lines[index] = text
        return index

    def clear(self):
        self._lines.clear()
        self._scroll_offset = 0

    # --- event handling ---

    def handle_event(self, event) -> bool:
        if event.type == pygame.MOUSEWHEEL:
            if self._content_rect().collidepoint(pygame.mouse.get_pos()):
                self._clamp_scroll(self._scroll_offset - event.y)
                return True

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self._scrollbar_track_rect().collidepoint(event.pos):
                thumb = self._thumb_rect()
                if thumb.collidepoint(event.pos):
                    self._dragging = True
                    self._drag_start_y = event.pos[1]
                    self._drag_start_offset = self._scroll_offset
                else:
                    # Click on track above/below thumb: page scroll
                    if event.pos[1] < thumb.top:
                        self._clamp_scroll(self._scroll_offset + self._visible_lines())
                    else:
                        self._clamp_scroll(self._scroll_offset - self._visible_lines())
                return True

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self._dragging:
                self._dragging = False
                return True

        elif event.type == pygame.MOUSEMOTION:
            if self._dragging:
                dy = event.pos[1] - self._drag_start_y
                track = self._scrollbar_track_rect()
                scrollable_px = track.height - self._thumb_height()
                if scrollable_px > 0:
                    max_off = max(0, len(self._lines) - self._visible_lines())
                    delta = round(dy / scrollable_px * max_off)
                    self._clamp_scroll(self._drag_start_offset - delta)
                return True

        return False

    # --- drawing ---

    def draw(self):
        content = self._content_rect()
        track = self._scrollbar_track_rect()
        lh = self.map.text_line_height
        visible = self._visible_lines()

        # Background + border for the whole area
        full_rect = pygame.Rect(content.x, content.y,
                                content.width + track.width, content.height)
        pygame.draw.rect(self.screen, self.map.WHITE, full_rect)
        pygame.draw.rect(self.screen, self.map.BLACK, full_rect, 2)

        # Divider between content and scrollbar
        pygame.draw.line(self.screen, self.map.BLACK,
                         (track.x, track.y), (track.x, track.bottom), 1)

        # Scrollbar track
        pygame.draw.rect(self.screen, (210, 210, 210), track)

        # Thumb
        if len(self._lines) > visible:
            pygame.draw.rect(self.screen, (130, 130, 130), self._thumb_rect())

        # Text lines (clip to content rect)
        font = pygame.font.Font(None, self.map.ui_font_size)
        total = len(self._lines)
        end = total - self._scroll_offset
        start = max(0, end - visible)
        clip = self.screen.get_clip()
        self.screen.set_clip(content)
        for i, line in enumerate(self._lines[start:end]):
            surf = font.render(line, True, self.map.BLACK)
            y = content.y + round(3) + i * lh
            self.screen.blit(surf, (content.x + self.map.text_padding, y))
        self.screen.set_clip(clip)

    # --- geometry helpers ---

    def _content_rect(self) -> pygame.Rect:
        return pygame.Rect(
            self.map.board_x,
            self.map.text_area_y,
            self.map.board_width - self.map.scrollbar_width,
            self.map.text_area_height,
        )

    def _scrollbar_track_rect(self) -> pygame.Rect:
        c = self._content_rect()
        return pygame.Rect(c.right, c.y, self.map.scrollbar_width, c.height)

    def _visible_lines(self) -> int:
        return max(1, self.map.text_area_height // self.map.text_line_height)

    def _max_offset(self) -> int:
        return max(0, len(self._lines) - self._visible_lines())

    def _clamp_scroll(self, value: int):
        self._scroll_offset = max(0, min(value, self._max_offset()))

    def _thumb_height(self) -> int:
        track_h = self._scrollbar_track_rect().height
        total = len(self._lines)
        visible = self._visible_lines()
        if total <= visible:
            return track_h
        ratio = visible / total
        return max(self.THUMB_MIN_HEIGHT, round(track_h * ratio))

    def _thumb_rect(self) -> pygame.Rect:
        track = self._scrollbar_track_rect()
        thumb_h = self._thumb_height()
        scrollable_px = track.height - thumb_h
        max_off = self._max_offset()
        # scroll_offset 0 → thumb at bottom; max_offset → thumb at top
        ratio = 1.0 - (self._scroll_offset / max_off) if max_off > 0 else 1.0
        thumb_y = track.y + round(ratio * scrollable_px)
        return pygame.Rect(track.x, thumb_y, track.width, thumb_h)
