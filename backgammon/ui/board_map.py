import pygame
from backgammon.game.position import Position, Player
from backgammon.ui.text_area import TextArea


class BoardMap:
    # Colors
    BROWN_LIGHT = (205, 133, 63)
    BROWN_DARK = (139, 69, 19)
    WHITE = (255, 255, 255)
    RED = (220, 20, 60)
    RED_LIGHT = (255, 120, 120)   # readable red text on the dark border
    BLACK = (0, 0, 0)
    BORDER = (101, 67, 33)
    HIGHLIGHT_ME = (0, 255, 0)
    HIGHLIGHT_OPPONENT = (255, 165, 0)
    MENU_BG = (240, 240, 240)
    MENU_HOVER = (200, 200, 255)

    # Checker display
    MAX_VISIBLE_CHECKERS = 5

    # Share of the window width for the analysis panel on the left
    PANEL_FRACTION = 0.24

    # Doubling cube
    MAX_CUBE_VALUE = 64

    def __init__(self, width=1024, height=768):
        self.width = width
        self.height = height

        sx = width / 1024    # horizontal scale factor
        sy = height / 768    # vertical scale factor
        s = min(sx, sy)      # uniform scale factor for square/font values

        # Font sizes (scale uniformly to preserve readability)
        self.label_font_size = round(20 * s)
        self.ui_font_size = round(24 * s)

        # Menu bar
        self.menu_height = round(30 * sy)
        self.menu_item_padding = round(20 * sx)       # horizontal padding added to text width per item
        self.menu_item_x = round(10 * sx)             # x offset of first menu item
        self.menu_item_y = round(5 * sy)              # y offset of item rect within menu bar
        self.menu_text_y = round(8 * sy)              # y position of text within menu bar

        # Submenu
        self.submenu_x = round(10 * sx)
        self.submenu_width = round(200 * sx)
        self.submenu_item_height = round(25 * sy)
        self.submenu_text_x_offset = round(5 * sx)   # x padding inside submenu items

        # Analysis panel: a column on the left, below the menu bar; the board takes the rest.
        self.panel_width = round(width * self.PANEL_FRACTION)
        self.panel_x = 0
        self.panel_y = self.menu_height
        self.panel_height = height - self.menu_height

        # Board
        # board_margin is the spacing reserved around the board: above it (menu + labels), below it
        # (labels + text area) and at its sides (cube, turn arrow).
        self.board_margin = round(70 * s) + self.menu_height   # (50 + 20) spatial margin + menu

        # Point labels
        self.label_margin = round(15 * sy)            # offset from board edge to label center

        # Text area (anchored to the bottom of the window; the board is sized to end above it)
        _font = pygame.font.Font(None, self.ui_font_size)
        self.text_line_height = _font.get_linesize()
        self.text_area_height = self.text_line_height * TextArea.VISIBLE_LINES + round(6 * sy)
        self.text_area_y = height - self.board_margin + round(5 * sy)
        self.text_padding = round(10 * sx)
        self.scrollbar_width = round(14 * sx)

        # Space for the board: between the side margins horizontally, and vertically between the
        # top margin and the point labels that sit just above the text area.
        available_width = width - self.panel_width - 2 * self.board_margin
        available_height = self.text_area_y - 2 * self.label_margin - self.board_margin

        # Points and checkers — all sizes derive from point_width, which is the largest value for
        # which the board fits both horizontally and vertically. The board then keeps its own
        # proportions whatever the window's aspect ratio.
        # Horizontally: 12 points + bar + bear-off column = 14 point widths.
        # Vertically: board_height ≈ 20 checker radii (see below), with radius = point_width // 2 - 2.
        max_pw_for_width = available_width // 14
        max_pw_for_height = 2 * (available_height // 20 + 2)
        self.point_width = max(min(max_pw_for_width, max_pw_for_height), 1)
        self.checker_radius = max(self.point_width // 2 - 2, 2)
        self.point_height = round(4.5 * self.checker_radius * 2)   # 4.5 checker diameters tall
        # board_height: 5 checkers top + 1 checker gap in middle + 5 checkers bottom = 20 radii
        self.board_height = 2 * self.point_height + self.checker_radius * 2

        if max_pw_for_width <= max_pw_for_height:
            # Width is the constraint: fill the available width, as before.
            self.board_width = available_width
        else:
            # Height is the constraint: use exactly the space the points need.
            self.board_width = 14 * self.point_width

        # Board origin: centered horizontally right of the panel, directly below the top margin.
        self.board_x = self.panel_width + (width - self.panel_width - self.board_width) // 2
        self.board_y = self.board_margin

        # Fixed x positions
        self.bar_x = self.board_x + 6 * self.point_width + self.point_width // 2
        self.bear_off_bar_width = self.point_width // 4
        self.bear_off_bar_x = self.board_x + 13 * self.point_width
        self.bear_off_x = self.board_x + self.board_width - self.checker_radius - 1

        # Turn indicator arrow
        board_bottom = self.board_y + self.board_height
        self.arrow_size = round(40 * s)
        self.arrow_x = self.board_x + self.board_width + self.board_margin * 2 // 3
        self.arrow_y_me = board_bottom - round(30 * sy)
        self.arrow_y_opponent = self.board_y + round(30 * sy)

        # Doubling cube
        self.cube_size = round(40 * s)
        self.cube_font_size = round(28 * s)
        self.cube_x = self.board_x - self.board_margin // 2
        self.cube_y_center = self.board_y + self.board_height // 2
        self.cube_y_me = self.board_y + self.board_height - self.cube_size // 2
        self.cube_y_opponent = self.board_y + self.cube_size // 2

        # Possible checker positions for hit-testing clicks
        self.me_possible_positions = []
        self.opponent_possible_positions = []
        self._initialize_possible_positions()

    def get_point_coordinates(self, point: int, player: Player):
        """Return (x, y, is_top) for the first checker slot on a point. y is the position of the first checker."""
        if point == 25:
            if player == Player.ME:
                return self.bar_x, self.board_y + self.board_height - self.checker_radius, False
            else:
                return self.bar_x, self.board_y + self.checker_radius, True

        if player == Player.ME:
            if point >= 13:
                i = point - 13
                x = self.board_x + i * self.point_width + self.point_width // 2
                if i >= 6:
                    x += self.point_width
                return x, self.board_y + self.checker_radius, True
            else:
                i = 12 - point
                x = self.board_x + i * self.point_width + self.point_width // 2
                if i >= 6:
                    x += self.point_width
                return x, self.board_y + self.board_height - self.checker_radius, False
        else:
            if point >= 13:
                i = point - 13
                x = self.board_x + i * self.point_width + self.point_width // 2
                if i >= 6:
                    x += self.point_width
                return x, self.board_y + self.board_height - self.checker_radius, False
            else:
                i = 12 - point
                x = self.board_x + i * self.point_width + self.point_width // 2
                if i >= 6:
                    x += self.point_width
                return x, self.board_y + self.checker_radius, True

    def point_at(self, x: int, y: int, player: Player):
        """The point under (x, y) from player's perspective: 1-24, 25 for the bar, 0 for the
        bear-off column. None when (x, y) is outside the board."""
        if not (self.board_y <= y <= self.board_y + self.board_height):
            return None
        if self.bear_off_bar_x <= x <= self.board_x + self.board_width:
            return 0
        column = (x - self.board_x) // self.point_width if x >= self.board_x else -1
        if not (0 <= column < 13):
            return None
        if column == 6:
            return 25
        index = column if column < 6 else column - 1          # 0-11 from left to right
        is_top = y < self.board_y + self.board_height // 2
        me_point = 13 + index if is_top else 12 - index     # ME's numbering, as labelled
        return me_point if player == Player.ME else 25 - me_point

    def dice_center(self, player: Player):
        """Where the dice are drawn: centered in the right half for ME, the left half for OPPONENT."""
        half_offset = 3 * self.point_width if player == Player.OPPONENT else 10 * self.point_width
        return self.board_x + half_offset, self.board_y + self.board_height // 2

    def _initialize_possible_positions(self):
        """Precompute all possible checker positions for hit-testing mouse clicks."""
        self.me_possible_positions = []
        self.opponent_possible_positions = []

        for point in range(26):
            for player in [Player.ME, Player.OPPONENT]:
                positions = self.me_possible_positions if player == Player.ME else self.opponent_possible_positions

                if point == 0:
                    for checker_num in range(1, self.MAX_VISIBLE_CHECKERS + 1):
                        if player == Player.OPPONENT:
                            y = self.board_y + self.checker_radius + (checker_num - 1) * self.checker_radius // 2
                        else:
                            y = self.board_y + self.board_height - self.checker_radius - (checker_num - 1) * self.checker_radius // 2
                        positions.append((point, checker_num, (self.bear_off_x, y)))

                else:
                    x, base_y, is_top = self.get_point_coordinates(point, player)
                    for checker_num in range(0, self.MAX_VISIBLE_CHECKERS + 1):
                        if is_top:
                            y = base_y + (checker_num - 1) * self.checker_radius * 2
                        else:
                            y = base_y - (checker_num - 1) * self.checker_radius * 2
                        positions.append((point, checker_num, (x, y)))

    def cube_center_y(self, position: Position) -> int:
        """Return the y-center of the cube given the current position."""
        if position.cube_owner is None:
            return self.cube_y_center
        elif position.cube_owner == Player.ME:
            return self.cube_y_me
        else:
            return self.cube_y_opponent

    def handle_cube_click(self, mouse_pos, is_left_click: bool, position: Position) -> bool:
        """Handle a click on the cube. Returns True if the cube was clicked."""
        cube_y = self.cube_center_y(position)
        half = self.cube_size // 2
        x, y = mouse_pos
        if not (self.cube_x - half <= x <= self.cube_x + half and
                cube_y - half <= y <= cube_y + half):
            return False

        owner = position.cube_owner
        value = position.cube_value
        if owner is None:
            owner = Player.ME if is_left_click else Player.OPPONENT
            value = 2
        else:
            # Clicking the owner's button doubles; the other button halves.
            owner_click = is_left_click == (owner == Player.ME)
            value = min(value * 2, self.MAX_CUBE_VALUE) if owner_click else value // 2
            if value == 1:
                owner = None
        position.set_cube(value, owner)

        return True

    def increment_checkers(self, position: Position, point: int, player: Player) -> None:
        """Add one checker to a point, taking it from the borne-off pile if available."""
        available = position.get_checkers(player, 0)
        if available > 0:
            position.set_checkers(player, point, position.get_checkers(player, point) + 1)
            position.set_checkers(player, 0, available - 1)

    def decrement_checkers(self, position: Position, point: int, player: Player) -> None:
        """Remove one checker from a point (only when count exceeds MAX_VISIBLE_CHECKERS)."""
        count = position.get_checkers(player, point)
        if count > self.MAX_VISIBLE_CHECKERS:
            position.set_checkers(player, point, count - 1)
            position.set_checkers(player, 0, position.get_checkers(player, 0) + 1)

    def adjust_checkers(self, position: Position, point: int, checker_num: int, player: Player):
        """Adjust the number of checkers on a point to checker_num, moving excess to/from borne-off."""
        current = position.get_checkers(player, point)

        if point > 0 and current != checker_num:
            other_player = player.other_player()
            other_point = 25 - point

            if point != 25 and position.get_checkers(other_player, other_point) > 0:
                other_checkers_on_point = position.get_checkers(other_player, other_point)
                current_off = position.get_checkers(other_player, 0)
                position.set_checkers(other_player, 0, current_off + other_checkers_on_point)
                position.set_checkers(other_player, other_point, 0)

            if checker_num > current:
                diff = checker_num - current
                available = position.get_checkers(player, 0)
                new_checkers = min(diff, available)
                position.set_checkers(player, point, current + new_checkers)
                position.set_checkers(player, 0, available - new_checkers)
            else:
                diff = current - checker_num
                position.set_checkers(player, point, current - diff)
                position.set_checkers(player, 0, position.get_checkers(player, 0) + diff)
