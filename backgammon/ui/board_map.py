from backgammon.game.position import Position, Player


class BoardMap:
    # Colors
    BROWN_LIGHT = (205, 133, 63)
    BROWN_DARK = (139, 69, 19)
    WHITE = (255, 255, 255)
    RED = (220, 20, 60)
    BLACK = (0, 0, 0)
    BORDER = (101, 67, 33)
    HIGHLIGHT_ME = (0, 255, 0)
    HIGHLIGHT_OPPONENT = (255, 165, 0)
    MENU_BG = (240, 240, 240)
    MENU_HOVER = (200, 200, 255)

    # Checker display
    MAX_VISIBLE_CHECKERS = 5

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

        # Board
        self.board_margin = round(70 * s) + self.menu_height   # (50 + 20) spatial margin + menu
        self.board_width = width - 2 * self.board_margin
        self.board_height = height - 2 * self.board_margin

        # Points and checkers
        self.point_width = self.board_width // 14     # 12 points + bar + bear-off column
        self.point_height = self.board_height // 3
        self.checker_radius = max(self.point_width // 2 - 2, 2)

        # Fixed x positions
        self.bar_x = self.board_margin + 6 * self.point_width + self.point_width // 2
        self.bear_off_x = self.board_margin + 13 * self.point_width + self.point_width // 2

        # Turn indicator arrow
        self.arrow_size = round(40 * s)
        self.arrow_x = width - self.board_margin // 3
        self.arrow_y_me = height - self.board_margin - round(30 * sy)
        self.arrow_y_opponent = self.board_margin + round(30 * sy)

        # Text area
        self.text_area_height = round(30 * sy)
        self.text_area_y = height - self.board_margin + round(30 * sy)
        self.text_padding = round(10 * sx)            # left padding inside text area

        # Point labels
        self.label_margin = round(15 * sy)            # offset from board edge to label center

        # Possible checker positions for hit-testing clicks
        self.me_possible_positions = []
        self.opponent_possible_positions = []
        self._initialize_possible_positions()

    def get_point_coordinates(self, point: int, player: Player):
        """Return (x, y, is_top) for the first checker slot on a point. y is the position of the first checker."""
        if player == Player.ME:
            if point >= 13:
                i = point - 13
                x = self.board_margin + i * self.point_width + self.point_width // 2
                if i >= 6:
                    x += self.point_width
                return x, self.board_margin + self.checker_radius, True
            else:
                i = 12 - point
                x = self.board_margin + i * self.point_width + self.point_width // 2
                if i >= 6:
                    x += self.point_width
                return x, self.height - self.board_margin - self.checker_radius, False
        else:
            if point >= 13:
                i = point - 13
                x = self.board_margin + i * self.point_width + self.point_width // 2
                if i >= 6:
                    x += self.point_width
                return x, self.height - self.board_margin - self.checker_radius, False
            else:
                i = 12 - point
                x = self.board_margin + i * self.point_width + self.point_width // 2
                if i >= 6:
                    x += self.point_width
                return x, self.board_margin + self.checker_radius, True

    def _initialize_possible_positions(self):
        """Precompute all possible checker positions for hit-testing mouse clicks."""
        self.me_possible_positions = []
        self.opponent_possible_positions = []

        for point in range(26):
            for player in [Player.ME, Player.OPPONENT]:
                positions = self.me_possible_positions if player == Player.ME else self.opponent_possible_positions

                if point == 0:
                    for checker_num in range(1, self.MAX_VISIBLE_CHECKERS + 1):
                        if player == Player.ME:
                            y = self.board_margin + self.checker_radius + (checker_num - 1) * self.checker_radius // 2
                        else:
                            y = self.height - self.board_margin - self.checker_radius - (checker_num - 1) * self.checker_radius // 2
                        positions.append((point, checker_num, (self.bear_off_x, y)))

                elif point == 25:
                    for checker_num in range(1, self.MAX_VISIBLE_CHECKERS + 1):
                        if player == Player.ME:
                            y = self.board_margin + self.point_height + self.checker_radius - (checker_num - 1) * self.checker_radius * 2
                        else:
                            y = self.height - self.board_margin - self.point_height - self.checker_radius + (checker_num - 1) * self.checker_radius * 2
                        positions.append((point, checker_num, (self.bar_x, y)))

                else:
                    x, base_y, is_top = self.get_point_coordinates(point, player)
                    for checker_num in range(0, self.MAX_VISIBLE_CHECKERS + 1):
                        if is_top:
                            y = base_y + (checker_num - 1) * self.checker_radius * 2
                        else:
                            y = base_y - (checker_num - 1) * self.checker_radius * 2
                        positions.append((point, checker_num, (x, y)))

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
