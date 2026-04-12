import pygame

from backgammon.game.analyzer import Analyzer
from backgammon.game.position import Position, Player
from backgammon.ui.board_map import BoardMap


class CheckerPosition:
    def __init__(self, center, player, point):
        self.center = center  # Tuple of (x, y)
        self.player = player
        self.point = point


# TODO option to show dice roll
class Board:
    def __init__(self, width=1024, height=768):
        pygame.init()
        self.map = BoardMap(width, height)
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        pygame.display.set_caption("Backgammon Board")

        # Menu state
        self.show_menu = True
        self.menu_items = ["Analyze"]
        self.selected_menu = None
        self.show_analyze_submenu = False
        self.analyze_items = ["Show winning chances"]

        # Selection state
        self.position = None
        self.selected_point = None
        self.selected_player = None
        self.checker_positions = []  # List of CheckerPosition objects

        # Text display area
        self.display_text = ""

        # Initialize possible checker positions for both players
        self.me_possible_positions = []  # List of (point, checker_number, coordinates)
        self.opponent_possible_positions = []  # List of (point, checker_number, coordinates)
        self._initialize_possible_positions()

    def draw(self, position: Position):
        self.position = position

        """Draw the backgammon board with the given position."""
        self.screen.fill(self.map.BORDER)

        board_rect = pygame.Rect(self.map.board_margin, self.map.board_margin,
                                 self.map.board_width, self.map.board_height)
        pygame.draw.rect(self.screen, self.map.BROWN_LIGHT, board_rect)

        self._draw_points()
        self._draw_checkers(position)
        self._draw_labels()
        self._draw_turn_indicator()
        self._draw_text_area()
        self._draw_menu()

        pygame.display.flip()

    def _draw_points(self):
        """Draw the triangular points on the board."""
        # Top points (13-24)
        for i in range(12):
            x = self.map.board_margin + i * self.map.point_width
            if i >= 6:  # Skip bar space
                x += self.map.point_width

            color = self.map.BROWN_DARK if i % 2 == 0 else self.map.WHITE
            self._draw_triangle(x, self.map.board_margin, self.map.point_width, self.map.point_height, color, True)

        # Bottom points (1-12)
        for i in range(12):
            x = self.map.board_margin + (11 - i) * self.map.point_width
            if i < 6:  # Skip bar space
                x += self.map.point_width

            color = self.map.BROWN_DARK if i % 2 == 0 else self.map.WHITE
            self._draw_triangle(x, self.map.board_margin + 2 * self.map.point_height,
                                self.map.point_width, self.map.point_height, color, False)

    def _draw_triangle(self, x, y, width, height, color, pointing_down):
        """Draw a triangular point."""
        if pointing_down:
            points = [(x, y), (x + width, y), (x + width // 2, y + height)]
        else:
            points = [(x, y + height), (x + width, y + height), (x + width // 2, y)]

        pygame.draw.polygon(self.screen, color, points)
        pygame.draw.polygon(self.screen, self.map.BLACK, points, 2)

    def _draw_checkers(self, position: Position):
        """Draw checkers on the board according to the position."""
        self.checker_positions = []

        for point in range(1, 25):
            white_count = position.get_checkers(Player.ME, point)
            black_count = position.get_checkers(Player.OPPONENT, point)

            if white_count > 0:
                self._draw_checkers_on_point(point, white_count, Player.ME)
            if black_count > 0:
                self._draw_checkers_on_point(point, black_count, Player.OPPONENT)

        self._draw_checkers_on_bar(position)
        self._draw_checkers_borne_off(position)

    def _draw_checkers_on_point(self, point: int, count: int, player: Player):
        """Draw checkers on a specific point."""
        x, y, is_top = self._get_point_coordinates(point, player)

        color = self.map.WHITE if player == Player.ME else self.map.RED
        is_selected = (self.selected_point == point and self.selected_player == player)

        for i in range(min(count, self.map.MAX_VISIBLE_CHECKERS)):
            checker_y = y + (i * self.map.checker_radius * 2) if is_top else y - (i * self.map.checker_radius * 2)

            self.checker_positions.append(CheckerPosition((x, checker_y), player, point))

            if is_selected:
                highlight_color = self.map.HIGHLIGHT_ME if player == Player.ME else self.map.HIGHLIGHT_OPPONENT
                pygame.draw.circle(self.screen, highlight_color, (x, checker_y), self.map.checker_radius)
                pygame.draw.circle(self.screen, self.map.BLACK, (x, checker_y), self.map.checker_radius, 2)
            else:
                pygame.draw.circle(self.screen, color, (x, checker_y), self.map.checker_radius)
                pygame.draw.circle(self.screen, self.map.BLACK, (x, checker_y), self.map.checker_radius, 2)

        if count > self.map.MAX_VISIBLE_CHECKERS:
            font = pygame.font.Font(None, self.map.ui_font_size)
            text = font.render(str(count), True, self.map.BLACK)
            text_rect = text.get_rect(center=(x, y))
            self.screen.blit(text, text_rect)

    def _get_point_coordinates(self, point: int, player: Player):
        """Get the center coordinates for a point and whether it's on top. y is the position of the first checker."""
        if player == Player.ME:
            # ME moves counter-clockwise from right top to right bottom
            if point >= 13:
                i = point - 13
                x = self.map.board_margin + i * self.map.point_width + self.map.point_width // 2
                if i >= 6:
                    x += self.map.point_width
                y = self.map.board_margin + self.map.checker_radius
                return x, y, True
            else:  # Bottom points (1-12)
                i = 12 - point
                x = self.map.board_margin + i * self.map.point_width + self.map.point_width // 2
                if i >= 6:
                    x += self.map.point_width
                y = self.map.height - self.map.board_margin - self.map.checker_radius
                return x, y, False
        else:
            # OPPONENT moves clockwise right bottom to right top (reverse of ME)
            if point >= 13:
                i = point - 13
                x = self.map.board_margin + i * self.map.point_width + self.map.point_width // 2
                if i >= 6:
                    x += self.map.point_width
                y = self.map.height - self.map.board_margin - self.map.checker_radius
                return x, y, False
            else:  # Top points (1-12)
                i = 12 - point
                x = self.map.board_margin + i * self.map.point_width + self.map.point_width // 2
                if i >= 6:
                    x += self.map.point_width
                y = self.map.board_margin + self.map.checker_radius
                return x, y, True

    def _draw_checkers_on_bar(self, position: Position):
        """Draw checkers on the bar."""
        me_count = position.get_checkers(Player.ME, 25)
        opponent_count = position.get_checkers(Player.OPPONENT, 25)

        for i in range(me_count):
            y = self.map.board_margin + self.map.point_height + self.map.checker_radius - i * self.map.checker_radius * 2
            self.checker_positions.append(CheckerPosition((self.map.bar_x, y), Player.ME, 25))
            pygame.draw.circle(self.screen, self.map.WHITE, (self.map.bar_x, y), self.map.checker_radius)
            pygame.draw.circle(self.screen, self.map.BLACK, (self.map.bar_x, y), self.map.checker_radius, 2)

        for i in range(opponent_count):
            y = self.map.height - self.map.board_margin - self.map.point_height - self.map.checker_radius + i * self.map.checker_radius * 2
            self.checker_positions.append(CheckerPosition((self.map.bar_x, y), Player.OPPONENT, 25))
            pygame.draw.circle(self.screen, self.map.RED, (self.map.bar_x, y), self.map.checker_radius)
            pygame.draw.circle(self.screen, self.map.BLACK, (self.map.bar_x, y), self.map.checker_radius, 2)

    def _draw_checkers_borne_off(self, position: Position):
        """Draw borne off checkers."""
        me_count = position.get_checkers(Player.ME, 0)
        opponent_count = position.get_checkers(Player.OPPONENT, 0)

        for i in range(min(me_count, 10)):
            y = self.map.board_margin + self.map.checker_radius + i * self.map.checker_radius // 2
            pygame.draw.circle(self.screen, self.map.WHITE, (self.map.bear_off_x, y), self.map.checker_radius)
            pygame.draw.circle(self.screen, self.map.BLACK, (self.map.bear_off_x, y), self.map.checker_radius, 2)

        for i in range(min(opponent_count, 10)):
            y = self.map.height - self.map.board_margin - self.map.checker_radius - i * self.map.checker_radius // 2
            pygame.draw.circle(self.screen, self.map.RED, (self.map.bear_off_x, y), self.map.checker_radius)
            pygame.draw.circle(self.screen, self.map.BLACK, (self.map.bear_off_x, y), self.map.checker_radius, 2)

    def _draw_labels(self):
        """Draw point numbers and other labels."""
        font = pygame.font.Font(None, self.map.label_font_size)

        for point in range(13, 25):
            x, _, _ = self._get_point_coordinates(point, Player.OPPONENT)
            text = font.render(str(point), True, self.map.BLACK)
            text_rect = text.get_rect(center=(x, self.map.board_margin - self.map.label_margin))
            self.screen.blit(text, text_rect)

        for point in range(1, 13):
            x, _, _ = self._get_point_coordinates(point, Player.OPPONENT)
            text = font.render(str(point), True, self.map.BLACK)
            text_rect = text.get_rect(center=(x, self.map.board_margin + self.map.board_height + self.map.label_margin))
            self.screen.blit(text, text_rect)

    def _draw_turn_indicator(self):
        """Draw turn indicator arrow in the right margin."""
        if not self.position:
            return

        if self.position.turn == Player.ME:
            arrow_y = self.map.arrow_y_me
            color = self.map.WHITE
        else:
            arrow_y = self.map.arrow_y_opponent
            color = self.map.RED

        points = [
            (self.map.arrow_x - self.map.arrow_size, arrow_y - self.map.arrow_size // 2),
            (self.map.arrow_x - self.map.arrow_size, arrow_y + self.map.arrow_size // 2),
            (self.map.arrow_x, arrow_y)
        ]

        pygame.draw.polygon(self.screen, color, points)
        pygame.draw.polygon(self.screen, self.map.BLACK, points, 2)

    def _draw_text_area(self):
        """Draw the text display area in the bottom margin."""
        text_area_rect = pygame.Rect(self.map.board_margin, self.map.text_area_y,
                                     self.map.board_width, self.map.text_area_height)

        pygame.draw.rect(self.screen, self.map.WHITE, text_area_rect)
        pygame.draw.rect(self.screen, self.map.BLACK, text_area_rect, 2)

        font = pygame.font.Font(None, self.map.ui_font_size)
        text_surface = font.render(self.display_text, True, self.map.BLACK)

        text_x = text_area_rect.x + self.map.text_padding
        text_y = text_area_rect.y + (self.map.text_area_height - text_surface.get_height()) // 2
        self.screen.blit(text_surface, (text_x, text_y))

    def show_text(self, text: str):
        """Set text to display in the text area."""
        self.display_text = text
        self.draw(self.position)

    def handle_turn_indicator_click(self, mouse_pos):
        """Handle click on turn indicator arrow to switch turns."""
        if not self.position:
            return False

        x, y = mouse_pos

        arrow_y = self.map.arrow_y_me if self.position.turn == Player.ME else self.map.arrow_y_opponent

        if (self.map.arrow_x - self.map.arrow_size <= x <= self.map.arrow_x and
                arrow_y - self.map.arrow_size // 2 <= y <= arrow_y + self.map.arrow_size // 2):
            self.position.switch_turn()
            return True

        return False

    def handle_click_entering(self, mouse_pos, player):
        """Handle mouse click: have checkers on point for player ME until the mouse cursor."""
        x, y = mouse_pos

        for point, checker_num, coordinates in (self.me_possible_positions if player == Player.ME else self.opponent_possible_positions):
            center_x, center_y = coordinates
            distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5

            if distance <= self.map.checker_radius:
                self.adjust_checkers(point, checker_num, player)
                self.draw(self.position)
                return

    def handle_click(self, mouse_pos):
        """Handle mouse click to select checkers using CheckerPosition list."""
        x, y = mouse_pos

        for checker_pos in self.checker_positions:
            center_x, center_y = checker_pos.center
            distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5

            if distance <= self.map.checker_radius:
                self.selected_point = checker_pos.point
                self.selected_player = checker_pos.player
                return

        self.selected_point = None
        self.selected_player = None

    def _initialize_possible_positions(self):
        """Initialize lists of all possible checker positions for both players."""
        self.me_possible_positions = []
        self.opponent_possible_positions = []

        for point in range(26):
            for player in [Player.ME, Player.OPPONENT]:
                if point == 0:
                    for checker_num in range(1, self.map.MAX_VISIBLE_CHECKERS + 1):
                        if player == Player.ME:
                            y = self.map.board_margin + self.map.checker_radius + (checker_num - 1) * self.map.checker_radius // 2
                        else:
                            y = self.map.height - self.map.board_margin - self.map.checker_radius - (checker_num - 1) * self.map.checker_radius // 2

                        pos_tuple = (point, checker_num, (self.map.bear_off_x, y))
                        if player == Player.ME:
                            self.me_possible_positions.append(pos_tuple)
                        else:
                            self.opponent_possible_positions.append(pos_tuple)

                elif point == 25:
                    for checker_num in range(1, self.map.MAX_VISIBLE_CHECKERS + 1):
                        if player == Player.ME:
                            y = self.map.board_margin + self.map.point_height + self.map.checker_radius - (checker_num - 1) * self.map.checker_radius * 2
                        else:
                            y = self.map.height - self.map.board_margin - self.map.point_height - self.map.checker_radius + (checker_num - 1) * self.map.checker_radius * 2

                        pos_tuple = (point, checker_num, (self.map.bar_x, y))
                        if player == Player.ME:
                            self.me_possible_positions.append(pos_tuple)
                        else:
                            self.opponent_possible_positions.append(pos_tuple)

                elif 1 <= point <= 24:
                    x, base_y, is_top = self._get_point_coordinates(point, player)
                    for checker_num in range(0, self.map.MAX_VISIBLE_CHECKERS + 1):
                        y = base_y + ((checker_num - 1) * self.map.checker_radius * 2) if is_top else base_y - ((checker_num - 1) * self.map.checker_radius * 2)

                        pos_tuple = (point, checker_num, (x, y))
                        if player == Player.ME:
                            self.me_possible_positions.append(pos_tuple)
                        else:
                            self.opponent_possible_positions.append(pos_tuple)

    def handle_resize(self, width, height):
        """Handle window resize: rebuild layout and reinitialize positions."""
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        self.map = BoardMap(width, height)
        self._initialize_possible_positions()

    def quit(self):
        """Clean up pygame."""
        pygame.quit()

    def adjust_checkers(self, point, checker_num, player):
        """Adjusts the number of checkers on the point to checker_num."""
        current = self.position.get_checkers(player, point)

        if point > 0 and current != checker_num:
            other_player = player.other_player()
            other_point = 25 - point

            if point != 25 and self.position.get_checkers(other_player, other_point) > 0:
                other_checkers_on_point = self.position.get_checkers(other_player, other_point)
                current_off = self.position.get_checkers(other_player, 0)

                self.position.set_checkers(other_player, 0, current_off + other_checkers_on_point)
                self.position.set_checkers(other_player, other_point, 0)

            if checker_num > current:
                diff = checker_num - current
                available = self.position.get_checkers(player, 0)
                new_checkers = min(diff, available)
                new_checkers_off = available - new_checkers

                self.position.set_checkers(player, point, current + new_checkers)
                self.position.set_checkers(player, 0, new_checkers_off)
            else:
                diff = current - checker_num
                new_checkers_off = self.position.get_checkers(player, 0) + diff
                self.position.set_checkers(player, point, current - diff)
                self.position.set_checkers(player, 0, new_checkers_off)

    def _draw_menu(self):
        """Draw the menu bar."""
        if not self.show_menu and not self.show_analyze_submenu:
            return

        font = pygame.font.Font(None, self.map.ui_font_size)

        if self.show_menu:
            menu_rect = pygame.Rect(0, 0, self.map.width, self.map.menu_height)
            pygame.draw.rect(self.screen, self.map.MENU_BG, menu_rect)
            pygame.draw.line(self.screen, self.map.BLACK, (0, self.map.menu_height), (self.map.width, self.map.menu_height), 1)

            x_offset = self.map.menu_item_x
            for i, item in enumerate(self.menu_items):
                item_width = font.size(item)[0] + self.map.menu_item_padding
                item_rect = pygame.Rect(x_offset, self.map.menu_item_y, item_width, self.map.menu_height - 10)

                if self.selected_menu == i:
                    pygame.draw.rect(self.screen, self.map.MENU_HOVER, item_rect)

                text = font.render(item, True, self.map.BLACK)
                self.screen.blit(text, (x_offset + self.map.menu_item_padding // 2, self.map.menu_text_y))
                x_offset += item_width

        if self.show_analyze_submenu:
            submenu_y = self.map.menu_height if self.show_menu else 0
            submenu_height = len(self.analyze_items) * self.map.submenu_item_height + 10
            submenu_rect = pygame.Rect(self.map.submenu_x, submenu_y, self.map.submenu_width, submenu_height)
            pygame.draw.rect(self.screen, self.map.MENU_BG, submenu_rect)
            pygame.draw.rect(self.screen, self.map.BLACK, submenu_rect, 1)

            for i, item in enumerate(self.analyze_items):
                item_y = submenu_y + 5 + i * self.map.submenu_item_height
                text = font.render(item, True, self.map.BLACK)
                self.screen.blit(text, (self.map.submenu_x + self.map.submenu_text_x_offset, item_y))

    def handle_menu_click(self, mouse_pos):
        """Handle menu clicks."""
        x, y = mouse_pos

        if self.show_menu and y < self.map.menu_height:
            font = pygame.font.Font(None, self.map.ui_font_size)
            x_offset = self.map.menu_item_x
            for i, item in enumerate(self.menu_items):
                item_width = font.size(item)[0] + self.map.menu_item_padding
                if x_offset <= x <= x_offset + item_width:
                    if item == "Analyze":
                        self.show_analyze_submenu = not self.show_analyze_submenu
                    return True
                x_offset += item_width

        if self.show_analyze_submenu:
            submenu_y = self.map.menu_height if self.show_menu else 0
            submenu_height = len(self.analyze_items) * self.map.submenu_item_height + 10
            submenu_right = self.map.submenu_x + self.map.submenu_width
            if self.map.submenu_x <= x <= submenu_right and submenu_y <= y <= submenu_y + submenu_height:
                item_index = (y - submenu_y - 5) // self.map.submenu_item_height
                if 0 <= item_index < len(self.analyze_items):
                    if self.analyze_items[item_index] == "Show winning chances":
                        self.analyze()
                        self.show_analyze_submenu = False
                    return True

        return False

    def handle_key_press(self, key):
        """Handle key press events."""
        if key == pygame.K_F10 or (key == pygame.K_m and pygame.key.get_pressed()[pygame.K_LALT]):
            self.show_menu = not self.show_menu
            if not self.show_menu:
                self.show_analyze_submenu = False

    def analyze(self):
        """Analyze the current position and show winning chances."""
        self.show_text("Analyzing position for winning chances...")
        pw = Analyzer.winning_chances(position=self.position, player=Player.ME)
        self.show_text(f"My winning chances: {pw}")