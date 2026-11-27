import pygame

from backgammon.game.analysis_worker import AnalysisWorker
from backgammon.game.position import Position, Player
from backgammon.game.settings import Settings
from backgammon.logger import Logger
from backgammon.ui.analysis_panel import AnalysisPanel
from backgammon.ui.board_map import BoardMap
from backgammon.ui.game_controller import GameController
from backgammon.ui.play_setup_dialog import PlaySetupDialog
from backgammon.ui.settings_dialog import SettingsDialog
from backgammon.ui.text_area import TextArea


class CheckerPosition:
    def __init__(self, center, player, point):
        self.center = center  # Tuple of (x, y)
        self.player = player
        self.point = point


class Board:
    def __init__(self, width=1340, height=768):
        pygame.init()
        self.map = BoardMap(width, height)
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        pygame.display.set_caption("Backgammon Board")

        # Menu state
        self.show_menu = True
        self.menus = {
            "Edit": ["Clear Board", "Starting Position"],
            "Play": ["Start...", "Stop"],
            "Analyze": ["Analyze", "Settings"],
        }
        self.menu_items = list(self.menus)
        self.active_submenu = None  # name of the currently open submenu, or None
        self.item_shortcuts = {"Analyze": "Ctrl+A"}

        # Selection state
        self.position = Position()
        self.position.setup_starting_position()
        self.selected_point = None
        self.selected_player = None
        self.checker_positions = []  # List of CheckerPosition objects

        self.text_area = TextArea(self.screen, self.map)
        Logger.set_text_area(self.text_area)
        self.analysis_panel = AnalysisPanel(self.screen, self.map)

        # Background work: analysis and engine moves share one worker thread
        self.analysis_worker = AnalysisWorker()
        self.analysis_worker.set_rust_cache_size(Settings.rust_cache_mb)
        self.analysis_worker.set_rust_network(Settings.rust_network).add_done_callback(self._log_evaluator)
        self.analysis_worker.set_rust_bearoff(Settings.rust_bearoff_db).add_done_callback(self._log_bearoff)
        self._pending_analysis = None  # Future of the running analysis, or None
        self.game_controller = GameController(self.analysis_worker)

    @staticmethod
    def _log_bearoff(future) -> None:
        """Report whether the Rust engine uses the bearoff database (runs on the worker thread)."""
        error = future.exception()
        Logger.log(f"Rust engine bearoff: {future.result()}" if error is None else f"Rust engine bearoff: none ({error})")

    @staticmethod
    def _log_evaluator(future) -> None:
        """Report which evaluator the Rust engine uses (runs on the worker thread)."""
        error = future.exception()
        if error is None:
            Logger.log(f"Rust engine evaluator: {future.result()}")
        else:
            Logger.log(f"Rust engine evaluator: placeholder ({error})")

    def update(self):
        """Per-frame housekeeping on the main thread: log output, analyses and the game."""
        self.game_controller.update()
        game_position = self.game_controller.display_position()
        if game_position is not None:
            self.position = game_position
        Logger.flush()

        # An analysis of another position or roll than the current one is no longer shown.
        analysis = self.analysis_panel.analysis
        if analysis is not None and not analysis.matches(*self._analysis_target()):
            self.analysis_panel.clear()

        future = self._pending_analysis
        if future is None or not future.done():
            return
        self._pending_analysis = None
        error = future.exception()
        if error is not None:
            self.analysis_panel.show_status(f"Analysis failed: {error}")
        elif future.result().matches(*self._analysis_target()):
            self.analysis_panel.set_analysis(future.result())
        else:
            self.analysis_panel.clear()

    @property
    def is_game_running(self) -> bool:
        return self.game_controller.is_running

    def draw(self, position: Position = None):
        """Draw the backgammon board with the given position (default: the current one)."""
        if position is not None:
            self.position = position
        position = self.position

        # While a human is entering a move, highlight the selected checker.
        controller = self.game_controller
        if controller.is_running:
            self.selected_point = controller.selected_point
            self.selected_player = controller.game.current_player
        else:
            self.selected_point = self.selected_player = None

        self.screen.fill(self.map.BORDER)
        # A move selected in the analysis panel is shown instead of the current position.
        position = self.analysis_panel.preview() or position

        board_rect = pygame.Rect(self.map.board_x, self.map.board_y,
                                 self.map.board_width, self.map.board_height)
        pygame.draw.rect(self.screen, self.map.BROWN_LIGHT, board_rect)

        self._draw_points()
        self._draw_bar()
        self._draw_bear_off_line()
        self._draw_checkers(position)
        self._draw_labels()
        self._draw_pipcounts(position)
        self._draw_cube(position)
        self._draw_turn_indicator()
        self._draw_dice()
        self.text_area.draw()
        self.analysis_panel.draw()
        self._draw_menu()

        pygame.display.flip()

    def _draw_points(self):
        """Draw the triangular points on the board."""
        # Top points (13-24)
        for i in range(12):
            x = self.map.board_x + i * self.map.point_width
            if i >= 6:  # Skip bar space
                x += self.map.point_width

            color = self.map.BROWN_DARK if i % 2 == 0 else self.map.WHITE
            self._draw_triangle(x, self.map.board_y, self.map.point_width, self.map.point_height, color, True)

        # Bottom points (1-12)
        for i in range(12):
            x = self.map.board_x + (11 - i) * self.map.point_width
            if i < 6:  # Skip bar space
                x += self.map.point_width

            color = self.map.BROWN_DARK if i % 2 == 0 else self.map.WHITE
            self._draw_triangle(x, self.map.board_y + self.map.board_height - self.map.point_height,
                                self.map.point_width, self.map.point_height, color, False)

    def _draw_bear_off_line(self):
        """Draw a narrow bar separating the last point from the bear-off area."""
        bar_rect = pygame.Rect(
            self.map.bear_off_bar_x,
            self.map.board_y,
            self.map.bear_off_bar_width,
            self.map.board_height
        )
        pygame.draw.rect(self.screen, self.map.BORDER, bar_rect)

    def _draw_bar(self):
        """Draw the bar in the middle of the board."""
        bar_rect = pygame.Rect(
            self.map.board_x + 6 * self.map.point_width,
            self.map.board_y,
            self.map.point_width,
            self.map.board_height
        )
        pygame.draw.rect(self.screen, self.map.BORDER, bar_rect)

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

        for point in range(1, 26):
            white_count = position.get_checkers(Player.ME, point)
            black_count = position.get_checkers(Player.OPPONENT, point)

            if white_count > 0:
                self._draw_checkers_on_point(point, white_count, Player.ME)
            if black_count > 0:
                self._draw_checkers_on_point(point, black_count, Player.OPPONENT)

        self._draw_checkers_borne_off(position)

    def _draw_checkers_on_point(self, point: int, count: int, player: Player):
        """Draw checkers on a specific point."""
        x, y, is_top = self.map.get_point_coordinates(point, player)

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
            last_i = self.map.MAX_VISIBLE_CHECKERS - 1
            last_y = y + last_i * self.map.checker_radius * 2 if is_top else y - last_i * self.map.checker_radius * 2
            font = pygame.font.Font(None, self.map.ui_font_size)
            text = font.render(str(count), True, self.map.BLACK)
            text_rect = text.get_rect(center=(x, last_y))
            self.screen.blit(text, text_rect)

    def _draw_checkers_borne_off(self, position: Position):
        """Draw borne off checkers at the side of each player's home board."""
        me_count = position.get_checkers(Player.ME, 0)
        opponent_count = position.get_checkers(Player.OPPONENT, 0)

        for i in range(min(opponent_count, 10)):
            y = self.map.board_y + self.map.checker_radius + i * self.map.checker_radius // 2
            pygame.draw.circle(self.screen, self.map.RED, (self.map.bear_off_x, y), self.map.checker_radius)
            pygame.draw.circle(self.screen, self.map.BLACK, (self.map.bear_off_x, y), self.map.checker_radius, 2)

        for i in range(min(me_count, 10)):
            y = self.map.board_y + self.map.board_height - self.map.checker_radius - i * self.map.checker_radius // 2
            pygame.draw.circle(self.screen, self.map.WHITE, (self.map.bear_off_x, y), self.map.checker_radius)
            pygame.draw.circle(self.screen, self.map.BLACK, (self.map.bear_off_x, y), self.map.checker_radius, 2)

    def _draw_labels(self):
        """Draw point numbers and other labels."""
        font = pygame.font.Font(None, self.map.label_font_size)

        for point in range(13, 25):
            x, _, _ = self.map.get_point_coordinates(point, Player.OPPONENT)
            text = font.render(str(point), True, self.map.BLACK)
            text_rect = text.get_rect(center=(x, self.map.board_y - self.map.label_margin))
            self.screen.blit(text, text_rect)

        for point in range(1, 13):
            x, _, _ = self.map.get_point_coordinates(point, Player.OPPONENT)
            text = font.render(str(point), True, self.map.BLACK)
            text_rect = text.get_rect(center=(x, self.map.board_y + self.map.board_height + self.map.label_margin))
            self.screen.blit(text, text_rect)

    def _draw_pipcounts(self, position: Position):
        """Draw each player's pipcount in the label row at the side of their home board."""
        font = pygame.font.Font(None, self.map.label_font_size)
        x = self.map.bear_off_x
        top_y = self.map.board_y - self.map.label_margin
        bottom_y = self.map.board_y + self.map.board_height + self.map.label_margin
        for player, color, y in ((Player.ME, self.map.WHITE, bottom_y),
                                 (Player.OPPONENT, self.map.RED_LIGHT, top_y)):
            text = font.render(str(position.pipcount(player)), True, color)
            self.screen.blit(text, text.get_rect(center=(x, y)))

    def _draw_cube(self, position: Position):
        """Draw the doubling cube on the left margin at the side of the owner."""
        if position.cube_owner is None:
            cube_y = self.map.cube_y_center
            label = "64"
        elif position.cube_owner == Player.ME:
            cube_y = self.map.cube_y_me
            label = str(position.cube_value)
        else:
            cube_y = self.map.cube_y_opponent
            label = str(position.cube_value)

        cube_size = self.map.cube_size
        x = self.map.cube_x - cube_size // 2
        y = cube_y - cube_size // 2

        pygame.draw.rect(self.screen, self.map.WHITE, (x, y, cube_size, cube_size))
        pygame.draw.rect(self.screen, self.map.BLACK, (x, y, cube_size, cube_size), 2)

        font = pygame.font.Font(None, self.map.cube_font_size)
        text = font.render(label, True, self.map.BLACK)
        text_rect = text.get_rect(center=(self.map.cube_x, cube_y))
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

    def _dice_rects(self) -> list:
        """Where the two dice are drawn (left, right), or [] when no dice are shown."""
        controller = self.game_controller
        if not controller.is_running or controller.is_hidden or controller.game.dice is None:
            return []
        size = round(self.map.checker_radius * 1.6)
        center_x, center_y = self.map.dice_center(controller.game.current_player)
        return [pygame.Rect(center_x + (i - 1) * (size + size // 4) + size // 8, center_y - size // 2, size, size)
                for i in range(2)]

    def _draw_dice(self):
        """Draw the current roll on the mover's half of the board during a displayed game: high
        left, low right (unless swapped), dice already played faded."""
        rects = self._dice_rects()
        if not rects:
            return
        player = self.game_controller.game.current_player
        die_color, pip_color = ((self.map.WHITE, self.map.BLACK) if player == Player.ME
                                else (self.map.RED, self.map.WHITE))
        for rect, (value, played) in zip(rects, self.game_controller.display_dice()):
            face = tuple((c + 2 * 160) // 3 for c in die_color) if played else die_color
            pips = tuple((c + 2 * 160) // 3 for c in pip_color) if played else pip_color
            radius = rect.width // 6
            pygame.draw.rect(self.screen, face, rect, border_radius=radius)
            pygame.draw.rect(self.screen, self.map.BLACK, rect, 2, border_radius=radius)
            self._draw_pips(rect, value, pips)

    PIP_LAYOUT = {
        1: [(0, 0)], 2: [(-1, -1), (1, 1)], 3: [(-1, -1), (0, 0), (1, 1)],
        4: [(-1, -1), (1, -1), (-1, 1), (1, 1)], 5: [(-1, -1), (1, -1), (0, 0), (-1, 1), (1, 1)],
        6: [(-1, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (1, 1)],
    }

    def _draw_pips(self, rect, value, color):
        offset = rect.width // 4
        radius = max(rect.width // 10, 2)
        for dx, dy in self.PIP_LAYOUT[value]:
            pygame.draw.circle(self.screen, color, (rect.centerx + dx * offset, rect.centery + dy * offset), radius)

    def handle_game_click(self, mouse_pos, is_left_click: bool):
        """A click on the board during a game: part of entering a human move."""
        controller = self.game_controller
        if not controller.is_human_turn:
            return
        if is_left_click and any(rect.collidepoint(mouse_pos) for rect in self._dice_rects()):
            controller.click_dice()
            return
        point = self.map.point_at(*mouse_pos, controller.game.current_player)
        controller.handle_click(point, is_left_click)

    def show_text(self, text: str):
        self.text_area.show_text(text)
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

    def handle_click_entering(self, mouse_pos, player, is_left_click: bool):
        """Handle mouse click: have checkers on point for player ME until the mouse cursor."""
        x, y = mouse_pos
        is_increase_click = is_left_click if player == Player.ME else not is_left_click

        # Check for overflow click on the 5th slot of either player's checkers
        for check_player in [Player.ME, Player.OPPONENT]:
            check_positions = self.map.me_possible_positions if check_player == Player.ME else self.map.opponent_possible_positions
            for point, checker_num, coordinates in check_positions:
                if checker_num != self.map.MAX_VISIBLE_CHECKERS:
                    continue
                center_x, center_y = coordinates
                if ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5 > self.map.checker_radius:
                    continue
                count = self.position.get_checkers(check_player, point)
                check_player_is_increase = is_left_click if check_player == Player.ME else not is_left_click
                if check_player_is_increase and count >= self.map.MAX_VISIBLE_CHECKERS:
                    self.map.increment_checkers(self.position, point, check_player)
                    self.draw(self.position)
                    return
                if not check_player_is_increase and count > self.map.MAX_VISIBLE_CHECKERS:
                    self.map.decrement_checkers(self.position, point, check_player)
                    self.draw(self.position)
                    return

        positions = self.map.me_possible_positions if player == Player.ME else self.map.opponent_possible_positions
        for point, checker_num, coordinates in positions:
            center_x, center_y = coordinates
            distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5

            if distance <= self.map.checker_radius:
                self.map.adjust_checkers(self.position, point, checker_num, player)
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

    def handle_resize(self, width, height):
        """Handle window resize: rebuild layout and reinitialize positions."""
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        self.map = BoardMap(width, height)
        self.text_area = TextArea(self.screen, self.map)
        Logger.set_text_area(self.text_area)
        old, self.analysis_panel = self.analysis_panel, AnalysisPanel(self.screen, self.map)
        self.analysis_panel.analysis, self.analysis_panel.selected = old.analysis, old.selected
        self.analysis_panel.status = old.status

    def handle_panel_click(self, mouse_pos) -> bool:
        """A left click on the analysis panel: select a move (shown on the board), or play the
        selected move when it's the human's turn. Returns whether the click was on the panel."""
        action = self.analysis_panel.handle_click(mouse_pos)
        if action == "play":
            candidate = self.analysis_panel.selected_move()
            controller = self.game_controller
            if (candidate is not None and candidate.move is not None and controller.is_human_turn and
                    self.analysis_panel.analysis.matches(*self._analysis_target())):
                controller.play_move(candidate.move)
                self.analysis_panel.clear()
        return action is not None

    def quit(self):
        """Clean up pygame."""
        pygame.quit()

    def _get_submenu_width(self, items, font) -> int:
        padding = self.map.submenu_text_x_offset * 2 + self.map.menu_item_padding
        max_width = max(
            font.size(item)[0] + (font.size("  " + self.item_shortcuts[item])[0] if item in self.item_shortcuts else 0)
            for item in items
        )
        return max_width + padding

    def _get_submenu_x(self, menu_name: str) -> int:
        """Return the x position of the named menu item."""
        font = pygame.font.Font(None, self.map.ui_font_size)
        x_offset = self.map.menu_item_x
        for item in self.menu_items:
            item_width = font.size(item)[0] + self.map.menu_item_padding
            if item == menu_name:
                return x_offset
            x_offset += item_width
        return self.map.menu_item_x

    def _draw_menu(self):
        """Draw the menu bar."""
        if not self.show_menu:
            return

        font = pygame.font.Font(None, self.map.ui_font_size)

        menu_rect = pygame.Rect(0, 0, self.map.width, self.map.menu_height)
        pygame.draw.rect(self.screen, self.map.MENU_BG, menu_rect)
        pygame.draw.line(self.screen, self.map.BLACK, (0, self.map.menu_height), (self.map.width, self.map.menu_height), 1)

        x_offset = self.map.menu_item_x
        for item in self.menu_items:
            item_width = font.size(item)[0] + self.map.menu_item_padding
            item_rect = pygame.Rect(x_offset, self.map.menu_item_y, item_width, self.map.menu_height - 10)

            if self.active_submenu == item:
                pygame.draw.rect(self.screen, self.map.MENU_HOVER, item_rect)

            text = font.render(item, True, self.map.BLACK)
            self.screen.blit(text, (x_offset + self.map.menu_item_padding // 2, self.map.menu_text_y))
            x_offset += item_width

        if self.active_submenu:
            submenu_x = self._get_submenu_x(self.active_submenu)
            items = self.menus[self.active_submenu]
            submenu_y = self.map.menu_height
            submenu_width = self._get_submenu_width(items, font)
            submenu_height = len(items) * self.map.submenu_item_height + 10
            submenu_rect = pygame.Rect(submenu_x, submenu_y, submenu_width, submenu_height)
            pygame.draw.rect(self.screen, self.map.MENU_BG, submenu_rect)
            pygame.draw.rect(self.screen, self.map.BLACK, submenu_rect, 1)

            for i, item in enumerate(items):
                item_y = submenu_y + 5 + i * self.map.submenu_item_height
                color = self.map.BLACK if self.is_menu_item_enabled(self.active_submenu, item) else self.MENU_DISABLED
                text = font.render(item, True, color)
                self.screen.blit(text, (submenu_x + self.map.submenu_text_x_offset, item_y))
                if item in self.item_shortcuts:
                    shortcut_text = font.render(self.item_shortcuts[item], True, color)
                    shortcut_x = submenu_x + submenu_width - shortcut_text.get_width() - self.map.submenu_text_x_offset
                    self.screen.blit(shortcut_text, (shortcut_x, item_y))

    def handle_menu_click(self, mouse_pos):
        """Handle menu clicks."""
        x, y = mouse_pos

        if self.show_menu and y < self.map.menu_height:
            font = pygame.font.Font(None, self.map.ui_font_size)
            x_offset = self.map.menu_item_x
            for item in self.menu_items:
                item_width = font.size(item)[0] + self.map.menu_item_padding
                if x_offset <= x <= x_offset + item_width:
                    self.active_submenu = item if self.active_submenu != item else None
                    return True
                x_offset += item_width
            self.active_submenu = None
            return False

        if self.active_submenu:
            submenu_x = self._get_submenu_x(self.active_submenu)
            items = self.menus[self.active_submenu]
            submenu_y = self.map.menu_height
            submenu_width = self._get_submenu_width(items, pygame.font.Font(None, self.map.ui_font_size))
            submenu_height = len(items) * self.map.submenu_item_height + 10
            if submenu_x <= x <= submenu_x + submenu_width and submenu_y <= y <= submenu_y + submenu_height:
                item_index = (y - submenu_y - 5) // self.map.submenu_item_height
                if 0 <= item_index < len(items):
                    menu, item = self.active_submenu, items[item_index]
                    if not self.is_menu_item_enabled(menu, item):
                        return True   # disabled: ignore the click, keep the menu open
                    self.active_submenu = None
                    self._handle_submenu_action(menu, item)
                return True
            self.active_submenu = None

        return False

    MENU_DISABLED = (150, 150, 150)

    def is_menu_item_enabled(self, menu: str, item: str) -> bool:
        if menu == "Play":
            return self.game_controller.can_stop if item == "Stop" else not self.is_game_running
        if menu == "Edit":
            return not self.is_game_running
        return True

    def _board_snapshot(self):
        self.draw(self.position)
        return self.screen.copy()

    def _handle_submenu_action(self, menu: str, item: str):
        if menu == "Analyze" and item == "Analyze":
            self.analyze()
        elif menu == "Analyze" and item == "Settings":
            SettingsDialog(self.screen, self.map, self.analysis_worker).show(self._board_snapshot())
        elif menu == "Play" and item == "Start...":
            config = PlaySetupDialog(self.screen, self.map).run(self._board_snapshot())
            if config is not None:
                self.text_area.clear()
                self.game_controller.start(config)
                self.position = self.game_controller.display_position()
        elif menu == "Play" and item == "Stop":
            self.game_controller.stop()
        elif menu == "Edit" and item == "Clear Board":
            self.clear_board()
        elif menu == "Edit" and item == "Starting Position":
            self.restore_starting_position()

    def clear_board(self):
        self.position.clear()
        self.text_area.clear()
        self.draw(self.position)

    def restore_starting_position(self):
        self.position.setup_starting_position()
        self.text_area.clear()
        self.draw(self.position)

    def handle_key_press(self, key):
        """Handle key press events."""
        if key == pygame.K_F10 or (key == pygame.K_m and pygame.key.get_pressed()[pygame.K_LALT]):
            self.show_menu = not self.show_menu
            if not self.show_menu:
                self.active_submenu = None
        elif key == pygame.K_a and pygame.key.get_mods() & pygame.KMOD_CTRL:
            self.analyze()
        elif key == pygame.K_BACKSPACE:
            self.game_controller.undo()

    def _analysis_target(self):
        """(position, dice) to analyze: during a game the game's position and dice, otherwise the
        edited position before its roll (dice None)."""
        game = self.game_controller.game
        if game is not None:
            return game.position, game.dice
        return self.position, None

    def analyze(self):
        """Analyze in the background with the Rust engine at the analysis depth; update() shows the
        result in the analysis panel. On a human's turn in a game: all moves with the dice, for
        the game's scoring. Without a game: the edited position, for the side on roll."""
        if self._pending_analysis is not None:
            self.analysis_panel.show_status("Analysis already in progress...")
            return
        game = self.game_controller.game
        if game is not None and (not self.game_controller.is_human_turn or game.dice is None):
            self.analysis_panel.show_status("During a game, moves are analyzed on a human player's turn")
            return
        position, dice = self._analysis_target()
        money = game is not None and not game.config.dmp
        depth = Settings.analysis_depth
        what = f"the moves with {dice[0]}-{dice[1]}" if dice else "the position"
        self.analysis_panel.show_status(f"Analyzing {what} at depth {depth}...")
        self._pending_analysis = self.analysis_worker.analyze(position, dice, depth, money)
