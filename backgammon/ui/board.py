import copy
import io
import os
import time
from typing import Optional

import pygame

from backgammon.game.analysis_worker import AnalysisWorker
from backgammon.game import position_formats
from backgammon.game.move_analysis import RolloutProgress, position_key
from backgammon.game.position_formats import Format, SavedPosition, to_xgid
from backgammon.game.position import GameVariant, Position, Player
from backgammon.game.settings import APP_NAME, Settings
from backgammon.logger import Logger
from backgammon.ui.analysis_panel import AnalysisPanel
from backgammon.ui.board_map import BoardMap
from backgammon.ui.game_controller import GameController
from backgammon.ui.play_setup_dialog import PlaySetupDialog
from backgammon.ui.rollout_dialog import RolloutDialog
from backgammon.ui.save_position_dialog import SavePositionDialog
from backgammon.ui.system_dialogs import (choose_file_to_open, choose_file_to_save, copy_to_clipboard,
                                          paste_from_clipboard)
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
        self.position_file: Optional[str] = None   # the file last loaded or saved, see set_position_file
        pygame.display.set_caption(APP_NAME)

        # Menu state
        self.show_menu = True
        self.menus = {
            "File": ["Load position...", "Save position..."],
            "Edit": ["Clear Board", "Starting Position", "Copy position", "Paste position"],
            "Play": ["Start...", "Stop"],
            "Analyze": ["Analyze position", "Rollout...", "Stop rollout", "Settings"],
        }
        self.menu_items = list(self.menus)
        self.active_submenu = None  # name of the currently open submenu, or None
        self.item_shortcuts = {"Analyze position": "Ctrl+A", "Load position...": "Ctrl+O",
                               "Save position...": "Ctrl+S", "Copy position": "Ctrl+C",
                               "Paste position": "Ctrl+V"}

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
        self.analysis_worker.set_rust_threads(Settings.rust_threads or os.cpu_count() or 1)
        self.analysis_worker.set_rust_move_filter(Settings.search_space_margin, Settings.search_space_moves)
        self.analysis_worker.set_rust_network(Settings.rust_network).add_done_callback(self._log_evaluator)
        self.analysis_worker.set_rust_bearoff(Settings.rust_bearoff_db).add_done_callback(self._log_bearoff)
        self._pending_analysis = None  # Future of the running analysis or rollout, or None
        self._rollout_progress = None  # RolloutProgress of the running rollout, or None
        self._rollout_start = 0.0      # time.monotonic() when it started
        self._rollout_line = None      # the text area line showing the rollout's progress
        # Dice of the edited position (outside a game): (left die, right die) for the side on
        # roll, or None for the position before the roll. Set by clicking the dice.
        self.edit_dice = None
        # Scoring of the edited position (outside a game): True for money ("Unlimited": gammons
        # and backgammons count), False for double match point. Set with the Game control above
        # the board, and from a loaded position.
        self.edit_money = True
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
        if analysis is not None and (not analysis.matches(*self._analysis_target())
                                     or analysis.money != self.money_scoring):
            self.analysis_panel.clear()

        future = self._pending_analysis
        if future is not None and not future.done() and self._rollout_progress is not None:
            self._show_rollout_progress()
        if future is None or not future.done():
            return
        self._pending_analysis = None
        error = future.exception()
        if self._rollout_progress is not None:
            progress, self._rollout_progress = self._rollout_progress, None
            seconds = time.monotonic() - self._rollout_start
            per_move = f" for each of {progress.moves} moves" if progress.moves > 1 else ""
            if error is not None:
                text = f"Rollout failed: {error}"
            elif progress.game < progress.games:
                text = f"Rollout stopped after {progress.game} of {progress.games} games{per_move} ({seconds:.1f} s)"
            else:
                text = f"Rollout finished: {progress.games} games{per_move} in {seconds:.1f} s"
            self.text_area.show_or_replace(self._rollout_line, text)
            self._rollout_line = None
            if error is None and future.result() is None:
                self.analysis_panel.show_status("Rollout stopped before any games were finished")
                return
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
        self._draw_scoring_control()
        self._draw_dice()
        self.text_area.draw()
        self.analysis_panel.draw()
        self._draw_menu()

        pygame.display.flip()

    def _show_rollout_progress(self):
        """The running rollout's progress, on one line of the text area below the board."""
        progress = self._rollout_progress
        moves = f" {progress.moves} moves" if progress.moves > 1 else ""
        if progress.stop:
            text = f"Stopping the rollout after {progress.game} games..."
        else:
            text = f"Rolling out{moves}: game {min(progress.game + 1, progress.games)} of {progress.games}"
        self._rollout_line = self.text_area.show_or_replace(self._rollout_line, text)

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

    @property
    def money_scoring(self) -> bool:
        """Whether positions are analyzed, saved and copied for money (gammons count): the running
        game's scoring, or outside a game the Game control's."""
        game = self.game_controller.game
        return not game.config.dmp if game is not None else self.edit_money

    SCORING_OPTIONS = ((True, "Unlimited"), (False, "DMP"))

    def _scoring_layout(self):
        """(label position, [(money, radio rect, text position)]) of the Game control: one line in
        the margin above the board, starting at its left edge."""
        font = pygame.font.Font(None, self.map.label_font_size)
        y = (self.map.menu_height + self.map.board_y - 2 * self.map.label_margin) // 2
        radius = max(font.get_height() // 3, 5)
        x = self.map.board_x
        label_pos = (x, y - font.get_height() // 2)
        x += font.size("Game:")[0] + 2 * radius
        options = []
        for money, text in self.SCORING_OPTIONS:
            rect = pygame.Rect(x, y - radius, 2 * radius, 2 * radius)
            text_pos = (rect.right + radius, y - font.get_height() // 2)
            options.append((money, rect, text_pos))
            x = text_pos[0] + font.size(text)[0] + 3 * radius
        return label_pos, options

    def _draw_scoring_control(self):
        """The Game control: Unlimited (money) or DMP. Editable outside a game; during a game it
        shows the game's scoring, greyed."""
        font = pygame.font.Font(None, self.map.label_font_size)
        enabled = not self.is_game_running
        color = self.map.WHITE if enabled else (170, 150, 130)
        label_pos, options = self._scoring_layout()
        self.screen.blit(font.render("Game:", True, color), label_pos)
        current = self.money_scoring
        for (money, rect, text_pos), (_, text) in zip(options, self.SCORING_OPTIONS):
            pygame.draw.circle(self.screen, color, rect.center, rect.width // 2, 2)
            if money == current:
                pygame.draw.circle(self.screen, color, rect.center, max(rect.width // 4, 2))
            self.screen.blit(font.render(text, True, color), text_pos)

    def handle_scoring_click(self, mouse_pos) -> bool:
        """A click on the Game control (outside a game: choose the scoring). Returns whether the
        click was on it."""
        _, options = self._scoring_layout()
        font = pygame.font.Font(None, self.map.label_font_size)
        for (money, rect, text_pos), (_, text) in zip(options, self.SCORING_OPTIONS):
            area = rect.union(pygame.Rect(text_pos, font.size(text))).inflate(6, 6)
            if area.collidepoint(mouse_pos):
                if not self.is_game_running:
                    self.edit_money = money
                return True
        return False

    def _use_scoring_network(self, money: bool) -> None:
        """Outside a game: make the engine use the net for this scoring (the money net for money),
        as a game does when it starts."""
        if self.game_controller.game is None:
            self.analysis_worker.use_rust_network(Settings.rust_network_for(GameVariant.BACKGAMMON, dmp=not money))

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

    def _dice_player(self):
        """Whose dice are drawn: the mover's during a displayed game with dice, the side on roll
        of the edited position outside a game (its edited dice or the placeholders to click),
        else None."""
        controller = self.game_controller
        if not controller.is_running:
            return self.position.get_turn()
        if controller.is_hidden or controller.game.dice is None:
            return None
        return controller.game.current_player

    def _dice_rects(self) -> list:
        """Where the two dice are drawn (left, right), or [] when no dice are shown."""
        player = self._dice_player()
        if player is None:
            return []
        size = round(self.map.checker_radius * 1.6)
        center_x, center_y = self.map.dice_center(player)
        return [pygame.Rect(center_x + (i - 1) * (size + size // 4) + size // 8, center_y - size // 2, size, size)
                for i in range(2)]

    def _draw_dice(self):
        """Draw the dice on the mover's half of the board. During a displayed game the current
        roll: high left, low right (unless swapped), dice already played faded. Outside a game the
        edited dice, or faded empty dice (placeholders to click) when none are set."""
        rects = self._dice_rects()
        if not rects:
            return
        player = self._dice_player()
        die_color, pip_color = ((self.map.WHITE, self.map.BLACK) if player == Player.ME
                                else (self.map.RED, self.map.WHITE))
        if self.game_controller.is_running:
            dice = self.game_controller.display_dice()
        elif self.edit_dice is not None:
            dice = [(value, False) for value in self.edit_dice]
        else:
            dice = [(None, True), (None, True)]
        for rect, (value, faded) in zip(rects, dice):
            face = tuple((c + 2 * 160) // 3 for c in die_color) if faded else die_color
            pips = tuple((c + 2 * 160) // 3 for c in pip_color) if faded else pip_color
            radius = rect.width // 6
            pygame.draw.rect(self.screen, face, rect, border_radius=radius)
            pygame.draw.rect(self.screen, self.map.BLACK if value else self.MENU_DISABLED, rect, 2,
                             border_radius=radius)
            if value:
                self._draw_pips(rect, value, pips)

    def handle_dice_edit_click(self, mouse_pos, is_left_click: bool) -> bool:
        """A click on the dice of the edited position (outside a game): a left click raises that
        die (6 wraps to 1), a right click lowers it, and lowering a 1 removes the dice (the
        position before the roll). Without dice, a click sets them to 1-1 and then applies.
        Returns whether the click was on the dice."""
        if self.game_controller.is_running:
            return False
        for i, rect in enumerate(self._dice_rects()):
            if not rect.collidepoint(mouse_pos):
                continue
            if self.edit_dice is None:
                self.edit_dice = (1, 1)
                if not is_left_click:
                    return True
            dice = list(self.edit_dice)
            if is_left_click:
                dice[i] = dice[i] % 6 + 1
            elif dice[i] == 1:
                self.edit_dice = None
                return True
            else:
                dice[i] -= 1
            self.edit_dice = tuple(dice)
            return True
        return False

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
            self.edit_dice = None   # the dice were the other side's
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
        self.analysis_panel.marked = old.marked
        self.analysis_panel.status = old.status

    def handle_panel_click(self, mouse_pos) -> bool:
        """A left click on the analysis panel: select a move (shown on the board), or play the
        selected move when it's the human's turn. Returns whether the click was on the panel."""
        toggle = bool(pygame.key.get_mods() & (pygame.KMOD_SHIFT | pygame.KMOD_META))
        action = self.analysis_panel.handle_click(mouse_pos, toggle)
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
            # Copying the position of a game is fine; changing it isn't.
            return item == "Copy position" or not self.is_game_running
        if menu == "File":
            return item == "Save position..." or not self.is_game_running
        if menu == "Analyze" and item == "Stop rollout":
            return self._rollout_progress is not None and not self._rollout_progress.stop
        return True

    def _board_snapshot(self):
        self.draw(self.position)
        return self.screen.copy()

    def _handle_submenu_action(self, menu: str, item: str):
        if menu == "Analyze" and item == "Analyze position":
            self.analyze()
        elif menu == "Analyze" and item == "Rollout...":
            self.rollout()
        elif menu == "Analyze" and item == "Stop rollout":
            self.stop_rollout()
        elif menu == "Analyze" and item == "Settings":
            SettingsDialog(self.screen, self.map, self.analysis_worker).show(self._board_snapshot())
        elif menu == "Play" and item == "Start...":
            config = PlaySetupDialog(self.screen, self.map).run(self._board_snapshot())
            if config is not None:
                self.text_area.clear()
                self.set_position_file(None)
                self.game_controller.start(config)
                self.position = self.game_controller.display_position()
        elif menu == "Play" and item == "Stop":
            self.game_controller.stop()
        elif menu == "File" and item == "Load position...":
            self.load_position()
        elif menu == "File" and item == "Save position...":
            self.save_position()
        elif menu == "Edit" and item == "Copy position":
            self.copy_position()
        elif menu == "Edit" and item == "Paste position":
            self.paste_position()
        elif menu == "Edit" and item == "Clear Board":
            self.clear_board()
        elif menu == "Edit" and item == "Starting Position":
            self.restore_starting_position()

    # Ctrl (or Cmd) + key: the File and Edit position items.
    POSITION_KEYS = {pygame.K_o: ("File", "Load position..."), pygame.K_s: ("File", "Save position..."),
                     pygame.K_c: ("Edit", "Copy position"), pygame.K_v: ("Edit", "Paste position")}

    def _saved_position(self) -> SavedPosition:
        """What Save and Copy write: during a game its position and dice, otherwise the edited
        position with its dice; for the scoring shown by the Game control: a money game
        ("Unlimited"), or double match point as a 1-point match, which is what that means to other
        programs."""
        position, dice = self._analysis_target()
        return SavedPosition(copy.deepcopy(position), tuple(dice) if dice else None,
                             match_length=0 if self.money_scoring else 1)

    def _apply_saved(self, saved: SavedPosition, fmt: Format, source: str) -> None:
        """Show a loaded or pasted position for editing and analysis."""
        self.position = saved.position
        self.edit_dice = saved.dice
        # A 1-point match is double match point; a money game "Unlimited". Longer matches can't be
        # shown yet: gammons count there, as in a money game.
        self.edit_money = saved.match_length != 1
        self.analysis_panel.clear()
        scoring = "Unlimited (money)" if self.edit_money else "DMP"
        Logger.log(f"{source} ({fmt.label}, {scoring}): {to_xgid(saved)}")
        if saved.match_length > 1:
            Logger.log(f"  {saved.match_length}-point match at {saved.score_me}-{saved.score_opponent} (ME-OPPONENT): "
                       f"match play isn't supported yet, so it's analyzed as Unlimited (gammons count)")

    def load_position(self) -> None:
        path = choose_file_to_open(self.screen, self.map, self._board_snapshot(), "Load position")
        if path is None:
            return
        try:
            saved, fmt = position_formats.load(path)
        except (OSError, position_formats.FormatError) as e:
            Logger.log(f"Could not load {path}: {e}")
            return
        self._apply_saved(saved, fmt, f"Loaded {os.path.basename(path)}")
        self.set_position_file(path)

    def save_position(self) -> None:
        fmt = SavePositionDialog(self.screen, self.map).run(self._board_snapshot())
        if fmt is None:
            return
        path = choose_file_to_save(self.screen, self.map, self._board_snapshot(), "Save position",
                                   "position" + fmt.extension)
        if path is None:
            return
        saved = self._saved_position()
        try:
            thumbnail = self._thumbnail_jpeg(saved) if fmt == Format.XG else None
            position_formats.save(path, saved, fmt, thumbnail)
        except OSError as e:
            Logger.log(f"Could not save {path}: {e}")
            return
        Logger.log(f"Saved the position to {path} ({fmt.label})")
        self.set_position_file(path)

    def _thumbnail_jpeg(self, saved: SavedPosition) -> Optional[bytes]:
        """The board with the saved position as a 192 x 113 JPEG (an XG file's picture), or None
        when it can't be made (then the file gets the template's picture)."""
        position, dice = self.position, self.edit_dice
        try:
            self.position, self.edit_dice = saved.position, saved.dice
            self.draw()
            m = self.map
            area = pygame.Rect(m.board_x - m.board_margin, m.board_y - 2 * m.label_margin,
                               m.board_width + 2 * m.board_margin, m.board_height + 4 * m.label_margin)
            picture = pygame.transform.smoothscale(self.screen.subsurface(area.clip(self.screen.get_rect())), (192, 113))
            buffer = io.BytesIO()
            pygame.image.save(picture, buffer, "thumbnail.jpg")
            return buffer.getvalue()
        except (pygame.error, ValueError):
            return None
        finally:
            self.position, self.edit_dice = position, dice

    def copy_position(self) -> None:
        """Copy the position to the clipboard as an XGID (what XG, gnubg and most programs paste);
        the GNUbg ID is shown too."""
        saved = self._saved_position()
        xgid = to_xgid(saved)
        try:
            copy_to_clipboard(xgid)
        except Exception as e:   # no clipboard available
            Logger.log(f"Could not copy the position: {e}")
            return
        Logger.log(f"Copied {xgid}   (GNUbg ID: {position_formats.to_gnubg_id(saved)})")

    def paste_position(self) -> None:
        """Set up the position on the clipboard: an XGID, GNUbg ID, Snowie text or SGF."""
        try:
            text = paste_from_clipboard()
        except Exception as e:
            Logger.log(f"Could not read the clipboard: {e}")
            return
        try:
            saved, fmt = position_formats.from_text(text)
        except position_formats.FormatError as e:
            Logger.log(f"No position on the clipboard: {e}")
            return
        self._apply_saved(saved, fmt, "Pasted")
        self.set_position_file(None)

    def set_position_file(self, path: Optional[str]) -> None:
        """The file the position was last loaded from or saved to, shown in the title bar after the
        app's name ('Alma Gammon - position.xgp'); None when the position no longer comes from a
        file (a new board, a pasted position, a game)."""
        self.position_file = path
        name = os.path.basename(path) if path else None
        pygame.display.set_caption(f"{APP_NAME} - {name}" if name else APP_NAME)

    def clear_board(self):
        self.set_position_file(None)
        self.position.clear()
        self.edit_dice = None
        self.text_area.clear()
        self.draw(self.position)

    def restore_starting_position(self):
        self.set_position_file(None)
        self.position.setup_starting_position()
        self.edit_dice = None
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
        elif key in self.POSITION_KEYS and pygame.key.get_mods() & (pygame.KMOD_CTRL | pygame.KMOD_META):
            menu, item = self.POSITION_KEYS[key]
            if self.is_menu_item_enabled(menu, item):
                self._handle_submenu_action(menu, item)
        elif key == pygame.K_BACKSPACE:
            self.game_controller.undo()

    def _analysis_target(self):
        """(position, dice) to analyze: during a game the game's position and dice, otherwise the
        edited position with its edited dice (None: before the roll)."""
        game = self.game_controller.game
        if game is not None:
            return game.position, game.dice
        return self.position, self.edit_dice

    def analyze(self):
        """Analyze in the background with the Rust engine at the analysis depth; update() shows the
        result in the analysis panel. On a human's turn in a game: all moves with the dice, for
        the game's scoring. Without a game: the moves with the edited dice, or the edited position
        when no dice are set, for the side on roll, for the scoring of the Game control."""
        if self._pending_analysis is not None:
            self.analysis_panel.show_status("Analysis already in progress...")
            return
        game = self.game_controller.game
        if game is not None and (not self.game_controller.is_human_turn or game.dice is None):
            self.analysis_panel.show_status("During a game, moves are analyzed on a human player's turn")
            return
        position, dice = self._analysis_target()
        money = self.money_scoring
        self._use_scoring_network(money)
        depth = Settings.analysis_depth
        what = f"the moves with {dice[0]}-{dice[1]}" if dice else "the position"
        self.analysis_panel.show_status(f"Analyzing {what} at {depth + 1}-ply...")
        self._pending_analysis = self.analysis_worker.analyze(position, dice, depth, money)

    def stop_rollout(self):
        """Stop the running rollout after the engine call in progress (a fraction of a second at
        1-ply); update() then shows the games finished so far."""
        if self._rollout_progress is not None:
            self._rollout_progress.stop = True

    def rollout(self):
        """Ask for the rollout settings, then roll out in the background with the Rust engine
        what analyze() would analyze: on a human's turn in a game every move with the dice, for
        the game's scoring; without a game the edited position, for the side on roll, or every
        move with the edited dice when they're set, for the scoring of the Game control. The text area below the
        board shows the progress; update() shows the result in the
        analysis panel."""
        if self._pending_analysis is not None:
            self.analysis_panel.show_status("Analysis already in progress...")
            return
        game = self.game_controller.game
        if game is not None and (not self.game_controller.is_human_turn or game.dice is None):
            self.analysis_panel.show_status("During a game, moves are rolled out on a human player's turn")
            return
        position, dice = self._analysis_target()
        # The candidates: the moves selected in the analysis shown (when it is of these dice)
        # first, then the others in its order; their number starts at the selection's size.
        shown = self.analysis_panel.analysis
        preferred, selected = None, []
        if dice is not None and shown is not None and shown.dice is not None and shown.matches(position, dice):
            selected = self.analysis_panel.marked_moves()
            preferred = [position_key(m.position) for m in selected]
            preferred += [position_key(m.position) for m in shown.moves if m not in selected]
        if dice is None:
            dialog = RolloutDialog(self.screen, self.map, candidates=None)
        elif selected:
            note = f"{len(selected)} move{'s' if len(selected) > 1 else ''} selected in the analysis"
            dialog = RolloutDialog(self.screen, self.map, candidates=len(selected), note=note)
        else:
            tip = "Tip: Shift- or Cmd-click moves in the analysis to choose them" if preferred else ""
            dialog = RolloutDialog(self.screen, self.map, note=tip)
        choice = dialog.run(self._board_snapshot())
        if choice is None:
            return
        games, depth, candidates = choice
        money = self.money_scoring
        self._use_scoring_network(money)
        what = f"the moves with {dice[0]}-{dice[1]}" if dice else "the position"
        self.analysis_panel.show_status(f"Rolling out {what}, {games} games at {depth + 1}-ply...")
        self._rollout_progress = RolloutProgress(games=games)
        self._rollout_line = None   # a new line for this rollout
        self._rollout_start = time.monotonic()
        self._pending_analysis = self.analysis_worker.rollout(position, dice, games, depth, money,
                                                              self._rollout_progress, candidates, preferred)
