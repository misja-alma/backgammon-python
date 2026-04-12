import pygame

from analyzer import Analyzer
from backgammon_position import Position, Player


class CheckerPosition:
    def __init__(self, center, player, point):
        self.center = center  # Tuple of (x, y)
        self.player = player
        self.point = point


# TODO option to show dice roll
class Board:
    def __init__(self, width=1024, height=768):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Backgammon Board")
        
        # Colors
        self.BROWN_LIGHT = (205, 133, 63)
        self.BROWN_DARK = (139, 69, 19)
        self.WHITE = (255, 255, 255)
        self.RED = (220, 20, 60)
        self.BLACK = (0, 0, 0)
        self.BORDER = (101, 67, 33)
        self.HIGHLIGHT_ME = (0, 255, 0)  # Green highlight for ME
        self.HIGHLIGHT_OPPONENT = (255, 165, 0)  # Orange highlight for OPPONENT
        self.MENU_BG = (240, 240, 240)  # Light gray for menu
        self.MENU_HOVER = (200, 200, 255)  # Light blue for menu hover
        
        # Menu state
        self.menu_height = 30
        self.show_menu = True
        self.menu_items = ["Analyze"]
        self.selected_menu = None
        self.show_analyze_submenu = False
        self.analyze_items = ["Show winning chances"]
        
        # Board dimensions
        self.board_margin = 50 + self.menu_height + 20  # Extra space for menu on all sides
        self.board_width = width - 2 * self.board_margin
        self.board_height = height - 2 * self.board_margin
        
        # Point dimensions
        self.point_width = self.board_width // 14  # 12 points + 1 bar space + 1 bear off
        self.point_height = self.board_height // 3
        self.checker_radius = max(self.point_width // 2 - 2, 2)
        
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
        self.screen.fill(self.BORDER)
        
        # Draw the main board background
        board_rect = pygame.Rect(self.board_margin, self.board_margin, 
                               self.board_width, self.board_height)
        pygame.draw.rect(self.screen, self.BROWN_LIGHT, board_rect)
        
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
            x = self.board_margin + i * self.point_width
            if i >= 6:  # Skip bar space
                x += self.point_width
            
            color = self.BROWN_DARK if i % 2 == 0 else self.WHITE
            self._draw_triangle(x, self.board_margin, self.point_width, self.point_height, color, True)
        
        # Bottom points (1-12)
        for i in range(12):
            x = self.board_margin + (11 - i) * self.point_width
            if i < 6:  # Skip bar space
                x += self.point_width
            
            color = self.BROWN_DARK if i % 2 == 0 else self.WHITE
            self._draw_triangle(x, self.board_margin + 2 * self.point_height, 
                              self.point_width, self.point_height, color, False)
    
    def _draw_triangle(self, x, y, width, height, color, pointing_down):
        """Draw a triangular point."""
        if pointing_down:
            points = [(x, y), (x + width, y), (x + width // 2, y + height)]
        else:
            points = [(x, y + height), (x + width, y + height), (x + width // 2, y)]
        
        pygame.draw.polygon(self.screen, color, points)
        pygame.draw.polygon(self.screen, self.BLACK, points, 2)
    
    def _draw_checkers(self, position: Position):
        """Draw checkers on the board according to the position."""
        # Clear checker positions list
        self.checker_positions = []
        
        # Draw checkers on points 1-24
        for point in range(1, 25):
            white_count = position.get_checkers(Player.ME, point)
            black_count = position.get_checkers(Player.OPPONENT, point)
            
            if white_count > 0:
                self._draw_checkers_on_point(point, white_count, Player.ME)
            if black_count > 0:
                self._draw_checkers_on_point(point, black_count, Player.OPPONENT)
        
        # Draw checkers on bar (point 25)
        self._draw_checkers_on_bar(position)
        
        # Draw checkers borne off (point 0)
        self._draw_checkers_borne_off(position)
    
    def _draw_checkers_on_point(self, point: int, count: int, player: Player):
        """Draw checkers on a specific point."""
        x, y, is_top = self._get_point_coordinates(point, player)
        
        color = self.WHITE if player == Player.ME else self.RED
        is_selected = (self.selected_point == point and self.selected_player == player)
        
        for i in range(min(count, 5)):  # Show max 5 checkers visually
            checker_y = y + (i * self.checker_radius * 2) if is_top else y - (i * self.checker_radius * 2)
            
            # Record checker position for all checkers
            self.checker_positions.append(CheckerPosition((x, checker_y), player, point))
            
            # Add highlight if this point is selected
            if is_selected:
                highlight_color = self.HIGHLIGHT_ME if player == Player.ME else self.HIGHLIGHT_OPPONENT
                # Draw checker with normal color
                pygame.draw.circle(self.screen, highlight_color, (x, checker_y), self.checker_radius)
                pygame.draw.circle(self.screen, self.BLACK, (x, checker_y), self.checker_radius, 2)
            else:
                # Draw checker with normal color
                pygame.draw.circle(self.screen, color, (x, checker_y), self.checker_radius)
                pygame.draw.circle(self.screen, self.BLACK, (x, checker_y), self.checker_radius, 2)
        
        # If more than 5 checkers, show count
        if count > 5:
            font = pygame.font.Font(None, 24)
            text = font.render(str(count), True, self.BLACK)
            text_rect = text.get_rect(center=(x, y))
            self.screen.blit(text, text_rect)
    
    def _get_point_coordinates(self, point: int, player: Player):
        """Get the center coordinates for a point and whether it's on top."""
        if player == Player.ME:
            # ME moves counter-clockwise from right top to right bottom
            if point >= 13:
                i = point - 13
                x = self.board_margin + i * self.point_width + self.point_width // 2
                if i >= 6:
                    x += self.point_width
                y = self.board_margin + self.checker_radius
                return x, y, True
            else:  # Bottom points (1-12)
                i = 12 - point
                x = self.board_margin + i * self.point_width + self.point_width // 2
                if i >= 6:
                    x += self.point_width
                y = self.height - self.board_margin - self.checker_radius
                return x, y, False
        else:
            # OPPONENT moves clockwise right bottom to right top (reverse of ME)
            if point >= 13:
                i = point - 13
                x = self.board_margin + i * self.point_width + self.point_width // 2
                if i >= 6:
                    x += self.point_width
                y = self.height -self.board_margin - self.checker_radius
                return x, y, False
            else:  # Top points (1-12)
                i = 12 - point
                x = self.board_margin + i * self.point_width + self.point_width // 2
                if i >= 6:
                    x += self.point_width
                y = self.board_margin + self.checker_radius
                return x, y, True
    
    def _draw_checkers_on_bar(self, position: Position):
        """Draw checkers on the bar."""
        bar_x = self.board_margin + 6 * self.point_width + self.point_width // 2
        
        me_count = position.get_checkers(Player.ME, 25)
        opponent_count = position.get_checkers(Player.OPPONENT, 25)
        
        # My checkers on bar (top half)
        for i in range(me_count):
            y = self.board_margin + self.point_height + self.checker_radius - i * self.checker_radius * 2
            self.checker_positions.append(CheckerPosition((bar_x, y), Player.ME, 25))
            pygame.draw.circle(self.screen, self.WHITE, (bar_x, y), self.checker_radius)
            pygame.draw.circle(self.screen, self.BLACK, (bar_x, y), self.checker_radius, 2)
        
        # Opponent checkers on bar (bottom half)
        for i in range(opponent_count):
            y = self.height - self.board_margin - self.point_height - self.checker_radius + i * self.checker_radius * 2
            self.checker_positions.append(CheckerPosition((bar_x, y), Player.OPPONENT, 25))
            pygame.draw.circle(self.screen, self.RED, (bar_x, y), self.checker_radius)
            pygame.draw.circle(self.screen, self.BLACK, (bar_x, y), self.checker_radius, 2)
    
    def _draw_checkers_borne_off(self, position: Position):
        """Draw borne off checkers."""
        bear_off_x = self.board_margin + 13 * self.point_width + self.point_width // 2
        
        me_count = position.get_checkers(Player.ME, 0)
        opponent_count = position.get_checkers(Player.OPPONENT, 0)
        
        # My checkers borne off (bottom)
        for i in range(min(me_count, 10)):
            x = bear_off_x
            y = self.board_margin +  self.checker_radius + i * self.checker_radius // 2
            pygame.draw.circle(self.screen, self.WHITE, (x, y), self.checker_radius)
            pygame.draw.circle(self.screen, self.BLACK, (x, y), self.checker_radius, 2)
        
        # Opponent checkers borne off (top)
        for i in range(min(opponent_count, 10)):
            x = bear_off_x
            y = self.height - self.board_margin - self.checker_radius - i * self.checker_radius // 2
            pygame.draw.circle(self.screen, self.RED, (x, y), self.checker_radius)
            pygame.draw.circle(self.screen, self.BLACK, (x, y), self.checker_radius, 2)
    
    def _draw_labels(self):
        """Draw point numbers and other labels."""
        font = pygame.font.Font(None, 20)
        
        # Top point labels (13-24) - use ME's perspective for positioning
        for point in range(13, 25):
            x, _, _ = self._get_point_coordinates(point, Player.OPPONENT)
            text = font.render(str(point), True, self.BLACK)
            text_rect = text.get_rect(center=(x, self.board_margin - 15))
            self.screen.blit(text, text_rect)
        
        # Bottom point labels (1-12) - use ME's perspective for positioning
        for point in range(1, 13):
            x, _, _ = self._get_point_coordinates(point, Player.OPPONENT)
            text = font.render(str(point), True, self.BLACK)
            text_rect = text.get_rect(center=(x, self.board_margin + self.board_height + 15))
            self.screen.blit(text, text_rect)

    def _draw_turn_indicator(self):
        """Draw turn indicator arrow in the right margin."""
        if not self.position:
            return
            
        # Arrow position in right margin
        arrow_x = self.width - self.board_margin // 3
        arrow_size = 40
        
        if self.position.turn == Player.ME:
            # White's turn - arrow at bottom in white
            arrow_y = self.height - self.board_margin - 30
            color = self.WHITE
        else:
            # Red's turn - arrow at top in red  
            arrow_y = self.board_margin + 30
            color = self.RED
        
        # Draw arrow pointing right
        points = [
            (arrow_x - arrow_size, arrow_y - arrow_size // 2),
            (arrow_x - arrow_size, arrow_y + arrow_size // 2),
            (arrow_x, arrow_y)
        ]
        
        pygame.draw.polygon(self.screen, color, points)
        pygame.draw.polygon(self.screen, self.BLACK, points, 2)

    def _draw_text_area(self):
        """Draw the text display area in the bottom margin."""
            
        # Text area positioned in bottom margin
        text_area_height = 30
        text_area_y = self.height - self.board_margin + 30
        text_area_rect = pygame.Rect(self.board_margin, text_area_y, 
                                   self.board_width, text_area_height)
        
        # Draw white background
        pygame.draw.rect(self.screen, self.WHITE, text_area_rect)
        pygame.draw.rect(self.screen, self.BLACK, text_area_rect, 2)
        
        # Draw text
        font = pygame.font.Font(None, 24)
        text_surface = font.render(self.display_text, True, self.BLACK)
        
        # Center text vertically in the area
        text_x = text_area_rect.x + 10  # Small padding from left
        text_y = text_area_rect.y + (text_area_height - text_surface.get_height()) // 2
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
        
        # Arrow position and size (same as in _draw_turn_indicator)
        arrow_x = self.width - self.board_margin // 3
        arrow_size = 40
        
        if self.position.turn == Player.ME:
            arrow_y = self.height - self.board_margin - 30
        else:
            arrow_y = self.board_margin + 30
        
        # Check if click is within arrow bounds (rectangular approximation)
        if (arrow_x - arrow_size <= x <= arrow_x and 
            arrow_y - arrow_size // 2 <= y <= arrow_y + arrow_size // 2):
            self.position.switch_turn()
            return True
        
        return False

    def handle_click_entering(self, mouse_pos, player):
        """Handle mouse click: have checkers on point for player ME until the mouse cursor."""
        x, y = mouse_pos

        # Look up the mouse position in player's possible checker positions
        for point, checker_num, coordinates in (self.me_possible_positions if player == Player.ME else self.opponent_possible_positions):
            center_x, center_y = coordinates
            distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
            
            if distance <= self.checker_radius:
                # Found a match - call adjust_checkers
                self.adjust_checkers(point, checker_num, player)
                self.draw(self.position)
                return
        
        # Not found - no action needed




    def handle_click(self, mouse_pos):
        """Handle mouse click to select checkers using CheckerPosition list."""
        x, y = mouse_pos
        
        # Check all recorded checker positions
        for checker_pos in self.checker_positions:
            center_x, center_y = checker_pos.center
            distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
            
            if distance <= self.checker_radius:
                self.selected_point = checker_pos.point
                self.selected_player = checker_pos.player
                return
        
        # If no checker was clicked, clear selection
        self.selected_point = None
        self.selected_player = None
    
    def _initialize_possible_positions(self):
        """Initialize lists of all possible checker positions for both players."""
        self.me_possible_positions = []
        self.opponent_possible_positions = []
        
        # For each point (0-25), calculate coordinates for up to 5 checkers
        for point in range(26):  # Points 0-25
            for player in [Player.ME, Player.OPPONENT]:
                if point == 0:
                    # Borne off area - special handling
                    bear_off_x = self.board_margin + 13 * self.point_width + self.point_width // 2
                    for checker_num in range(1, 6):  # 1-5 checkers
                        if player == Player.ME:
                            y = self.board_margin + self.checker_radius + (checker_num - 1) * self.checker_radius // 2
                        else:
                            y = self.height - self.board_margin - self.checker_radius - (checker_num - 1) * self.checker_radius // 2
                        
                        pos_tuple = (point, checker_num, (bear_off_x, y))
                        if player == Player.ME:
                            self.me_possible_positions.append(pos_tuple)
                        else:
                            self.opponent_possible_positions.append(pos_tuple)
                
                elif point == 25:
                    # Bar - special handling
                    bar_x = self.board_margin + 6 * self.point_width + self.point_width // 2
                    for checker_num in range(1, 6):  # 1-5 checkers
                        if player == Player.ME:
                            y = self.board_margin + self.point_height + self.checker_radius - (checker_num - 1) * self.checker_radius * 2
                        else:
                            y = self.height - self.board_margin - self.point_height - self.checker_radius + (checker_num - 1) * self.checker_radius * 2
                        
                        pos_tuple = (point, checker_num, (bar_x, y))
                        if player == Player.ME:
                            self.me_possible_positions.append(pos_tuple)
                        else:
                            self.opponent_possible_positions.append(pos_tuple)
                
                elif 1 <= point <= 24:
                    # Regular points
                    x, y, is_top = self._get_point_coordinates(point, player)
                    for checker_num in range(0, 6):  # 0-5 checkers
                        checker_y = y + ((checker_num - 1) * self.checker_radius * 2) if is_top else y - ((checker_num - 1) * self.checker_radius * 2)
                        
                        pos_tuple = (point, checker_num, (x, checker_y))
                        if player == Player.ME:
                            self.me_possible_positions.append(pos_tuple)
                        else:
                            self.opponent_possible_positions.append(pos_tuple)
    
    def quit(self):
        """Clean up pygame."""
        pygame.quit()

    def adjust_checkers(self, point, checker_num, player):
        current =  self.position.get_checkers(player, point)
        
        if point > 0 and current != checker_num:
            other_player = player.other_player()
            other_point = 25 - point
            
            if point != 25 and self.position.get_checkers(other_player, other_point) > 0:
                # remove any opponent's checkers from the point
                other_checkers_on_point = self.position.get_checkers(other_player, other_point)
                current_off = self.position.get_checkers(other_player, 0)
                
                self.position.set_checkers(other_player, 0, current_off + other_checkers_on_point)
                self.position.set_checkers(other_player, other_point, 0)

            if checker_num > current:
                # add checkers to point
                diff = checker_num - current
                available = self.position.get_checkers(player, 0)
                new_checkers = min(diff, available)
                new_checkers_off = available - new_checkers
                
                self.position.set_checkers(player, point, current + new_checkers)
                self.position.set_checkers(player, 0, new_checkers_off)
            else:
                # remove checkers from point
                diff = current - checker_num
                new_checkers_off = self.position.get_checkers(player, 0) + diff
                self.position.set_checkers(player, point, current - diff)
                self.position.set_checkers(player, 0, new_checkers_off)

    def _draw_menu(self):
        """Draw the menu bar."""
        if not self.show_menu and not self.show_analyze_submenu:
            return
            
        font = pygame.font.Font(None, 24)
        
        # Draw main menu bar
        if self.show_menu:
            menu_rect = pygame.Rect(0, 0, self.width, self.menu_height)
            pygame.draw.rect(self.screen, self.MENU_BG, menu_rect)
            pygame.draw.line(self.screen, self.BLACK, (0, self.menu_height), (self.width, self.menu_height), 1)
            
            x_offset = 10
            for i, item in enumerate(self.menu_items):
                item_width = font.size(item)[0] + 20
                item_rect = pygame.Rect(x_offset, 5, item_width, self.menu_height - 10)
                
                if self.selected_menu == i:
                    pygame.draw.rect(self.screen, self.MENU_HOVER, item_rect)
                
                text = font.render(item, True, self.BLACK)
                self.screen.blit(text, (x_offset + 10, 8))
                x_offset += item_width
        
        # Draw analyze submenu
        if self.show_analyze_submenu:
            submenu_y = self.menu_height if self.show_menu else 0
            submenu_height = len(self.analyze_items) * 25 + 10
            submenu_rect = pygame.Rect(10, submenu_y, 200, submenu_height)
            pygame.draw.rect(self.screen, self.MENU_BG, submenu_rect)
            pygame.draw.rect(self.screen, self.BLACK, submenu_rect, 1)
            
            for i, item in enumerate(self.analyze_items):
                item_y = submenu_y + 5 + i * 25
                text = font.render(item, True, self.BLACK)
                self.screen.blit(text, (15, item_y))

    def handle_menu_click(self, mouse_pos):
        """Handle menu clicks."""
        x, y = mouse_pos
        
        # Check main menu clicks
        if self.show_menu and y < self.menu_height:
            font = pygame.font.Font(None, 24)
            x_offset = 10
            for i, item in enumerate(self.menu_items):
                item_width = font.size(item)[0] + 20
                if x_offset <= x <= x_offset + item_width:
                    if item == "Analyze":
                        self.show_analyze_submenu = not self.show_analyze_submenu
                    return True
                x_offset += item_width
        
        # Check analyze submenu clicks
        if self.show_analyze_submenu:
            submenu_y = self.menu_height if self.show_menu else 0
            submenu_height = len(self.analyze_items) * 25 + 10
            if 10 <= x <= 210 and submenu_y <= y <= submenu_y + submenu_height:
                item_index = (y - submenu_y - 5) // 25
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

