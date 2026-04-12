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