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

    # Font sizes
    LABEL_FONT_SIZE = 20
    UI_FONT_SIZE = 24

    # Checker display
    MAX_VISIBLE_CHECKERS = 5

    def __init__(self, width=1024, height=768):
        self.width = width
        self.height = height

        # Menu bar
        self.menu_height = 30
        self.menu_item_padding = 20       # horizontal padding added to text width per item
        self.menu_item_x = 10            # x offset of first menu item
        self.menu_item_y = 5             # y offset of item rect within menu bar
        self.menu_text_y = 8             # y position of text within menu bar

        # Submenu
        self.submenu_x = 10
        self.submenu_width = 200
        self.submenu_item_height = 25
        self.submenu_text_x_offset = 5   # x padding inside submenu items

        # Board
        self.board_margin = 50 + self.menu_height + 20
        self.board_width = width - 2 * self.board_margin
        self.board_height = height - 2 * self.board_margin

        # Points and checkers
        self.point_width = self.board_width // 14   # 12 points + bar + bear-off column
        self.point_height = self.board_height // 3
        self.checker_radius = max(self.point_width // 2 - 2, 2)

        # Fixed x positions
        self.bar_x = self.board_margin + 6 * self.point_width + self.point_width // 2
        self.bear_off_x = self.board_margin + 13 * self.point_width + self.point_width // 2

        # Turn indicator arrow
        self.arrow_size = 40
        self.arrow_x = width - self.board_margin // 3
        self.arrow_y_me = height - self.board_margin - 30
        self.arrow_y_opponent = self.board_margin + 30

        # Text area
        self.text_area_height = 30
        self.text_area_y = height - self.board_margin + 30
        self.text_padding = 10           # left padding inside text area

        # Point labels
        self.label_margin = 15           # offset from board edge to label center