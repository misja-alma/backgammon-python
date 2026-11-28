from backgammon.game.position_formats import Format
from backgammon.ui.dialog import Dialog


class SavePositionDialog(Dialog):
    """The format to save a position in. run() returns a Format, or None when cancelled."""

    DESCRIPTIONS = {
        Format.XGID: "XGID (eXtreme Gammon, most programs)",
        Format.GNUBG_ID: "GNUbg Position ID and Match ID",
        Format.SNOWIE: "Snowie text (.txt)",
        Format.SGF: "GNU Backgammon (.sgf)",
        Format.XG: "eXtreme Gammon (.xgp)",
    }

    _last_format = Format.XGID   # remembered between openings

    def __init__(self, screen, board_map):
        super().__init__(screen, board_map, "Save position", 360, 90 + 30 * len(Format) + 60)
        self.format = SavePositionDialog._last_format
        self.radios = {fmt: self.rect(30, 80 + 30 * i, 16, 16) for i, fmt in enumerate(Format)}

    def draw_content(self) -> None:
        self.draw_text(20, 52, "Format:")
        for fmt, rect in self.radios.items():
            self.draw_radio(rect, self.format == fmt, self.DESCRIPTIONS[fmt])

    def handle_click(self, pos) -> None:
        for fmt, rect in self.radios.items():
            if rect.collidepoint(pos):
                self.format = fmt

    def result(self):
        SavePositionDialog._last_format = self.format
        return self.format
