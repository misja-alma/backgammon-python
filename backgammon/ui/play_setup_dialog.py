from typing import Optional

from backgammon.game.game import GameConfig, PlayerConfig, PlayerType
from backgammon.game.position import GameVariant, Player
from backgammon.ui.dialog import Dialog


class PlaySetupDialog(Dialog):
    """Choose the two players (human or an engine, with search depth), the variant (backgammon or
    nackgammon), double match point (no gammons) or not, and whether a game between two engines
    is displayed. run() returns a GameConfig,
    or None when cancelled."""

    MIN_DEPTH, MAX_DEPTH = 0, 4
    COLUMN_WIDTH = 210

    # Remembered between openings of the dialog.
    _last_config = GameConfig()

    def __init__(self, screen, board_map):
        super().__init__(screen, board_map, "Start game", 2 * self.COLUMN_WIDTH + 40, 390)
        last = PlaySetupDialog._last_config
        self.players = {Player.ME: PlayerConfig(last.me.type, last.me.depth),
                        Player.OPPONENT: PlayerConfig(last.opponent.type, last.opponent.depth)}
        self.show_engine_game = last.show_engine_game
        self.variant = last.variant
        self.dmp = last.dmp

        self.radios = {}    # (player, PlayerType) → rect
        self.steppers = {}  # player → (minus rect, plus rect)
        for column, player in enumerate((Player.ME, Player.OPPONENT)):
            x = 20 + column * self.COLUMN_WIDTH
            for row, player_type in enumerate(PlayerType):
                self.radios[(player, player_type)] = self.rect(x + 10, 85 + row * 28, 16, 16)
            self.steppers[player] = (self.rect(x + 70, 198, 28, 28), self.rect(x + 140, 198, 28, 28))
        self.variant_radios = {variant: self.rect(90 + i * self.COLUMN_WIDTH, 246, 16, 16)
                               for i, variant in enumerate(GameVariant)}
        self.checkbox_dmp = self.rect(30, 280, 16, 16)
        self.checkbox_show = self.rect(30, 308, 16, 16)

    def _all_engines(self) -> bool:
        return all(config.type.is_engine for config in self.players.values())

    def draw_content(self) -> None:
        for column, player in enumerate((Player.ME, Player.OPPONENT)):
            x = 20 + column * self.COLUMN_WIDTH
            config = self.players[player]
            self.draw_text(x, 58, "ME (white):" if player == Player.ME else "OPPONENT (red):")
            for player_type in PlayerType:
                self.draw_radio(self.radios[(player, player_type)], config.type == player_type, player_type.value)
            is_engine = config.type.is_engine
            self.draw_text(x + 10, 205, "Depth:", enabled=is_engine)
            minus, plus = self.steppers[player]
            self.draw_stepper(minus, plus, config.depth, enabled=is_engine)

        self.draw_text(30, 246, "Game:")
        for variant, rect in self.variant_radios.items():
            self.draw_radio(rect, self.variant == variant, variant.value)

        self.draw_checkbox(self.checkbox_dmp, self.dmp, "Double match point (no gammons)")
        self.draw_checkbox(self.checkbox_show, self.show_engine_game,
                           "Show game (engine vs engine)", enabled=self._all_engines())

    def handle_click(self, pos) -> None:
        for (player, player_type), rect in self.radios.items():
            if rect.collidepoint(pos):
                config = self.players[player]
                config.type = player_type
                if player_type.is_engine:
                    config.depth = player_type.engine.clamp_depth(config.depth)
                return
        for player, (minus, plus) in self.steppers.items():
            config = self.players[player]
            if not config.type.is_engine:
                continue
            if minus.collidepoint(pos):
                config.depth = max(self.MIN_DEPTH, config.depth - 1)
                return
            if plus.collidepoint(pos):
                config.depth = config.type.engine.clamp_depth(min(self.MAX_DEPTH, config.depth + 1))
                return
        for variant, rect in self.variant_radios.items():
            if rect.collidepoint(pos):
                self.variant = variant
                return
        if self.checkbox_dmp.collidepoint(pos):
            self.dmp = not self.dmp
            return
        if self.checkbox_show.collidepoint(pos) and self._all_engines():
            self.show_engine_game = not self.show_engine_game

    def result(self) -> Optional[GameConfig]:
        config = GameConfig(self.players[Player.ME], self.players[Player.OPPONENT],
                            show_engine_game=self.show_engine_game or not self._all_engines(),
                            variant=self.variant, dmp=self.dmp)
        PlaySetupDialog._last_config = config
        return config
