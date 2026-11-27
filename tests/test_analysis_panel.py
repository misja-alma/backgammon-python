import math
import os
import random
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame  # noqa: E402

from backgammon.game.analysis_worker import AnalysisWorker  # noqa: E402
from backgammon.game.game import GameConfig, PlayerConfig, PlayerType  # noqa: E402
from backgammon.game.move_analysis import analyze, format_probs  # noqa: E402
from backgammon.game.analyzer_proxy import AnalyzerProxy  # noqa: E402
from backgammon.game.position import Position, Player  # noqa: E402
from backgammon.ui.analysis_panel import AnalysisPanel  # noqa: E402
from backgammon.ui.board_map import BoardMap  # noqa: E402
from backgammon.ui.game_controller import GameController  # noqa: E402


def starting_position(turn=Player.ME) -> Position:
    position = Position()
    position.setup_starting_position()
    position.set_turn(turn)
    return position


class MoveAnalysisTest(unittest.TestCase):

    def setUp(self):
        AnalyzerProxy.set_rust_network("nets/td-eval.bgnet")

    def tearDown(self):
        AnalyzerProxy.set_rust_network(None)   # other tests expect the placeholder

    def test_moves_are_ranked_with_notation_and_win_chances(self):
        analysis = analyze(starting_position(), (3, 1), 1, money=False)
        self.assertEqual(16, len(analysis.moves))
        equities = [m.equity for m in analysis.moves]
        self.assertEqual(sorted(equities, reverse=True), equities)
        self.assertEqual("8/5 6/5", analysis.moves[0].notation)
        for m in analysis.moves:
            # Double match point net: equity = 2 * winning chance - 1, gammons unknown.
            self.assertAlmostEqual(m.equity, 2 * m.probs[0] - 1, places=6)
            self.assertTrue(all(math.isnan(p) for p in m.probs[1:]))
        self.assertTrue(analysis.matches(starting_position(), (3, 1)))
        self.assertFalse(analysis.matches(starting_position(), (3, 2)))

    def test_position_analysis(self):
        analysis = analyze(starting_position(), None, 0, money=False)
        self.assertEqual([], analysis.moves)
        self.assertAlmostEqual(analysis.equity, 2 * analysis.probs[0] - 1, places=6)

    def test_format_probs(self):
        self.assertEqual("50.4 16.3 1.8", format_probs([0.504, 0.163, 0.018]))
        self.assertEqual("52.5 - -", format_probs([0.525, math.nan, math.nan]))


class AnalysisPanelTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pygame.init()
        cls.screen = pygame.display.set_mode((1340, 768))
        AnalyzerProxy.set_rust_network("nets/td-eval.bgnet")

    @classmethod
    def tearDownClass(cls):
        AnalyzerProxy.set_rust_network(None)
        pygame.quit()

    def test_click_selects_previews_and_then_plays(self):
        board_map = BoardMap(1340, 768)
        panel = AnalysisPanel(self.screen, board_map)
        panel.set_analysis(analyze(starting_position(), (3, 1), 1, money=False))
        panel.draw()
        x = board_map.panel_width // 2
        first_row = board_map.panel_y + panel._header_height() + 5
        second_row = first_row + panel._row_height()
        self.assertEqual("select", panel.handle_click((x, second_row)))
        self.assertEqual(1, panel.selected)
        self.assertIs(panel.analysis.moves[1].position, panel.preview())
        self.assertEqual("play", panel.handle_click((x, second_row)))
        self.assertIsNone(panel.handle_click((board_map.panel_width + 50, second_row)))
        panel.clear()
        self.assertIsNone(panel.preview())

    def test_controller_plays_a_legal_move_on_a_human_turn(self):
        controller = GameController(AnalysisWorker())
        config = GameConfig(PlayerConfig(PlayerType.HUMAN, 0), PlayerConfig(PlayerType.HUMAN, 0))
        controller.start(config, rng=random.Random(1))
        game = controller.game
        mover = game.current_player
        move = sorted(game.legal_moves())[0]
        self.assertFalse(controller.play_move(((24, 1),)))   # not a legal move
        self.assertTrue(controller.play_move(move))
        self.assertNotEqual(mover, game.current_player)


if __name__ == '__main__':
    unittest.main()
