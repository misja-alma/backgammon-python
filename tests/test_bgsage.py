import unittest

from backgammon.game.analyzer_proxy import AnalyzerProxy
from backgammon.game.bgsage_analyzer import BgsageAnalyzer
from backgammon.game.game import Game, GameConfig, PlayerConfig, PlayerType
from backgammon.game.position import Position, Player
from backgammon.game.settings import Analyzer


def sample_position() -> Position:
    """An asymmetric position with checkers on both bars and borne off."""
    position = Position()
    position.clear()
    for player, point, count in [(Player.ME, 25, 1), (Player.ME, 20, 3), (Player.ME, 6, 4), (Player.ME, 1, 2),
                                 (Player.OPPONENT, 25, 2), (Player.OPPONENT, 13, 5), (Player.OPPONENT, 3, 3)]:
        position.set_checkers(player, point, count)
        position.set_checkers(player, 0, position.get_checkers(player, 0) - count)
    return position


class BoardConversionTest(unittest.TestCase):
    def test_starting_position_matches_bgsage(self):
        position = Position()
        position.setup_starting_position()
        bgsage_start = [0, -2, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, -5, 5, 0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 2, 0]
        self.assertEqual(bgsage_start, BgsageAnalyzer.to_board(position, Player.ME))
        self.assertEqual(bgsage_start, BgsageAnalyzer.to_board(position, Player.OPPONENT))

    def test_round_trip(self):
        position = sample_position()
        for player in Player:
            board = BgsageAnalyzer.to_board(position, player)
            back = BgsageAnalyzer.from_board(board, player, Player.ME)
            self.assertEqual(position.me_checkers, back.me_checkers)
            self.assertEqual(position.opponent_checkers, back.opponent_checkers)

    def test_bars(self):
        board = BgsageAnalyzer.to_board(sample_position(), Player.ME)
        self.assertEqual(1, board[25])  # own bar
        self.assertEqual(2, board[0])   # other side's bar

    def test_depth_out_of_range(self):
        position = Position()
        position.setup_starting_position()
        with self.assertRaises(ValueError):
            BgsageAnalyzer.equity(position, Player.ME, BgsageAnalyzer.MAX_DEPTH + 1)

    def test_clamp_depth(self):
        self.assertEqual(BgsageAnalyzer.MAX_DEPTH, Analyzer.BGSAGE.clamp_depth(10))
        self.assertEqual(10, Analyzer.RUST.clamp_depth(10))


@unittest.skipUnless(BgsageAnalyzer.is_available(), "bgsage is not installed (make bgsage)")
class BgsageEngineTest(unittest.TestCase):
    def test_equity_is_zero_sum_and_favours_side_on_roll(self):
        position = Position()
        position.setup_starting_position()
        for depth in range(BgsageAnalyzer.MAX_DEPTH):
            me = AnalyzerProxy.equity(position, Player.ME, engine=Analyzer.BGSAGE, depth=depth)
            opponent = AnalyzerProxy.equity(position, Player.OPPONENT, engine=Analyzer.BGSAGE, depth=depth)
            self.assertAlmostEqual(me, -opponent)
            self.assertTrue(0 < me < 0.2, me)   # being on roll at the start is worth a few percent

    def test_equity_is_symmetric_between_players(self):
        position = Position()
        position.setup_starting_position()
        position.set_checkers(Player.ME, 6, 0)          # ME is missing 5 checkers: a clear underdog
        position.set_checkers(Player.ME, 0, 5)
        position.set_checkers(Player.ME, 24, 7)
        mirrored = Position()
        mirrored.me_checkers = list(position.opponent_checkers)
        mirrored.opponent_checkers = list(position.me_checkers)
        mirrored.set_turn(Player.OPPONENT)
        self.assertAlmostEqual(AnalyzerProxy.equity(position, Player.ME, engine=Analyzer.BGSAGE, depth=0),
                               AnalyzerProxy.equity(mirrored, Player.OPPONENT, engine=Analyzer.BGSAGE, depth=0))

    def test_game_over(self):
        position = Position()
        position.clear()
        position.set_checkers(Player.ME, 1, 1)
        position.set_checkers(Player.ME, 0, 14)
        self.assertEqual(-1.0, AnalyzerProxy.equity(position, Player.ME, engine=Analyzer.BGSAGE, depth=0))

    def test_best_move_is_legal(self):
        for player in Player:
            position = sample_position()
            position.set_cube(2, Player.ME)
            position.set_turn(player)
            new_pos, equity = AnalyzerProxy.best_move(position, 6, 3, engine=Analyzer.BGSAGE, depth=1)
            self.assertEqual(player.other_player(), new_pos.get_turn())
            self.assertEqual((2, Player.ME), (new_pos.cube_value, new_pos.cube_owner))
            self.assertTrue(-1 <= equity <= 1)
            game = Game(GameConfig())
            game.position = position
            game.dice = (6, 3)
            game.find_move(new_pos)   # raises when the move is not legal

    def test_last_checker_off_with_one_die(self):
        # bgsage bears off 2/off with the 4; our move generator only knows 2/1*/off. Both win.
        position = Position()
        position.clear()
        position.set_checkers(Player.ME, 0, 0)
        position.set_checkers(Player.ME, 1, 13)
        position.set_checkers(Player.ME, 9, 1)
        position.set_checkers(Player.ME, 24, 1)
        position.set_checkers(Player.OPPONENT, 0, 14)
        position.set_checkers(Player.OPPONENT, 2, 1)
        position.set_turn(Player.OPPONENT)
        config = GameConfig(PlayerConfig(PlayerType.HUMAN), PlayerConfig(PlayerType.BGSAGE_ENGINE, 0))
        game = Game(config)
        game.position = position
        game.dice = (1, 4)
        game.play(game.engine_move())
        self.assertEqual(Player.OPPONENT, game.winner)

    def test_engine_game(self):
        config = GameConfig(PlayerConfig(PlayerType.BGSAGE_ENGINE, 0), PlayerConfig(PlayerType.RUST_ENGINE, 0))
        game = Game(config)
        game.start()
        game.play_to_end()
        self.assertTrue(game.is_over)


if __name__ == '__main__':
    unittest.main()
