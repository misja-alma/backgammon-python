import unittest
from unittest.mock import patch

from backgammon.game.position import Position, Player
from backgammon.game.analyzer_proxy import AnalyzerProxy
from backgammon.game.settings import Settings, Analyzer


class AnalyzerTest(unittest.TestCase):

    def setUp(self):
        Settings.analyzer = Analyzer.PYTHON
        Settings.analysis_depth = 3

    def test_winning_chances(self):
        position = Position()
        position.set_turn(Player.ME)
        position.set_checkers(Player.ME, 1, 1)
        position.set_checkers(Player.ME, 0, 14)
        position.set_checkers(Player.OPPONENT, 0, 15)

        self.assertEqual(1.0, AnalyzerProxy.winning_chances(position, Player.OPPONENT))
        self.assertEqual(0.0, AnalyzerProxy.winning_chances(position, Player.ME))

    def test_best_move(self):
        position = Position()
        position.set_turn(Player.ME)
        position.set_checkers(Player.ME, 1, 1)
        position.set_checkers(Player.ME, 0, 14)
        position.set_checkers(Player.OPPONENT, 1, 1)
        position.set_checkers(Player.OPPONENT, 0, 14)

        best_pos, best_pw = AnalyzerProxy.best_move(position, 2, 1)
        self.assertEqual(1.0, best_pw)
        self.assertEqual(15, best_pos.get_checkers(Player.ME, 0))

        position.set_checkers(Player.ME, 2, 1)
        position.set_checkers(Player.ME, 0, 13)
        best_pos, best_pw = AnalyzerProxy.best_move(position, 2, 1)
        self.assertEqual(1.0, best_pw)
        self.assertEqual(15, best_pos.get_checkers(Player.ME, 0))

        position.set_checkers(Player.ME, 3, 1)
        position.set_checkers(Player.ME, 2, 0)
        best_pos, best_pw = AnalyzerProxy.best_move(position, 2, 1)
        self.assertEqual(0.0, best_pw)

        best_pos, best_pw = AnalyzerProxy.best_move(position, 1, 1)
        self.assertEqual(1.0, best_pw)
        self.assertEqual(15, best_pos.get_checkers(Player.ME, 0))

    def test_proxy_calls_python_analyzer_winning_chances(self):
        position = Position()
        position.set_turn(Player.ME)

        with patch('backgammon.game.analyzer_proxy.PythonAnalyzer.winning_chances',
                   return_value=0.6) as mock:
            result = AnalyzerProxy.winning_chances(position, Player.ME)

        mock.assert_called_once_with(position, Player.ME, Settings.analysis_depth)
        self.assertEqual(0.6, result)

    def test_proxy_calls_python_analyzer_best_move(self):
        position = Position()
        position.set_turn(Player.ME)
        fake_result = (Position(), 0.7)

        with patch('backgammon.game.analyzer_proxy.PythonAnalyzer.best_move',
                   return_value=fake_result) as mock:
            result = AnalyzerProxy.best_move(position, 3, 2)

        mock.assert_called_once_with(position, 3, 2, Settings.analysis_depth)
        self.assertEqual(fake_result, result)


if __name__ == '__main__':
    unittest.main()
