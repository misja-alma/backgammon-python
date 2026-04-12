import unittest
from backgammon_position import Position, Player
from analyzer import Analyzer


class AnalyzerTest(unittest.TestCase):

    def test_winning_chances(self):
        position = Position()
        position.set_turn(Player.ME)
        position.set_checkers(Player.ME, 1, 1)
        position.set_checkers(Player.ME, 0, 14)
        position.set_checkers(Player.OPPONENT, 0, 15)

        self.assertEqual(1.0, Analyzer.winning_chances(position, Player.OPPONENT))
        self.assertEqual(0.0, Analyzer.winning_chances(position, Player.ME))

    def test_best_move(self):
        position = Position()
        position.set_turn(Player.ME)
        position.set_checkers(Player.ME, 1, 1)
        position.set_checkers(Player.ME, 0, 14)
        position.set_checkers(Player.OPPONENT, 1, 1)
        position.set_checkers(Player.OPPONENT, 0, 14)

        best_pos, best_pw = Analyzer.best_move(position, 2, 1)
        self.assertEqual(1.0, best_pw)
        self.assertEqual(15, best_pos.get_checkers(Player.ME, 0))

        position.set_checkers(Player.ME, 2, 1)
        position.set_checkers(Player.ME, 0, 13)
        best_pos, best_pw = Analyzer.best_move(position, 2, 1)
        self.assertEqual(1.0, best_pw)
        self.assertEqual(15, best_pos.get_checkers(Player.ME, 0))

        position.set_checkers(Player.ME, 3, 1)
        position.set_checkers(Player.ME, 2, 0)
        best_pos, best_pw = Analyzer.best_move(position, 2, 1)
        self.assertEqual(0.0, best_pw)

        best_pos, best_pw = Analyzer.best_move(position, 1, 1)
        self.assertEqual(1.0, best_pw)
        self.assertEqual(15, best_pos.get_checkers(Player.ME, 0))


if __name__ == '__main__':
    unittest.main()