import unittest
from backgammon_position import Position, Player
from utils import Utils


class UtilsTest(unittest.TestCase):
    
    def test_pipcount_initial_position(self):
        """Test pipcount calculation for both players in starting position."""
        position = Position()
        position.setup_starting_position()
        
        expected_pipcount = 167
        
        white_pipcount = Utils.pipcount(position, Player.OPPONENT)
        black_pipcount = Utils.pipcount(position, Player.ME)
        
        self.assertEqual(white_pipcount, expected_pipcount)
        self.assertEqual(black_pipcount, expected_pipcount)
    
    def test_can_move_basic(self):
        """Test basic move validation."""
        position = Position()
        position.setup_starting_position()
        position.set_turn(Player.OPPONENT)  # Set turn to white
        
        # Valid move: from point 24 to point 20 (white)
        self.assertTrue(Utils.can_move(position, 24, 20))
        
        # Invalid move: to opponent's made point (point 19 has 5 black checkers)
        self.assertFalse(Utils.can_move(position, 24, 19))
    
    def test_can_move_from_bar(self):
        """Test move validation when player has checkers on bar."""
        position = Position()
        position.set_checkers(Player.OPPONENT, 25, 1)  # OPPONENT checker on bar
        position.set_checkers(Player.OPPONENT, 24, 1)  # OPPONENT checker on point 24
        position.set_turn(Player.OPPONENT)  # Set turn to white
        
        # Cannot move from anywhere except bar when on bar
        self.assertFalse(Utils.can_move(position, 24, 20))
        
        # Can move from bar to open point
        self.assertTrue(Utils.can_move(position, 25, 20))
    
    def test_possible_moves_simple(self):
        """Test possible moves with a simple position."""
        position = Position()
        position.set_checkers(Player.OPPONENT, 10, 1)  # Single checker on point 10
        position.set_turn(Player.OPPONENT)
        
        moves = Utils.possible_moves(position, [3, 2])
        
        # Should have moves: (10,7) then (7,5), or (10,8) then (8,5)
        expected_moves = {
            ((10, 7), (7, 5)),
            ((10, 8), (8, 5))
        }
        self.assertEqual(moves, expected_moves)
    
    def test_possible_moves_no_moves(self):
        """Test possible moves when no moves are possible."""
        position = Position()
        position.set_checkers(Player.OPPONENT, 10, 1)
        position.set_checkers(Player.ME, 18, 2)  # Block target
        position.set_checkers(Player.ME, 17, 2)  # Block other target
        position.set_turn(Player.OPPONENT)  # Set turn to white
        
        moves = Utils.possible_moves(position, [3, 2])
        
        # No moves should be possible
        self.assertEqual(moves, set())
    
    def test_possible_moves_single_die(self):
        """Test possible moves with single die."""
        position = Position()
        position.set_checkers(Player.OPPONENT, 5, 1)
        position.set_turn(Player.OPPONENT)
        
        moves = Utils.possible_moves(position, [3])
        
        expected_moves = {
            ((5, 2),)
        }
        self.assertEqual(moves, expected_moves)

        # verify that when there is no other choice, it is allowed to use a 6 to bear off from the 5 point
        moves = Utils.possible_moves(position, [6])

        expected_moves = {
            ((5, 0),)
        }
        self.assertEqual(moves, expected_moves)

    def test_valid_possible_moves(self):
        position = Position()
        position.set_checkers(Player.OPPONENT, 6, 1)
        position.set_checkers(Player.ME, 24, 2)  # Block target
        position.set_turn(Player.OPPONENT)

        moves = Utils.valid_possible_moves(position, 3, 2)

        # Only one die could be played, so it should be the highest one
        expected_moves = {
            ((6, 3),)
        }
        self.assertEqual(moves, expected_moves)

    def test_valid_possible_moves_order(self):
        position = Position()
        position.set_checkers(Player.OPPONENT, 6, 1)
        position.set_checkers(Player.ME, 6, 1)
        position.set_turn(Player.ME)

        moves = Utils.valid_possible_moves(position, 4, 2)

        # There are 2 possible moves which are basically the same, but what matters is that they are listed in order of playing the dice
        expected_moves = {
            ((6, 4), (4, 0)),
            ((6, 2), (2, 0)),
        }
        self.assertEqual(moves, expected_moves)


        position.set_checkers(Player.OPPONENT, 6, 0)
        position.set_checkers(Player.ME, 6, 0)
        position.set_checkers(Player.ME, 1, 1)
        position.set_checkers(Player.ME, 3, 1)
        position.set_turn(Player.ME)

        moves = Utils.valid_possible_moves(position, 1, 1)

        expected_moves = {
            ((3, 2), (2, 1), (1, 0), (1, 0)),
            ((3, 2), (1, 0), (2, 1), (1, 0)),
            ((1, 0), (3, 2), (2, 1), (1, 0))
        }
        self.assertEqual(moves, expected_moves)

if __name__ == '__main__':
    unittest.main()