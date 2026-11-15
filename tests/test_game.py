import random
import unittest

from backgammon.game.game import (Game, GameConfig, PlayerConfig, PlayerType, MoveBuilder,
                                  format_move)
from backgammon.game.position import Position, Player


class ScriptedDice:
    """Returns the scripted values from randint, then falls back to a seeded generator."""

    def __init__(self, values):
        self.values = list(values)
        self.fallback = random.Random(0)

    def randint(self, a, b):
        return self.values.pop(0) if self.values else self.fallback.randint(a, b)


def engines(engine=PlayerType.RUST_ENGINE, depth=0):
    return GameConfig(PlayerConfig(engine, depth), PlayerConfig(engine, depth), show_engine_game=False)


def humans():
    return GameConfig(PlayerConfig(PlayerType.HUMAN), PlayerConfig(PlayerType.HUMAN))


class OpeningRollTest(unittest.TestCase):

    def test_higher_die_starts_and_plays_both_dice(self):
        game = Game(humans(), rng=ScriptedDice([2, 5]))
        game.start()
        self.assertEqual(Player.OPPONENT, game.current_player)
        self.assertEqual((5, 2), game.dice)

    def test_ties_are_rerolled(self):
        messages = []
        game = Game(humans(), rng=ScriptedDice([3, 3, 6, 6, 4, 1]), log=messages.append)
        game.start()
        self.assertEqual(Player.ME, game.current_player)
        self.assertEqual((4, 1), game.dice)
        self.assertEqual(3, len(messages))
        self.assertIn("ME starts with 4-1", messages[-1])


class PlayTest(unittest.TestCase):

    def test_legal_move_switches_turn_and_rolls(self):
        game = Game(humans(), rng=ScriptedDice([3, 1, 6, 5]))
        game.start()
        game.play(((8, 5), (6, 5)))
        self.assertEqual(Player.OPPONENT, game.current_player)
        self.assertEqual(2, game.position.get_checkers(Player.ME, 5))
        self.assertEqual((6, 5), game.dice)

    def test_illegal_move_is_rejected(self):
        game = Game(humans(), rng=ScriptedDice([3, 1]))
        game.start()
        with self.assertRaises(ValueError):
            game.play(((13, 7),))
        with self.assertRaises(ValueError):
            game.play(())  # passing is only allowed without legal moves

    def test_pass_when_no_legal_move(self):
        game = Game(humans(), rng=ScriptedDice([6, 5, 2, 3]))
        game.start()  # ME to move with 6-5
        position = Position()
        position.set_checkers(Player.ME, 25, 1)
        position.set_checkers(Player.ME, 0, 14)
        for point in range(1, 7):  # OPPONENT holds its whole home board
            position.set_checkers(Player.OPPONENT, point, 2)
        position.set_checkers(Player.OPPONENT, 0, 3)
        game.position = position

        self.assertEqual(set(), game.legal_moves())
        game.play(())
        self.assertEqual(Player.OPPONENT, game.current_player)
        self.assertEqual(1, game.position.get_checkers(Player.ME, 25))

    def test_winner_is_reported(self):
        messages = []
        game = Game(humans(), rng=ScriptedDice([2, 1]), log=messages.append)
        game.start()  # ME to move with 2-1
        position = Position()
        position.set_checkers(Player.ME, 1, 1)
        position.set_checkers(Player.ME, 0, 14)
        position.set_checkers(Player.OPPONENT, 6, 15)
        game.position = position

        game.play(((1, 0),))
        self.assertTrue(game.is_over)
        self.assertEqual(Player.ME, game.winner)
        self.assertEqual("ME wins", messages[-1])
        with self.assertRaises(ValueError):
            game.play(())

    def test_engine_move_is_legal(self):
        for engine in (PlayerType.PYTHON_ENGINE, PlayerType.RUST_ENGINE):
            with self.subTest(engine=engine):
                game = Game(engines(engine), rng=ScriptedDice([3, 1]))
                game.start()
                move = game.engine_move()
                self.assertIn(move, game.legal_moves())


class FullGameTest(unittest.TestCase):

    def test_engine_game_finishes_with_a_winner(self):
        for seed in range(3):
            with self.subTest(seed=seed):
                game = Game(engines(depth=0), rng=random.Random(seed))
                game.start()
                game.play_to_end()
                self.assertTrue(game.is_over)
                winner, loser = game.winner, game.winner.other_player()
                self.assertEqual(15, game.position.get_checkers(winner, 0))
                self.assertLess(game.position.get_checkers(loser, 0), 15)
                # Checkers are conserved for both players throughout.
                for player in (Player.ME, Player.OPPONENT):
                    self.assertEqual(15, sum(game.position.get_checkers(player, p) for p in range(26)))

    def test_play_to_end_can_be_cancelled(self):
        game = Game(engines(), rng=random.Random(1))
        game.start()
        game.play_to_end(is_cancelled=lambda: True)
        self.assertFalse(game.is_over)


class RustGameTest(unittest.TestCase):

    def test_runs_in_rust_only_for_hidden_rust_vs_rust(self):
        self.assertTrue(Game(engines(PlayerType.RUST_ENGINE)).runs_in_rust)
        self.assertFalse(Game(engines(PlayerType.PYTHON_ENGINE)).runs_in_rust)
        shown = engines(PlayerType.RUST_ENGINE)
        shown.show_engine_game = True
        self.assertFalse(Game(shown).runs_in_rust)
        with self.assertRaises(ValueError):
            Game(humans()).play_in_rust(1)

    def test_play_in_rust_finishes_and_is_reproducible(self):
        results = []
        for _ in range(2):
            game = Game(engines(PlayerType.RUST_ENGINE, depth=0))
            game.play_in_rust(seed=1234)
            self.assertTrue(game.is_over)
            self.assertEqual(15, game.position.get_checkers(game.winner, 0))
            for player in (Player.ME, Player.OPPONENT):
                self.assertEqual(15, sum(game.position.get_checkers(player, p) for p in range(26)))
            results.append((game.winner, game.turns, game.position.me_checkers, game.position.opponent_checkers))
        self.assertEqual(results[0], results[1])


class MoveBuilderTest(unittest.TestCase):

    def setUp(self):
        position = Position()
        position.setup_starting_position()
        from backgammon.game.utils import Utils
        self.legal = Utils.valid_possible_moves(position, 3, 1)

    def test_single_steps_complete_a_move(self):
        builder = MoveBuilder(self.legal)
        self.assertTrue(builder.try_move(8, 5))
        self.assertFalse(builder.is_complete)
        self.assertTrue(builder.try_move(6, 5))
        self.assertTrue(builder.is_complete)
        self.assertIn(builder.move, self.legal)

    def test_invalid_step_is_refused(self):
        builder = MoveBuilder(self.legal)
        self.assertFalse(builder.try_move(8, 3))
        self.assertFalse(builder.try_move(24, 18))
        self.assertEqual((), builder.move)

    def test_one_checker_with_both_dice(self):
        builder = MoveBuilder(self.legal)
        self.assertTrue(builder.try_move(24, 20))
        self.assertTrue(builder.is_complete)
        self.assertEqual(2, len(builder.move))

    def test_undo(self):
        builder = MoveBuilder(self.legal)
        builder.try_move(8, 5)
        builder.undo()
        self.assertEqual((), builder.move)
        self.assertTrue(builder.can_move_from(24))
        self.assertFalse(builder.can_move_from(3))


class FormatMoveTest(unittest.TestCase):

    def test_format(self):
        self.assertEqual("8/5 6/5", format_move(((8, 5), (6, 5))))
        self.assertEqual("bar/22 3/off", format_move(((25, 22), (3, 0))))
        self.assertEqual("no legal move", format_move(()))


if __name__ == '__main__':
    unittest.main()
