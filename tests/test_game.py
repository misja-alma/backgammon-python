import random
import unittest

from backgammon.game.game import (Game, GameConfig, PlayerConfig, PlayerType, MoveBuilder, WinType,
                                  format_move)
from backgammon.game.position import GameVariant, Position, Player


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


class GammonTest(unittest.TestCase):
    """Win types and points: a gammon when the loser has borne off nothing, a backgammon when the
    loser also still has a checker on the bar or in the winner's home board."""

    @staticmethod
    def final_position(opponent: dict) -> Position:
        """ME has borne off everything; OPPONENT's checkers (own numbering) as given."""
        position = Position()
        position.set_checkers(Player.ME, 0, 15)
        for point, count in opponent.items():
            position.set_checkers(Player.OPPONENT, point, count)
        return position

    def test_win_types(self):
        cases = [({0: 1, 6: 14}, WinType.SINGLE),
                 ({0: 1, 6: 13, 25: 1}, WinType.SINGLE),        # borne off one: never more than single
                 ({6: 15}, WinType.GAMMON),
                 ({18: 1, 6: 14}, WinType.GAMMON),               # 18 is just outside ME's home board
                 ({19: 1, 6: 14}, WinType.BACKGAMMON),           # ME's 6 point
                 ({24: 1, 6: 14}, WinType.BACKGAMMON),           # ME's 1 point
                 ({25: 1, 6: 14}, WinType.BACKGAMMON)]           # on the bar
        for opponent, expected in cases:
            self.assertEqual(expected, WinType.of(self.final_position(opponent), Player.ME), opponent)

    def play_last_move(self, dmp: bool, opponent: dict):
        """ME bears off the last checker (2-1, 1/off) against OPPONENT's checkers as given."""
        messages = []
        config = humans()
        config.dmp = dmp
        game = Game(config, rng=ScriptedDice([2, 1]), log=messages.append)
        game.start()
        position = self.final_position(opponent)
        position.set_checkers(Player.ME, 0, 14)
        position.set_checkers(Player.ME, 1, 1)
        game.position = position
        game.play(((1, 0),))
        return game, messages[-1]

    def test_points_without_dmp(self):
        for opponent, points, text in [({0: 3, 6: 12}, 1, "ME wins (1 point)"),
                                       ({6: 15}, 2, "ME wins a gammon (2 points)"),
                                       ({20: 2, 6: 13}, 3, "ME wins a backgammon (3 points)")]:
            game, message = self.play_last_move(False, opponent)
            self.assertEqual(points, game.points)
            self.assertEqual(text, message)
            self.assertEqual(text, game.result_text())

    def test_dmp_counts_every_win_as_one_point(self):
        game, message = self.play_last_move(True, {20: 2, 6: 13})
        self.assertEqual(WinType.BACKGAMMON, game.win_type)   # still known
        self.assertEqual(1, game.points)
        self.assertEqual("ME wins", message)

    def test_dmp_is_the_default(self):
        self.assertTrue(GameConfig().dmp)
        self.assertIsNone(Game(humans()).points)

    def test_game_in_rust_gets_a_win_type(self):
        config = engines()
        config.dmp = False
        game = Game(config, rng=random.Random(3))
        game.play_in_rust(seed=11)
        self.assertEqual(WinType.of(game.position, game.winner), game.win_type)
        self.assertEqual(game.win_type.value, game.points)


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


class NackgammonTest(unittest.TestCase):

    def test_starting_position(self):
        position = Position()
        position.setup_starting_position(GameVariant.NACKGAMMON)
        expected = [0] * 26
        for point, count in {24: 2, 23: 2, 13: 4, 8: 3, 6: 4}.items():
            expected[point] = count
        self.assertEqual(expected, position.me_checkers)
        self.assertEqual(expected, position.opponent_checkers)
        self.assertEqual(15, sum(position.me_checkers))
        self.assertEqual(194, position.pipcount(Player.ME))  # 167 - 13 - 6 + 2 * 23

    def test_game_uses_the_configured_variant(self):
        config = humans()
        self.assertEqual(0, Game(config).position.me_checkers[23])
        config.variant = GameVariant.NACKGAMMON
        game = Game(config)
        self.assertEqual(2, game.position.me_checkers[23])
        self.assertEqual(2, game.position.opponent_checkers[23])

    def test_nackgammon_game_in_rust(self):
        config = engines(PlayerType.RUST_ENGINE, depth=0)
        config.variant = GameVariant.NACKGAMMON
        results = []
        for _ in range(2):
            game = Game(config)
            game.play_in_rust(seed=99)
            self.assertTrue(game.is_over)
            results.append((game.winner, game.turns, game.position.me_checkers))
        self.assertEqual(results[0], results[1])
        backgammon = Game(engines(PlayerType.RUST_ENGINE, depth=0))
        backgammon.play_in_rust(seed=99)
        self.assertNotEqual(results[0], (backgammon.winner, backgammon.turns, backgammon.position.me_checkers))


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


class MoveBuilderStepTest(unittest.TestCase):
    """Steps by one die (the UI's way of entering moves)."""

    @staticmethod
    def builder(position, die1, die2):
        from backgammon.game.utils import Utils
        return MoveBuilder(Utils.valid_possible_moves(position, die1, die2), (die1, die2))

    def test_steps_by_die(self):
        position = Position()
        position.setup_starting_position()
        builder = self.builder(position, 3, 1)
        self.assertEqual([3, 1], builder.remaining_dice())
        self.assertFalse(builder.try_step(8, 2))     # not one of the dice
        self.assertTrue(builder.try_step(8, 3))
        self.assertEqual([1], builder.remaining_dice())
        self.assertFalse(builder.try_step(8, 3))     # the 3 is used
        self.assertTrue(builder.try_step(6, 1))
        self.assertTrue(builder.is_complete)
        self.assertEqual(((8, 5), (6, 5)), builder.move)
        builder.undo()
        self.assertEqual([1], builder.remaining_dice())
        builder.reset()
        self.assertEqual((), builder.move)

    def test_doubles_take_four_steps(self):
        position = Position()
        position.setup_starting_position()
        builder = self.builder(position, 2, 2)
        for start in (13, 13, 6, 6):
            self.assertTrue(builder.try_step(start, 2))
        self.assertTrue(builder.is_complete)
        self.assertEqual([], builder.remaining_dice())

    def test_bearing_off_with_a_higher_die(self):
        position = Position()
        position.set_checkers(Player.ME, 3, 1)
        position.set_checkers(Player.ME, 2, 1)
        position.set_checkers(Player.ME, 0, 13)
        position.set_checkers(Player.OPPONENT, 1, 15)   # far away: nothing is blocked
        position.set_turn(Player.ME)
        builder = self.builder(position, 6, 1)
        # The 6 bears off the checker on the 3; then the 1 has to play the other checker.
        self.assertTrue(builder.try_step(3, 6))
        self.assertEqual([1], builder.remaining_dice())
        self.assertFalse(builder.try_step(3, 1))
        self.assertTrue(builder.try_step(2, 1))
        self.assertTrue(builder.is_complete)


class FormatMoveTest(unittest.TestCase):

    def test_format(self):
        self.assertEqual("8/5 6/5", format_move(((8, 5), (6, 5))))
        self.assertEqual("bar/22 3/off", format_move(((25, 22), (3, 0))))
        self.assertEqual("no legal move", format_move(()))

    @staticmethod
    def position(me: dict, opponent: dict, turn=Player.ME) -> Position:
        p = Position()
        for point, n in me.items():
            p.set_checkers(Player.ME, point, n)
        for point, n in opponent.items():
            p.set_checkers(Player.OPPONENT, point, n)
        p.set_turn(turn)
        return p

    def test_hits_are_marked(self):
        # OPPONENT's blot on its 21 point is on ME's 4 point; two of its checkers on its 20 (ME's 5).
        before = self.position({18: 1, 8: 2, 6: 2}, {21: 1, 20: 2})
        self.assertEqual("18/13 8/4*", format_move(((18, 13), (8, 4)), before))
        self.assertEqual("6/1 8/5", format_move(((6, 1), (8, 5)), before))     # 2 checkers: no hit
        # Listed from the highest point down, the marks stay with their half-moves.
        self.assertEqual("18/13 8/4*", format_move(((8, 4), (18, 13)), before, highest_first=True))

    def test_hit_on_the_way_and_from_the_bar(self):
        # A blot on ME's 8 point (OPPONENT's 17): 13/8* then on with the same checker; then from the bar.
        before = self.position({13: 1, 25: 1}, {17: 1, 3: 1})
        self.assertEqual("13/8* 8/3", format_move(((13, 8), (8, 3)), before))
        self.assertEqual("bar/22* 13/8*", format_move(((25, 22), (13, 8)), before))   # OPPONENT's 3 is ME's 22

    def test_the_opponent_hits_too(self):
        before = self.position({4: 1}, {13: 2}, turn=Player.OPPONENT)          # ME's 4 is OPPONENT's 21
        self.assertEqual("13/8 13/9 9/4 8/4", format_move(((13, 8), (13, 9), (9, 4), (8, 4)), before))
        self.assertEqual("13/8 8/4", format_move(((13, 8), (8, 4)), before))
        self.assertEqual("13/9 9/4*", format_move(((13, 9), (9, 4)), self.position({21: 1}, {13: 2}, Player.OPPONENT)))


if __name__ == '__main__':
    unittest.main()


class PipcountTest(unittest.TestCase):
    def test_starting_position(self):
        position = Position()
        position.setup_starting_position()
        self.assertEqual(position.pipcount(Player.ME), 167)
        self.assertEqual(position.pipcount(Player.OPPONENT), 167)

    def test_bar_counts_25_and_off_counts_0(self):
        position = Position()
        position.clear()
        position.set_checkers(Player.ME, 25, 1)
        position.set_checkers(Player.ME, 3, 2)
        self.assertEqual(position.pipcount(Player.ME), 31)
        self.assertEqual(position.pipcount(Player.OPPONENT), 0)
