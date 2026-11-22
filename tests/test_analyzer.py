import unittest
from unittest.mock import patch

from backgammon.game.position import Position, Player
from backgammon.game.analyzer_proxy import AnalyzerProxy, PROJECT_ROOT
from backgammon.game.settings import Settings, Analyzer


class AnalyzerTest(unittest.TestCase):

    def setUp(self):
        Settings.analyzer = Analyzer.PYTHON
        Settings.analysis_depth = 3

    def test_equity(self):
        position = Position()
        position.set_turn(Player.ME)
        position.set_checkers(Player.ME, 1, 1)
        position.set_checkers(Player.ME, 0, 14)
        position.set_checkers(Player.OPPONENT, 0, 15)

        self.assertEqual(1.0, AnalyzerProxy.equity(position, Player.OPPONENT))
        self.assertEqual(-1.0, AnalyzerProxy.equity(position, Player.ME))

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
        self.assertEqual(-1.0, best_pw)

        best_pos, best_pw = AnalyzerProxy.best_move(position, 1, 1)
        self.assertEqual(1.0, best_pw)
        self.assertEqual(15, best_pos.get_checkers(Player.ME, 0))

    def test_proxy_calls_python_analyzer_equity(self):
        position = Position()
        position.set_turn(Player.ME)

        with patch('backgammon.game.analyzer_proxy.PythonAnalyzer.equity',
                   return_value=0.6) as mock:
            result = AnalyzerProxy.equity(position, Player.ME)

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

    def test_proxy_passes_cube_to_rust(self):
        Settings.analyzer = Analyzer.RUST
        position = Position()
        position.setup_starting_position()
        position.set_cube(4, Player.OPPONENT)

        with patch('backgammon.game.analyzer_proxy.backgammon_rust.equity',
                   return_value=0.4) as mock:
            result = AnalyzerProxy.equity(position, Player.ME)

        mock.assert_called_once_with(position.me_checkers, position.opponent_checkers,
                                     1, 4, 0, 1, Settings.analysis_depth)
        self.assertEqual(0.4, result)

    def test_rust_best_move_keeps_cube(self):
        Settings.analyzer = Analyzer.RUST
        position = Position()
        position.set_turn(Player.ME)
        position.set_checkers(Player.ME, 1, 1)
        position.set_checkers(Player.ME, 0, 14)
        position.set_checkers(Player.OPPONENT, 1, 1)
        position.set_checkers(Player.OPPONENT, 0, 14)
        position.set_cube(2, Player.ME)

        best_pos, best_pw = AnalyzerProxy.best_move(position, 2, 1)
        self.assertEqual(1.0, best_pw)
        self.assertEqual(15, best_pos.get_checkers(Player.ME, 0))
        self.assertEqual(2, best_pos.cube_value)
        self.assertEqual(Player.ME, best_pos.cube_owner)

    def test_rust_and_python_agree(self):
        position = Position()
        position.set_turn(Player.ME)
        position.set_checkers(Player.ME, 3, 1)
        position.set_checkers(Player.ME, 0, 14)
        position.set_checkers(Player.OPPONENT, 2, 1)
        position.set_checkers(Player.OPPONENT, 0, 14)
        position.set_cube(2, Player.OPPONENT)

        AnalyzerProxy.clear_caches()
        for depth in (0, 1, 2, 3):
            python_equity = AnalyzerProxy.equity(position, Player.ME, Analyzer.PYTHON, depth)
            rust_equity = AnalyzerProxy.equity(position, Player.ME, Analyzer.RUST, depth)
            self.assertAlmostEqual(python_equity, rust_equity)

    def test_explicit_engine_and_depth_override_settings(self):
        Settings.analyzer = Analyzer.RUST
        Settings.analysis_depth = 5
        position = Position()
        with patch('backgammon.game.analyzer_proxy.PythonAnalyzer.equity', return_value=0.1) as mock:
            AnalyzerProxy.equity(position, Player.ME, engine=Analyzer.PYTHON, depth=1)
        mock.assert_called_once_with(position, Player.ME, 1)

    def test_equity_depth_semantics(self):
        # ME has 2 checkers on the 2-point: 26/36 rolls win at once. OPPONENT then wins for sure.
        position = Position()
        position.set_turn(Player.ME)
        position.set_checkers(Player.ME, 2, 2)
        position.set_checkers(Player.ME, 0, 13)
        position.set_checkers(Player.OPPONENT, 1, 1)
        position.set_checkers(Player.OPPONENT, 0, 14)
        for engine in (Analyzer.PYTHON, Analyzer.RUST):
            AnalyzerProxy.clear_caches()
            self.assertEqual(0.0, AnalyzerProxy.equity(position, Player.ME, engine, 0))
            self.assertAlmostEqual(26 / 36, AnalyzerProxy.equity(position, Player.ME, engine, 1))
            self.assertAlmostEqual(16 / 36, AnalyzerProxy.equity(position, Player.ME, engine, 2))

    def test_rust_rejects_two_players_on_one_point(self):
        position = Position()
        position.set_checkers(Player.ME, 8, 3)
        position.set_checkers(Player.OPPONENT, 17, 2)  # the same point as ME's 8
        with self.assertRaises(ValueError):
            AnalyzerProxy.equity(position, Player.ME, Analyzer.RUST, 1)


class RustNetworkTest(unittest.TestCase):
    def tearDown(self):
        AnalyzerProxy.set_rust_network(None)

    def test_placeholder(self):
        self.assertIn("placeholder", AnalyzerProxy.set_rust_network(None))

    def test_missing_file(self):
        with self.assertRaises(ValueError):
            AnalyzerProxy.set_rust_network("nets/does-not-exist.bgnet")

    def test_bad_file(self):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".bgnet") as f:
            f.write(b"not a net\n")
            f.flush()
            with self.assertRaises(ValueError):
                AnalyzerProxy.set_rust_network(f.name)


class RustBearoffTest(unittest.TestCase):
    def tearDown(self):
        AnalyzerProxy.set_rust_bearoff(None)

    def test_none(self):
        self.assertIn("no bearoff", AnalyzerProxy.set_rust_bearoff(None))

    def test_missing_file(self):
        with self.assertRaises(ValueError):
            AnalyzerProxy.set_rust_bearoff("data/does-not-exist.bd")

    @unittest.skipUnless((PROJECT_ROOT / Settings.rust_bearoff_db).exists(), "bearoff database not built")
    def test_exact_bearoff_value(self):
        self.assertIn("6x15", AnalyzerProxy.set_rust_bearoff(Settings.rust_bearoff_db))
        # One checker each on the 6-point and the 1-point: the mover wins with 27 of 36 rolls.
        position = Position()
        position.set_checkers(Player.ME, 6, 1)
        position.set_checkers(Player.ME, 0, 14)
        position.set_checkers(Player.OPPONENT, 1, 1)
        position.set_checkers(Player.OPPONENT, 0, 14)
        position.set_turn(Player.ME)
        for depth in (0, 1, 2):
            equity = AnalyzerProxy.equity(position, Player.ME, Analyzer.RUST, depth)
            self.assertAlmostEqual(2 * 27 / 36 - 1, equity, places=4)


if __name__ == '__main__':
    unittest.main()
