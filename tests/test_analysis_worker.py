import threading
import unittest
from unittest.mock import patch

from backgammon.game.analysis_worker import AnalysisWorker
from backgammon.game.position import Position, Player
from backgammon.game.settings import Settings, Analyzer
from backgammon.logger import Logger


class AnalysisWorkerTest(unittest.TestCase):

    def setUp(self):
        Settings.analyzer = Analyzer.PYTHON
        Settings.analysis_depth = 3
        self.worker = AnalysisWorker()

    def test_equity_runs_off_main_thread(self):
        calls = []

        def fake(position, player):
            calls.append(threading.current_thread())
            return 0.25

        with patch('backgammon.game.analysis_worker.AnalyzerProxy.equity', side_effect=fake):
            result = self.worker.equity(Position(), Player.ME).result(timeout=5)

        self.assertEqual(0.25, result)
        self.assertIsNot(threading.main_thread(), calls[0])

    def test_all_tasks_run_on_the_same_thread(self):
        threads = []
        with patch('backgammon.game.analysis_worker.AnalyzerProxy.equity',
                   side_effect=lambda p, pl: threads.append(threading.current_thread())), \
             patch('backgammon.game.analysis_worker.AnalyzerProxy.clear_caches',
                   side_effect=lambda: threads.append(threading.current_thread())):
            self.worker.equity(Position(), Player.ME)
            self.worker.clear_caches().result(timeout=5)

        self.assertEqual(2, len(threads))
        self.assertIs(threads[0], threads[1])

    def test_position_is_snapshotted(self):
        position = Position()
        position.set_checkers(Player.ME, 0, 15)
        gate = threading.Event()
        seen = []

        def fake(snapshot, player):
            gate.wait(timeout=5)
            seen.append(snapshot.get_checkers(Player.ME, 0))
            return 1.0

        with patch('backgammon.game.analysis_worker.AnalyzerProxy.equity', side_effect=fake):
            future = self.worker.equity(position, Player.ME)
            position.set_checkers(Player.ME, 0, 0)  # edit while analysis is running
            gate.set()
            future.result(timeout=5)

        self.assertEqual([15], seen)

    def test_submit_runs_any_function(self):
        self.assertEqual(42, self.worker.submit(lambda: 6 * 7).result(timeout=5))

    def test_rust_cache_size_is_set_on_the_worker_thread(self):
        from backgammon.game.analyzer_proxy import AnalyzerProxy
        self.worker.set_rust_cache_size(64).result(timeout=5)
        self.assertEqual(64, self.worker.submit(AnalyzerProxy.rust_cache_size_mb).result(timeout=5))
        # The engine still works with the new cache.
        position = Position()
        position.set_turn(Player.ME)
        position.set_checkers(Player.ME, 1, 1)
        position.set_checkers(Player.ME, 0, 14)
        position.set_checkers(Player.OPPONENT, 0, 15)
        Settings.analyzer = Analyzer.RUST
        self.assertEqual(-1.0, self.worker.equity(position, Player.ME).result(timeout=5))

    def test_exception_is_propagated(self):
        with patch('backgammon.game.analysis_worker.AnalyzerProxy.equity',
                   side_effect=RuntimeError("boom")):
            future = self.worker.equity(Position(), Player.ME)
            with self.assertRaises(RuntimeError):
                future.result(timeout=5)

    def test_real_analysis(self):
        position = Position()
        position.set_turn(Player.ME)
        position.set_checkers(Player.ME, 1, 1)
        position.set_checkers(Player.ME, 0, 14)
        position.set_checkers(Player.OPPONENT, 0, 15)
        self.assertEqual(-1.0, self.worker.equity(position, Player.ME).result(timeout=5))


class LoggerTest(unittest.TestCase):

    class FakeTextArea:
        def __init__(self):
            self.lines = []

        def show_text(self, text):
            self.lines.append(text)

    def tearDown(self):
        Logger.set_text_area(None)

    def test_messages_from_other_threads_are_queued_until_flush(self):
        area = self.FakeTextArea()
        Logger.set_text_area(area)

        t = threading.Thread(target=lambda: Logger.log("from worker"))
        t.start()
        t.join()
        self.assertEqual([], area.lines)

        Logger.flush()
        self.assertEqual(["from worker"], area.lines)

    def test_main_thread_messages_are_shown_immediately(self):
        area = self.FakeTextArea()
        Logger.set_text_area(area)
        Logger.log("hello")
        self.assertEqual(["hello"], area.lines)


class PositionCubeTest(unittest.TestCase):

    def test_set_cube_invalidates_hash(self):
        position = Position()
        position.setup_starting_position()
        before = position.hash_code()
        position.set_cube(2, Player.ME)
        self.assertNotEqual(before, position.hash_code())
        self.assertEqual(2, position.cube_value)
        self.assertEqual(Player.ME, position.cube_owner)


if __name__ == '__main__':
    unittest.main()
