import copy
import queue
import threading
from concurrent.futures import Future
from typing import Callable

from backgammon.game.analyzer_proxy import AnalyzerProxy
from backgammon.game.position import Position, Player


class AnalysisWorker:
    """Runs analysis on a single background thread so the UI stays responsive.

    All engine calls go through this one thread, in submission order. That matters for the
    Rust engine, whose analyzer and cache are thread-local: using one long-lived thread keeps
    the cache alive across analyses, and makes clear_caches() clear the cache that is actually
    used.

    The thread is a daemon, so quitting the app never waits for a running search.
    """

    def __init__(self):
        self._tasks: queue.Queue[tuple[Future, Callable]] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="analysis-worker", daemon=True)
        self._thread.start()
        self.rust_network = None   # path of the net last submitted with set_rust_network

    def _run(self) -> None:
        while True:
            future, fn = self._tasks.get()
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(fn())
            except BaseException as e:
                future.set_exception(e)

    def submit(self, fn: Callable) -> Future:
        """Run fn() on the worker thread; the returned future resolves to its result."""
        future = Future()
        self._tasks.put((future, fn))
        return future

    def equity(self, position: Position, player: Player) -> Future:
        """Compute the equity for player in the background, using the global Settings.

        The position is copied, so the caller may keep editing the original while the analysis
        runs. The returned future resolves to the equity.
        """
        snapshot = copy.deepcopy(position)
        return self.submit(lambda: AnalyzerProxy.equity(snapshot, player))

    def analyze(self, position: Position, dice, depth: int, money: bool) -> Future:
        """Analyze the moves with dice (or the position itself for dice None) in the background,
        see move_analysis.analyze. The future resolves to an Analysis."""
        from backgammon.game.move_analysis import analyze
        snapshot = copy.deepcopy(position)
        return self.submit(lambda: analyze(snapshot, dice, depth, money))

    def set_rust_cache_size(self, megabytes: int) -> Future:
        """Resize the Rust engine's cache on the worker thread (where the engine runs), after any
        queued task."""
        return self.submit(lambda: AnalyzerProxy.set_rust_cache_size(megabytes))

    def set_rust_network(self, path) -> Future:
        """Switch the Rust engine's evaluator on the worker thread (see
        AnalyzerProxy.set_rust_network). The future resolves to the evaluator's description."""
        self.rust_network = path
        return self.submit(lambda: AnalyzerProxy.set_rust_network(path))

    def set_rust_bearoff(self, path) -> Future:
        """Load the bearoff database on the worker thread (see AnalyzerProxy.set_rust_bearoff).
        The future resolves to its description."""
        return self.submit(lambda: AnalyzerProxy.set_rust_bearoff(path))

    def use_rust_network(self, path) -> Future:
        """Like set_rust_network, but does nothing (an already resolved future) when path is the
        net already in use, so the engine's cache is kept."""
        if path == self.rust_network:
            future = Future()
            future.set_result(None)
            return future
        return self.set_rust_network(path)

    def clear_caches(self) -> Future:
        """Clear the engine caches on the worker thread, after any queued analysis."""
        return self.submit(AnalyzerProxy.clear_caches)
