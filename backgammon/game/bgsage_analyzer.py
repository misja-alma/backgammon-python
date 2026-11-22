import json
import os
import subprocess
import threading
from pathlib import Path
from typing import List, Optional

from backgammon.game.evaluator import Evaluator
from backgammon.game.position import Position, Player

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SERVER_SCRIPT = Path(__file__).resolve().with_name("bgsage_server.py")


class BgsageAnalyzer:
    """The bgsage engine (/Users/misja/workspace/bgsage), playing a 1-point match.

    bgsage needs a newer Python than the app, so it is installed in its own virtualenv and runs
    as a subprocess (bgsage_server.py) that is started on first use and kept alive. Set the
    BGSAGE_PYTHON environment variable to use another interpreter than .venv-bgsage.

    Depth means the same as for the other engines: the number of roll levels searched beyond the
    position itself. bgsage counts plies the XG way, so depth 0 is bgsage's 1-ply (the raw
    network evaluation) and depth d is (d + 1)-ply. bgsage goes up to 4-ply, so MAX_DEPTH is 3.
    """

    MAX_DEPTH = 3
    CHECKERS = 15

    _process: Optional[subprocess.Popen] = None
    _lock = threading.Lock()

    # --- board conversion ---------------------------------------------------------------------

    @staticmethod
    def _checkers(position: Position, player: Player) -> List[int]:
        return position.me_checkers if player == Player.ME else position.opponent_checkers

    @classmethod
    def to_board(cls, position: Position, player: Player) -> List[int]:
        """The position as a bgsage board from player's perspective: index 1-24 holds player's
        checkers (positive) and the other side's (negative) in player's numbering, index 25
        player's bar and index 0 the other side's bar."""
        own = cls._checkers(position, player)
        other = cls._checkers(position, player.other_player())
        board = [0] * 26
        for point in range(1, 25):
            board[point] = own[point] - other[25 - point]
        board[25] = own[25]
        board[0] = other[25]
        return board

    @classmethod
    def from_board(cls, board: List[int], player: Player, turn: Player) -> Position:
        """Inverse of to_board; borne-off counts are whatever is missing from the 15 checkers."""
        own = [0] * 26
        other = [0] * 26
        for point in range(1, 25):
            if board[point] > 0:
                own[point] = board[point]
            elif board[point] < 0:
                other[25 - point] = -board[point]
        own[25] = board[25]
        other[25] = board[0]
        own[0] = cls.CHECKERS - sum(own)
        other[0] = cls.CHECKERS - sum(other)

        position = Position()
        position.me_checkers, position.opponent_checkers = (own, other) if player == Player.ME else (other, own)
        position.set_turn(turn)
        return position

    # --- engine calls -------------------------------------------------------------------------

    @classmethod
    def _ply(cls, depth: int) -> int:
        if not 0 <= depth <= cls.MAX_DEPTH:
            raise ValueError(f"bgsage supports depth 0 to {cls.MAX_DEPTH} (1- to 4-ply), not {depth}")
        return depth + 1

    @classmethod
    def equity(cls, position: Position, player: Player, depth: int) -> float:
        """Equity of the position for player, in [-1, 1] (2 * winning chance - 1)."""
        on_roll = position.get_turn()
        if Evaluator.is_game_over(position):
            equity_on_roll = Evaluator.evaluate(position)
        else:
            # The position before on_roll's roll is the position right after the other side moved.
            just_moved = on_roll.other_player()
            answer = cls._request({"cmd": "evaluate", "board": cls.to_board(position, just_moved),
                                   "ply": cls._ply(depth)})
            equity_on_roll = -answer["equity"]
        return equity_on_roll if player == on_roll else -equity_on_roll

    @classmethod
    def best_move(cls, position: Position, die1: int, die2: int, depth: int):
        """Best move for the side to move: (position after the move, equity for the mover)."""
        mover = position.get_turn()
        answer = cls._request({"cmd": "best_move", "board": cls.to_board(position, mover),
                               "dice": [die1, die2], "ply": cls._ply(depth)})
        new_pos = cls.from_board(answer["board"], mover, mover.other_player())
        new_pos.set_cube(position.cube_value, position.cube_owner)
        return new_pos, answer["equity"]

    # --- subprocess ---------------------------------------------------------------------------

    @staticmethod
    def python_executable() -> str:
        return os.environ.get("BGSAGE_PYTHON", str(_PROJECT_ROOT / ".venv-bgsage" / "bin" / "python"))

    @classmethod
    def is_available(cls) -> bool:
        return Path(cls.python_executable()).exists()

    @classmethod
    def _request(cls, request: dict) -> dict:
        with cls._lock:
            process = cls._ensure_process()
            try:
                process.stdin.write(json.dumps(request) + "\n")
                process.stdin.flush()
                line = process.stdout.readline()
            except (BrokenPipeError, OSError):
                line = ""
            if not line:
                cls._process = None
                raise RuntimeError("The bgsage engine process stopped unexpectedly")
        answer = json.loads(line)
        if "error" in answer:
            raise RuntimeError(f"bgsage: {answer['error']}")
        return answer

    @classmethod
    def _ensure_process(cls) -> subprocess.Popen:
        if cls._process is None or cls._process.poll() is not None:
            python = cls.python_executable()
            if not Path(python).exists():
                raise RuntimeError(f"bgsage is not installed: {python} not found. Install it with "
                                   f"'make bgsage'.")
            cls._process = subprocess.Popen(
                [python, "-I", str(_SERVER_SCRIPT)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        return cls._process

    @classmethod
    def shutdown(cls) -> None:
        with cls._lock:
            if cls._process is not None:
                cls._process.stdin.close()
                cls._process.wait(timeout=5)
                cls._process = None
