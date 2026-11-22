"""Line-based JSON server around the bgsage engine.

bgsage needs Python 3.10 or later, so it runs in its own virtualenv (.venv-bgsage), separate from
the app's interpreter. This script is started there by BgsageAnalyzer and must therefore not import
anything from the backgammon package.

Each request is one JSON object on a line of stdin; each answer is one JSON object on a line of
stdout. Boards use bgsage's layout, always from the perspective of the player in question.
Every evaluation is for a 1-point match, so gammons and the cube don't count.

  {"cmd": "evaluate", "board": [...], "ply": n}
      board is a post-move position: the player who just moved is about to hand over the dice.
      → {"equity": e}  for the player who just moved, in [-1, 1]
  {"cmd": "best_move", "board": [...], "dice": [d1, d2], "ply": n}
      board is a pre-roll position of the player on roll.
      → {"board": [...], "equity": e}  the board after the best move (still from the mover's
                                       perspective) and its equity for the mover
Errors are answered with {"error": message}.
"""
import json
import sys

from bgsage import BgBotAnalyzer

MATCH = {"away1": 1, "away2": 1}
_analyzers = {}


def analyzer(ply: int) -> BgBotAnalyzer:
    if ply not in _analyzers:
        _analyzers[ply] = BgBotAnalyzer(eval_level=f"{ply}ply")
    return _analyzers[ply]


def handle(request: dict) -> dict:
    cmd = request["cmd"]
    if cmd == "evaluate":
        result = analyzer(request["ply"]).post_move_analytics(request["board"], **MATCH)
        return {"equity": result.cubeful_equity}
    if cmd == "best_move":
        die1, die2 = request["dice"]
        result = analyzer(request["ply"]).checker_play(request["board"], die1, die2, **MATCH)
        best = result.moves[0]
        return {"board": list(best.board), "equity": best.equity}
    raise ValueError(f"Unknown command: {cmd}")


def main() -> None:
    for line in sys.stdin:
        try:
            answer = handle(json.loads(line))
        except Exception as e:
            answer = {"error": f"{type(e).__name__}: {e}"}
        sys.stdout.write(json.dumps(answer) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
