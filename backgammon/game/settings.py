from enum import Enum
from pathlib import Path
from typing import Optional


APP_NAME = "Alma Gammon"

# The app's folder in the home folder: the default place for saved and loaded positions.
APP_FOLDER = Path.home() / "almagammon"


class Analyzer(Enum):
    PYTHON = "python"
    RUST = "rust"
    BGSAGE = "bgsage"

    def clamp_depth(self, depth: int) -> int:
        """depth limited to what this engine supports (bgsage stops at 4-ply)."""
        if self == Analyzer.BGSAGE:
            from backgammon.game.bgsage_analyzer import BgsageAnalyzer
            return min(depth, BgsageAnalyzer.MAX_DEPTH)
        return depth


class Settings:
    analyzer: Analyzer = Analyzer.RUST
    # Search depth of the Analyze menu in roll levels; the UI shows it in plies (depth + 1).
    analysis_depth: int = 2
    # Search space of the Rust engine (gnubg's move filter): at every move decision of a search of
    # 2-ply or more, only the moves within search_space_margin equity of the best by their 1-ply
    # equity are searched deeper, at most search_space_moves of them. None: no limit (both None:
    # every move, the full search). Set in Analyze > Settings. Default 0.16 / 10: on 1000 reference
    # decisions at 3-ply as good as the full search (0.01 millipoints lost), 25x faster; see
    # src/bin/search_space_benchmark.rs.
    search_space_margin: Optional[float] = 0.16
    search_space_moves: Optional[int] = 10
    # Memory for the Rust engine's cache (a fixed-size transposition table). More memory means
    # more cache hits in deep searches; 1 GB is as fast as an unbounded cache at depth 3.
    rust_cache_mb: int = 1024
    # Threads for the Rust engine's searches of one roll level or more (the results are the same
    # as with one thread; about 7x faster with 10) and for rollouts. 0 means one per CPU core.
    # Set in Analyze > Settings.
    rust_threads: int = 0
    # Neural net used as the Rust engine's evaluator (a weight file written by the Rust `train`
    # binary, or its half-precision evaluation copy from `make eval-net`, about 3x faster),
    # relative to the project root. None, or a missing file, means the placeholder
    # evaluator, which only recognizes finished games.
    rust_network: Optional[str] = "nets/td-eval.bgnet"
    # The net for nackgammon games; same conventions. The default net (512-256, trained on
    # backgammon and nackgammon starts alternately) is the best for both variants, so it's the same
    # file. The previous default (576-192) is nets/td-576-192-eval.bgnet. A nackgammon specialist (128-128) is in nets/nack/nack-4M-avg.bgnet.
    rust_nackgammon_network: Optional[str] = "nets/td-eval.bgnet"
    # The nets for games that aren't double match point (cubeless money: gammons count), per
    # variant: money nets (5 outputs, `train --objective money`); None means the double match
    # point net above, which plays to win and ignores gammons. Default: the f16 copy of
    # nets/td-money.bgnet, a 512-256 money net trained on both variants (52M games, alpha decayed to
    # 0.0025; +0.007 ppg against bgsage). The previous default (256-256) is nets/td-money-256-256-52M-eval.bgnet.
    rust_money_network: Optional[str] = "nets/td-money-eval.bgnet"
    rust_nackgammon_money_network: Optional[str] = "nets/td-money-eval.bgnet"
    # Two-sided bearoff database (exact winning chances once both sides are in their home boards),
    # built with `make bearoff` (5.9 GB, not in git). None, or a missing file, means the net is used
    # for bearoff positions too.
    rust_bearoff_db: Optional[str] = "data/bearoff-6x15.bd"

    @classmethod
    def rust_network_for(cls, variant, dmp: bool = True) -> Optional[str]:
        """The Rust engine's net for a GameVariant, at double match point (dmp) or for money
        (the money net, falling back to the double match point net when there is none)."""
        from backgammon.game.position import GameVariant
        nack = variant == GameVariant.NACKGAMMON
        dmp_net = cls.rust_nackgammon_network if nack else cls.rust_network
        if dmp:
            return dmp_net
        return (cls.rust_nackgammon_money_network if nack else cls.rust_money_network) or dmp_net
