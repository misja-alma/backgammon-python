from enum import Enum
from typing import Optional


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
    analysis_depth: int = 3
    # Memory for the Rust engine's cache (a fixed-size transposition table). More memory means
    # more cache hits in deep searches; 1 GB is as fast as an unbounded cache at depth 3.
    rust_cache_mb: int = 1024
    # Neural net used as the Rust engine's evaluator (a weight file written by the Rust `train`
    # binary), relative to the project root. None, or a missing file, means the placeholder
    # evaluator, which only recognizes finished games.
    rust_network: Optional[str] = "nets/td.bgnet"
    # The net for nackgammon games; same conventions. The default net (256-256, trained on
    # backgammon and nackgammon starts alternately) is the best for both variants, so it's the same
    # file. A nackgammon specialist (128-128) is in nets/nack/nack-4M-avg.bgnet.
    rust_nackgammon_network: Optional[str] = "nets/td.bgnet"
    # Two-sided bearoff database (exact winning chances once both sides are in their home boards),
    # built with `make bearoff` (5.9 GB, not in git). None, or a missing file, means the net is used
    # for bearoff positions too.
    rust_bearoff_db: Optional[str] = "data/bearoff-6x15.bd"

    @classmethod
    def rust_network_for(cls, variant) -> Optional[str]:
        """The Rust engine's net for a GameVariant."""
        from backgammon.game.position import GameVariant
        return cls.rust_nackgammon_network if variant == GameVariant.NACKGAMMON else cls.rust_network
