from enum import Enum


class Analyzer(Enum):
    PYTHON = "python"
    RUST = "rust"


class Settings:
    analyzer: Analyzer = Analyzer.RUST
    analysis_depth: int = 3
    # Memory for the Rust engine's cache (a fixed-size transposition table). More memory means
    # more cache hits in deep searches; 1 GB is as fast as an unbounded cache at depth 3.
    rust_cache_mb: int = 1024
