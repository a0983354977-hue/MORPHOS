"""Phase III: Memory & Counterfactual Engine - Public API."""

from core.memory.buffer import (
    RollingTemporalBuffer,
    TemporalConfig,
    make_temporal_buffer,
)

from core.memory.associative import (
    AssociativeMemory,
    RetrievalConfig,
    MatchResult,
    make_associative_memory,
)

from core.memory.history import (
    TrajectoryBuffer,
    make_snapshot,
    load_snapshot,
)

__all__ = [
    # T3.1 - Rolling Temporal Buffer
    "RollingTemporalBuffer",
    "TemporalConfig",
    "make_temporal_buffer",
    # T3.2 - Associative Memory
    "AssociativeMemory",
    "RetrievalConfig",
    "MatchResult",
    "make_associative_memory",
    # Legacy
    "TrajectoryBuffer",
    "make_snapshot",
    "load_snapshot",
]