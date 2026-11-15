"""Phase III: Counterfactual Engine - Public API."""

from core.counterfactual.engine import (
    CounterfactualEngine,
    CounterfactualConfig,
    SimulatorSnapshot,
    BranchConfig,
    BranchState,
)

from core.counterfactual.observer import (
    DivergenceObserver,
    DivergenceConfig,
    DivergenceMetrics,
    ScenarioReport,
    make_divergence_observer,
)

__all__ = [
    # T3.3 - Counterfactual Engine
    "CounterfactualEngine",
    "CounterfactualConfig",
    "SimulatorSnapshot",
    "BranchConfig",
    "BranchState",
    # T3.4 - Divergence Observer
    "DivergenceObserver",
    "DivergenceConfig",
    "DivergenceMetrics",
    "ScenarioReport",
    "make_divergence_observer",
]