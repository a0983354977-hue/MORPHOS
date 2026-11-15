"""MORPHOS Dynamics Module.

Exports:
- NengoDynamics: Single-scale neural dynamics engine
- MultiScaleDynamics: Multi-scale temporal dynamics with hierarchical τ
- ScaleConfig, MultiScaleConfig: Configuration dataclasses
- GatedConnection: Cross-scale gated connections
- create_default_multiscale, make_multiscale_dynamics: Factory functions
"""

from core.dynamics.engine import NengoDynamics
from core.dynamics.multiscale import (
    MultiScaleDynamics,
    ScaleConfig,
    MultiScaleConfig,
    GatedConnection,
    create_default_multiscale,
    make_multiscale_dynamics,
)

__all__ = [
    "NengoDynamics",
    "MultiScaleDynamics",
    "ScaleConfig",
    "MultiScaleConfig",
    "GatedConnection",
    "create_default_multiscale",
    "make_multiscale_dynamics",
]