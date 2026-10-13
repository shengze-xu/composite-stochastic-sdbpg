"""
SDBPG core library.

This package contains the reusable implementation of:
- Bregman divergences and utilities
- SDBPG algorithm and baseline decentralized optimizers
"""

from .core import (
    BregmanDivergence,
    QuadraticRegressionProblem,
    compute_consensus_gap,
    compute_optimality_gap,
    entropy_bregman,
    exact_group_l21_quadratic_prox_step,
    exact_l1_quadratic_prox_step,
    exact_quadratic_primal_step,
    exact_quadratic_smoothing_step,
    generate_network_topology,
    quadratic_bregman,
    soft_threshold,
)
from .algorithm import (
    SDBPGAgent,
    SDBPG,
    DecentralizedGradientDescent,
    DecentralizedGradientTracking,
)

__all__ = [
    # core
    "BregmanDivergence",
    "quadratic_bregman",
    "entropy_bregman",
    "exact_quadratic_primal_step",
    "exact_quadratic_smoothing_step",
    "exact_l1_quadratic_prox_step",
    "exact_group_l21_quadratic_prox_step",
    "soft_threshold",
    "QuadraticRegressionProblem",
    "generate_network_topology",
    "compute_consensus_gap",
    "compute_optimality_gap",
    # algorithm
    "SDBPGAgent",
    "SDBPG",
    "DecentralizedGradientDescent",
    "DecentralizedGradientTracking",
]
