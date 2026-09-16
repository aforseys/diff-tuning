"""
Abstract base class for reward models r(φ; ω).

Interface:
    __call__(omega, phi) -> np.ndarray (N,)
        omega : (N, d) batch of weight vectors, one per particle.
        phi   : (d,)   feature vector of a single trajectory.
        returns: (N,) per-particle reward values.
"""

from abc import ABC, abstractmethod
import numpy as np


class RewardModel(ABC):
    """Abstract base for a parameterised reward function r(φ; ω)."""

    @abstractmethod
    def __call__(self, omega: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """
        Args:
            omega : (N, d) weight vectors, one per particle.
            phi   : (d,)   feature vector of a single trajectory.

        Returns:
            (N,) reward values, one per particle.
        """
        ...
