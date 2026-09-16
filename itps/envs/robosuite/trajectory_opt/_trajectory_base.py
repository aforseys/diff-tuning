"""
Shared mathematical utilities for trajectory definitions.

_slerp  — spherical linear interpolation between two unit quaternions (xyzw).
_min_jerk — minimum-jerk timing law (Flash & Hogan 1985).
"""

import numpy as np


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation between two unit quaternions (xyzw).
    t=0 returns q0, t=1 returns q1.
    """
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = np.clip(np.dot(q0, q1), -1.0, 1.0)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return (q0 + t * (q1 - q0)) / np.linalg.norm(q0 + t * (q1 - q0))
    theta_0 = np.arccos(dot)
    theta   = theta_0 * t
    q_perp  = (q1 - q0 * dot) / np.sin(theta_0)
    return q0 * np.cos(theta) + q_perp * np.sin(theta)


def _min_jerk(t: float) -> float:
    """
    Minimum-jerk timing law (Flash & Hogan 1985).

    Maps normalised time t ∈ [0, 1] to a smooth arc parameter u ∈ [0, 1]
    with zero velocity and zero acceleration at both endpoints.

        u(t) = 10t³ − 15t⁴ + 6t⁵
    """
    return 10*t**3 - 15*t**4 + 6*t**5
