"""
Gradient-based trajectory optimizer.

Uses L-BFGS-B for unconstrained problems and SLSQP for constrained problems.
Pass jac= for an analytical gradient function (~10x faster than finite
differences). Use trajectory_reward_grad() from linear_reward_model for
analytical gradients of linear feature combinations.

Usage:
    from itps.trajectory_opt.gradient_optimizer import GradientOptimizer
    from itps.trajectory_opt.geometric_features import DistanceToObject, Smoothness
    from itps.trajectory_opt.linear_reward_model import trajectory_reward_arrays, trajectory_reward_grad

    features = [Smoothness(), DistanceToObject(some_pos)]
    weights  = [1.0, 0.5]

    def reward_fn(positions, quats):
        return trajectory_reward_arrays(positions, quats, features, weights)

    def jac_fn(positions, quats):
        return trajectory_reward_grad(positions, quats, features, weights)

    opt = GradientOptimizer(max_iter=300)
    waypoints = opt.optimize_trajectory(
        pos_start, pos_goal, quat_start, quat_goal, reward_fn, jac=jac_fn,
    )
    # waypoints: list of dicts {'pos': (3,), 'quat': (4,)}
    # Extract positions for execute_spline:
    #   positions = np.array([wp['pos'] for wp in waypoints])
"""

import time

import numpy as np
from scipy.optimize import minimize

from ._base_trajectory_optimizer import TrajectoryOptimizer


class GradientOptimizer(TrajectoryOptimizer):
    """
    Gradient-based trajectory optimizer.

    - Unconstrained: L-BFGS-B (memory-efficient quasi-Newton).
    - Constrained:   SLSQP (sequential least-squares programming).

    Args:
        max_iter (int):   maximum optimizer iterations (default 200).
        tol      (float): convergence tolerance (default 1e-6).
    """

    def optimize_trajectory(
        self,
        pos_start:  np.ndarray,
        pos_goal:   np.ndarray,
        quat_start: np.ndarray,
        quat_goal:  np.ndarray,
        reward_fn,
        initial_waypoints: "list[dict] | int" = 20,
        *,
        jac=None,
        constraints=None,
        return_time: bool = False,
    ) -> "list[dict] | tuple":
        x0, objective, scipy_jac, scipy_constraints = self._prepare(
            pos_start, pos_goal, quat_start, quat_goal,
            reward_fn, initial_waypoints, jac, constraints,
        )

        method = "SLSQP" if scipy_constraints is not None else "L-BFGS-B"
        t0 = time.perf_counter()
        result = minimize(
            objective, x0,
            method      = method,
            jac         = scipy_jac,
            constraints = scipy_constraints,
            options     = {"maxiter": self.max_iter, "ftol": self.tol}
                        | ({"gtol": self.tol} if method == "L-BFGS-B" else {}),
        )

        return self._finalize(result, time.perf_counter() - t0, return_time)
