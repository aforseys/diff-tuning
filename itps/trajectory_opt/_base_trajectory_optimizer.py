"""
Abstract base class for waypoint trajectory optimizers.

Subclasses implement optimize_trajectory() and use _prepare() / _finalize()
to handle the shared geometry, packing/unpacking, gradient caching, and
constraint wrapping.
"""

from abc import ABC, abstractmethod

import numpy as np

from ._trajectory_base import _slerp


class TrajectoryOptimizer(ABC):
    """
    Abstract base class for waypoint trajectory optimizers.

    Args:
        max_iter (int):   maximum optimizer iterations (default 200).
        tol      (float): convergence tolerance (default 1e-6).
    """

    def __init__(self, max_iter: int = 200, tol: float = 1e-6):
        self.max_iter = max_iter
        self.tol      = tol

    @abstractmethod
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
        """
        Optimise k intermediate waypoints to maximise reward_fn.

        Start and goal are fixed; only the intermediate waypoints move.

        Args:
            pos_start, pos_goal   (np.ndarray, 3): Fixed endpoint positions.
            quat_start, quat_goal (np.ndarray, 4): Fixed endpoint orientations (xyzw).
            reward_fn             callable(positions, quats) -> float.
                                  Receives (k+2, 3)/(k+2, 4) arrays with
                                  endpoints included.
            initial_waypoints     List of k dicts with 'pos'/'quat' keys
                                  (warm-start), or int k for a straight-line
                                  initial guess (default 20).
            jac                   Optional gradient callable jac(positions, quats)
                                  -> np.ndarray (7*k,).
            constraints           Optional callable(positions, quats) -> float or
                                  np.ndarray. Must return <= 0 when satisfied.
            return_time           If True, return (waypoints, elapsed_seconds).

        Returns:
            list of k optimised waypoint dicts {'pos': (3,), 'quat': (4,)},
            or (list, float) if return_time=True.
        """
        ...

    # ------------------------------------------------------------------
    # Protected helpers
    # ------------------------------------------------------------------

    def _prepare(
        self,
        pos_start, pos_goal, quat_start, quat_goal,
        reward_fn,
        initial_waypoints,
        jac,
        constraints,
    ):
        """
        Build the initial parameter vector and wrap reward/jac/constraints for
        scipy.optimize.minimize.

        Returns:
            x0                : (7k,) packed initial parameter vector.
            objective         : callable(x) -> float  (negated reward).
            scipy_jac         : callable(x) -> (7k,)  or None.
            scipy_constraints : scipy constraint dict, or None.
        """
        if isinstance(initial_waypoints, int):
            k = initial_waypoints
            if k < 1:
                raise ValueError("initial_waypoints must be >= 1.")
            alphas = np.linspace(0, 1, k + 2)[1:-1]
            initial_waypoints = [
                {"pos":  (1 - a) * pos_start + a * pos_goal,
                 "quat": _slerp(quat_start, quat_goal, a)}
                for a in alphas
            ]
        elif not initial_waypoints:
            raise ValueError("initial_waypoints list must not be empty.")

        k = len(initial_waypoints)

        pos_boundary  = np.stack([pos_start, pos_goal])
        quat_boundary = np.stack([quat_start, quat_goal])

        def _to_arrays(x):
            wp        = x.reshape(k, 7)
            positions = wp[:, :3]
            quats_raw = wp[:, 3:]
            norms     = np.linalg.norm(quats_raw, axis=1, keepdims=True)
            quats     = quats_raw / np.maximum(norms, 1e-8)
            return positions, quats

        def _with_endpoints(positions, quats):
            return (
                np.concatenate([pos_boundary[:1],  positions, pos_boundary[1:]], axis=0),
                np.concatenate([quat_boundary[:1], quats,     quat_boundary[1:]], axis=0),
            )

        def objective(x):
            return -reward_fn(*_with_endpoints(*_to_arrays(x)))

        # Gradient caching: SLSQP calls fun and jac separately on the same x.
        scipy_jac = None
        if jac is not None:
            _cache: dict = {}

            def _combined(x):
                key = x.tobytes()
                if key not in _cache:
                    pos_f, q_f       = _to_arrays(x)
                    pos_full, q_full = _with_endpoints(pos_f, q_f)
                    val  = -reward_fn(pos_full, q_full)
                    grad = -np.asarray(jac(pos_full, q_full), dtype=float)
                    _cache.clear()
                    _cache[key] = (val, grad)
                return _cache[key]

            objective = lambda x: _combined(x)[0]
            scipy_jac = lambda x: _combined(x)[1]

        scipy_constraints = None
        if constraints is not None:
            def _constraint_fun(x):
                pos_f, q_f = _to_arrays(x)
                pos_full, q_full = _with_endpoints(pos_f, q_f)
                return -np.atleast_1d(constraints(pos_full, q_full))

            # Let scipy estimate the Jacobian via finite differences — constraint
            # functions may return variable-length outputs (e.g. combine_constraints)
            # and the active set is often discontinuous, so a hardcoded approximation
            # is both fragile and potentially wrong-shaped.
            scipy_constraints = {"type": "ineq", "fun": _constraint_fun}

        x0 = np.concatenate([np.concatenate([wp["pos"], wp["quat"]]) for wp in initial_waypoints])
        return x0, objective, scipy_jac, scipy_constraints

    def _finalize(self, result, elapsed: float, return_time: bool):
        """
        Unpack the scipy OptimizeResult into a list of waypoint dicts.
        Re-normalises quaternions and logs a warning on failure.
        """
        if not result.success:
            print(f"[{type(self).__name__}] warning: {result.message}")

        k = len(result.x) // 7
        x = result.x
        waypoints = []
        for i in range(k):
            pos  = x[7*i : 7*i + 3].copy()
            quat = x[7*i + 3 : 7*i + 7].copy()
            quat /= max(np.linalg.norm(quat), 1e-8)
            waypoints.append({"pos": pos, "quat": quat})

        return (waypoints, elapsed) if return_time else waypoints
