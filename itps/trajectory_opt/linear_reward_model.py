"""
Linear reward model: r(φ; ω) = ω · φ.

  trajectory_reward_arrays   — fast path for the optimization loop.
  trajectory_reward_grad     — analytical gradient; pass as `jac` to the optimizer.
  trajectory_reward          — convenience wrapper for dict-based waypoints.
  make_trajectory_reward_fns — build a (reward_fn, jac_fn) pair.
"""

import numpy as np

from ._base_reward_model import RewardModel


class LinearRewardModel(RewardModel):
    """Linear reward: r(φ; ω) = ω · φ."""

    def __call__(self, omega: np.ndarray, phi: np.ndarray) -> np.ndarray:
        return omega @ np.asarray(phi, dtype=float)


def trajectory_reward_arrays(
    positions: np.ndarray,
    quats: np.ndarray,
    features,
    weights,
) -> float:
    """
    Compute the weighted sum of features over a trajectory.

    Args:
        positions : (k, 3) world-frame positions.
        quats     : (k, 4) xyzw quaternions.
        features  : list of n Feature instances.
        weights   : array-like of n floats.

    Returns:
        float: dot(weights, [feature(positions, quats).mean() for feature in features])
    """
    w = np.asarray(weights)
    feature_means = np.array([f(positions, quats).mean() for f in features])
    return float(w @ feature_means)


def trajectory_reward_grad(
    positions: np.ndarray,
    quats: np.ndarray,
    features,
    weights,
) -> np.ndarray:
    """
    Analytical gradient of trajectory_reward_arrays w.r.t. the FREE intermediate
    waypoints (indices 1..k in the full k+2 array).

    Returns a flat (7*k,) vector packed as [grad_pos[1], grad_quat[1], ...].
    Pass this as the `jac` argument to GradientOptimizer.
    """
    w = np.asarray(weights)
    n = len(positions)
    k = n - 2

    grad_pos_acc  = np.zeros_like(positions)
    grad_quat_acc = np.zeros_like(quats)

    for wi, feature in zip(w, features):
        gp, gq = feature.gradient(positions, quats)
        grad_pos_acc  += wi * gp
        grad_quat_acc += wi * gq

    # Quaternion normalisation chain rule.
    dots = (quats * grad_quat_acc).sum(axis=1)
    grad_quat_raw = grad_quat_acc - quats * dots[:, None]

    free = np.concatenate([grad_pos_acc[1:k+1], grad_quat_raw[1:k+1]], axis=1)
    return free.ravel()


def make_trajectory_reward_fns(fixed_features, learnable_features, weight_fn):
    """
    Build a (reward_fn, jac_fn) pair for use with GradientOptimizer.

    Combines fixed features (constant weights) with learnable features whose
    weights are fetched from weight_fn() on every call.

    Args:
        fixed_features      : list of (Feature, float) pairs — constant weight.
        learnable_features  : list of Feature instances — weights from weight_fn().
        weight_fn           : callable() -> np.ndarray of shape (len(learnable_features),).

    Returns:
        reward_fn : callable(positions, quats) -> float
        jac_fn    : callable(positions, quats) -> np.ndarray (7*k,)
    """
    def _all_features_weights():
        fixed_w   = np.array([w for _, w in fixed_features])
        fixed_f   = [f for f, _ in fixed_features]
        learned_w = np.asarray(weight_fn())
        all_f = fixed_f + list(learnable_features)
        all_w = np.concatenate([fixed_w, learned_w]) if len(fixed_w) else learned_w
        return all_f, all_w

    def reward_fn(positions, quats):
        all_f, all_w = _all_features_weights()
        return trajectory_reward_arrays(positions, quats, all_f, all_w)

    def jac_fn(positions, quats):
        all_f, all_w = _all_features_weights()
        return trajectory_reward_grad(positions, quats, all_f, all_w)

    return reward_fn, jac_fn


def trajectory_reward(waypoints, features, weights) -> float:
    """
    Convenience wrapper for dict-based waypoints.

    Args:
        waypoints : list of k dicts with 'pos' (3,) and 'quat' (4,) keys.
        features  : list of n Feature instances.
        weights   : list of n floats.
    """
    positions = np.array([wp["pos"]  for wp in waypoints])
    quats     = np.array([wp["quat"] for wp in waypoints])
    return trajectory_reward_arrays(positions, quats, features, weights)
