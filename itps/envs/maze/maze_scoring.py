"""
Pygame-free maze trajectory scoring functions.

Shared between offline training evaluation (`eval_maze`, called repeatedly
during training and must not depend on pygame/MazeGUI) and preference-pair
generation (`extract_preference_pairs`, which does depend on MazeGUI for
GUI<->XY conversion but delegates scoring here).

All functions operate on trajectories in maze XY coordinate space (not GUI
pixel space) and take the raw boolean maze grid (True = wall) as an argument.
"""

import numpy as np


def check_maze_collision(xy_traj, maze):
    """
    xy_traj: (batch, steps, 2) numpy array in maze coordinate space
    maze: 2D boolean array where True = wall
    Returns: (batch,) boolean array, True if trajectory has any collision
    """
    batch_size, num_steps, _ = xy_traj.shape
    xy_flat = xy_traj.reshape(-1, 2)
    nan_mask = np.any(np.isnan(xy_flat), axis=1)
    xy_flat = np.nan_to_num(xy_flat, nan=0.0)
    xy_flat = np.clip(xy_flat, [0, 0], [maze.shape[0] - 1, maze.shape[1] - 1])
    mx = np.round(xy_flat[:, 0]).astype(int)
    my = np.round(xy_flat[:, 1]).astype(int)
    collisions = maze[mx, my].reshape(batch_size, num_steps)
    collisions |= nan_mask.reshape(batch_size, num_steps)
    return np.any(collisions, axis=1)


def score_center(xy_traj, maze):
    """
    Center preference — 1 minus mean normalized distance from maze center over
    all trajectory steps. Higher is better (closer to center on average).

    xy_traj: (batch, steps, 2) numpy array in maze coordinate space
    maze: 2D boolean array (used only for its shape)
    Returns: (batch,) float array
    """
    rows, cols = maze.shape
    x_center = (rows - 1) / 2.0
    y_center = (cols - 1) / 2.0
    max_dist = np.sqrt(x_center**2 + y_center**2)
    dx = xy_traj[:, :, 0] - x_center
    dy = xy_traj[:, :, 1] - y_center
    return 1 - np.sqrt(dx**2 + dy**2).mean(axis=1) / max_dist


def score_bottom_half(xy_traj, maze):
    """
    Bottom-half preference — fraction of trajectory steps with x > x_mid
    (higher row index = bottom of maze). Higher is better.

    xy_traj: (batch, steps, 2) numpy array in maze coordinate space
    maze: 2D boolean array (used only for its shape)
    Returns: (batch,) float array
    """
    rows, cols = maze.shape
    x_mid = (rows - 1) / 2.0
    return (xy_traj[:, :, 0] > x_mid).astype(float).mean(axis=1)


def score_goal_dist(xy_traj, goal):
    """
    Distance from trajectory endpoint to a goal. Lower is better — callers
    that want a "higher is better" score should invert/normalize this
    themselves (see `extract_preference_pairs`'s `endpoint_distance` metric).

    xy_traj: (batch, steps, 2) numpy array in maze coordinate space
    goal: (2,) array-like in maze coordinate space, broadcastable against
        `xy_traj[:, -1, :]` (so a per-trajectory (batch, 2) goal also works)
    Returns: (batch,) float array
    """
    goal = np.asarray(goal, dtype=float)
    return np.linalg.norm(xy_traj[:, -1, :] - goal, axis=1)


def score_goal_progress(xy_traj, goal, start, clip=True):
    """
    Percentage-to-goal — the fraction of the initial start->goal distance that
    the trajectory endpoint covered: (ref_dist - endpoint_dist) / ref_dist.
    1.0 means the endpoint reached the goal, 0 means it ended no closer than it
    started. Higher is better.

    By default (clip=True) negative progress (ending farther from the goal than
    it started) is clipped to 0 -- this is THE goal preference metric, used by
    `eval_maze` for `<metric>_pct_clipped`, by preference-pair selection, and by
    the energy-ranking eval scripts, so all three score `obs_goal_dist` /
    `finetune_goal_dist` identically under the same names. Pass clip=False for
    the raw signed value (`eval_maze`'s `<metric>_pct` diagnostic).

    xy_traj: (batch, steps, 2) numpy array in maze coordinate space
    goal:  (2,) or (batch, 2) goal in maze coordinate space
    start: (2,) or (batch, 2) start position in maze coordinate space
        (ref_dist is measured from here to the goal, matching eval_maze's
        `ref_dist = ||goal - state||`)
    clip: clip negative progress to 0 (default True; the standard metric)
    Returns: (batch,) float array
    """
    goal = np.asarray(goal, dtype=float)
    start = np.asarray(start, dtype=float)
    endpoint_dist = score_goal_dist(xy_traj, goal)
    ref_dist = np.linalg.norm(goal - start, axis=-1)
    progress = (ref_dist - endpoint_dist) / ref_dist
    if clip:
        progress = np.clip(progress, 0, None)
    return progress


def score_goal_dist_relative(xy_traj, goal):
    """
    Endpoint distance to a goal, normalized by the WORST endpoint distance in the
    batch: `1 - dist / max(dist)`. Higher is better.

    This is the original `endpoint_distance` formula (see
    `extract_preference_pairs`), generalized to take an explicit goal instead of
    reading one from a human-drawn guide. It differs from `score_goal_progress`
    in two ways that matter for preference selection:

      - Normalizer. The yardstick is the worst of the batch's own endpoint
        distances, not the start->goal distance, so it is re-derived per trial
        from whatever the sampler produced rather than from task geometry.
      - No floor. Trajectories that ended FARTHER from the goal than they started
        stay ordered relative to each other, where `score_goal_progress` clips
        them all to 0 and makes them mutually untied. Pairs can therefore be
        minted inside the "moved away from the goal" region.

    Preference-generation only -- this is NOT an eval metric, and unlike
    `score_goal_progress` its value depends on the other trajectories in the
    batch, so it is not comparable across trials or against an eval log.

    xy_traj: (batch, steps, 2) numpy array in maze coordinate space
    goal: (2,) or (batch, 2) goal in maze coordinate space
    Returns: (batch,) float array
    """
    dists = score_goal_dist(xy_traj, goal)
    return 1.0 - dists / (dists.max() + 1e-6)


# Minimum |score_i - score_j| for a pair to count as an informative preference
# rather than a tie, per maze metric. Used by `extract_preference_pairs` to
# decide which pairs to mint as winner/loser training data. (The energy-ranking
# eval scripts instead throw out only exact ground-truth ties, via
# `preference_scoring.pairwise_win_rate`'s default tie_tol=0.)
DEFAULT_SCORE_THRESHOLDS = {
    'similarity_score': 0.5,
    'endpoint_distance': 0.3,
    'collision_rate': 0.9,
    'center_rate': 0.3,
    'bottom_half_rate': 0.3,
    'obs_goal_dist': 0.3,
    'finetune_goal_dist': 0.3,
    # Same 0.3 as `endpoint_distance`, whose formula this metric reuses -- but note
    # the two 0.3s gate different things. Here the bar is 0.3 * (worst endpoint
    # distance in the batch); for `finetune_goal_dist` it is 0.3 * (start->goal
    # distance). Only the second is a fixed function of the observation.
    'finetune_goal_dist_relative': 0.3,
}


def score_metric(metric, xy_traj, maze, start=None, obs_goal=None, fixed_goal=None):
    """Per-trajectory scores for one metric. Higher = more preferred.

    The goal metrics use the same percentage-to-goal calc as eval_maze
    (score_goal_progress): obs_goal_dist toward the per-obs goal-conditioned
    goal, finetune_goal_dist toward a fixed goal.

    This is the preference convention -- direction normalized so higher wins,
    matching how `extract_preference_pairs` labels pairs. It is deliberately NOT
    what `eval_maze` reports: there `collision_rate` is the fraction that DID
    collide, because that is the useful number to log.
    """
    if metric == 'collision_rate':
        return (~check_maze_collision(xy_traj, maze)).astype(float)
    elif metric == 'center_rate':
        return score_center(xy_traj, maze)
    elif metric == 'bottom_half_rate':
        return score_bottom_half(xy_traj, maze)
    elif metric == 'obs_goal_dist':
        assert obs_goal is not None, \
            "'obs_goal_dist' requires a goal-conditioned obs file (x,y,goal_x,goal_y)"
        assert start is not None, "'obs_goal_dist' requires the start position"
        return score_goal_progress(xy_traj, obs_goal, start)
    elif metric == 'finetune_goal_dist':
        assert fixed_goal is not None, \
            "'finetune_goal_dist' requires --goal (or eval.goal in the checkpoint config)"
        assert start is not None, "'finetune_goal_dist' requires the start position"
        return score_goal_progress(xy_traj, fixed_goal, start)
    else:
        raise NotImplementedError(f"Metric '{metric}' is not implemented.")
