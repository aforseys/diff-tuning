"""
Feature functions for trajectory reward design.

A Feature maps a batch of poses to a vector of scalar values, one per pose.

Feature interface:
    feature(positions, quats) -> np.ndarray (k,)
        positions : np.ndarray (k, 3)  positions in world frame (metres)
        quats     : np.ndarray (k, 4)  orientations as xyzw quaternions
"""

import numpy as np
from abc import ABC, abstractmethod


class Feature(ABC):
    """Abstract base class for pose features."""

    @abstractmethod
    def __call__(self, positions: np.ndarray, quats: np.ndarray) -> np.ndarray:
        """
        Args:
            positions : (k, 3)
            quats     : (k, 4)
        Returns:
            (k,) array of scalar feature values, one per waypoint.
        """
        ...

    def gradient(self, positions: np.ndarray, quats: np.ndarray) -> tuple:
        """
        Gradient of mean(self(positions, quats)) w.r.t. positions and quats.

        Returns:
            grad_pos  : (n, 3)
            grad_quat : (n, 4)

        Default raises NotImplementedError; subclasses override to enable
        analytical gradients in the optimizer.
        """
        raise NotImplementedError(f"{type(self).__name__} has no analytical gradient")


class DistanceToObject(Feature):
    """
    Soft proximity feature based on a Gaussian kernel.

    Returns exp(-distance² / (2 * temperature²)) for each waypoint:
        - value near 1.0 when close to the object
        - value near 0.0 when far from the object

    Use with a positive weight for attraction, negative weight for avoidance.

    Args:
        object_pos  (array-like, 3): world-frame position of the object.
        temperature (float):         length scale in metres (default 0.2).
    """

    def __init__(self, object_pos, temperature: float = 0.2):
        self.object_pos  = np.asarray(object_pos, dtype=float)
        self.temperature = temperature

    def __call__(self, positions: np.ndarray, _quats: np.ndarray) -> np.ndarray:
        distances_sq = ((positions - self.object_pos) ** 2).sum(axis=1)
        return np.exp(-distances_sq / (2 * self.temperature ** 2)) / len(positions)

    def gradient(self, positions, quats):
        n = len(positions)
        diff = positions - self.object_pos
        vals = self(positions, quats)
        grad_pos = (vals[:, None] / n) * (-diff / self.temperature ** 2)
        return grad_pos, np.zeros_like(quats)


class MaintainOrientation(Feature):
    """
    Returns the negative angular distance between each quat and a desired quaternion.

    Higher (less negative) means more aligned with desired_quat.
    Use with a positive weight to keep the EEF orientation stable.

    Args:
        desired_quat (array-like, 4): desired EEF orientation as xyzw quaternion.
    """

    def __init__(self, desired_quat):
        q = np.asarray(desired_quat, dtype=float)
        self.desired_quat = q / np.linalg.norm(q)

    def __call__(self, positions: np.ndarray, quats: np.ndarray) -> np.ndarray:
        norms  = np.linalg.norm(quats, axis=1, keepdims=True)
        q_norm = quats / norms
        dots   = np.clip(np.abs(q_norm @ self.desired_quat), 0.0, 1.0)
        angles = 2.0 * np.arccos(dots)
        return -angles

    def gradient(self, positions, quats):
        n = len(quats)
        norms    = np.linalg.norm(quats, axis=1, keepdims=True)
        q_norm   = quats / np.maximum(norms, 1e-8)
        raw_dots = q_norm @ self.desired_quat
        abs_dots = np.clip(np.abs(raw_dots), 0.0, 1.0 - 1e-7)
        signs    = np.where(raw_dots >= 0, 1.0, -1.0)
        denom    = np.sqrt(1.0 - abs_dots ** 2)
        proj     = self.desired_quat[None, :] - q_norm * raw_dots[:, None]
        coeff    = 2.0 * signs / (denom * n)
        grad_quat = coeff[:, None] * proj
        return np.zeros_like(positions), grad_quat


class Smoothness(Feature):
    """
    Penalises large steps between consecutive waypoints.

    Returns the negative squared distance between each pair of neighbours
    as a per-waypoint array. Use with a positive weight for smooth paths.
    """

    def __call__(self, positions: np.ndarray, _quats: np.ndarray) -> np.ndarray:
        diffs    = positions[1:] - positions[:-1]
        sq_dists = (diffs * diffs).sum(axis=1)
        return -np.append(sq_dists, 0.0)

    def gradient(self, positions, quats):
        n = len(positions)
        grad_pos = np.zeros_like(positions)
        grad_pos[1:-1] = (2.0 / n) * (positions[:-2] - 2.0 * positions[1:-1] + positions[2:])
        return grad_pos, np.zeros_like(quats)


class Jerk(Feature):
    """
    Penalises large jerk (3rd finite difference of position).

    Jerk at waypoint i = p[i+3] - 3*p[i+2] + 3*p[i+1] - p[i].
    Returns -||jerk||² per waypoint (zero-padded at the tail).
    Use with a positive weight to discourage abrupt direction changes.
    """

    def __call__(self, positions: np.ndarray, _quats: np.ndarray) -> np.ndarray:
        if len(positions) < 4:
            return np.zeros(len(positions))
        jerks    = np.diff(positions, n=3, axis=0)          # (n-3, 3)
        sq_jerks = (jerks * jerks).sum(axis=1)              # (n-3,)
        return -np.append(sq_jerks, np.zeros(3))            # pad tail to length n

    def gradient(self, positions, quats):
        n = len(positions)
        if n < 4:
            return np.zeros_like(positions), np.zeros_like(quats)
        jerks    = np.diff(positions, n=3, axis=0)          # (n-3, 3)
        grad_pos = np.zeros_like(positions)
        grad_pos[:n-3]  += -2.0 * jerks   # coeff of p[i]   in j[i] is -1
        grad_pos[1:n-2] +=  6.0 * jerks   # coeff of p[i+1] in j[i] is +3
        grad_pos[2:n-1] += -6.0 * jerks   # coeff of p[i+2] in j[i] is -3
        grad_pos[3:]    +=  2.0 * jerks   # coeff of p[i+3] in j[i] is +1
        return grad_pos / n, np.zeros_like(quats)


class MaintainDistanceThreshold(Feature):
    """
    Sigmoid penalty that rises from 0 to 1 as distance from a point exceeds a threshold.

    Use with a negative weight to keep the trajectory near the point (penalise
    drifting beyond the threshold), or with a positive weight to push it away.

    Args:
        position   (array-like, 3): world-frame reference position.
        threshold  (float):         distance at which cost reaches 0.5 (metres).
        sharpness  (float):         steepness of the sigmoid (default 20.0).
    """

    def __init__(self, position, threshold: float, sharpness: float = 20.0):
        self.object_pos = np.asarray(position, dtype=float)
        self.threshold  = threshold
        self.sharpness  = sharpness

    def __call__(self, positions: np.ndarray, _quats: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(positions - self.object_pos, axis=1)
        return 1.0 / (1.0 + np.exp(-self.sharpness * (distances - self.threshold)))

    def gradient(self, positions, quats):
        n = len(positions)
        diff   = positions - self.object_pos
        d      = np.linalg.norm(diff, axis=1)
        safe_d = np.maximum(d, 1e-12)
        sigma  = 1.0 / (1.0 + np.exp(-self.sharpness * (d - self.threshold)))
        coeff  = sigma * (1.0 - sigma) * self.sharpness / (safe_d * n)
        return coeff[:, None] * diff, np.zeros_like(quats)


class DesiredHeight(Feature):
    """
    Quadratic penalty for deviation from a desired world-frame z-height.

    Returns -(z - desired_z)² for each waypoint: zero at the target height,
    increasingly negative further away. Use with a positive weight to pull
    the trajectory toward desired_z.

    Args:
        height (float): target world-frame z-coordinate (metres).
    """

    def __init__(self, height: float):
        self.desired_z = height

    def __call__(self, positions: np.ndarray, _quats: np.ndarray) -> np.ndarray:
        dz = positions[:, 2] - self.desired_z
        return -(dz * dz)

    def gradient(self, positions, quats):
        n = len(positions)
        grad_pos = np.zeros_like(positions)
        grad_pos[:, 2] = -2.0 * (positions[:, 2] - self.desired_z) / n
        return grad_pos, np.zeros_like(quats)


class HeightThreshold(Feature):
    """
    Sigmoid penalty that rises from 0 to 1 as the EEF height crosses a threshold.

        σ(z - threshold) ≈ 0 when z << threshold (below)
                         ≈ 1 when z >> threshold (above)

    Use with a positive weight to encourage staying above the threshold (reward
    being high), or with a negative weight to penalise going below it.

    Args:
        height    (float): z-coordinate of the threshold (metres).
        sharpness (float): steepness of the sigmoid (default 20.0).
    """

    def __init__(self, height: float, sharpness: float = 20.0):
        self.threshold = height
        self.sharpness = sharpness

    def __call__(self, positions: np.ndarray, _quats: np.ndarray) -> np.ndarray:
        z = positions[:, 2]
        return 1.0 / (1.0 + np.exp(-self.sharpness * (z - self.threshold)))

    def gradient(self, positions, quats):
        n = len(positions)
        z     = positions[:, 2]
        sigma = 1.0 / (1.0 + np.exp(-self.sharpness * (z - self.threshold)))
        grad_pos = np.zeros_like(positions)
        grad_pos[:, 2] = sigma * (1.0 - sigma) * self.sharpness / n
        return grad_pos, np.zeros_like(quats)


class ZTableDistance(Feature):
    """
    Per-waypoint signed distance above the table surface: z - table_z.

    Use with a positive weight to encourage the arm to stay high above the
    table, or a negative weight to encourage staying close to it.

    Args:
        table_z (float): world-frame z-height of the table surface.
    """

    def __init__(self, table_z: float):
        self.table_z = table_z

    def __call__(self, positions: np.ndarray, _quats: np.ndarray) -> np.ndarray:
        return positions[:, 2] - self.table_z

    def gradient(self, positions, quats):
        n = len(positions)
        grad_pos = np.zeros_like(positions)
        grad_pos[:, 2] = 1.0 / n
        return grad_pos, np.zeros_like(quats)


class BinYAlignment(Feature):
    """
    Encourages the trajectory to align with a target bin's y-coordinate,
    with optional extra weight on early timesteps for earlier commitment.

    Per-waypoint value: -weight_i * (y_i - y_bin)²

    where weight_i = 1 + early_weight * (1 - i / (k-1))

    At i=0:   weight = 1 + early_weight  (highest pressure)
    At i=k-1: weight = 1.0               (base pressure)

    Setting early_weight=0 gives uniform total y-alignment.
    Higher early_weight front-loads the commitment to the target bin's y-lane.

    Use with a positive weight.

    Args:
        y_bin        (float): target bin's y-coordinate.
        early_weight (float): extra multiplier on early timesteps (default 1.0).
    """

    def __init__(self, y_bin: float, early_weight: float = 1.0):
        self.y_bin        = y_bin
        self.early_weight = early_weight

    def _timestep_weights(self, k: int) -> np.ndarray:
        return 1.0 + self.early_weight * np.linspace(1.0, 0.0, k)

    def __call__(self, positions: np.ndarray, _quats: np.ndarray) -> np.ndarray:
        k      = len(positions)
        y_dev  = (positions[:, 1] - self.y_bin) ** 2
        return -self._timestep_weights(k) * y_dev

    def gradient(self, positions, quats):
        n      = len(positions)
        w      = self._timestep_weights(n)
        y_dev  = positions[:, 1] - self.y_bin
        grad_pos = np.zeros_like(positions)
        grad_pos[:, 1] = -2.0 * w * y_dev / n
        return grad_pos, np.zeros_like(quats)


class BinWallAvoidance(Feature):
    """
    Soft penalty for trajectories that pass through bin walls.

    For each waypoint the penalty is the product of three sigmoid terms:
        inside_x * inside_y * below_wall

    Each term is near 1 when the waypoint is inside the bin footprint
    laterally AND below the wall top — i.e. when it would collide with a wall.
    The penalty is near 0 outside the footprint or above the walls.

    Summed over all bins, so any bin can trigger the penalty.

    Use with a NEGATIVE weight so the optimizer avoids wall regions.

    Args:
        bin_xys    : list of (bx, by) bin centre coordinates.
        ox         : bin half-extent in x (BinTableArena.OX).
        oy         : bin half-extent in y (BinTableArena.OY).
        wall_top_z : z-height of the top of the bin walls.
        sharpness  : sigmoid steepness (default 20.0, ≈ sharp over ~0.1 m).
    """

    def __init__(self, bin_xys, ox: float, oy: float, wall_top_z: float,
                 sharpness: float = 20.0):
        self.bin_xys    = [(float(bx), float(by)) for bx, by in bin_xys]
        self.ox         = ox
        self.oy         = oy
        self.wall_top_z = wall_top_z
        self.s          = sharpness

    def _sigma(self, u):
        return 1.0 / (1.0 + np.exp(-self.s * u))

    def __call__(self, positions: np.ndarray, _quats: np.ndarray) -> np.ndarray:
        s   = self.s
        penalty = np.zeros(len(positions))
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        below = self._sigma(self.wall_top_z - z)          # near 1 when below wall top
        for bx, by in self.bin_xys:
            in_x = self._sigma(x - (bx - self.ox)) * self._sigma((bx + self.ox) - x)
            in_y = self._sigma(y - (by - self.oy)) * self._sigma((by + self.oy) - y)
            penalty += in_x * in_y * below
        return penalty

    def gradient(self, positions, quats):
        n   = len(positions)
        s   = self.s
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

        grad_pos = np.zeros_like(positions)

        below   = self._sigma(self.wall_top_z - z)
        d_below = -s * below * (1.0 - below)             # d(below)/dz

        for bx, by in self.bin_xys:
            a1 = self._sigma(x - (bx - self.ox))
            a2 = self._sigma((bx + self.ox) - x)
            a3 = self._sigma(y - (by - self.oy))
            a4 = self._sigma((by + self.oy) - y)

            in_x = a1 * a2
            in_y = a3 * a4

            d_in_x = s * a1 * (1 - a1) * a2 - s * a2 * (1 - a2) * a1   # d(in_x)/dx
            d_in_y = s * a3 * (1 - a3) * a4 - s * a4 * (1 - a4) * a3   # d(in_y)/dy

            grad_pos[:, 0] += d_in_x * in_y  * below  / n
            grad_pos[:, 1] += in_x  * d_in_y * below  / n
            grad_pos[:, 2] += in_x  * in_y   * d_below / n

        return grad_pos, np.zeros_like(quats)


def make_max_step_constraint(max_dist: float):
    """
    Hard constraint: consecutive waypoints must be no more than max_dist apart.

    Prevents negative smoothness from creating extreme zigzag patterns while
    still allowing some jerkiness. Set max_dist larger than
    total_path_length / n_waypoints to avoid making straight-line paths infeasible.

    Returns a constraint function compatible with GradientOptimizer (SLSQP mode):
        constraint(positions, quats) -> np.ndarray of shape (k-1,)
        Values <= 0 mean the constraint is satisfied.

    Args:
        max_dist (float): maximum allowed distance between consecutive waypoints (metres).
    """
    def constraint(positions, _quats):
        diffs = np.diff(positions, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        return dists - max_dist   # <= 0 when step is small enough
    return constraint


def make_table_constraint(table_z: float):
    """
    Hard constraint: all waypoints must be at or above the table surface.

    Returns a constraint function compatible with GradientOptimizer (SLSQP mode):
        constraint(positions, quats) -> np.ndarray of shape (k,)
        Values <= 0 mean the constraint is satisfied.

    Args:
        table_z (float): world-frame z-height of the table surface.
    """
    def constraint(positions, _quats):
        return table_z - positions[:, 2]   # <= 0 when above table
    return constraint


def make_workspace_constraint(center, radius: float):
    """
    Hard constraint: all waypoints must lie within a sphere of the given radius.

    Intended to keep the optimizer within the robot's reachable workspace.
    For the Panda in the BinPlacing setup, use:

        center = np.array([-0.66, 0.0, 0.912])   # robot0_base / link0 world position
        radius = 0.855                             # Panda max reach in metres

    Returns a constraint function compatible with GradientOptimizer (SLSQP mode):
        constraint(positions, quats) -> np.ndarray of shape (k,)
        Values <= 0 mean the constraint is satisfied.

    Args:
        center (array-like, 3): world-frame centre of the reachable sphere.
        radius (float):         maximum allowable distance from centre (metres).
    """
    center = np.asarray(center, dtype=float)

    def constraint(positions, _quats):
        dists = np.linalg.norm(positions - center, axis=1)   # (k,)
        return dists - radius                                  # <= 0 when within reach

    return constraint


class GoalProgress(Feature):
    """
    Rewards the trajectory for approaching the goal position quickly.

    Returns -(weight_i * distance_to_goal) per waypoint, where weights decrease
    linearly from early_weight at the first waypoint to 1.0 at the last. This
    penalises lingering far from the goal early in the trajectory.

    Use with a positive weight to pull the trajectory toward goal_pos as soon
    as possible.

    Args:
        goal_pos     (array-like, 3): target world-frame position (the release point).
        early_weight (float):         weight multiplier at the first waypoint (default 3.0).
    """

    def __init__(self, goal_pos, early_weight: float = 3.0):
        self.goal_pos    = np.asarray(goal_pos, dtype=float)
        self.early_weight = early_weight

    def __call__(self, positions: np.ndarray, _quats: np.ndarray) -> np.ndarray:
        n       = len(positions)
        dists   = np.linalg.norm(positions[:, :2] - self.goal_pos[:2], axis=1)
        weights = np.linspace(self.early_weight, 1.0, n)
        return -weights * dists

    def gradient(self, positions, quats):
        n       = len(positions)
        diff_xy = positions[:, :2] - self.goal_pos[:2]     # (n, 2)
        dists   = np.linalg.norm(diff_xy, axis=1)          # (n,)
        safe_d  = np.maximum(dists, 1e-8)
        weights = np.linspace(self.early_weight, 1.0, n)
        coeff   = -weights / (safe_d * n)
        grad_pos = np.zeros_like(positions)
        grad_pos[:, :2] = coeff[:, None] * diff_xy
        return grad_pos, np.zeros_like(quats)


def make_bin_wall_constraint(bin_xys, ox: float, oy: float, wall_top_z: float,
                             margin: float = 0.0, wall_thickness: float = 0.008):
    """
    Hard constraint: waypoints over a bin WALL (not the interior) must be above the wall top.

    The constraint fires only in the annular region between the outer footprint and
    the inner floor footprint, so the goal waypoint can descend into the bin interior
    without being blocked.

    Returns a constraint function compatible with GradientOptimizer (SLSQP mode):
        constraint(positions, quats) -> np.ndarray
        Values <= 0 mean the constraint is satisfied.

    Args:
        bin_xys        : list of (bx, by) bin centre coordinates.
        ox             : bin outer half-extent in x.
        oy             : bin outer half-extent in y.
        wall_top_z     : z-height of the bin wall tops.
        margin         : extra clearance added to wall_top_z (default 0.0).
        wall_thickness : half-thickness of bin walls used to define the inner
                         footprint (default 0.008 m = BinTableArena.WT).
    """
    z_min  = wall_top_z + margin
    ox_in  = ox - 2 * wall_thickness   # inner half-extent x
    oy_in  = oy - 2 * wall_thickness   # inner half-extent y

    def constraint(positions, _quats):
        out = np.full(len(positions), -1.0)   # -1 = satisfied with margin
        for i, pos in enumerate(positions):
            x, y, z = pos
            for bx, by in bin_xys:
                in_outer = (bx - ox)    <= x <= (bx + ox)    and (by - oy)    <= y <= (by + oy)
                in_inner = (bx - ox_in) <= x <= (bx + ox_in) and (by - oy_in) <= y <= (by + oy_in)
                if in_outer and not in_inner:   # over a wall, not the floor
                    out[i] = max(out[i], z_min - z)
        return out

    return constraint


def combine_constraints(*constraint_fns):
    """
    Combine multiple constraint functions into one for use with GradientOptimizer.

    Each constraint function must have the signature:
        fn(positions, quats) -> np.ndarray  (values <= 0 when satisfied)

    The combined function concatenates all outputs into a single array.

    Example:
        workspace = make_workspace_constraint(center, radius=0.855)
        walls     = make_bin_wall_constraint(BIN_XY, OX, OY, BIN_WALL_TOP)
        constraint = combine_constraints(workspace, walls)

        waypoints = opt.optimize_trajectory(..., constraints=constraint)
    """
    def combined(positions, quats):
        return np.concatenate([fn(positions, quats) for fn in constraint_fns])
    return combined
