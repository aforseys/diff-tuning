"""
Sanity check for the trajectory optimizer in the BinPlacing world.

Tests both soft (feature) and hard (constraint) wall avoidance approaches.
No renderer needed.

Run from the itps/ directory:
    conda run -n diffpreff python scripts/test_trajectory_opt.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scripts.bin_placing import BinTableArena

from itps.trajectory_opt.gradient_optimizer import GradientOptimizer
from itps.trajectory_opt.geometric_features import (
    Smoothness, MaintainOrientation, HeightThreshold,
    BinWallAvoidance, make_bin_wall_constraint,
    ZTableDistance, BinYAlignment,
)
from itps.trajectory_opt.linear_reward_model import trajectory_reward_arrays, trajectory_reward_grad

# ---------------------------------------------------------------------------
# Scene constants
# ---------------------------------------------------------------------------

TABLE_Z      = 0.8
BIN_WALL_TOP = TABLE_Z + BinTableArena.H * 2
BIN_XY       = BinTableArena.BIN_XY
OX, OY       = BinTableArena.OX, BinTableArena.OY

# EEF start: near arm home
pos_start  = np.array([-0.215,  0.009, 1.007])

# Goal: INSIDE bin 0 — below the wall top
bx0, by0   = BIN_XY[0]
pos_goal   = np.array([bx0, by0, TABLE_Z + 0.02])   # just above bin floor

quat_down  = np.array([1.0, 0.0, 0.0, 0.0])
quat_start = quat_down.copy()
quat_goal  = quat_down.copy()

opt = GradientOptimizer(max_iter=300)


def report(label, waypoints, elapsed):
    positions = np.array([wp["pos"] for wp in waypoints])
    all_pos   = np.vstack([pos_start, positions, pos_goal])
    min_z     = all_pos[:, 2].min()

    # Count intermediate waypoints inside a bin footprint AND below wall top.
    # Excludes endpoints — the goal is intentionally inside the bin below wall top.
    violations = 0
    for pos in positions:   # intermediate only
        x, y, z = pos
        for bx, by in BIN_XY:
            if (bx - OX) <= x <= (bx + OX) and (by - OY) <= y <= (by + OY):
                if z < BIN_WALL_TOP:
                    violations += 1

    print(f"\n{'='*50}")
    print(f"{label}")
    print(f"  {elapsed*1000:.1f} ms   min_z={min_z:.4f}   wall violations={violations}")
    print(f"  path (z only): {' '.join(f'{p[2]:.3f}' for p in all_pos)}")


# ---------------------------------------------------------------------------
# Option A — Soft avoidance (BinWallAvoidance feature, L-BFGS-B)
# also exercises ZTableDistance and BinYAlignment
# ---------------------------------------------------------------------------

features_soft = [
    Smoothness(),
    MaintainOrientation(quat_down),
    BinWallAvoidance(BIN_XY, OX, OY, BIN_WALL_TOP),
    ZTableDistance(TABLE_Z),
    BinYAlignment(y_bin=by0, early_weight=2.0),
]
weights_soft = [
    2.0,    # smoothness
    1.0,    # keep gripper level
   -5.0,    # penalise wall regions
    1.0,    # reward height above table
    1.0,    # y-alignment with target bin, front-loaded
]

def reward_soft(positions, quats):
    return trajectory_reward_arrays(positions, quats, features_soft, weights_soft)

def jac_soft(positions, quats):
    return trajectory_reward_grad(positions, quats, features_soft, weights_soft)

wps_soft, t_soft = opt.optimize_trajectory(
    pos_start, pos_goal, quat_start, quat_goal,
    reward_soft, initial_waypoints=20, jac=jac_soft, return_time=True,
)
report("Soft avoidance (L-BFGS-B)", wps_soft, t_soft)


# ---------------------------------------------------------------------------
# Option B — Hard constraint (make_bin_wall_constraint, SLSQP)
# ---------------------------------------------------------------------------

features_hard = [
    Smoothness(),
    MaintainOrientation(quat_down),
]
weights_hard = [2.0, 1.0]

wall_constraint = make_bin_wall_constraint(BIN_XY, OX, OY, BIN_WALL_TOP, margin=0.01)

def reward_hard(positions, quats):
    return trajectory_reward_arrays(positions, quats, features_hard, weights_hard)

def jac_hard(positions, quats):
    return trajectory_reward_grad(positions, quats, features_hard, weights_hard)

wps_hard, t_hard = opt.optimize_trajectory(
    pos_start, pos_goal, quat_start, quat_goal,
    reward_hard, initial_waypoints=20, jac=jac_hard,
    constraints=wall_constraint, return_time=True,
)
report("Hard constraint (SLSQP)", wps_hard, t_hard)
