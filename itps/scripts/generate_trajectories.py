"""
Generate optimized waypoint trajectories for each (start, bin, observation) triple.

Reads a .npz file produced by sample_generation / save_observations and runs
GradientOptimizer on each (prism, bin, obs) pair. The i-th start observation is
paired with the i-th goal observation for each bin.

Multiple weights per feature are supported (e.g. --smoothness -1 0 1). The
Cartesian product of all weight lists defines the reward combinations. Observations
within each (prism, bin) group are split evenly across combinations; an error is
raised if the count is not evenly divisible.

Output .npz  (N = n_prisms * n_bins * n_obs):
    waypoints      : (N, n_wpts+2, 3)         full path including endpoints
    prism_idx      : (N,)                      which start prism
    start_obs_idx  : (N,)                      observation index within that prism
    bin_idx        : (N,)                      which goal bin (0-3)
    goal_obs_idx   : (N,)                      observation index within that bin
    combo_idx      : (N,)                      which reward combination
    combo_weights  : (n_combos, n_features)    weight per feature per combo
    start_positions: (n_prisms, n_obs, 3)      from input file
    goal_positions : (n_bins,   n_obs, 3)      from input file
    config_json    : string                    full feature/weight/constraint spec

Run from the itps/ directory:
    conda run -n diffpreff python scripts/generate_trajectories.py \\
        --obs-file data/obs.npz --save-path data/trajs.npz \\
        --smoothness -1 0 1 --bin-wall-avoidance -5 \\
        --workspace-constraint --bin-wall-constraint
"""

import argparse
import json
import os
import sys
from itertools import product as cart_product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scripts.bin_placing import BinTableArena
from itps.trajectory_opt.gradient_optimizer import GradientOptimizer
from itps.trajectory_opt.geometric_features import (
    Smoothness,
    Jerk,
    MaintainOrientation,
    HeightThreshold,
    DesiredHeight,
    ZTableDistance,
    BinYAlignment,
    BinWallAvoidance,
    GoalProgress,
    make_max_step_constraint,
    make_table_constraint,
    make_workspace_constraint,
    make_bin_wall_constraint,
    combine_constraints,
)
from itps.trajectory_opt.linear_reward_model import (
    trajectory_reward_arrays,
    trajectory_reward_grad,
)

# ---------------------------------------------------------------------------
# Scene constants
# ---------------------------------------------------------------------------

TABLE_Z      = 0.8
BIN_WALL_TOP = TABLE_Z + BinTableArena.H * 2
BIN_XY       = BinTableArena.BIN_XY
OX, OY       = BinTableArena.OX, BinTableArena.OY
BIN_WT       = BinTableArena.WT
ROBOT_BASE   = np.array([-0.66, 0.0, 0.912])
QUAT_DOWN    = np.array([1.0, 0.0, 0.0, 0.0])   # gripper pointing down (xyzw)

# Ordered names for the weight Cartesian product — must match argparse dest names.
FEATURE_NAMES = [
    "smoothness",
    "jerk",
    "maintain_orientation",
    "height_threshold",
    "desired_height",
    "z_table_distance",
    "bin_wall_avoidance",
    "bin_y_alignment",
    "goal_progress",
]


# ---------------------------------------------------------------------------
# Feature / constraint builders
# ---------------------------------------------------------------------------

def build_features_and_weights(args, target_bin_x: float, target_bin_y: float,
                               weight_combo: dict):
    """Return (features, weights) for one (prism, bin, combo) triple."""
    features, weights = [], []

    w = weight_combo["smoothness"]
    if w != 0.0:
        features.append(Smoothness())
        weights.append(w)

    w = weight_combo["jerk"]
    if w != 0.0:
        features.append(Jerk())
        weights.append(w)

    w = weight_combo["maintain_orientation"]
    if w != 0.0:
        features.append(MaintainOrientation(QUAT_DOWN))
        weights.append(w)

    w = weight_combo["height_threshold"]
    if w != 0.0:
        features.append(HeightThreshold(height=args.height_threshold_z))
        weights.append(w)

    w = weight_combo["desired_height"]
    if w != 0.0:
        features.append(DesiredHeight(height=args.desired_height_z))
        weights.append(w)

    w = weight_combo["z_table_distance"]
    if w != 0.0:
        features.append(ZTableDistance(table_z=TABLE_Z))
        weights.append(w)

    w = weight_combo["bin_wall_avoidance"]
    if w != 0.0:
        features.append(BinWallAvoidance(BIN_XY, OX, OY, BIN_WALL_TOP))
        weights.append(w)

    w = weight_combo["bin_y_alignment"]
    if w != 0.0:
        features.append(BinYAlignment(y_bin=target_bin_y,
                                      early_weight=args.bin_y_alignment_early))
        weights.append(w)

    w = weight_combo["goal_progress"]
    if w != 0.0:
        bin_center = np.array([target_bin_x, target_bin_y, BIN_WALL_TOP])
        features.append(GoalProgress(bin_center,
                                     early_weight=args.goal_progress_early))
        weights.append(w)

    return features, weights


def build_reward_fns(features, weights):
    """Return (reward_fn, jac_fn) closures; jac_fn is None if any feature lacks a gradient."""
    if not features:
        return (lambda pos, q: 0.0), None

    def reward_fn(positions, quats, f=features, w=weights):
        return trajectory_reward_arrays(positions, quats, f, w)

    if all(hasattr(f, "gradient") for f in features):
        def jac_fn(positions, quats, f=features, w=weights):
            return trajectory_reward_grad(positions, quats, f, w)
    else:
        jac_fn = None

    return reward_fn, jac_fn


def build_constraints(args):
    fns = []
    if args.workspace_constraint:
        fns.append(make_workspace_constraint(ROBOT_BASE, radius=args.workspace_radius))
    if args.bin_wall_constraint:
        fns.append(make_bin_wall_constraint(BIN_XY, OX, OY, BIN_WALL_TOP,
                                            margin=args.bin_wall_margin,
                                            wall_thickness=BIN_WT))
    if args.table_constraint:
        fns.append(make_table_constraint(TABLE_Z))
    if args.max_step_constraint:
        fns.append(make_max_step_constraint(args.max_step_dist))
    return combine_constraints(*fns) if fns else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--obs-file",    required=True, help="Input observations .npz")
    parser.add_argument("--save-path",   required=True, help="Output .npz path")
    parser.add_argument("--n-waypoints", type=int, default=20,
                        help="Intermediate waypoints per trajectory (default 20)")
    parser.add_argument("--max-iter",    type=int, default=300,
                        help="Optimizer max iterations (default 300)")

    # Feature weights — nargs='+' allows multiple values for a weight sweep.
    # Cartesian product of all lists defines the reward combinations.
    parser.add_argument("--smoothness",            type=float, nargs="+", default=[0.0],
                        help="Smoothness weight(s) (default 0.0)")
    parser.add_argument("--jerk",                  type=float, nargs="+", default=[0.0],
                        help="Jerk weight(s); positive penalises, negative rewards jerk (default 0)")
    parser.add_argument("--maintain-orientation",  type=float, nargs="+", default=[0.0],
                        help="MaintainOrientation weight(s), gripper-down target (default 0)")
    parser.add_argument("--height-threshold",      type=float, nargs="+", default=[0.0],
                        help="HeightThreshold weight(s) (default 0, disabled)")
    parser.add_argument("--height-threshold-z",    type=float, default=BIN_WALL_TOP + 0.05,
                        help=f"HeightThreshold z value (default {BIN_WALL_TOP + 0.05:.3f})")
    parser.add_argument("--desired-height",        type=float, nargs="+", default=[0.0],
                        help="DesiredHeight weight(s) (default 0, disabled)")
    parser.add_argument("--desired-height-z",      type=float, default=1.00,
                        help="DesiredHeight target z (default 1.00)")
    parser.add_argument("--z-table-distance",      type=float, nargs="+", default=[0.0],
                        help="ZTableDistance weight(s) — reward height above table (default 0)")
    parser.add_argument("--bin-wall-avoidance",    type=float, nargs="+", default=[0.0],
                        help="BinWallAvoidance weight(s); use negative to penalise (default 0)")
    parser.add_argument("--bin-y-alignment",       type=float, nargs="+", default=[0.0],
                        help="BinYAlignment weight(s) toward target bin (default 0)")
    parser.add_argument("--bin-y-alignment-early", type=float, default=1.0,
                        help="BinYAlignment early_weight front-loading (default 1.0)")
    parser.add_argument("--goal-progress",         type=float, nargs="+", default=[0.0],
                        help="GoalProgress weight(s) — reward getting to goal quickly (default 0)")
    parser.add_argument("--goal-progress-early",   type=float, default=3.0,
                        help="GoalProgress early_weight multiplier at first waypoint (default 3.0)")

    # Hard constraints
    parser.add_argument("--workspace-constraint",  action="store_true",
                        help="Enforce reachable-workspace sphere constraint")
    parser.add_argument("--workspace-radius",      type=float, default=0.85,
                        help="Workspace sphere radius in metres (default 0.85)")
    parser.add_argument("--bin-wall-constraint",   action="store_true",
                        help="Enforce hard bin-wall height constraint (SLSQP)")
    parser.add_argument("--bin-wall-margin",       type=float, default=0.09,
                        help="Extra clearance above bin wall top in metres (default 0.09)")
    parser.add_argument("--table-constraint",      action="store_true",
                        help="Enforce hard table-surface floor constraint (z >= table_z)")
    parser.add_argument("--max-step-constraint",   action="store_true",
                        help="Enforce maximum distance between consecutive waypoints")
    parser.add_argument("--max-step-dist",         type=float, default=0.15,
                        help="Max allowed distance between consecutive waypoints in metres (default 0.15)")

    args = parser.parse_args()

    # --- Build reward combinations ---
    weight_lists = [getattr(args, name) for name in FEATURE_NAMES]
    combos       = list(cart_product(*weight_lists))   # list of tuples, len = n_combos
    n_combos     = len(combos)
    combo_weights_arr = np.array(combos, dtype=np.float64)   # (n_combos, n_features)

    # --- Load and validate observations ---
    data            = np.load(args.obs_file)
    start_positions = data["start_positions"]   # (n_prisms, n_obs, 3)
    goal_positions  = data["goal_positions"]    # (n_bins,   n_obs, 3)

    n_prisms, n_obs_start, _ = start_positions.shape
    n_bins,   n_obs_goal,  _ = goal_positions.shape

    assert n_obs_start == n_obs_goal, (
        f"n_obs_start ({n_obs_start}) != n_obs_goal ({n_obs_goal}); "
        "each start must pair 1-to-1 with a goal."
    )
    n_obs = n_obs_start

    assert n_bins == len(BIN_XY), (
        f"goal_positions has {n_bins} bins but BinTableArena has {len(BIN_XY)}."
    )

    if n_obs % n_combos != 0:
        raise ValueError(
            f"n_obs ({n_obs}) is not divisible by n_combos ({n_combos}). "
            f"Adjust weight lists so their Cartesian product size ({n_combos}) "
            f"evenly divides n_obs ({n_obs})."
        )
    obs_per_combo = n_obs // n_combos

    n_wpts = args.n_waypoints
    N      = n_prisms * n_bins * n_obs

    print(f"Loaded:  {n_prisms} prisms × {n_obs} obs,  {n_bins} bins × {n_obs} goals")
    print(f"Reward combos: {n_combos}  ({obs_per_combo} obs each)")
    print(f"Total trajectories: {N}  ({n_wpts} intermediate waypoints each)")
    for ci, combo in enumerate(combos):
        weights_str = "  ".join(f"{name}={w}" for name, w in zip(FEATURE_NAMES, combo)
                                if w != 0.0)
        print(f"  combo {ci}: {weights_str if weights_str else '(no active features)'}")

    opt         = GradientOptimizer(max_iter=args.max_iter)
    constraints = build_constraints(args)

    # --- Output buffers ---
    all_waypoints     = np.zeros((N, n_wpts + 2, 3), dtype=np.float64)
    all_prism_idx     = np.zeros(N, dtype=np.int32)
    all_start_obs_idx = np.zeros(N, dtype=np.int32)
    all_bin_idx       = np.zeros(N, dtype=np.int32)
    all_goal_obs_idx  = np.zeros(N, dtype=np.int32)
    all_combo_idx     = np.zeros(N, dtype=np.int32)

    flat = 0
    for pi in range(n_prisms):
        for bi in range(n_bins):
            bin_x, bin_y = BIN_XY[bi]

            for ci, combo in enumerate(combos):
                weight_combo      = dict(zip(FEATURE_NAMES, combo))
                features, weights = build_features_and_weights(
                    args, bin_x, bin_y, weight_combo,
                )
                reward_fn, jac_fn = build_reward_fns(features, weights)

                obs_start = ci * obs_per_combo
                obs_end   = obs_start + obs_per_combo

                for oi in range(obs_start, obs_end):
                    pos_start = start_positions[pi, oi]
                    pos_goal  = goal_positions[bi, oi]

                    result = opt.optimize_trajectory(
                        pos_start, pos_goal, QUAT_DOWN, QUAT_DOWN,
                        reward_fn,
                        initial_waypoints=n_wpts,
                        jac=jac_fn,
                        constraints=constraints,
                    )

                    path = np.array(
                        [pos_start]
                        + [wp["pos"] for wp in result]
                        + [pos_goal]
                    )
                    all_waypoints[flat]     = path
                    all_prism_idx[flat]     = pi
                    all_start_obs_idx[flat] = oi
                    all_bin_idx[flat]       = bi
                    all_goal_obs_idx[flat]  = oi
                    all_combo_idx[flat]     = ci
                    flat += 1

                    if flat % 500 == 0 or flat == N:
                        print(f"  {flat}/{N}")

    # --- Config record ---
    constraint_specs = []
    if args.workspace_constraint:
        constraint_specs.append({"name": "workspace_sphere",
                                  "center": ROBOT_BASE.tolist(),
                                  "radius": args.workspace_radius})
    if args.bin_wall_constraint:
        constraint_specs.append({"name": "bin_wall_height",
                                  "wall_top_z": BIN_WALL_TOP,
                                  "margin": args.bin_wall_margin})

    config = {
        "obs_file":        args.obs_file,
        "n_waypoints":     n_wpts,
        "max_iter":        args.max_iter,
        "n_combos":        n_combos,
        "obs_per_combo":   obs_per_combo,
        "feature_names":   FEATURE_NAMES,
        "weight_lists":    {name: getattr(args, name) for name in FEATURE_NAMES},
        "combos":          [list(c) for c in combos],
        "feature_params":  {
            "height_threshold_z":     args.height_threshold_z,
            "desired_height_z":       args.desired_height_z,
            "bin_y_alignment_early":  args.bin_y_alignment_early,
            "table_z":                TABLE_Z,
            "bin_wall_top_z":         BIN_WALL_TOP,
            "quat_down":              QUAT_DOWN.tolist(),
        },
        "constraints":     constraint_specs,
    }

    np.savez(
        args.save_path,
        waypoints       = all_waypoints,        # (N, n_wpts+2, 3)
        prism_idx       = all_prism_idx,        # (N,)
        start_obs_idx   = all_start_obs_idx,    # (N,)
        bin_idx         = all_bin_idx,          # (N,)
        goal_obs_idx    = all_goal_obs_idx,     # (N,)
        combo_idx       = all_combo_idx,        # (N,)
        combo_weights   = combo_weights_arr,    # (n_combos, n_features)
        start_positions = start_positions,      # (n_prisms, n_obs, 3)
        goal_positions  = goal_positions,       # (n_bins,   n_obs, 3)
        config_json     = np.bytes_(json.dumps(config, indent=2)),
    )

    print(f"\nSaved → {args.save_path}")
    print(f"waypoints shape: {all_waypoints.shape}")
    print(f"combo_weights:\n{combo_weights_arr}")


if __name__ == "__main__":
    main()
