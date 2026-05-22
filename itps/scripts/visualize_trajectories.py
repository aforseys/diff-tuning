"""
Visualize optimized spline trajectories in the BinPlacing environment.

Loads a trajectories .npz (output of generate_trajectories.py) and renders
waypoints as capsule tubes baked into the MuJoCo scene, one color per target
bin. Filters by prism, bin, combo, and goal-obs are all optional; --max-trajs
subsamples the result for readability.

With --execute the robot physically follows each trajectory one by one while
the tubes remain visible in the scene.

Run from the itps/ directory:
    conda run -n diffpreff python scripts/visualize_trajectories.py \\
        --traj-file data/trajs.npz --max-trajs 20
    # execute each trajectory with tubes visible:
    conda run -n diffpreff python scripts/visualize_trajectories.py \\
        --traj-file data/trajs.npz --max-trajs 5 --execute
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scripts.bin_placing import BinPlacing, BinTableArena, OBJECT_MAP, make_env, execute_spline


# One RGBA color per bin — semi-transparent so tubes don't obscure the scene.
SPLINE_COLORS = [
    [0.90, 0.15, 0.15, 0.55],   # bin 0 — red
    [0.15, 0.75, 0.15, 0.55],   # bin 1 — green
    [0.15, 0.15, 0.90, 0.55],   # bin 2 — blue
    [0.90, 0.55, 0.05, 0.55],   # bin 3 — orange
]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--traj-file",    required=True,
                        help="Trajectories .npz from generate_trajectories.py")
    parser.add_argument("--max-trajs",    type=int, default=20,
                        help="Max trajectories to display (default 20)")
    parser.add_argument("--prism",        type=int, default=None,
                        help="Filter: only show trajectories from this prism index")
    parser.add_argument("--bin",          type=int, default=None, choices=[0, 1, 2, 3],
                        help="Filter: only show trajectories going to this bin (0-3)")
    parser.add_argument("--combo",        type=int, default=None,
                        help="Filter: only show trajectories with this reward combo index")
    parser.add_argument("--show-samples", action="store_true",
                        help="Also render start/goal sample dots from the saved observations")
    parser.add_argument("--show-waypoints", action="store_true",
                        help="Render raw optimizer waypoints as yellow spheres (useful for debugging constraint violations)")
    parser.add_argument("--no-spline", action="store_true",
                        help="Skip rendering the spline tubes (useful when --show-waypoints is set)")
    parser.add_argument("--execute",      action="store_true",
                        help="Execute each trajectory one by one with tubes visible")
    parser.add_argument("--horizon",      type=int, default=150,
                        help="Keyframe steps per trajectory when --execute is set (default 64)")
    parser.add_argument("--camera",       default="f", choices=["f", "o", "s"],
                        help="Camera: f=front (default), o=overhead, s=side")
    parser.add_argument("--object",       default="can",
                        choices=list(OBJECT_MAP.keys()),
                        help="Object type (default: can)")
    args = parser.parse_args()

    # --- Load trajectories ---
    data      = np.load(args.traj_file, allow_pickle=False)
    waypoints = data["waypoints"]       # (N, n_pts, 3)
    prism_idx = data["prism_idx"]
    bin_idx   = data["bin_idx"]
    combo_idx = data["combo_idx"]

    # --- Filter ---
    mask = np.ones(len(waypoints), dtype=bool)
    if args.prism is not None:
        mask &= (prism_idx == args.prism)
    if args.bin is not None:
        mask &= (bin_idx == args.bin)
    if args.combo is not None:
        mask &= (combo_idx == args.combo)

    indices = np.where(mask)[0]
    if len(indices) == 0:
        raise ValueError("No trajectories match the given filters.")

    if len(indices) > args.max_trajs:
        indices = np.random.default_rng(0).choice(indices, size=args.max_trajs,
                                                   replace=False)
        indices.sort()

    selected_waypoints = waypoints[indices]
    selected_bins      = bin_idx[indices]
    colors = [SPLINE_COLORS[b] for b in selected_bins]

    print(f"Showing {len(indices)} trajectories "
          f"(filtered from {mask.sum()} of {len(waypoints)} total)")
    bin_counts = {b: int((selected_bins == b).sum()) for b in range(4)}
    print(f"  by bin: { {f'bin{k}': v for k, v in bin_counts.items() if v > 0} }")

    camera = {"f": "frontview", "o": "birdview", "s": "sideview"}[args.camera]

    # Execute mode needs OSC_POSE; static view uses JOINT_POSITION.
    controller = "OSC_POSE" if args.execute else "JOINT_POSITION"
    env = make_env(has_renderer=True, camera=camera,
                   mujoco_object=OBJECT_MAP[args.object],
                   controller=controller)

    if not args.no_spline:
        env._spline_data = (list(selected_waypoints), colors)

    if args.show_samples:
        sp = data["start_pos"]
        gp = data["goal_pos"]
        pi = data["prism_idx"]
        bi = data["bin_idx"]
        n_p = int(pi.max()) + 1
        n_b = int(bi.max()) + 1
        # Reconstruct list-of-arrays expected by _bake_sample_markers
        start_positions = [sp[(pi == p) & (bi == 0)] for p in range(n_p)]
        goal_positions  = [gp[(bi == b) & (pi == 0)] for b in range(n_b)]
        env._marker_data = (start_positions, goal_positions)

    if args.show_waypoints:
        env._waypoint_data = (list(selected_waypoints), colors)

    env.reset()

    if not args.execute:
        # Static view — just hold and render.
        print("Rendering — close the window to exit.")
        action = np.zeros(env.action_spec[0].shape)
        action[-1] = 1.0
        while True:
            env.step(action)
            env.render()

    else:
        # Execute each trajectory in turn, tubes stay visible throughout.
        print(f"Executing {len(indices)} trajectories — close the window to exit.")
        for i, (wp, goal_bin) in enumerate(zip(selected_waypoints, selected_bins)):
            print(f"  [{i+1}/{len(indices)}] bin {goal_bin}")
            env.reset()   # hard reset rebakes tubes + resets robot
            execute_spline(env, wp, horizon=args.horizon, record=False)

            # Brief pause so you can see the result before next reset.
            action = np.zeros(env.action_spec[0].shape)
            action[-1] = -1.0   # open gripper
            for _ in range(100):  # gripper open phase, matches collect_demos default
                env.step(action)
                env.render()


if __name__ == "__main__":
    main()
