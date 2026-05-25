"""
Collect diffusion-policy demonstrations by executing optimized spline trajectories.

Loads a trajectories .npz (from generate_trajectories.py), executes each spline
in the BinPlacing environment using OSC_POSE delta control, and saves all
observations and actions to an HDF5 file.

Two action representations are saved (both computed from consecutive snapshots):
    delta_eef   : (T, 4)  [dx, dy, dz, gripper]
    delta_joint : (T, 8)  [dj1..dj7, gripper]

Observations saved at each keyframe (T+1 total per episode):
    eef_pos, eef_quat, joint_pos, cube_pos, goal_pos,
    agentview_image, wrist_image (optional)

Run from the itps/ directory:
    conda run -n diffpreff python scripts/collect_demos.py \\
        --traj-file data/trajs.npz --save-path data/demos.hdf5
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np
from scipy.interpolate import CubicSpline, interp1d

from scripts.bin_placing import BinPlacing, OBJECT_MAP
from robosuite.controllers import load_part_controller_config


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def make_collection_env(img_size=84, use_wrist=True, has_renderer=False,
                         mujoco_object=None):
    cameras = ["agentview"]
    if use_wrist:
        cameras.append("robot0_eye_in_hand")

    part = load_part_controller_config(default_controller="OSC_POSE")
    part["gripper"] = {"type": "GRIP"}
    controller_cfg = {"type": "BASIC", "body_parts": {"right": part}}

    return BinPlacing(
        robots="Panda",
        controller_configs=controller_cfg,
        mujoco_object=mujoco_object,
        has_renderer=has_renderer,
        has_offscreen_renderer=True,
        render_camera="agentview",
        use_camera_obs=True,
        use_object_obs=True,
        reward_shaping=True,
        horizon=5000,
        control_freq=20,
        ignore_done=True,
        hard_reset=True,
        camera_names=cameras,
        camera_heights=img_size,
        camera_widths=img_size,
    )


# ---------------------------------------------------------------------------
# Episode collection
# ---------------------------------------------------------------------------

def _snapshot(obs, use_wrist):
    snap = {
        "eef_pos":         obs["robot0_eef_pos"].copy(),
        "eef_quat":        obs["robot0_eef_quat"].copy(),
        "joint_pos":       obs["robot0_joint_pos"].copy(),
        "cube_pos":        obs["cube_pos"].copy(),
        "agentview_image": obs["agentview_image"].copy(),
    }
    if use_wrist:
        snap["wrist_image"] = obs["robot0_eye_in_hand_image"].copy()
    return snap


def collect_episode(env, init_obs, waypoints, goal_xyz, target_bin, record_every=5,
                    gripper_steps=30, use_wrist=True, n_spline_pts=200):
    """
    Execute a spline and record observations at a fixed physics-step rate.

    Snapshots are taken every `record_every` physics steps, so longer
    trajectories naturally produce more observations.  The spline is
    interpolated to `n_spline_pts` targets purely for path-following
    resolution; recording is decoupled from that count.

    Returns:
        snapshots   : list of T+1 snapshot dicts (initial + one per record)
        gripper_seq : list of T gripper values (1.0 closed, -1.0 open)
        success     : bool
    """
    waypoints  = np.asarray(waypoints, dtype=float)
    action_dim = env.action_spec[0].shape[0]

    # Arc-length parameterised spline for smooth path following
    seg_len  = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    arc      = np.concatenate([[0.0], np.cumsum(seg_len)])
    arc_norm = arc / arc[-1]
    if len(waypoints) >= 4:
        targets = CubicSpline(arc_norm, waypoints)(np.linspace(0, 1, n_spline_pts))
    else:
        targets = interp1d(arc_norm, waypoints, axis=0)(np.linspace(0, 1, n_spline_pts))

    snapshots   = [_snapshot(init_obs, use_wrist)]
    gripper_seq = []
    obs         = init_obs
    step_count  = 0

    def _step(action, gripper_val):
        nonlocal obs, step_count
        obs, _, _, _ = env.step(action)
        step_count += 1
        if env.has_renderer:
            env.render()
        if step_count % record_every == 0:
            snapshots.append(_snapshot(obs, use_wrist))
            gripper_seq.append(gripper_val)

    # --- Spline phase: gripper closed ---
    goal_xyz = targets[-1]
    for target_xyz in targets:
        if np.linalg.norm(goal_xyz - env._eef_pos()) < 0.010:
            break

        moved = False
        for _ in range(20):
            eef   = env._eef_pos()
            delta = target_xyz - eef
            if np.linalg.norm(delta) < 0.010:
                break
            action     = np.zeros(action_dim)
            action[:3] = np.clip(delta / 0.10, -1.0, 1.0)
            action[-1] = 1.0
            _step(action, 1.0)
            moved = True

        if not moved:
            action     = np.zeros(action_dim)
            action[-1] = 1.0
            _step(action, 1.0)

    # --- Gripper open phase ---
    for _ in range(gripper_steps):
        action     = np.zeros(action_dim)
        action[-1] = -1.0
        _step(action, -1.0)

    return {
        "snapshots":   snapshots,
        "gripper_seq": gripper_seq,
        "goal_xyz":    goal_xyz,
        "success":     env.placement_success(target_bin),
    }


def episode_to_arrays(episode):
    """Convert raw episode dict to numpy arrays ready for HDF5 storage."""
    snaps    = episode["snapshots"]              # length T+1
    gripper  = np.array(episode["gripper_seq"])  # length T
    goal_xyz = episode["goal_xyz"]               # (3,)

    eef_pos   = np.array([s["eef_pos"]   for s in snaps])   # (T+1, 3)
    eef_quat  = np.array([s["eef_quat"]  for s in snaps])   # (T+1, 4)
    joint_pos = np.array([s["joint_pos"] for s in snaps])   # (T+1, 7)
    cube_pos  = np.array([s["cube_pos"]  for s in snaps])   # (T+1, 3)
    agentview = np.array([s["agentview_image"] for s in snaps])  # (T+1, H, W, 3)

    # Delta actions computed from consecutive keyframe positions
    delta_eef_xyz = np.diff(eef_pos,   axis=0)   # (T, 3)
    delta_jnt     = np.diff(joint_pos, axis=0)   # (T, 7)

    delta_eef   = np.concatenate([delta_eef_xyz, gripper[:, None]], axis=1)  # (T, 4)
    delta_joint = np.concatenate([delta_jnt,     gripper[:, None]], axis=1)  # (T, 8)

    arrays = {
        "obs/eef_pos":         eef_pos,
        "obs/eef_quat":        eef_quat,
        "obs/joint_pos":       joint_pos,
        "obs/cube_pos":        cube_pos,
        "obs/goal_pos":        np.tile(goal_xyz, (len(snaps), 1)),  # (T+1, 3)
        "obs/agentview_image": agentview,
        "action/delta_eef":    delta_eef,
        "action/delta_joint":  delta_joint,
    }
    if "wrist_image" in snaps[0]:
        arrays["obs/wrist_image"] = np.array([s["wrist_image"] for s in snaps])

    return arrays, episode["success"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--traj-file",      required=True,
                        help="Input trajectories .npz")
    parser.add_argument("--save-path",      required=True,
                        help="Output HDF5 file path")
    parser.add_argument("--record-every",   type=int, default=2,
                        help="Record a snapshot every N physics steps (default 2, gives 10 Hz at 20 Hz control)")
    parser.add_argument("--n-spline-pts",   type=int, default=150,
                        help="Spline interpolation points for path following (default 150)")
    parser.add_argument("--gripper-steps",  type=int, default=25,
                        help="Gripper-open steps after reaching goal (default 25, ~1.25s at 20 Hz)")
    parser.add_argument("--img-size",       type=int, default=84,
                        help="Camera image resolution (default 84)")
    parser.add_argument("--no-wrist",       action="store_true",
                        help="Disable wrist camera")
    parser.add_argument("--render",         action="store_true",
                        help="Show renderer during collection")
    parser.add_argument("--object",         default="can",
                        choices=list(OBJECT_MAP.keys()),
                        help="Object type (default: can)")
    parser.add_argument("--n-demos",        type=int, default=None,
                        help="Max demos to collect (default: all)")

    # Optional filters (same as visualize_trajectories.py)
    parser.add_argument("--prism",  type=int, default=None)
    parser.add_argument("--bin",    type=int, default=None, choices=[0, 1, 2, 3])
    parser.add_argument("--combo",  type=int, default=None)

    args = parser.parse_args()

    # --- Load and filter ---
    data         = np.load(args.traj_file, allow_pickle=False)
    waypoints    = data["waypoints"]    # (N, n_pts, 3)
    start_pos    = data["start_pos"]    # (N, 3)
    goal_pos     = data["goal_pos"]     # (N, 3)
    bin_idx      = data["bin_idx"]      # (N,)
    prism_idx    = data["prism_idx"]    # (N,)
    combo_idx    = data["combo_idx"]    # (N,)
    if "start_joints" not in data:
        raise ValueError(
            "Trajectory file is missing 'start_joints'. "
            "Re-generate observations with --solve-ik and re-run generate_trajectories.py."
        )
    start_joints = data["start_joints"]  # (N, n_joints)

    mask = np.ones(len(waypoints), dtype=bool)
    if args.prism is not None:
        mask &= (prism_idx == args.prism)
    if args.bin is not None:
        mask &= (bin_idx == args.bin)
    if args.combo is not None:
        mask &= (combo_idx == args.combo)

    indices = np.where(mask)[0]
    if args.n_demos is not None:
        indices = indices[:args.n_demos]

    n_demos   = len(indices)
    use_wrist = not args.no_wrist
    config_json = str(data["config_json"]) if "config_json" in data else "{}"

    print(f"Collecting {n_demos} demos")
    print(f"  record_every={args.record_every}  n_spline_pts={args.n_spline_pts}  gripper_steps={args.gripper_steps}")
    print(f"  cameras: agentview{' + wrist' if use_wrist else ''}  "
          f"img_size={args.img_size}")

    # --- Build env once and reuse ---
    env = make_collection_env(
        img_size=args.img_size,
        use_wrist=use_wrist,
        has_renderer=args.render,
        mujoco_object=OBJECT_MAP[args.object],
    )
    n_joints = env.robots[0].dof

    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)

    n_success = 0
    with h5py.File(args.save_path, "w") as f:
        f.attrs["n_demos"]     = n_demos
        f.attrs["object_type"] = args.object
        f.attrs["config_json"] = config_json
        data_grp = f.create_group("data")

        for demo_num, traj_i in enumerate(indices):
            wp       = waypoints[traj_i]          # (n_pts, 3)
            goal_xyz = goal_pos[traj_i]           # (3,)

            init_obs = env.reset()

            # Teleport arm to saved start joint configuration.
            env.sim.data.qpos[:n_joints] = start_joints[traj_i]
            env.sim.data.qvel[:n_joints] = 0.0
            env.sim.forward()
            # Re-place the can at the new EEF position so it is still grasped
            # (_reset_internal placed it at the rest-pose EEF, which is now wrong).
            eef_pos = env._eef_pos()
            env.sim.data.set_joint_qpos(
                env.grasp_obj.joints[0],
                np.concatenate([eef_pos, np.array([1, 0, 0, 0])])
            )
            env.sim.forward()
            zero_act = np.zeros(env.action_spec[0].shape)
            zero_act[-1] = 1.0   # gripper closed
            init_obs, _, _, _ = env.step(zero_act)

            if demo_num < 5:
                actual_eef = init_obs["robot0_eef_pos"]
                err = np.linalg.norm(actual_eef - start_pos[traj_i])
                print(f"  [demo {demo_num}] start_pos={start_pos[traj_i].round(3)}  "
                      f"eef={actual_eef.round(3)}  err={err:.4f} m")

            t0 = time.perf_counter()
            episode = collect_episode(
                env, init_obs, wp, goal_xyz=goal_xyz, target_bin=int(bin_idx[traj_i]),
                record_every=args.record_every,
                gripper_steps=args.gripper_steps,
                use_wrist=use_wrist,
                n_spline_pts=args.n_spline_pts,
            )
            elapsed = time.perf_counter() - t0
            arrays, success = episode_to_arrays(episode)
            n_obs = len(episode["snapshots"])
            if demo_num < 5 and not success:
                obj_pos  = env.sim.data.body_xpos[env.obj_body_id]
                loc_ok   = env.location_success(int(bin_idx[traj_i]))
                grip_ok  = env._gripper_is_open()
                total_fj = sum(
                    env.sim.data.qpos[env.sim.model.jnt_qposadr[i]]
                    for i in range(env.sim.model.njnt)
                    if "finger_joint" in env.sim.model.joint_id2name(i)
                )
                print(f"  [demo {demo_num}] FAIL  obj_pos={obj_pos.round(3)}  "
                      f"location={loc_ok}  gripper_open={grip_ok}  finger_qpos_sum={total_fj:.4f}  "
                      f"z_thresh={env.PLACED_Z_THRESHOLD:.3f}")
            if success:
                n_success += 1

            dg = data_grp.create_group(f"demo_{demo_num}")
            dg.attrs["traj_idx"]  = int(traj_i)
            dg.attrs["prism_idx"] = int(prism_idx[traj_i])
            dg.attrs["bin_idx"]   = int(bin_idx[traj_i])
            dg.attrs["combo_idx"] = int(combo_idx[traj_i])
            dg.attrs["success"]   = success

            for key, arr in arrays.items():
                dg.create_dataset(key, data=arr, compression="gzip",
                                  compression_opts=4)

            print(f"  demo {demo_num + 1}/{n_demos}  "
                  f"{elapsed:.1f}s  {n_obs} obs  "
                  f"({'success' if success else 'fail'})")

    print(f"\nSaved → {args.save_path}")
    print(f"Success rate: {n_success}/{n_demos} ({100*n_success/n_demos:.1f}%)")


if __name__ == "__main__":
    main()
