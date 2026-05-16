import os
import xml.etree.ElementTree as ET
import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config, load_part_controller_config
from scipy.interpolate import CubicSpline, interp1d

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.models.objects import (
    BoxObject, CylinderObject, BallObject,
    CanObject, BottleObject, MilkObject, CerealObject, BreadObject, LemonObject,
)
from robosuite.utils.mjcf_utils import array_to_string
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.transform_utils import convert_quat


# ---------------------------------------------------------------------------
# Arena: white table with 4 blue bin pads
# ---------------------------------------------------------------------------

class BinTableArena(TableArena):
    """TableArena with a white surface and 4 blue bin-zone pads in a 2x2 grid."""

    BIN_RGBA   = [0.25, 0.52, 0.80, 1.0]
    TABLE_RGBA = [0.95, 0.95, 0.95, 1.0]

    # outer half-extents and wall geometry
    OX = 0.14    # bin half-depth  (x, toward robot)
    OY = 0.10    # bin half-width  (y, side-to-side) — short side faces robot
    WT = 0.008   # wall half-thickness
    H  = 0.022   # wall half-height  → ~4.5 cm tall bin
    BT = 0.005   # floor half-thickness

    # Single row of 4 bins on a 1.0 x 1.0 table (half-extents x=0.50, y=0.50):
    #   all at x=0.15; y: 5 equal gaps of 0.04 → centers at ±0.12, ±0.36
    BIN_XY = [
        ( 0.05, -0.36),   # bin 0: left        (~0.72m from base)
        ( 0.05, -0.12),   # bin 1: centre-left (~0.63m from base)
        ( 0.05,  0.12),   # bin 2: centre-right (~0.63m from base)
        ( 0.05,  0.36),   # bin 3: right        (~0.72m from base)
    ]

    def configure_location(self):
        super().configure_location()
        self.table_visual.set("rgba", array_to_string(self.TABLE_RGBA))
        tz = self.table_half_size[2]   # table-top z in table-body local frame
        ox, oy, wt, h, bt = self.OX, self.OY, self.WT, self.H, self.BT

        for i, (bx, by) in enumerate(self.BIN_XY):
            parts = {
                "floor": ([ox,        oy,        bt], [bx,          by,          tz + bt]),
                "wx_pos": ([wt,        oy,        h],  [bx + ox - wt, by,          tz + h]),
                "wx_neg": ([wt,        oy,        h],  [bx - ox + wt, by,          tz + h]),
                "wy_pos": ([ox - 2*wt, wt,        h],  [bx,           by + oy - wt, tz + h]),
                "wy_neg": ([ox - 2*wt, wt,        h],  [bx,           by - oy + wt, tz + h]),
            }
            for part, (size, pos) in parts.items():
                g = ET.SubElement(self.table_body, "geom")
                g.set("name",  f"bin{i}_{part}")
                g.set("type",  "box")
                g.set("size",  array_to_string(size))
                g.set("pos",   array_to_string(pos))
                g.set("rgba",  array_to_string(self.BIN_RGBA))
                g.set("group", "1")


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class BinPlacing(ManipulationEnv):
    """
    Simple bin-placing task.
    A red cube sits on a white table with 4 blue target bins.
    Goal: move the cube into the indicated bin.
    """

    # Tight bounding box around default EEF (~[-0.215, 0.009, 1.007])
    # Size: 0.09 x 0.12 x 0.06 m  (~20% smaller than original)
    START_REGION = dict(
        low  = [-0.260, -0.055, 0.980],
        high = [-0.170,  0.065, 1.040],
    )

    # Reachable workspace bounds for prism centres when generating random regions
    # Kept to the robot-side of the table (negative x), left/right of the stand — not over the bins
    PRISM_CENTER_BOUNDS = dict(
        low  = [-0.50, -0.42, 0.88],
        high = [-0.08,  0.42, 1.25],
    )

    @classmethod
    def make_start_regions(cls, n=10, seed=0, max_attempts=10_000):
        """Return a list of n non-overlapping same-size start regions.

        Region 0 is always START_REGION. The remaining n-1 are placed via
        rejection sampling within PRISM_CENTER_BOUNDS: a candidate center is
        accepted only if it does not overlap any already-placed prism.

        Two axis-aligned boxes overlap iff they overlap on every axis, so the
        check is: reject if |c_new - c_existing| < 2*half on ALL three axes.

        Raises RuntimeError if a valid placement cannot be found within
        max_attempts tries (indicates workspace is too small for n prisms).
        """
        lo   = np.array(cls.START_REGION['low'])
        hi   = np.array(cls.START_REGION['high'])
        half = (hi - lo) / 2.0

        centers = [(lo + hi) / 2.0]
        regions = [cls.START_REGION]

        rng  = np.random.default_rng(seed)
        c_lo = np.array(cls.PRISM_CENTER_BOUNDS['low'])
        c_hi = np.array(cls.PRISM_CENTER_BOUNDS['high'])

        for i in range(n - 1):
            for _ in range(max_attempts):
                candidate = rng.uniform(c_lo + half, c_hi - half)
                if not any(np.all(np.abs(candidate - ec) < 2 * half) for ec in centers):
                    centers.append(candidate)
                    regions.append({
                        'low':  (candidate - half).tolist(),
                        'high': (candidate + half).tolist(),
                    })
                    break
            else:
                raise RuntimeError(
                    f"Could not place non-overlapping prism {i + 1} after "
                    f"{max_attempts} attempts. Reduce n or expand PRISM_CENTER_BOUNDS."
                )

        return regions

    # One color per bin for goal dots
    BIN_COLORS = [
        [0.9, 0.15, 0.15, 0.9],  # red
        [0.15, 0.75, 0.15, 0.9],  # green
        [0.15, 0.15, 0.9,  0.9],  # blue
        [0.9,  0.55, 0.05, 0.9],  # orange
    ]

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        table_full_size=(1.0, 1.0, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_shaping=False,
        target_bin=None,
        mujoco_object=None,
        n_starts=100,
        show_samples=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
    ):
        self.table_full_size   = table_full_size
        self.table_friction    = table_friction
        self.table_offset      = np.array((0, 0, 0.8))
        self.use_object_obs    = use_object_obs
        self.reward_shaping    = reward_shaping
        self._fixed_target_bin = target_bin
        self.current_goal_bin  = None
        self._input_object     = mujoco_object
        self.n_starts          = n_starts
        self.show_samples      = show_samples
        self.start_positions   = None   # (n_starts, 3)
        self.goal_positions    = None   # list of 4 arrays, each (n_starts, 3)

        # World-space bin centres (table surface z = 0.8)
        self.bin_positions = np.array([
            [xy[0], xy[1], 0.8] for xy in BinTableArena.BIN_XY
        ])

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def _load_model(self):
        super()._load_model()

        self.robots[0].robot_model.set_base_xpos(
            self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        )

        mujoco_arena = BinTableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        if self._input_object is not None:
            self.grasp_obj = self._input_object
        else:
            self.grasp_obj = BoxObject(
                name="cube",
                size_min=[0.020, 0.020, 0.020],
                size_max=[0.022, 0.022, 0.022],
                rgba=[1, 0, 0, 1],
                rng=self.rng,
            )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.grasp_obj,
        )

        if self.show_samples:
            self._generate_samples()
            self._bake_sample_markers()

    # ------------------------------------------------------------------
    # Sample generation + visualization
    # ------------------------------------------------------------------

    def _generate_samples(self, seed=42, start_regions=None):
        """Sample start (EEF) and per-bin goal positions. Stores results in self.

        Args:
            seed:          Base integer seed.
            start_regions: List of dicts with 'low'/'high' keys defining each prism.
                           Defaults to [self.START_REGION].

        Seeding contract:
          - Start positions for prism i use seed [seed, i]       → stable across bin changes
          - Goal positions for bin j   use seed [seed, 100 + j]  → stable across prism changes

        self.start_positions: list of (n_starts, 3) arrays, one per prism.
        self.goal_positions:  list of (n_starts, 3) arrays, one per bin.
        """
        if start_regions is None:
            start_regions = self.make_start_regions()

        # Start positions: one independent RNG per prism
        self.start_positions = []
        for prism_idx, region in enumerate(start_regions):
            rng = np.random.default_rng([seed, prism_idx])
            lo = np.array(region['low'])
            hi = np.array(region['high'])
            self.start_positions.append(rng.uniform(lo, hi, size=(self.n_starts, 3)))

        # Goal positions: one independent RNG per bin
        goal_z = self.table_offset[2] + BinTableArena.H * 2 + 0.01  # 1cm above rim
        ix = (BinTableArena.OX - 2 * BinTableArena.WT) * 0.25  # inner x half-extent
        iy = (BinTableArena.OY - 2 * BinTableArena.WT) * 0.25  # inner y half-extent

        self.goal_positions = []
        for bin_idx, (bx, by) in enumerate(BinTableArena.BIN_XY):
            rng_bin = np.random.default_rng([seed, 100 + bin_idx])
            gx = rng_bin.uniform(bx - ix, bx + ix, size=self.n_starts)
            gy = rng_bin.uniform(by - iy, by + iy, size=self.n_starts)
            self.goal_positions.append(
                np.stack([gx, gy, np.full(self.n_starts, goal_z)], axis=1)
            )

    def save_observations(self, save_path, n_obs_per_prism=None, n_obs_per_bin=None):
        """Save start and goal positions to a .npz file.

        Args:
            save_path:       Destination path (e.g. 'observations.npz').
            n_obs_per_prism: How many start positions to save per prism (default: all).
            n_obs_per_bin:   How many goal positions to save per bin   (default: all).

        Saved arrays:
            start_positions  — (n_prisms, n_obs_per_prism, 3)
            goal_positions   — (n_bins,   n_obs_per_bin,   3)
        """
        if self.start_positions is None or self.goal_positions is None:
            raise RuntimeError("Call _generate_samples() before save_observations().")

        starts = []
        for prism_idx, prism_pos in enumerate(self.start_positions):
            n = n_obs_per_prism if n_obs_per_prism is not None else len(prism_pos)
            if n > len(prism_pos):
                raise ValueError(
                    f"Prism {prism_idx}: requested {n} obs but only {len(prism_pos)} available."
                )
            starts.append(prism_pos[:n])

        goals = []
        for bin_idx, bin_pos in enumerate(self.goal_positions):
            n = n_obs_per_bin if n_obs_per_bin is not None else len(bin_pos)
            if n > len(bin_pos):
                raise ValueError(
                    f"Bin {bin_idx}: requested {n} obs but only {len(bin_pos)} available."
                )
            goals.append(bin_pos[:n])

        np.savez(
            save_path,
            start_positions=np.array(starts),   # (n_prisms, n_obs_per_prism, 3)
            goal_positions=np.array(goals),      # (n_bins,   n_obs_per_bin,   3)
        )
        print(
            f"Saved: {len(starts)} prisms × {len(starts[0])} starts, "
            f"{len(goals)} bins × {len(goals[0])} goals → {save_path}"
        )

    def _bake_sample_markers(self):
        """Add start/goal dots to the worldbody XML before sim compilation."""
        wb = self.model.worldbody

        # Start dots — yellow (one dot per sample per prism)
        dot_idx = 0
        for prism_positions in self.start_positions:
            for pos in prism_positions:
                g = ET.SubElement(wb, "geom")
                g.set("name",         f"start_dot_{dot_idx}")
                g.set("type",         "sphere")
                g.set("size",         "0.012")
                g.set("pos",          array_to_string(pos))
                g.set("rgba",         array_to_string([0.95, 0.85, 0.1, 0.9]))
                g.set("contype",      "0")
                g.set("conaffinity",  "0")
                g.set("group",        "1")
                dot_idx += 1

        # Goal dots — one color per bin
        for bin_idx, (goals, color) in enumerate(
            zip(self.goal_positions, self.BIN_COLORS)
        ):
            for j, pos in enumerate(goals):
                g = ET.SubElement(wb, "geom")
                g.set("name",         f"goal_dot_{bin_idx}_{j}")
                g.set("type",         "sphere")
                g.set("size",         "0.010")
                g.set("pos",          array_to_string(pos))
                g.set("rgba",         array_to_string(color))
                g.set("contype",      "0")
                g.set("conaffinity",  "0")
                g.set("group",        "1")

    # ------------------------------------------------------------------
    # References / Reset
    # ------------------------------------------------------------------

    def _setup_references(self):
        super()._setup_references()
        self.obj_body_id = self.sim.model.body_name2id(self.grasp_obj.root_body)

    def _reset_internal(self):
        super()._reset_internal()
        self.sim.forward()

        # Place object at grip site
        eef_pos = self._eef_pos()
        self.sim.data.set_joint_qpos(
            self.grasp_obj.joints[0],
            np.concatenate([eef_pos, np.array([1, 0, 0, 0])])
        )

        # Close gripper: zero both qpos and actuator ctrl for finger joints
        for i in range(self.sim.model.njnt):
            if "finger_joint" in self.sim.model.joint_id2name(i):
                self.sim.data.qpos[self.sim.model.jnt_qposadr[i]] = 0.0
        for i in range(self.sim.model.nu):
            if "finger_joint" in self.sim.model.actuator_id2name(i):
                self.sim.data.ctrl[i] = 0.0

        self.sim.forward()

        # Stabilize the grasp with a few physics steps
        for _ in range(50):
            self.sim.step()
        self.sim.forward()

        if self._fixed_target_bin is not None:
            self.current_goal_bin = self._fixed_target_bin
        else:
            self.current_goal_bin = int(self.rng.integers(0, 4))

    def _eef_pos(self):
        """Return the grip-site world position."""
        eef_id = self.robots[0].eef_site_id
        if isinstance(eef_id, dict):
            eef_id = next(iter(eef_id.values()))
        return self.sim.data.site_xpos[eef_id].copy()

    # ------------------------------------------------------------------
    # Observables
    # ------------------------------------------------------------------

    def _setup_observables(self):
        observables = super()._setup_observables()

        if self.use_object_obs:
            modality = "object"

            @sensor(modality=modality)
            def cube_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.obj_body_id])

            @sensor(modality=modality)
            def cube_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.obj_body_id]), to="xyzw")

            @sensor(modality=modality)
            def goal_bin(obs_cache):
                return np.array([self.current_goal_bin or 0], dtype=np.int64)

            @sensor(modality=modality)
            def goal_pos(obs_cache):
                return self.bin_positions[self.current_goal_bin or 0].copy()

            for s in [cube_pos, cube_quat, goal_bin, goal_pos]:
                observables[s.__name__] = Observable(
                    name=s.__name__, sensor=s, sampling_rate=self.control_freq
                )

        return observables

    # ------------------------------------------------------------------
    # Reward / success
    # ------------------------------------------------------------------

    def reward(self, action=None):
        if self.current_goal_bin is None:
            return 0.0
        obj_pos  = self.sim.data.body_xpos[self.obj_body_id]
        goal_pos = self.bin_positions[self.current_goal_bin]
        dist = np.linalg.norm(obj_pos - goal_pos)
        if self._check_success():
            return 1.0
        return -dist if self.reward_shaping else 0.0

    def _check_success(self):
        if self.current_goal_bin is None:
            return False
        obj_pos  = self.sim.data.body_xpos[self.obj_body_id]
        goal_pos = self.bin_positions[self.current_goal_bin]
        return bool(np.linalg.norm(obj_pos - goal_pos) < 0.05)

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def set_target_bin(self, idx):
        if not 0 <= idx < 4:
            raise ValueError(f"idx must be 0-3, got {idx}")
        self._fixed_target_bin = idx


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _composite_cfg(controller_type):
    """Build a BASIC composite controller config for Panda with the given arm controller."""
    part = load_part_controller_config(default_controller=controller_type)
    part["gripper"] = {"type": "GRIP"}
    return {"type": "BASIC", "body_parts": {"right": part}}


def make_env(target_bin=None, has_renderer=True, camera="frontview",
             mujoco_object=None, show_samples=False, n_starts=100,
             controller="JOINT_POSITION"):
    """
    controller="JOINT_POSITION"  — for diffusion policy evaluation
    controller="OSC_POSE"        — for scripted demo collection (Cartesian delta control)
    """
    return BinPlacing(
        robots="Panda",
        controller_configs=_composite_cfg(controller),
        target_bin=target_bin,
        mujoco_object=mujoco_object,
        show_samples=show_samples,
        n_starts=n_starts,
        has_renderer=has_renderer,
        has_offscreen_renderer=False,
        render_camera=camera,
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=True,
        horizon=500,
        control_freq=20,
        ignore_done=True,
    )


def scripted_episode(env, bin_idx):
    """
    Drive the arm to a sampled goal in bin_idx (0-3) then drop the object.
    Requires env created with controller='OSC_POSE'.

    OSC_POSE default: control_delta=True, output_max=0.05m for position.
    So action[:3] = 1.0 → move 0.05 m/step (at 20 Hz = 1 m/s max).
    """
    goal_pos = env.goal_positions[bin_idx][0]
    print(f"Target: bin {bin_idx+1}  goal_pos={goal_pos.round(3)}")

    action_dim = env.action_spec[0].shape[0]   # 7: [dx dy dz dRx dRy dRz gripper]

    # Phase 1 — move to goal (P-controller in Cartesian action space)
    for step in range(400):
        eef_pos = env._eef_pos()
        delta   = goal_pos - eef_pos
        dist    = np.linalg.norm(delta)
        if dist < 0.015:
            print(f"  reached in {step} steps  (dist={dist:.4f} m)")
            break
        action        = np.zeros(action_dim)
        action[:3]    = np.clip(delta / 0.60, -1.0, 1.0)   # cap at ~0.08 m/s (slow viewing speed)
        action[-1]    = 1.0                                  # gripper closed
        env.step(action)
        env.render()

    # Phase 2 — open gripper and let object drop
    for _ in range(80):
        action      = np.zeros(action_dim)
        action[-1]  = -1.0
        env.step(action)
        env.render()

    # Phase 3 — hold still and watch
    for _ in range(60):
        env.step(np.zeros(action_dim))
        env.render()


def execute_spline(env, waypoints, horizon=64, record=True):
    """
    Fit a smooth path through XYZ waypoints, sample `horizon` evenly-spaced
    Cartesian targets, then drive the arm to each via OSC_POSE delta control.

    Typical usage — demo collection:
        waypoints = np.array([[x0,y0,z0], ..., [xN,yN,zN]])  # ~20 control pts
        joint_traj = execute_spline(env, waypoints, horizon=64, record=True)
        # joint_traj: (64, 7) — 64 joint-position snapshots → diffusion policy data

    Args:
        env:       BinPlacing env with controller='OSC_POSE'
        waypoints: (N, 3) XYZ positions defining the desired path  (N >= 2)
        horizon:   number of evenly-spaced targets to sample and execute
        record:    if True, capture joint positions at each of the 64 steps

    Returns:
        np.ndarray (horizon, n_joints) if record=True, else None
    """
    waypoints = np.asarray(waypoints, dtype=float)
    if waypoints.ndim != 2 or waypoints.shape[1] != 3:
        raise ValueError("waypoints must be (N, 3)")
    if len(waypoints) < 2:
        raise ValueError("need at least 2 waypoints")

    # --- Arc-length parameterisation ---
    seg_len  = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    arc      = np.concatenate([[0.0], np.cumsum(seg_len)])
    arc_norm = arc / arc[-1]                   # [0, 1]

    # Cubic spline if enough points, else linear
    if len(waypoints) >= 4:
        cs      = CubicSpline(arc_norm, waypoints)
        targets = cs(np.linspace(0.0, 1.0, horizon))
    else:
        cs      = interp1d(arc_norm, waypoints, axis=0, kind='linear')
        targets = cs(np.linspace(0.0, 1.0, horizon))

    action_dim = env.action_spec[0].shape[0]
    n_joints   = env.robots[0].dof
    joint_traj = [] if record else None

    for target_xyz in targets:
        # Inner loop: step until arm reaches this waypoint or budget exhausted
        for _ in range(20):
            eef   = env._eef_pos()
            delta = target_xyz - eef
            if np.linalg.norm(delta) < 0.010:
                break
            action      = np.zeros(action_dim)
            action[:3]  = np.clip(delta / 0.20, -1.0, 1.0)
            action[-1]  = 1.0                               # gripper closed
            env.step(action)
            if env.has_renderer:
                env.render()

        if record:
            joint_traj.append(env.sim.data.qpos[:n_joints].copy())

    return np.array(joint_traj) if record else None


def demo():
    import argparse
    parser = argparse.ArgumentParser()
    view = parser.add_mutually_exclusive_group()
    view.add_argument("-o", "--overhead", action="store_true", help="Overhead (bird's-eye) view")
    view.add_argument("-s", "--side",     action="store_true", help="Side view")
    parser.add_argument("-v", "--viz-samples", action="store_true", help="Show start/goal sample dots")
    parser.add_argument("-e", "--episode", type=int, choices=[1,2,3,4], metavar="BIN",
                        help="Scripted episode: move to a goal in bin 1-4 and drop")
    parser.add_argument("--object", default="can",
                        choices=["can", "bottle", "milk", "cereal", "bread", "lemon",
                                 "box", "cylinder", "ball"],
                        help="Object to hold (default: can)")
    args = parser.parse_args()

    if args.overhead:
        camera = "birdview"
    elif args.side:
        camera = "sideview"
    else:
        camera = "frontview"

    obj_map = {
        # Realistic bin-packing objects (mesh-based)
        "can":      CanObject(name="obj"),
        "bottle":   BottleObject(name="obj"),
        "milk":     MilkObject(name="obj"),
        "cereal":   CerealObject(name="obj"),
        "bread":    BreadObject(name="obj"),
        "lemon":    LemonObject(name="obj"),
        # Geometric primitives
        "box":      BoxObject(name="obj", size_min=[0.020]*3, size_max=[0.022]*3, rgba=[1, 0, 0, 1]),
        "cylinder": CylinderObject(name="obj", size_min=[0.020, 0.030], size_max=[0.022, 0.032], rgba=[0.2, 0.6, 1, 1]),
        "ball":     BallObject(name="obj", size_min=[0.022], size_max=[0.024], rgba=[0.1, 0.9, 0.2, 1]),
    }
    mujoco_object = obj_map[args.object]

    # Scripted episodes need Cartesian delta control; normal view uses JOINT_POSITION
    ctrl = "OSC_POSE" if args.episode else "JOINT_POSITION"
    env  = make_env(has_renderer=True, camera=camera, show_samples=args.viz_samples,
                    mujoco_object=mujoco_object, controller=ctrl)
    obs  = env.reset()
    print(f"goal_bin={env.current_goal_bin}  camera={camera}  controller={ctrl}")

    if args.episode:
        scripted_episode(env, bin_idx=args.episode - 1)
        print("Episode done. Close the window to exit.")
        while True:
            env.render()
    else:
        print("Close the window to exit.")
        action = np.zeros(env.action_spec[0].shape)
        action[-1] = 1.0  # keep gripper closed
        while True:
            obs, reward, done, info = env.step(action)
            env.render()


if __name__ == "__main__":
    demo()
