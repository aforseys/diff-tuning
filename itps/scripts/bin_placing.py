import os
import argparse
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

OBJECT_MAP = {
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
        low  = [-0.2375, -0.025, 0.995],
        high = [-0.1925,  0.035, 1.025],
    )

    # Reachable workspace bounds for prism centres when generating random regions
    # Kept to the robot-side of the table (negative x), left/right of the stand — not over the bins
    PRISM_CENTER_BOUNDS = dict(
        low  = [-0.45, -0.42, 0.95],
        high = [-0.15,  0.42, 1.25],
    )

    @classmethod
    def make_start_regions(cls, n=10, seed=0, max_attempts=10_000, min_sep_factor=3.0):
        """Return a list of n same-size start regions with enforced minimum separation.

        Region 0 is always START_REGION. The remaining n-1 are placed via
        rejection sampling within PRISM_CENTER_BOUNDS: a candidate center is
        accepted only if every existing center is at least
        min_sep_factor * 2 * half away on all three axes simultaneously.

        min_sep_factor=1.0 → just non-overlapping; 2.0 → one full region-width
        gap between every pair of regions.

        Raises RuntimeError if a valid placement cannot be found within
        max_attempts tries (reduce n, increase PRISM_CENTER_BOUNDS, or lower
        min_sep_factor).
        """
        lo   = np.array(cls.START_REGION['low'])
        hi   = np.array(cls.START_REGION['high'])
        half = (hi - lo) / 2.0
        min_sep = min_sep_factor * 2 * half

        centers = [(lo + hi) / 2.0]
        regions = [cls.START_REGION]

        rng  = np.random.default_rng(seed)
        c_lo = np.array(cls.PRISM_CENTER_BOUNDS['low'])
        c_hi = np.array(cls.PRISM_CENTER_BOUNDS['high'])

        for i in range(n - 1):
            for _ in range(max_attempts):
                candidate = rng.uniform(c_lo + half, c_hi - half)
                if not any(np.all(np.abs(candidate - ec) < min_sep) for ec in centers):
                    centers.append(candidate)
                    regions.append({
                        'low':  (candidate - half).tolist(),
                        'high': (candidate + half).tolist(),
                    })
                    break
            else:
                raise RuntimeError(
                    f"Could not place region {i + 1} after {max_attempts} attempts. "
                    f"Reduce n, expand PRISM_CENTER_BOUNDS, or lower min_sep_factor."
                )

        return regions

    # One color per bin for goal dots
    BIN_COLORS = [
        [0.9, 0.15, 0.15, 0.9],  # red
        [0.15, 0.75, 0.15, 0.9],  # green
        [0.15, 0.15, 0.9,  0.9],  # blue
        [0.9,  0.55, 0.05, 0.9],  # orange
    ]

    # table surface z = 0.8; bin rim = 0.8 + 2*H ≈ 0.844
    # 4 cm clearance above rim covers the lower half of typical objects (can, bottle, etc.)
    PLACED_Z_THRESHOLD = 0.8 + BinTableArena.H * 2 + 0.04

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
        #target_bin=None,
        mujoco_object=None,
       # n_starts=100,
       # show_samples=False,
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
        self._input_object     = mujoco_object
        self._marker_data      = None
        self._spline_data      = None   # (waypoints_list, colors_list) — set before reset()
        self._waypoint_data    = None   # (waypoints_list, colors_list) — raw optimizer waypoints
        #self.n_starts          = n_starts
        #self.show_samples      = show_samples
        # self.start_positions   = None   # (n_starts, 3)
        # self.goal_positions    = None   # list of 4 arrays, each (n_starts, 3)

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

        for cam in self.model.worldbody.findall('.//camera[@name="agentview"]'):
            cam.set('pos', '0.8 0 1.6')
            cam.set('fovy', '70')

        if self._marker_data is not None:
            self._bake_sample_markers(*self._marker_data)

        if self._spline_data is not None:
            self._bake_spline_markers(*self._spline_data)

        if self._waypoint_data is not None:
            self._bake_waypoint_markers(*self._waypoint_data)

    # ------------------------------------------------------------------
    # Sample generation + visualization
    # ------------------------------------------------------------------

    def _generate_samples(self, seed=42, start_regions=None, n_obs=100):
        """Sample start (EEF) and per-bin goal positions with 1:1 pairing.

        Args:
            seed:          Base integer seed.
            start_regions: List of dicts with 'low'/'high' keys defining each prism.
                           Defaults to make_start_regions().
            n_obs:         Number of (start, goal) pairs per (prism, bin) combination.

        Seeding contract:
          - Start positions for prism i use seed [seed, i]       → stable across bin changes
          - Goal positions for bin j   use seed [seed, 100 + j]  → stable across prism changes

        Returns:
            start_positions: list of (n_obs, 3) arrays, one per prism.
            goal_positions:  list of (n_obs, 3) arrays, one per bin.
        """
        if start_regions is None:
            start_regions = self.make_start_regions()

        # Start positions: one independent RNG per prism
        start_positions = []
        for prism_idx, region in enumerate(start_regions):
            rng = np.random.default_rng([seed, prism_idx])
            lo = np.array(region['low'])
            hi = np.array(region['high'])
            start_positions.append(rng.uniform(lo, hi, size=(n_obs, 3)))

        # Goal positions: one independent RNG per bin
        goal_z = self.table_offset[2] + BinTableArena.H * 2 + 0.01  # 1cm above rim
        ix = (BinTableArena.OX - 2 * BinTableArena.WT) * 0.25  # inner x half-extent
        iy = (BinTableArena.OY - 2 * BinTableArena.WT) * 0.25  # inner y half-extent

        goal_positions = []
        for bin_idx, (bx, by) in enumerate(BinTableArena.BIN_XY):
            rng_bin = np.random.default_rng([seed, 100 + bin_idx])
            gx = rng_bin.uniform(bx - ix, bx + ix, size=n_obs)
            gy = rng_bin.uniform(by - iy, by + iy, size=n_obs)
            goal_positions.append(
                np.stack([gx, gy, np.full(n_obs, goal_z)], axis=1)
            )

        return (start_positions, goal_positions)

    def save_observations(self, start_positions, goal_positions, save_path,
                           object_type=None, start_joints=None):
        """Save all (start, goal) pairs as flat aligned arrays.

        For each (prism p, bin b), start[j] is paired 1:1 with goal[j].
        Total pairs N = n_prisms * n_bins * n_obs.

        Args:
            start_positions: list of (n_obs, 3) arrays, one per prism.
            goal_positions:  list of (n_obs, 3) arrays, one per bin.
            save_path:       Destination path (e.g. 'observations.npz').
            object_type:     String name of the object (e.g. 'can', 'bottle').
            start_joints:    Optional list of (n_obs, n_joints) arrays, one per prism.
                             Tiled across bins and saved as 'start_joints' (N, n_joints).
        """
        n_prisms = len(start_positions)
        n_bins   = len(goal_positions)
        n_obs    = start_positions[0].shape[0]
        N        = n_prisms * n_bins * n_obs

        out_start     = np.empty((N, 3), dtype=np.float64)
        out_goal      = np.empty((N, 3), dtype=np.float64)
        out_bin_idx   = np.empty(N,      dtype=np.int32)
        out_prism_idx = np.empty(N,      dtype=np.int32)
        if start_joints is not None:
            n_joints   = start_joints[0].shape[1]
            out_joints = np.empty((N, n_joints), dtype=np.float64)

        i = 0
        for p, starts in enumerate(start_positions):
            for b, goals in enumerate(goal_positions):
                out_start[i:i+n_obs]     = starts
                out_goal[i:i+n_obs]      = goals
                out_bin_idx[i:i+n_obs]   = b
                out_prism_idx[i:i+n_obs] = p
                if start_joints is not None:
                    out_joints[i:i+n_obs] = start_joints[p]
                i += n_obs

        save_dict = dict(
            start_pos=out_start,
            goal_pos=out_goal,
            bin_idx=out_bin_idx,
            prism_idx=out_prism_idx,
        )
        if start_joints is not None:
            save_dict["start_joints"] = out_joints
        if object_type is not None:
            save_dict["object_type"] = np.array(object_type)

        np.savez(save_path, **save_dict)
        print(
            f"Saved {N} pairs "
            f"({n_prisms} prisms × {n_bins} bins × {n_obs} obs each)"
            + (f"  object={object_type}" if object_type else "")
            + (f"  start_joints={out_joints.shape}" if start_joints is not None else "")
            + f" → {save_path}"
        )

    def _bake_sample_markers(self, start_positions=None, goal_positions=None):
        """Add start/goal dots to the worldbody XML before sim compilation."""
        wb = self.model.worldbody

        # Start dots — yellow (one dot per sample per prism)
        dot_idx = 0
        if start_positions is not None:
            for prism_positions in start_positions:
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
        if goal_positions is not None:
            for bin_idx, (goals, color) in enumerate(
                zip(goal_positions, self.BIN_COLORS)
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

    def _bake_spline_markers(self, waypoints_list, colors_list=None, radius=0.004, n_samples=100):
        """Draw spline trajectories as capsule tubes in the worldbody XML.

        Fits the same cubic spline used by execute_spline and samples it densely,
        so the tubes match the path the robot will actually follow.

        Args:
            waypoints_list : list of (n_pts, 3) arrays, one per spline.
            colors_list    : list of RGBA arrays (one per spline), or None for grey.
            radius         : tube radius in metres (default 0.004).
            n_samples      : points sampled from the spline for drawing (default 100).
        """
        wb = self.model.worldbody
        default_color = [0.75, 0.75, 0.75, 0.6]
        size_str = str(radius)

        for traj_idx, path in enumerate(waypoints_list):
            color = colors_list[traj_idx] if colors_list is not None else default_color
            path = np.asarray(path, dtype=float)

            seg_len  = np.linalg.norm(np.diff(path, axis=0), axis=1)
            arc      = np.concatenate([[0.0], np.cumsum(seg_len)])
            arc_norm = arc / arc[-1]

            if len(path) >= 4:
                dense = CubicSpline(arc_norm, path)(np.linspace(0.0, 1.0, n_samples))
            else:
                dense = interp1d(arc_norm, path, axis=0, kind='linear')(np.linspace(0.0, 1.0, n_samples))

            for seg_idx in range(len(dense) - 1):
                p0 = dense[seg_idx]
                p1 = dense[seg_idx + 1]
                g = ET.SubElement(wb, "geom")
                g.set("name",        f"spline_{traj_idx}_{seg_idx}")
                g.set("type",        "capsule")
                g.set("fromto",      array_to_string(np.concatenate([p0, p1])))
                g.set("size",        size_str)
                g.set("rgba",        array_to_string(color))
                g.set("contype",     "0")
                g.set("conaffinity", "0")
                g.set("group",       "1")

    def _bake_waypoint_markers(self, waypoints_list, colors_list=None, radius=0.008):
        """Draw raw optimizer waypoints as spheres in the worldbody XML."""
        wb = self.model.worldbody
        default_color = [1.0, 1.0, 0.0, 0.9]   # yellow
        size_str = str(radius)

        for traj_idx, path in enumerate(waypoints_list):
            color = colors_list[traj_idx] if colors_list is not None else default_color
            for pt_idx, pos in enumerate(np.asarray(path, dtype=float)):
                g = ET.SubElement(wb, "geom")
                g.set("name",        f"waypoint_{traj_idx}_{pt_idx}")
                g.set("type",        "sphere")
                g.set("pos",         array_to_string(pos))
                g.set("size",        size_str)
                g.set("rgba",        array_to_string(color))
                g.set("contype",     "0")
                g.set("conaffinity", "0")
                g.set("group",       "1")

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

            for s in [cube_pos, cube_quat]:
                observables[s.__name__] = Observable(
                    name=s.__name__, sensor=s, sampling_rate=self.control_freq
                )

        return observables

    # ------------------------------------------------------------------
    # Reward / success
    # ------------------------------------------------------------------

    def reward(self, action=None):
        return 0.0

    def _gripper_is_open(self, threshold=0.06):
        total = 0.0
        for i in range(self.sim.model.njnt):
            if "finger_joint" in self.sim.model.joint_id2name(i):
                total += abs(self.sim.data.qpos[self.sim.model.jnt_qposadr[i]])
        return total > threshold

    def location_success(self, bin_idx):
        """Object CoM is within bin XY footprint and within z bounds [table, PLACED_Z_THRESHOLD]."""
        obj_pos    = self.sim.data.body_xpos[self.obj_body_id]
        bx, by     = BinTableArena.BIN_XY[bin_idx]
        ox, oy     = BinTableArena.OX, BinTableArena.OY
        in_bin_xy  = (bx - ox) <= obj_pos[0] <= (bx + ox) and \
                     (by - oy) <= obj_pos[1] <= (by + oy)
        in_z_range = 0.8 < obj_pos[2] < self.PLACED_Z_THRESHOLD
        return bool(in_bin_xy and in_z_range)

    def placement_success(self, bin_idx):
        """Object is location_success AND gripper is open."""
        return self.location_success(bin_idx) and self._gripper_is_open()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _composite_cfg(controller_type):
    """Build a BASIC composite controller config for Panda with the given arm controller."""
    part = load_part_controller_config(default_controller=controller_type)
    part["gripper"] = {"type": "GRIP"}
    return {"type": "BASIC", "body_parts": {"right": part}}


def make_env(has_renderer=True, camera="frontview",
             mujoco_object=None, 
             controller="JOINT_POSITION"):
    """
    controller="JOINT_POSITION"  — for diffusion policy evaluation
    controller="OSC_POSE"        — for scripted demo collection (Cartesian delta control)
    """
    return BinPlacing(
        robots="Panda",
        controller_configs=_composite_cfg(controller),
        # target_bin=target_bin,
        mujoco_object=mujoco_object,
        # show_samples=show_samples,
        # n_starts=n_starts,
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


def make_eval_env(img_size=84, mujoco_object=None):
    """BinPlacing env for diffusion policy evaluation: JOINT_POSITION control with camera obs."""
    return BinPlacing(
        robots="Panda",
        controller_configs=_composite_cfg("JOINT_POSITION"),
        mujoco_object=mujoco_object,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="agentview",
        use_camera_obs=True,
        use_object_obs=True,
        reward_shaping=True,
        horizon=500,
        control_freq=20,
        ignore_done=True,
        hard_reset=True,
        camera_names=["agentview"],
        camera_heights=img_size,
        camera_widths=img_size,
    )


def scripted_episode(env, goal_positions, bin_idx):
    """
    Drive the arm to a sampled goal in bin_idx (0-3) then drop the object.
    Requires env created with controller='OSC_POSE'.

    OSC_POSE default: control_delta=True, output_max=0.05m for position.
    So action[:3] = 1.0 → move 0.05 m/step (at 20 Hz = 1 m/s max).
    """
    goal_pos = goal_positions[bin_idx][0]
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

def solve_start_joints(env_osc, start_positions):
    """Solve IK for each EEF start position by driving the arm there via OSC_POSE.

    For each target, resets the env and uses a P-controller to move the EEF to the
    target XYZ, then records joint angles at convergence. Results are per-prism so
    save_observations can tile them across bins without re-solving.

    Args:
        env_osc:         BinPlacing env built with controller='OSC_POSE'.
        start_positions: list of (n_obs, 3) arrays, one per start prism.

    Returns:
        list of (n_obs, n_joints) arrays, one per start prism.
    """
    n_joints   = env_osc.robots[0].dof
    action_dim = env_osc.action_spec[0].shape[0]
    total      = sum(len(p) for p in start_positions)
    done       = 0
    result     = []

    for pi, positions in enumerate(start_positions):
        n = len(positions)
        joints = np.zeros((n, n_joints), dtype=np.float64)
        for oi, target_xyz in enumerate(positions):
            env_osc.reset()
            for _ in range(500):
                eef   = env_osc._eef_pos()
                delta = target_xyz - eef
                if np.linalg.norm(delta) < 0.008:
                    break
                action     = np.zeros(action_dim)
                action[:3] = np.clip(delta / 0.20, -1.0, 1.0)
                action[-1] = 1.0
                env_osc.step(action)
            joints[oi] = env_osc.sim.data.qpos[:n_joints].copy()
            done += 1
            if done % 20 == 0 or done == total:
                err = np.linalg.norm(target_xyz - env_osc._eef_pos())
                print(f"  IK: {done}/{total}  prism={pi}  err={err:.4f} m")
        result.append(joints)

    return result


def sample_generation(env, save_dir, n_starts=None, n_obs=10, object_type=None, env_osc=None):
    all_regions = env.make_start_regions()

    if n_starts is None:
        start_regions = all_regions
    elif n_starts > len(all_regions):
        raise ValueError(
            f"n_starts ({n_starts}) exceeds the number of preset start regions ({len(all_regions)})"
        )
    else:
        start_regions = all_regions[:n_starts]

    start_positions, goal_positions = env._generate_samples(
        start_regions=start_regions,
        n_obs=n_obs,
    )

    start_joints = None
    if env_osc is not None:
        print(f"Solving IK for {sum(len(p) for p in start_positions)} start positions...")
        start_joints = solve_start_joints(env_osc, start_positions)

    if save_dir is None:
        print('NOTE: Samples will not be saved. No save directory passed in.')
    else:
        env.save_observations(start_positions, goal_positions, save_dir,
                               object_type=object_type, start_joints=start_joints)

    return (start_positions, goal_positions)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--camera_view", type=str, default="f", help="Camera view options: f (front), o (overhead), s (side)")
    parser.add_argument("-g", "--gen-samples", action="store_true", help="Generate start/goal samples")
    parser.add_argument("-s", "--save-dir", type=str, help="Path to save generated samples", default=None)
    parser.add_argument("--n-starts", type=int, default=None,
                        help="Number of preset start regions to sample (default: all)")
    parser.add_argument("--n-obs", type=int, default=10,
                        help="Observations per (prism, bin) pair (default: 10)")
    parser.add_argument("-v", "--viz", action="store_true", help="Show env")
    parser.add_argument("-d", "--run-demo", action="store_true", help="Show demo motion")
    parser.add_argument("-b", "--bin", type=int, choices=[1,2,3,4], metavar="BIN",
                        help="Scripted episode: move to a goal in bin 1-4 and drop")
    parser.add_argument("--object", default="can",
                        choices=["can", "bottle", "milk", "cereal", "bread", "lemon",
                                 "box", "cylinder", "ball"],
                        help="Object to hold (default: can)")
    parser.add_argument("--solve-ik", action="store_true",
                        help="Solve IK for start positions and save joint angles (requires --gen-samples)")

    args = parser.parse_args()

    # Set camera angle
    if args.camera_view=='o':
        camera = "birdview"
    elif args.camera_view=='s':
        camera = "sideview"
    else:
        camera = "frontview"

    # Set object and controller 
    mujoco_object = OBJECT_MAP[args.object]
    ctrl = "OSC_POSE" if args.run_demo else "JOINT_POSITION"

    # Initialize environment
    env = make_env(has_renderer=args.viz, camera=camera,
                   mujoco_object=mujoco_object, controller=ctrl)

    # Generate samples before reset so markers are baked into the model at compile time
    samples = None
    if args.gen_samples:
        env_osc = None
        if args.solve_ik:
            env_osc = make_env(has_renderer=False, mujoco_object=mujoco_object,
                               controller="OSC_POSE")
        samples = sample_generation(env, args.save_dir,
                                    n_starts=args.n_starts,
                                    n_obs=args.n_obs,
                                    object_type=args.object,
                                    env_osc=env_osc)
        if env_osc is not None:
            env_osc.close()
        if args.viz:
            env._marker_data = samples

    obs = env.reset()
    print(f"camera={camera}  controller={ctrl}")

    # Visualize environment
    if args.viz:

        # If running a demonstration
        if args.run_demo:
            bin_idx = args.bin - 1
            if samples is not None:
                _, goal_positions = samples
            else:
                _, goal_positions = env._generate_samples(start_regions=[], obs_per_goal=1)
            scripted_episode(env, goal_positions=goal_positions, bin_idx=bin_idx)
            print("Episode done. Close the window to exit.")
            while True:
                env.render()

        # Otherwise just render environment
        else:
            print("Close the window to exit.")
            action = np.zeros(env.action_spec[0].shape)
            action[-1] = 1.0  # keep gripper closed
            while True:
                obs, reward, done, info = env.step(action)
                env.render()

    



    
