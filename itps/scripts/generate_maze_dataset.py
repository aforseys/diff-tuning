#!/usr/bin/env python
"""
Generate custom maze training datasets using A* path planning + PD controller.

For each episode:
  1. Sample random start and goal positions in free (inflated) space
  2. Plan a collision-free path with A* on a clearance-inflated fine grid
  3. Shortcut the A* waypoints to keep only geometric turning points
  4. Follow the waypoints with a noisy PD controller for smooth, varied motion
  5. Chain multiple goal-to-goal paths to fill the episode length

All-pairs shortest paths are precomputed once per maze (one-time cost, ~5-15s
for the large maze), then each episode is just array lookups + PD simulation.

Output format: HDF5 (.hdf5) with 'observations' (N,4) [x,y,vx,vy] and
'timeouts' (N,) bool.  The ITPS loader applies 4x downsampling to HDF5 files
(capped at 1M raw frames), so generate at dt=0.025 (40 Hz):
  raw 40 Hz  →  4x downsample  →  10 Hz effective (matches training fps)

Episode length similarly scales: episode_length=600 raw → 150 effective steps.
Loader cap: only the first 1M raw frames are used; generating more is wasted.

Usage:
  python scripts/generate_maze_dataset.py --maze open --n-episodes 5000 \\
      --save data/maze2d-open-custom.hdf5

  python scripts/generate_maze_dataset.py --maze sparse --n-episodes 8000 \\
      --save data/maze2d-sparse-custom.hdf5

  # Custom maze: .npy file with a 2D bool array (True=wall)
  python scripts/generate_maze_dataset.py --maze-file my_maze.npy \\
      --n-episodes 8000 --save data/maze2d-custom.hdf5

  # Preview trajectories before a full run
  python scripts/generate_maze_dataset.py --maze open --n-episodes 10 --viz
"""

import argparse
import time
import numpy as np
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path as csgraph_sp

from itps.common.utils.maze_maps import MAZE_MAPS


# ── Planner: builds the inflated grid, graph, and precomputes all-pairs paths ──

class MazePlanner:
    """
    One-time setup per maze. After __init__, path queries are just array lookups.

    Coordinate convention (matches the rest of the codebase):
      maze[r, c]  where r = row (x-axis), c = col (y-axis)
      node_positions[:, 0] = x (row),  node_positions[:, 1] = y (col)
    """

    def __init__(self, maze: np.ndarray, scale: int = 5, clearance: float = 0.5):
        """
        Args:
            maze:      2D bool array, True=wall.
            scale:     Fine grid cells per maze unit. scale=5 → 0.2-unit spacing.
            clearance: Minimum wall distance (maze units) for navigable cells.
        """
        self.maze = maze
        self.scale = scale
        self.clearance = clearance

        print("  [1/3] Building inflated fine grid...")
        self.valid_mask = self._build_valid_mask()           # (fine_rows, fine_cols)
        self.fine_rows, self.fine_cols = self.valid_mask.shape

        # Map between flat fine-grid index and compressed node index
        valid_flat = self.valid_mask.ravel()
        self.flat_to_node = np.full(valid_flat.size, -1, dtype=np.int32)
        self.flat_to_node[valid_flat] = np.arange(valid_flat.sum(), dtype=np.int32)
        self.node_to_ij = np.argwhere(self.valid_mask)       # (n_nodes, 2)
        self.n_nodes = len(self.node_to_ij)
        # Shift so cell r has center at position r (matches MazeEnv's gui2xy offset=0.5).
        # Fine cell fi has center at (fi+0.5)/scale, then subtract 0.5 to align with env.
        self.node_positions = (self.node_to_ij + 0.5) / scale - 0.5

        print(f"  [1/3] Done — {self.n_nodes} valid nodes "
              f"({self.valid_mask.size} fine cells total).")

        print("  [2/3] Building adjacency graph...")
        graph = self._build_graph()

        print("  [3/3] Precomputing all-pairs shortest paths (one-time cost)...")
        t0 = time.time()
        self.dist_matrix, self.predecessors = csgraph_sp(
            graph, directed=False, return_predecessors=True
        )
        print(f"  [3/3] Done in {time.time() - t0:.1f}s.")

        # Identify the main connected component for safe start/goal sampling
        reachable_from_0 = np.isfinite(self.dist_matrix[0])
        if not reachable_from_0.all():
            # Label components, keep the largest
            labels = np.full(self.n_nodes, -1, dtype=int)
            comp_id = 0
            for seed in range(self.n_nodes):
                if labels[seed] != -1:
                    continue
                members = np.isfinite(self.dist_matrix[seed])
                labels[members] = comp_id
                comp_id += 1
            sizes = np.bincount(labels[labels >= 0])
            best = int(np.argmax(sizes))
            self._main = labels == best
            print(f"  Warning: graph has {comp_id} components; "
                  f"using largest ({sizes[best]} nodes).")
        else:
            self._main = np.ones(self.n_nodes, dtype=bool)

    # ── internals ──

    def _build_valid_mask(self) -> np.ndarray:
        """
        Upsample the maze to fine resolution, compute the distance transform,
        and mark cells that are physically free AND at least `clearance` maze-units
        from any wall. With (node_to_ij + 0.5)/scale - 0.5 as the position formula,
        round(pos) == fi // scale, so ~fine_maze already guarantees collision safety.
        """
        s = self.scale
        fine_maze = np.kron(self.maze.astype(np.uint8),
                            np.ones((s, s), dtype=np.uint8)).astype(bool)
        edt_pixels = distance_transform_edt(~fine_maze)
        edt_units  = edt_pixels / s
        return (~fine_maze) & (edt_units >= self.clearance)

    def _build_graph(self) -> csr_matrix:
        """8-connected sparse adjacency graph over valid fine-grid nodes."""
        ij = self.node_to_ij                 # (n, 2)
        n  = self.n_nodes
        FR, FC = self.fine_rows, self.fine_cols

        rows_all, cols_all, data_all = [], [], []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni = ij[:, 0] + di
                nj = ij[:, 1] + dj
                in_bounds = (ni >= 0) & (ni < FR) & (nj >= 0) & (nj < FC)
                # Use 0 as a dummy flat index for out-of-bounds (guarded by in_bounds mask)
                flat_nb = np.where(in_bounds, ni * FC + nj, 0)
                is_valid = in_bounds & self.valid_mask.ravel()[flat_nb]
                nb_node  = np.where(is_valid, self.flat_to_node[flat_nb], -1)
                keep = is_valid & (nb_node >= 0)
                src = np.where(keep)[0]
                dst = nb_node[keep]
                cost = float(np.sqrt(di**2 + dj**2)) / self.scale  # maze units
                rows_all.append(src)
                cols_all.append(dst)
                data_all.append(np.full(len(src), cost))

        r = np.concatenate(rows_all)
        c = np.concatenate(cols_all)
        d = np.concatenate(data_all)
        return csr_matrix((d, (r, c)), shape=(n, n))

    # ── public API ──

    def sample_node(self, rng: np.random.Generator) -> int:
        candidates = np.where(self._main)[0]
        return int(candidates[rng.integers(len(candidates))])

    def nearest_node(self, pos_xy: np.ndarray) -> int:
        """Return the index of the closest valid node to a continuous (x,y) position."""
        dists = np.linalg.norm(self.node_positions - pos_xy, axis=1)
        dists[~self._main] = np.inf
        return int(np.argmin(dists))

    def get_path_xy(self, start_node: int, goal_node: int):
        """
        Return (L, 2) array of maze-coordinate waypoints, or None if unreachable.
        """
        if not np.isfinite(self.dist_matrix[start_node, goal_node]):
            return None
        path_nodes = []
        cur = goal_node
        while cur != start_node:
            path_nodes.append(cur)
            cur = self.predecessors[start_node, cur]
            if cur < 0:
                return None
        path_nodes.append(start_node)
        path_nodes.reverse()
        return self.node_positions[path_nodes]   # (L, 2)


# ── Grid planner (d4rl-style: 4-connected BFS on integer cell centres) ────────

class GridMazePlanner:
    """
    Routes through integer cell centres using 4-connected BFS on the original
    maze grid (no fine-grid inflation, no diagonal moves).  Waypoints are exact
    cell-centre positions (r, c), so the PD controller produces axis-aligned,
    grid-following motion that matches the d4rl maze2d dataset style.
    """

    def __init__(self, maze: np.ndarray):
        self.maze = maze
        rows, cols = maze.shape

        free = np.argwhere(~maze)                          # (n_free, 2)
        self.cell_to_node = np.full(maze.shape, -1, dtype=np.int32)
        for idx, (r, c) in enumerate(free):
            self.cell_to_node[r, c] = idx
        self.node_positions = free.astype(float)           # cell (r,c) → pos (r,c)
        self.n_nodes = len(free)

        print("  [1/2] Building 4-connected grid graph...")
        graph = self._build_graph(free, rows, cols)

        print("  [2/2] Precomputing all-pairs shortest paths (one-time cost)...")
        t0 = time.time()
        self.dist_matrix, self.predecessors = csgraph_sp(
            graph, directed=False, return_predecessors=True
        )
        print(f"  [2/2] Done in {time.time() - t0:.1f}s.")

        reachable = np.isfinite(self.dist_matrix[0])
        if not reachable.all():
            labels = np.full(self.n_nodes, -1, dtype=int)
            comp_id = 0
            for seed in range(self.n_nodes):
                if labels[seed] != -1:
                    continue
                members = np.isfinite(self.dist_matrix[seed])
                labels[members] = comp_id
                comp_id += 1
            sizes = np.bincount(labels[labels >= 0])
            best = int(np.argmax(sizes))
            self._main = labels == best
            print(f"  Warning: {comp_id} components; using largest ({sizes[best]} nodes).")
        else:
            self._main = np.ones(self.n_nodes, dtype=bool)

    def _build_graph(self, free, rows, cols):
        src, dst, data = [], [], []
        for idx, (r, c) in enumerate(free):
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    nb = self.cell_to_node[nr, nc]
                    if nb >= 0:
                        src.append(idx)
                        dst.append(nb)
                        data.append(1.0)
        return csr_matrix((data, (src, dst)), shape=(self.n_nodes, self.n_nodes))

    def sample_node(self, rng: np.random.Generator) -> int:
        candidates = np.where(self._main)[0]
        return int(candidates[rng.integers(len(candidates))])

    def nearest_node(self, pos_xy: np.ndarray) -> int:
        dists = np.linalg.norm(self.node_positions - pos_xy, axis=1)
        dists[~self._main] = np.inf
        return int(np.argmin(dists))

    def get_path_xy(self, start_node: int, goal_node: int):
        if not np.isfinite(self.dist_matrix[start_node, goal_node]):
            return None
        path = []
        cur = goal_node
        while cur != start_node:
            path.append(cur)
            cur = self.predecessors[start_node, cur]
            if cur < 0:
                return None
        path.append(start_node)
        path.reverse()
        return self.node_positions[path]   # (L, 2) — integer cell centres


# ── Path shortcutting ─────────────────────────────────────────────────────────

def _line_free(p1, p2, valid_mask, scale, n_checks=20):
    """True if the straight line p1→p2 stays within the inflated valid mask."""
    pts = p1 + np.outer(np.linspace(0, 1, n_checks), (p2 - p1))
    # Inverse of node_positions formula: fi = (pos + 0.5) * scale - 0.5
    half = (scale - 1) / 2
    fi = np.round(np.clip(pts[:, 0] * scale + half, 0, valid_mask.shape[0] - 1)).astype(int)
    fj = np.round(np.clip(pts[:, 1] * scale + half, 0, valid_mask.shape[1] - 1)).astype(int)
    return valid_mask[fi, fj].all()


def shortcut_path(path_xy, valid_mask, scale):
    """
    Greedily skip A* waypoints: jump from each point to the furthest one
    reachable by a straight line that stays within the inflated valid mask.
    Returns a smaller array of (x, y) turning-point waypoints.
    """
    if len(path_xy) <= 2:
        return path_xy
    result = [path_xy[0]]
    i = 0
    while i < len(path_xy) - 1:
        j = len(path_xy) - 1
        while j > i + 1 and not _line_free(path_xy[i], path_xy[j], valid_mask, scale):
            j -= 1
        result.append(path_xy[j])
        i = j
    return np.array(result)


# ── PD controller ─────────────────────────────────────────────────────────────

def follow_waypoints(waypoints, planner, n_steps, dt, kp, kd, noise_std,
                     max_speed, wp_thresh, rng, lookahead=2, use_shortcut=True):
    """
    Track `waypoints` with a noisy PD controller for exactly `n_steps` steps.
    When the last waypoint is reached, a new goal is automatically planned so
    the agent keeps moving for the full episode — no stalling or spiralling.
    Wall violations are caught and the step is cancelled (rare given clearance).

    lookahead: target this many waypoints ahead of the current one, which
               smooths corners by making the agent start turning early.

    Returns (n_steps, 4) float32 array of [x, y, vx, vy].
    """
    waypoints = list(waypoints)
    pos = np.array(waypoints[0], dtype=float)
    vel = np.zeros(2)
    wp_idx = 1
    obs = np.empty((n_steps, 4), dtype=np.float32)

    for t in range(n_steps):
        obs[t] = (pos[0], pos[1], vel[0], vel[1])

        # Advance past waypoints already within reach
        while wp_idx < len(waypoints) and \
                np.linalg.norm(np.array(waypoints[wp_idx]) - pos) < wp_thresh:
            wp_idx += 1

        # Replan when the waypoint list is exhausted
        if wp_idx >= len(waypoints):
            cur_node  = planner.nearest_node(pos)
            goal_node = planner.sample_node(rng)
            attempts  = 0
            while goal_node == cur_node and attempts < 10:
                goal_node = planner.sample_node(rng)
                attempts += 1
            path_xy = planner.get_path_xy(cur_node, goal_node)
            if path_xy is not None:
                new_wps = shortcut_path(path_xy, planner.valid_mask, planner.scale) if use_shortcut else path_xy
                waypoints.extend(new_wps[1:].tolist())   # skip duplicate current pos

        target = np.array(waypoints[min(wp_idx + lookahead, len(waypoints) - 1)])
        force  = kp * (target - pos) - kd * vel + rng.standard_normal(2) * noise_std

        new_vel = vel + force * dt
        spd = np.linalg.norm(new_vel)
        if spd > max_speed:
            new_vel *= max_speed / spd

        new_pos = pos + new_vel * dt

        # Collision guard against the original (un-inflated) maze
        r = int(np.round(np.clip(new_pos[0], 0, planner.maze.shape[0] - 1)))
        c = int(np.round(np.clip(new_pos[1], 0, planner.maze.shape[1] - 1)))
        if not planner.maze[r, c]:
            pos, vel = new_pos, new_vel
        else:
            vel = np.zeros(2)   # stop; noise will push away next step

    return obs


# ── Episode generation ────────────────────────────────────────────────────────

def generate_episode(planner, n_steps, dt, kp, kd, noise_std, max_speed,
                     wp_thresh, n_goals, rng, lookahead=2, use_shortcut=True, start_node=None):
    """
    Chain `n_goals` A*-planned, shortcutted paths and follow them for n_steps.
    If the chain is exhausted before n_steps the agent rests near the last goal.
    """
    if start_node is None:
        start_node = planner.sample_node(rng)
    waypoints   = [planner.node_positions[start_node]]
    cur_node    = start_node

    for _ in range(n_goals):
        goal_node = planner.sample_node(rng)
        # Resample until goal is meaningfully far from current position
        attempts = 0
        while (goal_node == cur_node or
               np.linalg.norm(planner.node_positions[goal_node] -
                               planner.node_positions[cur_node]) < 0.5):
            goal_node = planner.sample_node(rng)
            attempts += 1
            if attempts > 20:
                break

        path_xy = planner.get_path_xy(cur_node, goal_node)
        if path_xy is None:
            continue

        shortcut = shortcut_path(path_xy, planner.valid_mask, planner.scale) if use_shortcut else path_xy
        waypoints.extend(shortcut[1:].tolist())   # skip duplicate start
        cur_node = goal_node

    if len(waypoints) < 2:
        waypoints.append(waypoints[0] + np.array([0.01, 0.01]))

    return follow_waypoints(
        np.array(waypoints), planner, n_steps,
        dt, kp, kd, noise_std, max_speed, wp_thresh, rng,
        lookahead=lookahead, use_shortcut=use_shortcut
    )


# ── Full dataset ──────────────────────────────────────────────────────────────

def generate_dataset(maze, n_episodes, episode_length, scale=5, clearance=0.3,
                     dt=0.025, kp=5.0, kd=3.0, noise_std=0.05, max_speed=2.5,
                     wp_thresh=0.25, n_goals=4, lookahead=0, seed=42):
    """
    Returns:
        observations : (N, 4) float32  [x, y, vx, vy]
        timeouts     : (N,)   bool     True at last step of each episode
    """
    rng = np.random.default_rng(seed)
    planner = MazePlanner(maze, scale=scale, clearance=clearance)

    N = n_episodes * episode_length
    observations = np.empty((N, 4), dtype=np.float32)
    timeouts     = np.zeros(N, dtype=bool)

    t0 = time.time()
    log_every = max(1, n_episodes // 10)
    for ep in range(n_episodes):
        ep_obs = generate_episode(planner, episode_length, dt, kp, kd,
                                  noise_std, max_speed, wp_thresh, n_goals, rng,
                                  lookahead=lookahead)
        start = ep * episode_length
        observations[start: start + episode_length] = ep_obs
        last_raw = start + episode_length - 1
        timeouts[last_raw - (last_raw % 4)] = True  # align to 4x downsampling grid

        if (ep + 1) % log_every == 0:
            elapsed = time.time() - t0
            eta     = elapsed / (ep + 1) * (n_episodes - ep - 1)
            print(f"  Episode {ep+1:>6}/{n_episodes}  |  "
                  f"{elapsed:.1f}s elapsed  |  ETA {eta:.1f}s")

    print(f"  Done — {N:,} total frames in {time.time()-t0:.1f}s")
    return observations, timeouts


def generate_dataset_grid(maze, n_episodes, episode_length,
                          dt=0.025, kp=5.0, kd=3.0, noise_std=0.05, max_speed=2.5,
                          wp_thresh=0.25, n_goals=4, seed=42):
    """
    Like generate_dataset but routes through integer cell centres (4-connected BFS).
    No fine-grid inflation, no path shortcutting — the PD controller follows each
    cell centre in sequence, producing axis-aligned grid-following motion that
    matches the d4rl maze2d dataset style.

    Returns:
        observations : (N, 4) float32  [x, y, vx, vy]
        timeouts     : (N,)   bool     True at last step of each episode
    """
    rng = np.random.default_rng(seed)
    planner = GridMazePlanner(maze)

    N = n_episodes * episode_length
    observations = np.empty((N, 4), dtype=np.float32)
    timeouts     = np.zeros(N, dtype=bool)

    t0 = time.time()
    log_every = max(1, n_episodes // 10)
    last_node = None
    for ep in range(n_episodes):
        ep_obs = generate_episode(
            planner, episode_length, dt, kp, kd,
            noise_std, max_speed,
            wp_thresh=0.05,
            n_goals=n_goals, rng=rng,
            lookahead=0, use_shortcut=False,
            start_node=last_node,
        )
        last_node = planner.nearest_node(ep_obs[-1, :2])
        start = ep * episode_length
        observations[start: start + episode_length] = ep_obs
        last_raw = start + episode_length - 1
        timeouts[last_raw - (last_raw % 4)] = True

        if (ep + 1) % log_every == 0:
            elapsed = time.time() - t0
            eta     = elapsed / (ep + 1) * (n_episodes - ep - 1)
            print(f"  Episode {ep+1:>6}/{n_episodes}  |  "
                  f"{elapsed:.1f}s elapsed  |  ETA {eta:.1f}s")

    print(f"  Done — {N:,} total frames in {time.time()-t0:.1f}s")
    return observations, timeouts


# ── Visualization ─────────────────────────────────────────────────────────────

def visualize(maze, observations, timeouts, n_show=8):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        print("matplotlib not available — skipping.")
        return

    rows, cols = maze.shape
    fig, ax = plt.subplots(figsize=(max(6, cols), max(5, rows)))

    for r in range(rows):
        for c in range(cols):
            if maze[r, c]:
                ax.add_patch(patches.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    facecolor='#333333', edgecolor='none'))

    ep_ends   = np.where(timeouts)[0]
    ep_starts = np.concatenate([[0], ep_ends[:-1] + 1])
    colors    = plt.cm.tab10(np.linspace(0, 1, min(n_show, len(ep_starts))))

    for i in range(min(n_show, len(ep_starts))):
        s, e = ep_starts[i], ep_ends[i] + 1
        traj = observations[s:e, :2]          # col 0 = x(row), col 1 = y(col)
        ax.plot(traj[:, 1], traj[:, 0], '-', color=colors[i], lw=0.9, alpha=0.85)
        ax.plot(traj[0,  1], traj[0,  0], 'o', color=colors[i], ms=5, zorder=5)
        ax.plot(traj[-1, 1], traj[-1, 0], 's', color=colors[i], ms=5, zorder=5)

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.set_title(f"{min(n_show, len(ep_starts))} episodes  (○ start, □ end)")
    ax.set_xlabel("y (column)")
    ax.set_ylabel("x (row)")
    plt.tight_layout()
    plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument('--maze', choices=list(MAZE_MAPS),
                     help="Predefined maze layout.")
    grp.add_argument('--maze-file',
                     help=".npy file with a 2D bool array (True=wall).")

    parser.add_argument('--save',
                        help="Output .hdf5 path (e.g. data/maze2d-open-custom.hdf5). "
                             "The ITPS loader routes .hdf5 files through 4x downsampling.")
    parser.add_argument('--n-episodes',     type=int,   default=5000)
    parser.add_argument('--episode-length', type=int,   default=600,
                        help="Raw steps per episode at 40 Hz (default 600 → 150 effective "
                             "steps after 4x downsampling). Raw must exceed "
                             "4 × (n_obs_steps + horizon + drop_n_last_frames) = 292.")
    parser.add_argument('--n-goals',        type=int,   default=4,
                        help="Goals chained per episode (default 4). "
                             "Increase for small/open mazes so episodes stay active longer.")
    parser.add_argument('--dt',             type=float, default=0.025,
                        help="Simulation timestep in seconds (default 0.025 = 40 Hz). "
                             "After 4x downsampling the effective rate is 10 Hz, matching fps.")
    parser.add_argument('--kp',             type=float, default=5.0,
                        help="PD proportional gain (default 5.0).")
    parser.add_argument('--kd',             type=float, default=3.0,
                        help="PD derivative gain / damping (default 3.0). "
                             "Higher → less oscillation, smoother paths.")
    parser.add_argument('--noise-std',      type=float, default=0.05,
                        help="Gaussian noise on control force (default 0.05). "
                             "Higher → more deviation from planned path.")
    parser.add_argument('--lookahead',      type=int,   default=0,
                        help="Waypoints to look ahead when targeting (default 0). "
                             "Only useful if shortcutting is disabled; with shortcutted "
                             "paths, values >0 cause the agent to skip corners entirely.")
    parser.add_argument('--max-speed',      type=float, default=2.5,
                        help="Speed cap in maze units/s (default 2.5).")
    parser.add_argument('--clearance',      type=float, default=0.3,
                        help="Wall clearance for A* grid in maze units (default 0.5). "
                             "Max useful value is 0.5 for mazes with 1-cell-wide corridors.")
    parser.add_argument('--scale',          type=int,   default=5,
                        help="Fine grid cells per maze unit (default 5 → 0.2-unit spacing).")
    parser.add_argument('--style',          choices=['continuous', 'grid'],
                        default='continuous',
                        help="'continuous': fine-grid A* + shortcutting (default). "
                             "'grid': 4-connected BFS on integer cell centres, "
                             "no shortcutting — produces d4rl-style axis-aligned motion.")
    parser.add_argument('--seed',           type=int,   default=0)
    parser.add_argument('--viz',            action='store_true',
                        help="Show a trajectory plot after generation.")
    parser.add_argument('--viz-n',          type=int,   default=8,
                        help="Episodes to show in the plot (default 8).")
    args = parser.parse_args()

    # ── Load maze ──
    if args.maze is not None:
        maze = MAZE_MAPS[args.maze]
        print(f"Maze: '{args.maze}'  shape={maze.shape}")
    else:
        maze = np.load(args.maze_file).astype(bool)
        assert maze.ndim == 2, "Maze file must be a 2D array."
        print(f"Maze: loaded from '{args.maze_file}'  shape={maze.shape}")

    # ── Validate save path ──
    save_path = Path(args.save) if args.save else None
    if save_path and save_path.suffix != '.hdf5':
        print(f"WARNING: '{save_path.name}' does not end in .hdf5. "
              "The ITPS loader checks for 'hdf5' in the path to apply 4x downsampling.")

    # Raw episode_length must yield enough effective steps after 4x downsampling:
    # effective = episode_length // 4  >= n_obs_steps + horizon + drop_n_last_frames
    min_raw = 4 * (2 + 64 + 7)   # = 292
    if args.episode_length < min_raw:
        print(f"WARNING: episode_length={args.episode_length} < {min_raw} "
              "(4 × (n_obs_steps + horizon + drop_n_last_frames)).")

    print(f"Seed: {args.seed}")

    total = args.n_episodes * args.episode_length
    effective = total // 4
    cap_note = f"  (loader caps at 1M raw / 250K effective)" if total > 1_000_000 else ""
    print(f"\nGenerating {args.n_episodes} episodes × {args.episode_length} raw steps "
          f"= {total:,} raw frames  →  ~{effective:,} effective after 4x downsample{cap_note}\n")

    if args.style == 'grid':
        print(f"Style: grid (4-connected BFS on integer cell centres, no shortcutting)")
        observations, timeouts = generate_dataset_grid(
            maze           = maze,
            n_episodes     = args.n_episodes,
            episode_length = args.episode_length,
            dt             = args.dt,
            kp             = args.kp,
            kd             = args.kd,
            noise_std      = args.noise_std,
            max_speed      = args.max_speed,
            wp_thresh      = 0.25,
            n_goals        = args.n_goals,
            seed           = args.seed,
        )
    else:
        print(f"Style: continuous (fine-grid A* + shortcutting)")
        observations, timeouts = generate_dataset(
            maze            = maze,
            n_episodes      = args.n_episodes,
            episode_length  = args.episode_length,
            scale           = args.scale,
            clearance       = args.clearance,
            dt              = args.dt,
            kp              = args.kp,
            kd              = args.kd,
            noise_std       = args.noise_std,
            max_speed       = args.max_speed,
            wp_thresh       = 0.25,
            n_goals         = args.n_goals,
            lookahead       = args.lookahead,
            seed            = args.seed,
        )

    # ── Sanity checks ──
    print(f"\nSanity checks:")
    print(f"  observations : {observations.shape}  dtype={observations.dtype}")
    print(f"  timeouts     : {int(timeouts.sum())} episodes marked")
    print(f"  x range      : [{observations[:,0].min():.3f}, {observations[:,0].max():.3f}]")
    print(f"  y range      : [{observations[:,1].min():.3f}, {observations[:,1].max():.3f}]")
    speeds = np.linalg.norm(observations[:, 2:4], axis=1)
    print(f"  speed range  : [{speeds.min():.3f}, {speeds.max():.3f}]")

    rs = np.round(np.clip(observations[:, 0], 0, maze.shape[0]-1)).astype(int)
    cs = np.round(np.clip(observations[:, 1], 0, maze.shape[1]-1)).astype(int)
    n_wall = int(maze[rs, cs].sum())
    if n_wall:
        print(f"  Collision check: {n_wall} wall violations ({n_wall/len(observations)*100:.3f}%)")
    else:
        print(f"  Collision check: PASSED (0 wall violations)")

    # ── Save ──
    if save_path is not None:
        import h5py
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(save_path, 'w') as f:
            f.create_dataset('observations', data=observations, compression='gzip')
            f.create_dataset('timeouts',     data=timeouts,     compression='gzip')
            f.attrs['seed'] = args.seed
            f.attrs['maze'] = args.maze if args.maze is not None else args.maze_file
            f.attrs['n_episodes'] = args.n_episodes
            f.attrs['episode_length'] = args.episode_length
        effective = len(observations) // 4
        print(f"\nSaved to '{save_path}'  ({save_path.stat().st_size/1e6:.1f} MB)")
        print(f"  {len(observations):,} raw frames  →  ~{effective:,} effective frames after 4x downsample")
        if len(observations) > 1_000_000:
            print(f"  NOTE: loader caps at 1M raw frames; only {1_000_000//4:,} effective frames will be used.")
        print(f"\nIn your training config set:")
        print(f"  dataset_root: '{save_path}'")

    if args.viz:
        visualize(maze, observations, timeouts, n_show=args.viz_n)


if __name__ == "__main__":
    main()
