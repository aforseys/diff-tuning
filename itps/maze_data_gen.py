import sys
import os
import argparse
import json
import time
import numpy as np
import pygame
from collections import deque

from interact_maze2d import MazeEnv


def pick_start_positions(maze_type='large', n_positions=None, savepath=None):
    """
    Display the maze and let the user click valid (non-collision) starting
    positions. Each accepted click is printed and saved.

    Controls
    --------
    Left-click  : accept position (skipped silently if in collision)
    U (undo)    : remove last accepted position
    Q / Enter   : finish and return

    Parameters
    ----------
    maze_type   : 'open' | 'sparse' | 'large'
    n_positions : stop automatically after this many valid picks (None = unlimited)
    savepath    : directory to save JSON file

    Returns
    -------
    list of [x, y] positions in maze XY space
    """
    env = MazeEnv(maze_type)
    pygame.font.init()
    font = pygame.font.SysFont(None, 30)
    positions = []
    running = True

    while running:
        mouse_pos = np.array(pygame.mouse.get_pos())
        mouse_xy = env.gui2xy(mouse_pos)
        in_collision = env.check_collision(mouse_xy.reshape(1, 1, 2))[0]

        env.draw_maze_background()

        for xy in positions:
            gui = env.xy2gui(np.array(xy))
            pygame.draw.circle(env.screen, env.GREEN, (int(gui[0]), int(gui[1])), 12)
            pygame.draw.circle(env.screen, (0, 180, 0), (int(gui[0]), int(gui[1])), 12, 2)

        cursor_color = (180, 180, 180) if in_collision else env.BLUE
        pygame.draw.circle(env.screen, cursor_color,
                           (int(mouse_pos[0]), int(mouse_pos[1])), 12)

        status = "COLLISION — move away" if in_collision else "valid"
        caption = (f"Picked: {len(positions)}"
                   + (f"/{n_positions}" if n_positions else "")
                   + f"   [{status}]   U=undo  Q=done")
        surf = font.render(caption, True, (30, 30, 30), (220, 220, 220))
        env.screen.blit(surf, (8, 8))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if in_collision:
                    print(f"  x ({mouse_xy[0]:.2f}, {mouse_xy[1]:.2f}) — in collision, skipped")
                else:
                    positions.append(mouse_xy.tolist())
                    print(f"  #{len(positions):02d}  xy = {mouse_xy.round(3).tolist()}")
                    if n_positions and len(positions) >= n_positions:
                        running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_RETURN, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_u and positions:
                    removed = positions.pop()
                    print(f"  removed {removed}")

        env.clock.tick(30)

    pygame.quit()

    print(f"\nCollected {len(positions)} start positions:")
    for i, p in enumerate(positions):
        print(f"  [{i:02d}]  {p}")

    filename = f"set_obs_{len(positions)}_" + time.strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(savepath, filename), "w") as f:
        json.dump(positions, f, indent=2)
    print(f"Saved to {os.path.join(savepath, filename)}")

    return positions


def _bfs_distance_map(maze, start_cell):
    """Return dict mapping (r, c) -> BFS path distance from start_cell."""
    rows, cols = maze.shape
    start = (int(start_cell[0]), int(start_cell[1]))
    dist_map = {start: 0}
    queue = deque([(start[0], start[1], 0)])
    while queue:
        r, c, d = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not maze[nr, nc] and (nr, nc) not in dist_map:
                dist_map[(nr, nc)] = d + 1
                queue.append((nr, nc, d + 1))
    return dist_map


def _build_candidate_pool(env, clearance=0.8, grid_spacing=0.1):
    """Precompute all positions on a fine grid that pass collision and clearance checks."""
    wall_cells = np.argwhere(env.maze).astype(float)
    xs = np.arange(0.0, env.maze.shape[0], grid_spacing)
    ys = np.arange(0.0, env.maze.shape[1], grid_spacing)
    grid = np.array([[x, y] for x in xs for y in ys])
    collisions = env.check_collision(grid.reshape(-1, 1, 2))
    grid = grid[~collisions]
    dists = np.min(np.linalg.norm(grid[:, None, :] - wall_cells[None, :, :], axis=2), axis=1)
    grid = grid[dists >= clearance]
    return grid


def _sample_free(env, rng, candidates, existing=None, min_sep=1.5):
    """Sample a position from the candidate pool, at least `min_sep` from all existing positions."""
    if existing is not None and len(existing) > 0:
        existing_arr = np.array(existing)
        dists = np.min(np.linalg.norm(candidates[:, None, :] - existing_arr[None, :, :], axis=2), axis=1)
        valid = candidates[dists >= min_sep]
    else:
        valid = candidates
    if len(valid) == 0:
        print(f"  Warning: no position found with min_sep={min_sep}, relaxing to min_sep=1.0")
        dists = np.min(np.linalg.norm(candidates[:, None, :] - existing_arr[None, :, :], axis=2), axis=1)
        valid = candidates[dists >= 1.0]
    if len(valid) == 0:
        print(f"  Warning: still no position found, using full candidate pool")
        valid = candidates
    return valid[rng.integers(len(valid))]


def generate_random_observations(maze_type='large', n=10, include_goals=False,
                                  min_goal_dist=5, max_goal_dist=7,
                                  savepath=None, seed=0, split=None):
    """
    Generate N random legal observations in the maze, optionally with goals.

    Parameters
    ----------
    maze_type      : 'open' | 'sparse' | 'large'
    n              : number of observations to generate
    include_goals  : if True, also generate a goal for each obs (each entry is [x, y, goal_x, goal_y])
    min_goal_dist  : minimum BFS path distance (in cells) from obs to goal (default 5)
    max_goal_dist  : maximum BFS path distance (in cells) from obs to goal (default 7)
    savepath       : directory to save JSON file
    seed           : random seed for reproducibility (default 0)
    split          : if provided (e.g. 0.8), fraction used for train; saves separate train/test files

    Returns
    -------
    list of [x, y] or [x, y, goal_x, goal_y], or (train_list, test_list) if split is set
    """
    env = MazeEnv(maze_type)
    rng = np.random.default_rng(seed)
    candidates = _build_candidate_pool(env)

    positions = []
    starts_so_far = []
    for _ in range(n):
        start = _sample_free(env, rng, candidates, existing=starts_so_far)
        starts_so_far.append(start)
        if include_goals:
            start_cell = tuple(np.round(start).astype(int))
            dist_map = _bfs_distance_map(env.maze, start_cell)
            for _ in range(10000):
                goal = _sample_free(env, rng, candidates)
                goal_cell = tuple(np.round(goal).astype(int))
                d = dist_map.get(goal_cell)
                if d is not None and min_goal_dist <= d <= max_goal_dist:
                    break
            else:
                print(f"  Warning: no goal found in path distance [{min_goal_dist}, {max_goal_dist}] from {start.tolist()}, using random free position")
                goal = _sample_free(env, rng, candidates)
            positions.append(start.tolist() + goal.tolist())
        else:
            positions.append(start.tolist())

    prefix = 'set_obs_gc' if include_goals else 'set_obs'
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if include_goals:
        base_filename = f"{prefix}_{n}_min{min_goal_dist}_max{max_goal_dist}_seed{seed}_{timestamp}"
    else:
        base_filename = f"{prefix}_{n}_seed{seed}_{timestamp}"

    if split is not None:
        rng.shuffle(positions)
        n_train = int(len(positions) * split)
        train, test = positions[:n_train], positions[n_train:]
        for subset, tag in [(train, 'train'), (test, 'test')]:
            fname = f"{base_filename}_{tag}"
            with open(os.path.join(savepath, fname), 'w') as f:
                json.dump(subset, f, indent=2)
            print(f"Saved {len(subset)} {tag} observations to {os.path.join(savepath, fname)}")
        return train, test
    else:
        with open(os.path.join(savepath, base_filename), 'w') as f:
            json.dump(positions, f, indent=2)
        print(f"Saved {n} observations{'  (with goals)' if include_goals else ''} to {os.path.join(savepath, base_filename)}")
        return positions


def visualize_observations(positions, maze_type='large'):
    """
    Visualize observation (and goal) positions on the maze.

    Parameters
    ----------
    positions  : list of [x, y] or [x, y, goal_x, goal_y]
    maze_type  : 'open' | 'sparse' | 'large'
    """
    env = MazeEnv(maze_type)
    pygame.font.init()
    font = pygame.font.SysFont(None, 28)
    has_goals = len(positions[0]) == 4
    running = True

    while running:
        env.draw_maze_background()

        for i, pos in enumerate(positions):
            start_gui = env.xy2gui(np.array(pos[:2]))
            pygame.draw.circle(env.screen, env.agent_color,
                               (int(start_gui[0]), int(start_gui[1])), 12)
            if has_goals:
                goal_gui = env.xy2gui(np.array(pos[2:]))
                pygame.draw.circle(env.screen, env.goal_color,
                                   (int(goal_gui[0]), int(goal_gui[1])), 12)
                pygame.draw.line(env.screen, (180, 180, 180),
                                 (int(start_gui[0]), int(start_gui[1])),
                                 (int(goal_gui[0]), int(goal_gui[1])), 1)

        caption = f"{len(positions)} obs" + (" + goals" if has_goals else "") + "   Q=quit"
        surf = font.render(caption, True, (30, 30, 30), (220, 220, 220))
        env.screen.blit(surf, (8, 8))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_RETURN, pygame.K_ESCAPE):
                    running = False
        env.clock.tick(30)

    pygame.quit()


_DEFAULT_THRESHOLDS = {
    'similarity_score': 0.5,
    'endpoint_distance': 0.3,
    'collision_rate': 0.9,
}

def extract_preference_pairs(loadpath, savepath, maze_type='large', score_threshold=None, metric='similarity_score', metric_kwargs=None, viz=False, prefix=None):
    prefix = (prefix + '_' if prefix else '') + 'maze_' + time.strftime("%Y%m%d_%H%M%S")

    if score_threshold is None:
        score_threshold = _DEFAULT_THRESHOLDS[metric]

    maze_env = MazeEnv(maze_type)
    metric_kwargs = metric_kwargs or {}

    pairs = []
    with open(loadpath, "r") as f:
        trials = [json.loads(line) for line in f]

    for trial in trials:
        if metric == 'similarity_score':
            guide = np.array(trial["guide"])
            if len(guide) == 0:
                continue
            pred_traj = np.asarray(trial["pred_traj"], dtype=float)
            samples, scores = maze_env.similarity_score(pred_traj, guide)
        elif metric == 'collision_rate':
            xy_traj = np.array([[maze_env.gui2xy(p) for p in traj] for traj in trial["pred_traj"]])
            collisions = maze_env.check_collision(xy_traj)
            scores = (~collisions).astype(float)
            samples = np.asarray(trial["pred_traj"], dtype=float)
            guide = None
        elif metric == 'endpoint_distance':
            guide = np.array(trial["guide"])
            if len(guide) == 0:
                continue
            goal_gui = guide[0]  # GUI space, consistent with similarity_score
            pred_traj = np.asarray(trial["pred_traj"], dtype=float)  # (B, T, 2) GUI space
            dists = np.linalg.norm(pred_traj[:, -1, :] - goal_gui, axis=1)
            scores = 1.0 - dists / (dists.max() + 1e-6)  # closer endpoint = higher score
            samples = pred_traj
        else:
            raise NotImplementedError(f"Metric '{metric}' is not implemented.")

        trial_pairs = []
        for i in range(len(scores)):
            for j in range(i + 1, len(scores)):
                if abs(scores[i] - scores[j]) >= score_threshold:
                    winner, loser = (i, j) if scores[i] > scores[j] else (j, i)
                    trial_pairs.append({
                        "metric": metric,
                        "metric_kwargs": metric_kwargs,
                        "obs_idx": trial["trial_idx"],
                        "agent_pos": trial["agent_pos"],
                        "winner_traj": samples[winner].tolist(),
                        "loser_traj": samples[loser].tolist(),
                        "winner_score": float(scores[winner]),
                        "loser_score": float(scores[loser]),
                        **({"guide": guide.tolist()} if guide is not None else {}),
                    })
        pairs.extend(trial_pairs)

    print(f"Generated {len(pairs)} total preference pairs.")

    if viz and len(pairs) > 0:
        pair_idx = 0
        running = True
        while running and pair_idx < len(pairs):
            pair = pairs[pair_idx]
            print(f"Visualizing pair {pair_idx + 1}/{len(pairs)}")
            winner_traj = np.asarray(pair["winner_traj"], dtype=float)[None]
            loser_traj = np.asarray(pair["loser_traj"], dtype=float)[None]
            viz_traj = np.concatenate([winner_traj, loser_traj], axis=0)
            maze_env.draw_traj = np.asarray(pair["guide"], dtype=float) if "guide" in pair else []
            maze_env.agent_gui_pos = np.asarray(pair["agent_pos"], dtype=float)
            maze_env.update_screen(
                xy_pred=viz_traj,
                collisions=np.array([False, True]),
                keep_drawing=("guide" in pair),
                traj_in_gui_space=True,
            )
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        waiting = False
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_n:
                            pair_idx += 1
                            waiting = False
                        elif event.key == pygame.K_q:
                            waiting = False
                            running = False
                maze_env.clock.tick(10)
        pygame.quit()

    if not pairs:
        print("No pairs to save.")
        return []

    T = len(pairs[0]['winner_traj'])
    N = len(pairs)
    winners = np.zeros((N, 2 + T, 2), dtype=np.float32)
    losers = np.zeros((N, 2 + T, 2), dtype=np.float32)
    meta = []

    for i, pair in enumerate(pairs):
        agent_xy = maze_env.gui2xy(np.array(pair['agent_pos'], dtype=float))
        w_xy = np.array([maze_env.gui2xy(np.array(p, dtype=float))
                         for p in pair['winner_traj']], dtype=np.float32)
        l_xy = np.array([maze_env.gui2xy(np.array(p, dtype=float))
                         for p in pair['loser_traj']], dtype=np.float32)
        winners[i, 0:2] = agent_xy
        winners[i, 2:] = w_xy
        losers[i, 0:2] = agent_xy
        losers[i, 2:] = l_xy
        entry = {
            'obs_idx': pair['obs_idx'],
            'winner_score': float(pair['winner_score']),
            'loser_score': float(pair['loser_score']),
        }
        if 'guide' in pair:
            entry['guide'] = [maze_env.gui2xy(np.array(p, dtype=float)).tolist()
                              for p in pair['guide']]
        meta.append(entry)

    step_size = 2 + T
    timeouts = np.zeros(N * step_size, dtype=bool)
    timeouts[np.arange(1, N) * step_size - 1] = True

    assert np.all(losers[:, 0] == winners[:, 0])

    np.savez(os.path.join(savepath, f"{prefix}_winners.npz"),
             observations=winners.reshape(-1, 2), timeouts=timeouts)
    np.savez(os.path.join(savepath, f"{prefix}_losers.npz"),
             observations=losers.reshape(-1, 2), timeouts=timeouts)
    with open(os.path.join(savepath, f"{prefix}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {N} pairs → {prefix}_{{winners,losers}}.npz + _meta.json")
    print(f"  observations shape: {winners.reshape(-1, 2).shape}  (N*(1+T)={N}*{step_size}, 2)")

    return pairs


def visualize_preference_pairs(winners_path, losers_path, meta_path=None, maze_type='large', goal=None):
    """
    Load saved preference pair .npz files and replay them in a pygame window.

    Winner trajectory is shown in full color; loser is tinted white (same as
    the collision rendering used elsewhere).  The guide sketch is drawn in gray
    when a meta .json file is provided.

    Controls: N = next pair,  P = previous pair,  Q / Esc = quit
    """
    winners_obs = np.load(winners_path)['observations']   # (N*(2+T), 2)
    losers_obs  = np.load(losers_path)['observations']

    meta = None
    if meta_path is not None:
        with open(meta_path) as f:
            meta = json.load(f)

    # Infer N and step_size (step_size = 2 obs steps + T action steps)
    if meta is not None:
        N = len(meta)
        step_size = len(winners_obs) // N
    else:
        timeouts = np.load(winners_path)['timeouts']
        if timeouts.any():
            step_size = int(np.argmax(timeouts)) + 1
        else:
            step_size = len(winners_obs)   # single pair
        N = len(winners_obs) // step_size

    T = step_size - 2
    print(f"Loaded {N} pairs  (T={T} trajectory steps each)")

    winners_eps = winners_obs.reshape(N, step_size, 2)
    losers_eps  = losers_obs.reshape(N, step_size, 2)

    env = MazeEnv(maze_type)
    pygame.font.init()
    font = pygame.font.SysFont(None, 28)

    pair_idx = 0
    running  = True

    while running:
        pair_idx = max(0, min(pair_idx, N - 1))

        agent_xy    = winners_eps[pair_idx, 0]          # maze XY
        winner_traj = winners_eps[pair_idx, 2:]         # (T, 2) maze XY
        loser_traj  = losers_eps[pair_idx, 2:]          # (T, 2) maze XY

        env.agent_gui_pos = env.xy2gui(agent_xy)

        # Stack into (2, T, 2); winner first so it's drawn on top
        combined   = np.stack([loser_traj, winner_traj], axis=0)
        collisions = np.array([True, False])   # loser tinted, winner full color

        # Guide from meta (stored as XY, convert to GUI for draw_traj)
        goal_dist_winner = goal_dist_loser = None
        if meta is not None and meta[pair_idx].get('guide'):
            guide_xy = np.array(meta[pair_idx]['guide'])
            env.draw_traj = [env.xy2gui(p) for p in guide_xy]
            keep_drawing  = True
        else:
            env.draw_traj = []
            keep_drawing  = False

        # finetune_goal_dist: L2 from trajectory endpoint to goal (maze XY)
        # --goal takes priority (matches cfg.eval.goal exactly); falls back to guide endpoint
        if goal is not None:
            goal_xy = goal
            goal_dist_winner = float(np.linalg.norm(winner_traj[-1] - goal_xy))
            goal_dist_loser  = float(np.linalg.norm(loser_traj[-1]  - goal_xy))
            print(f"Pair {pair_idx + 1}/{N}  |  goal_dist  winner={goal_dist_winner:.4f}  loser={goal_dist_loser:.4f}  (eval goal)")
        elif meta is not None and meta[pair_idx].get('guide'):
            goal_xy = np.array(meta[pair_idx]['guide'])[-1]
            goal_dist_winner = float(np.linalg.norm(winner_traj[-1] - goal_xy))
            goal_dist_loser  = float(np.linalg.norm(loser_traj[-1]  - goal_xy))
            print(f"Pair {pair_idx + 1}/{N}  |  goal_dist  winner={goal_dist_winner:.4f}  loser={goal_dist_loser:.4f}  (guide endpoint)")

        env.update_screen(combined, collisions, keep_drawing=keep_drawing)

        # HUD
        caption = f"Pair {pair_idx + 1}/{N}"
        if meta is not None:
            caption += (f"  |  winner={meta[pair_idx]['winner_score']:.3f}"
                        f"  loser={meta[pair_idx]['loser_score']:.3f}")
        if goal_dist_winner is not None:
            caption += f"  |  goal_dist W={goal_dist_winner:.3f} L={goal_dist_loser:.3f}"
        caption += "   N=next  P=prev  Q=quit"
        surf = font.render(caption, True, (30, 30, 30), (220, 220, 220))
        env.screen.blit(surf, (8, 8))
        pygame.display.flip()

        # Block until a navigation key is pressed
        waiting = True
        while waiting and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_n:
                        pair_idx += 1
                        waiting = False
                    elif event.key == pygame.K_p:
                        pair_idx -= 1
                        waiting = False
                    elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                        waiting = running = False
            env.clock.tick(10)

    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-mt', '--maze_type', default='large', type=str, help="Maze type: open | sparse | large")
    parser.add_argument('-s', '--savepath', type=str, default=None, help="Directory to save output files")
    parser.add_argument('--pick-obs', action='store_true', help="Manually pick start positions")
    parser.add_argument('--gen-obs', action='store_true', help="Generate random observations")
    parser.add_argument('--n-obs', type=int, default=10, help="Number of observations to generate")
    parser.add_argument('--include-goals', action='store_true', help="Generate goal for each observation")
    parser.add_argument('--min-goal-dist', type=int, default=5, help="Min BFS path distance (cells) from obs to goal")
    parser.add_argument('--max-goal-dist', type=int, default=7, help="Max BFS path distance (cells) from obs to goal")
    parser.add_argument('--viz-obs', action='store_true', help="Visualize observations from file")
    parser.add_argument('--obs-file', type=str, default=None, help="Path to obs JSON file for visualization")
    parser.add_argument('--gen-pref', action='store_true', help="Generate preference pairs from saved trials")
    parser.add_argument('-l', '--loadpath', type=str, default=None, help="Path to trials file for preference generation")
    parser.add_argument('--score-threshold', type=float, default=None, help="Score threshold for preference pairs (default: metric-specific)")
    parser.add_argument('--metric', type=str, default='similarity_score')
    parser.add_argument('--viz-pref', action='store_true', help="Visualize preference pairs during generation")
    parser.add_argument('--prefix', type=str, default=None, help="Optional prefix to prepend to output filenames")
    parser.add_argument('--seed', type=int, default=0, help="Random seed for reproducibility (used by --gen-obs)")
    parser.add_argument('--split', type=float, default=None, help="Train fraction for train/test split (e.g. 0.8 for 80/20)")
    parser.add_argument('--viz-pairs', action='store_true', help="Visualize saved preference pairs from .npz files")
    parser.add_argument('--winners-file', type=str, default=None, help="Path to winners .npz file")
    parser.add_argument('--losers-file', type=str, default=None, help="Path to losers .npz file")
    parser.add_argument('--meta-file', type=str, default=None, help="Path to meta .json file (optional, enables guide overlay)")
    parser.add_argument('--goal', type=float, nargs=2, default=None, metavar=('X', 'Y'),
                        help="Fixed eval goal in maze XY space (matches cfg.eval.goal). "
                             "If omitted, falls back to the guide's last point.")
    args = parser.parse_args()

    if args.viz_pairs:
        assert args.winners_file is not None and args.losers_file is not None, \
            "Pass --winners-file and --losers-file to visualize pairs"
        visualize_preference_pairs(
            winners_path=args.winners_file,
            losers_path=args.losers_file,
            meta_path=args.meta_file,
            maze_type=args.maze_type,
            goal=np.array(args.goal, dtype=np.float32) if args.goal is not None else None,
        )
        sys.exit(0)

    if args.pick_obs:
        positions = pick_start_positions(maze_type=args.maze_type, savepath=args.savepath)
        if args.viz_obs:
            visualize_observations(positions, maze_type=args.maze_type)
        sys.exit(0)

    if args.gen_obs:
        assert args.savepath is not None, "Pass --savepath to save generated observations"
        positions = generate_random_observations(
            maze_type=args.maze_type,
            n=args.n_obs,
            include_goals=args.include_goals,
            min_goal_dist=args.min_goal_dist if args.include_goals else None,
            max_goal_dist=args.max_goal_dist if args.include_goals else None,
            savepath=args.savepath,
            seed=args.seed,
            split=args.split,
        )
        if args.viz_obs:
            all_positions = positions[0] + positions[1] if args.split is not None else positions
            visualize_observations(all_positions, maze_type=args.maze_type)
        sys.exit(0)

    if args.viz_obs:
        assert args.obs_file is not None, "Pass --obs-file to visualize existing observations"
        with open(args.obs_file) as f:
            positions = json.load(f)
        visualize_observations(positions, maze_type=args.maze_type)
        sys.exit(0)

    if args.gen_pref:
        assert args.loadpath is not None, "Pass --loadpath to generate preference pairs"
        extract_preference_pairs(
            loadpath=args.loadpath,
            savepath=args.savepath,
            maze_type=args.maze_type,
            score_threshold=args.score_threshold,
            metric=args.metric,
            viz=args.viz_pref,
            prefix=args.prefix,
        )
        sys.exit(0)
