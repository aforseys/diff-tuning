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


def _bfs_sample_goal(maze, start, min_steps, max_steps, rng):
    rows, cols = maze.shape
    visited = np.zeros((rows, cols), dtype=bool)
    visited[start[0], start[1]] = True
    queue = deque([(int(start[0]), int(start[1]), 0)])
    candidates = []

    while queue:
        r, c, dist = queue.popleft()
        if (min_steps is None or dist >= min_steps) and (max_steps is None or dist <= max_steps):
            if not (r == start[0] and c == start[1]):
                candidates.append((r, c))
        if max_steps is not None and dist >= max_steps:
            continue
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not maze[nr, nc] and not visited[nr, nc]:
                visited[nr, nc] = True
                queue.append((nr, nc, dist + 1))

    if not candidates:
        return None
    return candidates[rng.integers(len(candidates))]


def generate_random_observations(maze_type='large', n=10, include_goals=False,
                                  min_goal_steps=None, max_goal_steps=None,
                                  savepath=None):
    """
    Generate N random legal observations in the maze, optionally with goals.

    Parameters
    ----------
    maze_type       : 'open' | 'sparse' | 'large'
    n               : number of observations to generate
    include_goals   : if True, also generate a goal for each obs
    min_goal_steps  : minimum BFS steps from obs to goal
    max_goal_steps  : maximum BFS steps from obs to goal
    savepath        : directory to save JSON file

    Returns
    -------
    list of [x, y] or [x, y, goal_x, goal_y] in maze XY space
    """
    env = MazeEnv(maze_type)
    maze = env.maze
    free_cells = np.argwhere(~maze)
    rng = np.random.default_rng()

    replace = n > len(free_cells)
    starts = free_cells[rng.choice(len(free_cells), size=n, replace=replace)]

    positions = []
    for start in starts:
        x, y = float(start[0]), float(start[1])
        if include_goals:
            goal = _bfs_sample_goal(maze, start, min_goal_steps, max_goal_steps, rng)
            if goal is None:
                goal = free_cells[rng.integers(len(free_cells))]
            positions.append([x, y, float(goal[0]), float(goal[1])])
        else:
            positions.append([x, y])

    prefix = 'set_obs_gc' if include_goals else 'set_obs'
    filename = f"{prefix}_{n}_" + time.strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(savepath, filename), 'w') as f:
        json.dump(positions, f, indent=2)
    print(f"Saved {n} observations to {os.path.join(savepath, filename)}")

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
            pygame.draw.circle(env.screen, env.GREEN,
                               (int(start_gui[0]), int(start_gui[1])), 12)
            if has_goals:
                goal_gui = env.xy2gui(np.array(pos[2:]))
                pygame.draw.circle(env.screen, env.goal_color,
                                   (int(goal_gui[0]), int(goal_gui[1])), 12)
                pygame.draw.line(env.screen, (180, 180, 180),
                                 (int(start_gui[0]), int(start_gui[1])),
                                 (int(goal_gui[0]), int(goal_gui[1])), 1)
            label = font.render(str(i), True, (255, 255, 255))
            env.screen.blit(label, (int(start_gui[0]) + 8, int(start_gui[1]) - 8))

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


def extract_preference_pairs(loadpath, savepath, maze_type='large', score_threshold=0.3, metric='similarity_score', metric_kwargs=None, viz=False):
    prefix = 'maze_' + time.strftime("%Y%m%d_%H%M%S")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-mt', '--maze_type', default='large', type=str, help="Maze type: open | sparse | large")
    parser.add_argument('-s', '--savepath', type=str, required=True, help="Directory to save output files")
    parser.add_argument('--pick-obs', action='store_true', help="Manually pick start positions")
    parser.add_argument('--gen-obs', action='store_true', help="Generate random observations")
    parser.add_argument('--n-obs', type=int, default=10, help="Number of observations to generate")
    parser.add_argument('--include-goals', action='store_true', help="Generate goal for each observation")
    parser.add_argument('--min-goal-steps', type=int, default=20, help="Min BFS steps from obs to goal")
    parser.add_argument('--max-goal-steps', type=int, default=40, help="Max BFS steps from obs to goal")
    parser.add_argument('--viz-obs', action='store_true', help="Visualize observations from file")
    parser.add_argument('--obs-file', type=str, default=None, help="Path to obs JSON file for visualization")
    parser.add_argument('--gen-pref', action='store_true', help="Generate preference pairs from saved trials")
    parser.add_argument('-l', '--loadpath', type=str, default=None, help="Path to trials file for preference generation")
    parser.add_argument('--score-threshold', type=float, default=0.3)
    parser.add_argument('--metric', type=str, default='similarity_score')
    parser.add_argument('--viz-pref', action='store_true', help="Visualize preference pairs during generation")
    args = parser.parse_args()

    if args.pick_obs:
        positions = pick_start_positions(maze_type=args.maze_type, savepath=args.savepath)
        if args.viz_obs:
            visualize_observations(positions, maze_type=args.maze_type)
        sys.exit(0)

    if args.gen_obs:
        positions = generate_random_observations(
            maze_type=args.maze_type,
            n=args.n_obs,
            include_goals=args.include_goals,
            min_goal_steps=args.min_goal_steps if args.include_goals else None,
            max_goal_steps=args.max_goal_steps if args.include_goals else None,
            savepath=args.savepath,
        )
        if args.viz_obs:
            visualize_observations(positions, maze_type=args.maze_type)
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
        )
        sys.exit(0)
