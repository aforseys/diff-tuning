#!/usr/bin/env python

# Alexandra Forsey-Smerek
"""Evaluation for the maze2d environment: samples trajectories for saved start observations and scores
them with the shared maze metrics (no gym env needed)."""
import json

import numpy as np
import torch

from itps.common.utils.utils import seeded_context
from itps.envs.maze.maze_maps import MAZE_MAPS
from itps.envs.maze.maze_scoring import (
    check_maze_collision,
    score_center,
    score_bottom_half,
    score_goal_dist,
    score_goal_progress,
)


def eval_maze(policy, cfg, split='test', seed=None):
    """
    Offline trajectory evaluation for maze2d without running the gym env.

    For each starting obs in eval_cfg.{split}_obs, generates cfg.eval.n_samples
    trajectories per obs and computes the requested metrics.

    Config fields:
      env_type:          "large" | "sparse" | "open" (top level, not under eval)
      eval.n_samples:    trajectories sampled per observation
      eval.methods:      list of "ired" | "ddim"
      eval.opt_params:   one entry per IRED variant to evaluate
      eval.goal:         [x, y] target, required by "finetune_goal_dist"
      eval.metrics:      list of "collision_rate" | "obs_goal_dist" |
                         "finetune_goal_dist" | "center_rate" | "bottom_half_rate".
                         The two goal metrics additionally report <metric>_pct
                         (signed percentage-to-goal) and <metric>_pct_clipped.
                         "nan_rate" is always reported.
      eval.train_obs / eval.test_obs: JSON file of [state_x, state_y] or
                         [state_x, state_y, goal_x, goal_y] for goal-conditioned policies
    """
    if seed is None:
        return _eval_maze(policy, cfg, split)
    # seeded_context, not set_global_seed: this is called from inside the training
    # loop, so seeding globally would leave torch/numpy/random reseeded for every
    # subsequent training step -- making the diffusion noise (torch.randn/randint in
    # modeling_diffusion) repeat with period eval_freq. Restoring the caller's state
    # on exit keeps eval reproducible -- the same trajectories are sampled every time
    # -- without perturbing training's stream.
    with seeded_context(seed):
        return _eval_maze(policy, cfg, split)


def _eval_maze(policy, cfg, split):
    obs_file = cfg.eval.train_obs if split == 'train' else cfg.eval.test_obs
    if obs_file is None:
        return {}

    maze = MAZE_MAPS[cfg.env_type]
    device = next(policy.parameters()).device
    n_samples = cfg.eval.n_samples
    metrics = list(cfg.eval.metrics)
    n_obs_steps = policy.config.n_obs_steps
    opt_params = list(cfg.eval.opt_params)

    #Read in obs
    with open(obs_file, 'r') as f:                                                                                    
        positions = json.load(f)  # [[x0,y0], [x1,y1], ...]                                                           
    obs_data = np.array(positions, dtype=np.float32)  # (N_obs, 2)                                                    
    N_obs = len(obs_data)  
    if policy.use_goal_cond: 
        start_pos = obs_data[:, :2]
        goal_pos = obs_data[:, 2:]
    else:
        start_pos = obs_data

    states = np.repeat(start_pos, n_samples, axis=0)
    state_t = torch.tensor(states, dtype=torch.float32, device=device)  

    # (N_obs*n_samples, n_obs_steps, 2) — repeat starting pos for all history steps
    obs = {
        'observation.state':
            state_t.unsqueeze(1).expand(-1, n_obs_steps, -1).clone(),
        'observation.environment_state':
            state_t.unsqueeze(1).expand(-1, n_obs_steps, -1).clone(),
    }

    if policy.use_goal_cond:
        goals = np.repeat(goal_pos, n_samples, axis=0)
        goal_t = torch.tensor(goals, dtype=torch.float32, device=device)
        obs['episode_goal'] = goal_t.unsqueeze(1).clone()

    chunk_size = 256
    total = state_t.shape[0]
    all_chunks = None
    with torch.no_grad():
        for start in range(0, total, chunk_size):
            chunk_obs = {k: v[start:start + chunk_size] for k, v in obs.items()}
            _, chunk_full_trajs = policy.run_inference(chunk_obs, methods=list(cfg.eval.methods), opt_params=opt_params, return_full=True)
            if all_chunks is None:
                all_chunks = [[t.cpu()] for t in chunk_full_trajs]
            else:
                for i, t in enumerate(chunk_full_trajs):
                    all_chunks[i].append(t.cpu())

    # run_inference unnormalizes --> trajectories are in coordinate space
    # each entry: (N_obs * n_samples, horizon, 2)
    trajs = [torch.cat(chunks).numpy() for chunks in all_chunks]

    metrics_dict ={}
    for i, traj in enumerate(trajs):
        per_obs ={}

        nan_steps = np.isnan(traj).any(axis=-1)  # (N_obs*n_samples, horizon)
        nan_traj = nan_steps.any(axis=-1)         # (N_obs*n_samples,)
        per_obs['nan_rate'] = nan_traj.reshape(N_obs, n_samples).mean(axis=1).tolist()

        for m in metrics:
            if m == 'collision_rate':
                collisions = check_maze_collision(traj, maze)
                per_obs[m] = collisions.reshape(N_obs, n_samples).mean(axis=1).tolist() 
            elif m == 'obs_goal_dist':
                assert policy.use_goal_cond, "obs_goal_dist requires a goal-conditioned policy"
                goals_repeated = np.repeat(goal_pos, n_samples, axis=0)
                dists = score_goal_dist(traj, goals_repeated)
                per_obs[m] = dists.reshape(N_obs, n_samples).mean(axis=1).tolist()
                progress = score_goal_progress(traj, goals_repeated, states, clip=False)
                per_obs[f'{m}_pct'] = progress.reshape(N_obs, n_samples).mean(axis=1).tolist()
                per_obs[f'{m}_pct_clipped'] = np.clip(progress, 0, None).reshape(N_obs, n_samples).mean(axis=1).tolist()
            elif m == 'finetune_goal_dist':
                assert cfg.eval.goal is not None, "finetune_goal_dist requires cfg.eval.goal to be set"
                goal = np.array(cfg.eval.goal, dtype=np.float32)
                dists = score_goal_dist(traj, goal)
                per_obs[m] = dists.reshape(N_obs, n_samples).mean(axis=1).tolist()
                progress = score_goal_progress(traj, goal, states, clip=False)
                per_obs[f'{m}_pct'] = progress.reshape(N_obs, n_samples).mean(axis=1).tolist()
                per_obs[f'{m}_pct_clipped'] = np.clip(progress, 0, None).reshape(N_obs, n_samples).mean(axis=1).tolist()
            elif m == 'center_rate':
                scores = score_center(traj, maze)  # (N_obs*n_samples,)
                per_obs[m] = scores.reshape(N_obs, n_samples).mean(axis=1).tolist()
            elif m == 'bottom_half_rate':
                scores = score_bottom_half(traj, maze)  # (N_obs*n_samples,)
                per_obs[m] = scores.reshape(N_obs, n_samples).mean(axis=1).tolist()

            else:
                raise NotImplementedError(f"Metric '{m}' not implemented") 

        if 'ddim' in list(cfg.eval.methods) and i == len(trajs) - 1:
            label = "DDIM"

        else:
            label = f'IRED_{opt_params[i]["n_opt"]}steps'
            if opt_params[i]["t_subset"] is not None:
                label+=f'_last{opt_params[i]["t_subset"]}'
            if opt_params[i]["denoise"]:
                label+='_denoise'

        metrics_dict[label] ={
            f"{split}_{m}": {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "per_obs": vals}
            for m, vals in per_obs.items()
        }
    
    return metrics_dict
