#!/usr/bin/env python

# Alexandra Forsey-Smerek
"""Evaluation for the maze2d environment: samples trajectories for saved start observations and scores
them with the shared maze metrics (no gym env needed)."""
import json
from contextlib import nullcontext

import numpy as np
import torch

from itps.common.utils.preference_scoring import dpo_reward, policy_win_rate
from itps.common.utils.utils import seeded_context
from itps.envs.maze.maze_maps import MAZE_MAPS
from itps.envs.maze.maze_scoring import (
    DEFAULT_SCORE_THRESHOLDS,
    check_maze_collision,
    score_center,
    score_bottom_half,
    score_goal_dist,
    score_goal_progress,
    score_metric,
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


_ITEMS_CACHE = {}


def _load_items(path, cfg, policy, device, episode_starts=False, n_items=None, seed=0):
    """
    Load a maze dataset file and return (trajs, obs) for a selection of its items.

    episode_starts: one item per episode (the preference test set, where each episode is
        one candidate trajectory). Otherwise `n_items` items drawn WITHOUT replacement
        from every row whose action window lies wholly inside its episode -- no padded
        slots, since the denoising error is computed without a mask.

    The draw is seeded, so the same states are scored at every eval and by every model;
    taking the first `n_items` instead would sample only the opening episodes of the file.

    That makes the result identical at every eval, so it is cached. Uncached, each eval
    rebuilds a LeRobotDataset per file -- for eval.dmse_dataset a re-read of the hdf5 and
    a reset_episode_index .map() over ~250k rows -- and re-reads every window.
    """
    key = (str(path), episode_starts, n_items, seed, policy.use_goal_cond, str(device))
    if key in _ITEMS_CACHE:
        return _ITEMS_CACHE[key]

    from itps.common.datasets.factory import resolve_delta_timestamps
    from itps.common.datasets.lerobot_dataset import LeRobotDataset

    resolve_delta_timestamps(cfg)  # idempotent; eval.py may run before make_dataset
    kwargs = {'delta_timestamps': cfg.training.get('delta_timestamps'), 'load_images': False}
    if policy.use_goal_cond:
        kwargs['goal_horizon'] = cfg.training.goal_horizon
    ds = LeRobotDataset(cfg.dataset_repo_id, root=path, **kwargs)

    starts = ds.episode_data_index['from'].tolist()
    ends = ds.episode_data_index['to'].tolist()
    if episode_starts:
        rows = np.array(starts)
    else:
        # An item at row r queries action rows r-(n_obs-1) .. r+(horizon-n_obs); keep only
        # the rows where that whole window is inside the episode.
        n_obs, horizon = policy.config.n_obs_steps, policy.config.horizon
        rows = np.array([r for f, t in zip(starts, ends)
                         for r in range(f + n_obs - 1, t - (horizon - n_obs))])
        assert len(rows) > 0, (
            f"No unpadded action windows in {path}: every episode is shorter than "
            f"horizon {horizon}."
        )

    if n_items is not None and n_items < len(rows):
        # Drawn from the row COUNT, not the row values, so winners and losers -- which have
        # the same episode count -- select the same pair indices and stay aligned.
        pick = np.random.default_rng(seed).choice(len(rows), size=n_items, replace=False)
        rows = rows[np.sort(pick)]
    rows = rows.tolist()

    items = [ds[r] for r in rows]
    assert not any(it['action_is_pad'].any() for it in items) or episode_starts, \
        "Expected unpadded action windows; the row-selection arithmetic is wrong."

    trajs = torch.stack([it['action'] for it in items]).to(device)
    keys = ['observation.state', 'observation.environment_state']
    if policy.use_goal_cond:
        keys.append('episode_goal')
    obs = {k: torch.stack([it[k] for it in items]).to(device) for k in keys}
    _ITEMS_CACHE[key] = (trajs, obs)
    return trajs, obs


def eval_maze_preference(policy, cfg, ref_policy=None, seed=None):
    """
    Preference metrics on fixed held-out sets, the maze counterpart of
    `envs.gmm.eval.eval_gmm_preference`. Returns a dict for info["aggregated"]:

      win_rate_<score>  over `eval.pref_test_set` -- pre-paired winner/loser trajectories
          that share a start observation, so only those comparisons are made. <score> is
          "energy" for energy-based policies and "dpo" for DPO-finetuned ones (both, when
          a DPO run uses an EBM).
      dmse_test         mean denoising error on `eval.dmse_dataset`, the ORIGINAL
          demonstration data -- how much of the pretrained behaviour finetuning kept.
          No reference model, so it is comparable across objectives.

    The ranking math is `preference_scoring.policy_win_rate`; this function only supplies
    the trajectories, their observations, and the ground-truth scores.
    """
    out = {}
    device = next(policy.parameters()).device
    has_energy = hasattr(policy, "get_energy")
    is_dpo = bool(cfg.policy.get("finetune_dpo") or cfg.policy.get("finetune_dpo_forward_kl"))

    with seeded_context(seed) if seed is not None else nullcontext():
        pref_set = cfg.eval.get("pref_test_set")
        if pref_set is not None:
            metric = cfg.eval.pref_metric
            # Same n_items and seed on both, so the same pair indices are drawn from each.
            cap = cfg.eval.get("pref_test_n")
            pick = dict(episode_starts=True, n_items=cap, seed=cfg.eval.get("pref_test_seed", 0))
            w_trajs, w_obs = _load_items(pref_set['winners'], cfg, policy, device, **pick)
            l_trajs, l_obs = _load_items(pref_set['losers'], cfg, policy, device, **pick)
            n = len(w_trajs)
            assert len(l_trajs) == n, f"{n} winners but {len(l_trajs)} losers"

            # One pool of 2N trajectories; pair k is (winner k, loser k), which share a start.
            trajs = torch.cat([w_trajs, l_trajs])
            obs = {k: torch.cat([w_obs[k], l_obs[k]]) for k in w_obs}
            pairs = np.stack([np.arange(n), np.arange(n) + n], axis=1)

            starts = obs['observation.state'][:, -1].cpu().numpy()   # current position
            goals = obs['episode_goal'][:, 0].cpu().numpy() if policy.use_goal_cond else None
            gt_scores = score_metric(metric, trajs.cpu().numpy(), MAZE_MAPS[cfg.env_type],
                                     start=starts, obs_goal=goals,
                                     fixed_goal=np.array(cfg.eval.goal, dtype=np.float32)
                                     if cfg.eval.get("goal") is not None else None)

            tie_tol = cfg.eval.get("default_tie_tol", True)
            tie_tol = DEFAULT_SCORE_THRESHOLDS[metric] if tie_tol is True else tie_tol[metric]

            for score in (["energy"] if has_energy else []) + (["dpo"] if is_dpo else []):
                win_rate = policy_win_rate(policy, trajs, [obs], gt_scores, score=score,
                                           ref_policy=ref_policy if cfg.policy.get("finetune_dpo") else None,
                                           tie_tol=tie_tol, pairs=pairs)[0]
                out[f"win_rate_{score}"] = win_rate

            # D-MSE on each side of the held-out pairs, alongside dmse_test on the demos
            # below: the same denoising error, so the three share (t, eps) draws and are
            # directly comparable. Reuses the tensors already loaded above. Note these
            # windows are episode starts, so one of the 64 action slots is edge padding
            # that dpo_reward does not mask, unlike the unpadded dmse_test windows.
            for side, (s_trajs, s_obs) in (("winners", (w_trajs, w_obs)),
                                           ("losers", (l_trajs, l_obs))):
                out[f"dmse_{side}"] = float(-dpo_reward(policy, s_trajs, [s_obs])[0].mean())

        dmse_set = cfg.eval.get("dmse_dataset")
        if dmse_set is not None:
            trajs, obs = _load_items(dmse_set, cfg, policy, device,
                                     n_items=cfg.eval.get("dmse_n", 1000),
                                     seed=cfg.eval.get("dmse_seed", 0))
            # No ref_policy: D-MSE is this policy's own error, comparable across objectives.
            out["dmse_test"] = float(-dpo_reward(policy, trajs, [obs])[0].mean())

    return out
