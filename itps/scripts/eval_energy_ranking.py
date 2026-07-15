#!/usr/bin/env python
"""
Check whether a fine-tuned diffusion policy's energy landscape ranks maze
trajectories the way a given preference metric would.

For each observation in an obs file:
  1. Sample `--n-samples` candidate trajectories from the ORIGINAL
     (pre-fine-tune) policy.
  2. Score each candidate under one or more maze preference metrics
     (collision_rate, center_rate, bottom_half_rate, obs_goal_dist,
     finetune_goal_dist).
  3. Compute the energy of each candidate under the FINE-TUNED policy, at one
     or more energy-landscape timesteps (--energy-timesteps, default just t=0;
     trajectories are scaled to the timestep with no noise added, as in
     get_traj_energies).
  4. Report the pairwise win rate (see
     itps.common.utils.preference_scoring.pairwise_win_rate): over every pair of
     candidates, how often does the fine-tuned model's (negative) energy order
     the pair the same way the metric does? Pairs whose metric scores are tied
     are thrown out; the fraction thrown out is also reported.

When more than one energy timestep is given, the win rate is reported once per
timestep plus two cross-timestep aggregate scorers:
  - mean_energy: candidates ranked by their energy averaged over the timesteps.
  - mean_rank: candidates ranked within each timestep, then ranked by their
    average rank (robust to energy scale differing across timesteps).

By default, --obs-file/--maze-type/--goal are read directly from the fine-tuned
checkpoint's own saved config.yaml (Logger.save_model writes one into every
pretrained_model dir alongside the weights) — so this reuses the exact same
eval.train_obs/test_obs, env_type, and eval.goal the fine-tuning run itself
used. Pass any of --obs-file/--maze-type/--goal explicitly to override.

Run from the itps/ directory:
    conda run -n diffpreff python scripts/eval_energy_ranking.py \\
        --pretrained-path data/maze2d_dp/large/general/train/.../pretrained_model \\
        --finetuned-path  data/maze2d_dp/large/general/tune/.../pretrained_model \\
        --split test --n-samples 32 \\
        --metrics collision_rate
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from itps.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from itps.common.utils.maze_maps import MAZE_MAPS
from itps.common.utils.maze_scoring import (
    check_maze_collision,
    score_center,
    score_bottom_half,
    score_goal_progress,
)
from itps.common.utils.preference_scoring import pairwise_win_rate
from itps.common.utils.utils import set_global_seed


def score_metric(metric, xy_traj, maze, start=None, obs_goal=None, fixed_goal=None):
    """Per-trajectory scores for one metric. Higher = more preferred.

    The goal metrics use the same percentage-to-goal calc as eval_maze
    (score_goal_progress): obs_goal_dist toward the per-obs goal-conditioned
    goal, finetune_goal_dist toward a fixed goal.
    """
    if metric == 'collision_rate':
        return (~check_maze_collision(xy_traj, maze)).astype(float)
    elif metric == 'center_rate':
        return score_center(xy_traj, maze)
    elif metric == 'bottom_half_rate':
        return score_bottom_half(xy_traj, maze)
    elif metric == 'obs_goal_dist':
        assert obs_goal is not None, \
            "'obs_goal_dist' requires a goal-conditioned obs file (x,y,goal_x,goal_y)"
        assert start is not None, "'obs_goal_dist' requires the start position"
        return score_goal_progress(xy_traj, obs_goal, start)
    elif metric == 'finetune_goal_dist':
        assert fixed_goal is not None, \
            "'finetune_goal_dist' requires --goal (or eval.goal in the checkpoint config)"
        assert start is not None, "'finetune_goal_dist' requires the start position"
        return score_goal_progress(xy_traj, fixed_goal, start)
    else:
        raise NotImplementedError(f"Metric '{metric}' is not implemented.")


def eval_energy_ranking(pretrained_policy, finetuned_policy, obs_data, maze_type,
                         metrics, n_samples=32, sampler='ddim', goal=None,
                         chunk_size=256, seed=None, energy_timesteps=(0,)):
    """
    obs_data: (N_obs, 2) or (N_obs, 4) array — [x,y] or [x,y,goal_x,goal_y], maze XY space.
    energy_timesteps: train-timestep indices of the energy landscapes to score
        candidates in. With num_inference_steps == num_train_timesteps (the
        default) the last k inference landscapes are simply t = 0..k-1.

    Returns: {metric: {scorer: {"per_obs": [...], "mean_win_rate": float, ...}}}
    where scorer is "t=<n>" per timestep, plus "mean_energy" and "mean_rank"
    aggregates when more than one timestep is given.
    """
    if seed is not None:
        set_global_seed(seed)

    maze = MAZE_MAPS[maze_type]
    device = next(pretrained_policy.parameters()).device

    is_goal_cond = pretrained_policy.use_goal_cond
    assert finetuned_policy.use_goal_cond == is_goal_cond, \
        "Pretrained and fine-tuned policies must agree on goal-conditioning."
    n_obs_steps = pretrained_policy.config.n_obs_steps

    N_obs = len(obs_data)
    start_pos = obs_data[:, :2]
    goal_pos = obs_data[:, 2:] if is_goal_cond else None

    states = np.repeat(start_pos, n_samples, axis=0)
    state_t = torch.tensor(states, dtype=torch.float32, device=device)
    obs = {
        'observation.state':
            state_t.unsqueeze(1).expand(-1, n_obs_steps, -1).clone(),
        'observation.environment_state':
            state_t.unsqueeze(1).expand(-1, n_obs_steps, -1).clone(),
    }
    if is_goal_cond:
        goals = np.repeat(goal_pos, n_samples, axis=0)
        goal_t = torch.tensor(goals, dtype=torch.float32, device=device)
        obs['episode_goal'] = goal_t.unsqueeze(1).clone()

    # Sample candidates from the ORIGINAL policy, score their energy under the
    # FINE-TUNED policy at each requested landscape timestep, chunked to bound
    # peak memory (same pattern as eval_maze).
    energy_timesteps = list(energy_timesteps)
    total = state_t.shape[0]
    traj_chunks = []
    energy_chunks = {t: [] for t in energy_timesteps}
    with torch.no_grad():
        for start in range(0, total, chunk_size):
            chunk_obs = {k: v[start:start + chunk_size] for k, v in obs.items()}
            _, chunk_full_trajs = pretrained_policy.run_inference(
                chunk_obs, methods=[sampler], return_full=True
            )
            chunk_traj = chunk_full_trajs[0]  # (chunk, horizon, 2), unnormalized maze XY space
            traj_chunks.append(chunk_traj.cpu())
            for t in energy_timesteps:
                chunk_energy = finetuned_policy.get_energy(
                    action_batch={'action': chunk_traj}, t=t, observation_batch=chunk_obs
                )
                energy_chunks[t].append(chunk_energy.squeeze(-1).cpu())

    traj_all = torch.cat(traj_chunks, dim=0).numpy().reshape(N_obs, n_samples, -1, 2)
    # (n_ts, N_obs, n_samples)
    energy_all = np.stack([
        torch.cat(energy_chunks[t], dim=0).numpy().reshape(N_obs, n_samples)
        for t in energy_timesteps
    ])

    # Each scorer maps to a (N_obs, n_samples) value array where higher = more
    # preferred by the model, matching the `scores` convention.
    scorers = {f"t={t}": -energy_all[i] for i, t in enumerate(energy_timesteps)}
    if len(energy_timesteps) > 1:
        scorers["mean_energy"] = -energy_all.mean(axis=0)
        # Rank candidates within each (timestep, obs): rank 0 = lowest energy =
        # most preferred. Averaging ranks is robust to the energy scale
        # differing across timesteps.
        ranks = energy_all.argsort(axis=-1).argsort(axis=-1)
        scorers["mean_rank"] = -ranks.mean(axis=0)

    results = {m: {s: {"per_obs": []} for s in scorers} for m in metrics}
    for obs_idx in range(N_obs):
        obs_goal = goal_pos[obs_idx] if is_goal_cond else None
        for m in metrics:
            scores = score_metric(
                m, traj_all[obs_idx], maze,
                start=start_pos[obs_idx], obs_goal=obs_goal, fixed_goal=goal,
            )
            for s, values in scorers.items():
                win_rate, n_used, n_tied = pairwise_win_rate(scores, values[obs_idx])
                results[m][s]["per_obs"].append({
                    "obs_idx": obs_idx, "win_rate": win_rate,
                    "n_pairs_used": n_used, "n_pairs_tied": n_tied,
                })

    for m in metrics:
        for s in scorers:
            r = results[m][s]
            per = r["per_obs"]
            valid = [x["win_rate"] for x in per if not np.isnan(x["win_rate"])]
            total_used = sum(x["n_pairs_used"] for x in per)
            total_tied = sum(x["n_pairs_tied"] for x in per)
            total_pairs = total_used + total_tied
            r["mean_win_rate"] = float(np.mean(valid)) if valid else float('nan')
            r["n_obs_with_pairs"] = len(valid)
            r["n_pairs_used"] = total_used
            r["n_pairs_tied"] = total_tied
            r["pct_pairs_thrown_out"] = (
                float(100.0 * total_tied / total_pairs) if total_pairs else float('nan')
            )

    return results


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pretrained-path", required=True,
                        help="Path to the original (pre-fine-tune) pretrained_model dir")
    parser.add_argument("--finetuned-path", required=True,
                        help="Path to the fine-tuned pretrained_model dir")
    parser.add_argument("--obs-file", default=None,
                        help="JSON file of [x,y] or [x,y,goal_x,goal_y] observations, maze XY space. "
                             "Defaults to eval.train_obs/test_obs (per --split) read from the "
                             "fine-tuned checkpoint's own config.yaml.")
    parser.add_argument("--split", default="test", choices=["train", "test"],
                        help="Which obs split to read from the fine-tuned checkpoint's "
                             "eval.train_obs/test_obs (default: test; ignored if --obs-file is "
                             "given explicitly)")
    parser.add_argument("--maze-type", default=None, choices=list(MAZE_MAPS.keys()),
                        help="Maze layout. Defaults to env_type read from the fine-tuned "
                             "checkpoint's config.yaml.")
    parser.add_argument("--n-samples", type=int, default=32,
                        help="Candidate trajectories sampled per obs (default 32)")
    parser.add_argument("--metrics", nargs="+", required=True,
                        help="Metric(s) to check energy ranking against, e.g. "
                             "'--metrics collision_rate' or '--metrics collision_rate "
                             "center_rate bottom_half_rate obs_goal_dist finetune_goal_dist'")
    parser.add_argument("--energy-timesteps", type=int, nargs="+", default=[0],
                        help="Train-timestep indices of the energy landscapes to score "
                             "candidates in (default: 0). E.g. '--energy-timesteps 0 1 2 3 4' "
                             "scores the last 5 inference landscapes (num_inference_steps == "
                             "num_train_timesteps by default). With multiple timesteps, "
                             "per-timestep win rates plus mean_energy/mean_rank aggregates "
                             "are reported.")
    parser.add_argument("--sampler", default="ddim", choices=["ddim", "ired"],
                        help="Sampling method used to generate candidates from the "
                             "pretrained (non-fine-tuned) policy (default: ddim)")
    parser.add_argument("--goal", type=float, nargs=2, default=None, metavar=("X", "Y"),
                        help="Fixed goal in maze XY space, used for 'finetune_goal_dist'. "
                             "Defaults to eval.goal read from the fine-tuned checkpoint's "
                             "config.yaml, if set there.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-path", default=None, help="Optional .json path to save detailed results")
    args = parser.parse_args()

    pretrained_policy = DiffusionPolicy.from_pretrained(args.pretrained_path)
    finetuned_policy = DiffusionPolicy.from_pretrained(args.finetuned_path)
    for p in (pretrained_policy, finetuned_policy):
        p.to(args.device)
        p.eval()

    # Read defaults for obs-file/maze-type/goal from the fine-tuned checkpoint's own
    # saved config.yaml (written by Logger.save_model next to the checkpoint weights),
    # so this reuses the exact same eval split the fine-tuning run itself used.
    obs_file, maze_type, goal = args.obs_file, args.maze_type, args.goal
    if obs_file is None or maze_type is None or goal is None:
        run_cfg_path = os.path.join(args.finetuned_path, "config.yaml")
        run_cfg = None
        if os.path.exists(run_cfg_path):
            from omegaconf import OmegaConf
            run_cfg = OmegaConf.load(run_cfg_path)

        if obs_file is None:
            assert run_cfg is not None, (
                f"--obs-file not given and no config.yaml found at {run_cfg_path}; "
                "pass --obs-file explicitly."
            )
            obs_file = run_cfg.eval.train_obs if args.split == "train" else run_cfg.eval.test_obs
            assert obs_file is not None, f"--config's eval.{args.split}_obs is not set"

        if maze_type is None:
            maze_type = str(run_cfg.env_type) if run_cfg is not None else "large"

        if goal is None and run_cfg is not None and run_cfg.eval.get("goal") is not None:
            goal = list(run_cfg.eval.goal)

    with open(obs_file, "r") as f:
        obs_data = np.array(json.load(f), dtype=np.float32)

    results = eval_energy_ranking(
        pretrained_policy, finetuned_policy, obs_data,
        maze_type=maze_type, metrics=args.metrics, n_samples=args.n_samples,
        sampler=args.sampler, goal=goal, seed=args.seed,
        energy_timesteps=args.energy_timesteps,
    )

    print(f"\n{len(obs_data)} obs  |  {args.n_samples} candidates/obs sampled from the pretrained "
          f"policy  |  energy scored under the fine-tuned policy at "
          f"t={args.energy_timesteps}\n")
    for m in args.metrics:
        print(f"  {m}:")
        for s, r in results[m].items():
            total_pairs = r['n_pairs_used'] + r['n_pairs_tied']
            print(f"    {s:>12s}:  mean win rate = {r['mean_win_rate']:.3f}  "
                  f"({r['n_obs_with_pairs']}/{len(obs_data)} obs had usable pairs)  |  "
                  f"{r['pct_pairs_thrown_out']:.1f}% of pairs thrown out as ties "
                  f"({r['n_pairs_tied']}/{total_pairs})")

    if args.save_path is not None:
        with open(args.save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved detailed per-obs results -> {args.save_path}")


if __name__ == "__main__":
    main()
