#!/usr/bin/env python
"""
WHERE along the horizon do wall crossings happen? (Not where along the denoising chain -- that
tells you nothing, since every x_t is in collision until the last step.)

Why ask: the EBM applies the padding mask INSIDE its energy (EBMWrapper.forward zeroes
energy_per_state before summing), while DiffusionConditionalUnet1d ignores `mask` entirely. So the
EBM trains `grad E_masked` and samples with `grad E_unmasked` (conditional_sample passes no mask),
whereas for the plain DP the mask only decides which loss entries count and never changes what the
network computes.

In the goal-conditioned maze those are coupled. A window is padded exactly when
steps_to_end <= 61, the padded count is `62 - steps_to_end`, and the goal is clamped to the
episode's final position whenever steps_to_end <= goal_horizon (common/datasets/utils.py:219). So
the goal's distance is a proxy for how many slots were masked, and the model CAN learn a separate
(masked) field for near goals -- one the sampler never uses. Unconditionally it has no way to tell
the regimes apart, which is why the large maze has the same padding fraction and no collision gap.

Prediction if that is the mechanism: the EBM's crossings sit in the TAIL slots (the ones masked
most often during training), and more so for near goals. The DP's should be spread over the
horizon and flat in goal distance.

Usage (from itps/):
  python scripts/eval/collision_by_slot.py \
      --a data/maze2d_dp/sparse/gc/train/2026.10.16/14.35.19_maze2d_ebm_diffusion/checkpoints/last/pretrained_model \
      --b data/maze2d_dp/sparse/gc/train/2026.10.15/23.08.49_maze2d_diffusion/checkpoints/last/pretrained_model \
      --label-a EBM --label-b DP --split test
"""

import argparse
import json

import numpy as np
import torch

# Sibling script; python puts this file's directory on sys.path.
from compare_fields import build_obs, load_policy

from itps.envs.maze.maze_maps import MAZE_MAPS
from itps.envs.maze.maze_scoring import maze_collision_steps


def sample_trajs(policy, obs, methods):
    """Full-horizon trajectories in maze coordinates, chunked as in envs/maze/eval.py:99-112."""
    total = next(iter(obs.values())).shape[0]
    chunks = []
    with torch.no_grad():
        for start in range(0, total, 256):
            chunk = {k: v[start:start + 256] for k, v in obs.items()}
            _, full = policy.run_inference(chunk, methods=methods, opt_params=[], return_full=True)
            chunks.append(full[0].cpu())
    return torch.cat(chunks).numpy()


def report(label, steps, n_obs, n_samples, goal_dist, n_bins):
    """steps: (n_obs*n_samples, H) bool. goal_dist: (n_obs,) or None."""
    H = steps.shape[1]
    hit = steps.any(axis=1)
    first = np.argmax(steps, axis=1)  # only meaningful where hit

    print(f"\n### {label}  collision_rate {hit.mean():.4f}   "
          f"({n_obs} observations x {n_samples} samples, horizon {H}) ###")
    edges = np.linspace(0, H, n_bins + 1).astype(int)
    print(f"{'slots':>10}{'in-wall freq':>14}{'first-crossing share':>22}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_wall = steps[:, lo:hi].mean()
        share = ((first >= lo) & (first < hi) & hit).sum() / max(hit.sum(), 1)
        print(f"{f'{lo}-{hi - 1}':>10}{in_wall:>14.4f}{share:>22.4f}")
    if hit.any():
        print(f"{'mean first-crossing slot':>34}: {first[hit].mean():.1f}  (midpoint would be {(H - 1) / 2:.1f})")

    if goal_dist is None:
        return
    # Does the effect track goal distance, i.e. the masked-slot count during training?
    per_obs_hit = hit.reshape(n_obs, n_samples)
    per_obs_first = np.where(hit, first, np.nan).reshape(n_obs, n_samples)
    med = np.median(goal_dist)
    for name, sel in (("near goals (<= median)", goal_dist <= med), ("far goals (> median)", goal_dist > med)):
        f = per_obs_first[sel]
        print(f"  {name:<24} dist {goal_dist[sel].mean():5.2f}  "
              f"collision {per_obs_hit[sel].mean():.4f}  "
              f"mean first slot {np.nanmean(f) if np.isfinite(f).any() else float('nan'):.1f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help="path to policy A's pretrained_model dir")
    p.add_argument("--b", required=True, help="path to policy B's pretrained_model dir")
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--n-bins", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    pa, cfg = load_policy(args.a, args.device)
    pb, _ = load_policy(args.b, args.device)
    maze = MAZE_MAPS[cfg.env_type]
    n_samples = cfg.eval.n_samples
    obs_file = cfg.eval.test_obs if args.split == "test" else cfg.eval.train_obs

    obs_data = np.array(json.load(open(obs_file)), dtype=np.float32)
    n_obs = len(obs_data)
    goal_dist = None
    if pa.use_goal_cond:
        goal_dist = np.linalg.norm(obs_data[:, 2:] - obs_data[:, :2], axis=1)

    for label, pol in ((args.label_a, pa), (args.label_b, pb)):
        torch.manual_seed(args.seed)  # same prior draws for both policies
        obs = build_obs(cfg, pol, args.device, n_samples, obs_file=obs_file)
        traj = sample_trajs(pol, obs, list(cfg.eval.methods))
        report(label, maze_collision_steps(traj, maze), n_obs, n_samples, goal_dist, args.n_bins)


if __name__ == "__main__":
    main()
