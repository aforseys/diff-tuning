#!/usr/bin/env python
"""
Compare two policies' noise predictions eps(x, t) at the SAME inputs, to find where their learned
fields disagree. Written for the maze EBM-vs-DP gap: both match on denoising loss over the data
yet the EBM collides ~5-7x more, so the disagreement has to be somewhere the loss never looks.

Three sets of query points:

  on-tube   x_t = add_noise(x0, eps, t) for real x0 from eval.test_obs' trajectories... except
            there is no trajectory dataset here, so x0 comes from each policy's own final samples.
            Ground truth IS the eps that was drawn, so this reports each model's ABSOLUTE error.
            Expect both to look similar -- it is the control that says the harness is sound.

  path A    every x_t policy A's sampler visited (conditional_sample(return_path=True)).
  path B    the same for policy B.

On the paths there is no ground truth -- no x0 generated those points -- so the two models are
compared to each other: ||eps_A|| / ||eps_B|| and cos(eps_A, eps_B), binned by t.

Reading it:
  cos ~ 1 on every set            the fields agree; the collision gap is not a per-point field
                                  difference and something else accumulates across steps.
  cos ~ 1 but norm ratio != 1     direction right, scale wrong -- testable immediately by rescaling.
  cos falls off only on path A    A drifts into a region where its OWN field is bad while B's is
                                  still fine there: a feedback loop, not a globally wrong field.
  cos low on both paths           the field is broadly wrong wherever you evaluate it.

Usage (from itps/):
  python scripts/eval/compare_fields.py \
      --a data/.../ebm/checkpoints/last/pretrained_model \
      --b data/.../dp/checkpoints/last/pretrained_model \
      --label-a EBM --label-b DP
"""

import argparse
import json

import numpy as np
import torch

from itps.common.policies.factory import make_policy
from itps.common.utils.utils import init_hydra_config


def load_policy(path, device):
    cfg = init_hydra_config(f"{path}/config.yaml", [f"device={device}"])
    policy = make_policy(hydra_cfg=cfg, pretrained_policy_name_or_path=path)
    policy.to(device).eval()
    return policy, cfg


def build_obs(cfg, policy, device, n_samples):
    """Start/goal observations, exactly as envs/maze/eval.py:_eval_maze builds them."""
    with open(cfg.eval.test_obs) as f:
        obs_data = np.array(json.load(f), dtype=np.float32)
    start_pos = obs_data[:, :2] if policy.use_goal_cond else obs_data
    states = np.repeat(start_pos, n_samples, axis=0)
    state_t = torch.tensor(states, dtype=torch.float32, device=device)
    n_obs_steps = policy.config.n_obs_steps
    obs = {
        "observation.state": state_t.unsqueeze(1).expand(-1, n_obs_steps, -1).clone(),
        "observation.environment_state": state_t.unsqueeze(1).expand(-1, n_obs_steps, -1).clone(),
    }
    if policy.use_goal_cond:
        goals = np.repeat(obs_data[:, 2:], n_samples, axis=0)
        obs["episode_goal"] = torch.tensor(goals, dtype=torch.float32, device=device).unsqueeze(1).clone()
    return obs


def real_tube_points(dataset_root, policy, device, n, horizon, goal_horizon, seed):
    """
    Real training-distribution x0, with the conditioning that goes WITH each window -- otherwise the
    model is asked to denoise a trajectory that does not start where it was conditioned, which is
    not on-tube at all.

    The action target for the maze hdf5 is observations[1:, :2], the next position
    (common/datasets/utils.py:228). So for an anchor i: state = obs[i], action = obs[i+1 : i+1+H],
    and the goal is the position goal_horizon ahead, i.e. obs[i + goal_horizon].
    """
    import h5py

    with h5py.File(dataset_root, "r") as f:
        xy = f["observations"][:, :2].astype(np.float32)
    span = max(horizon, goal_horizon) + 1
    rng = np.random.default_rng(seed)
    anchors = rng.choice(len(xy) - span, size=n, replace=False)

    state = np.stack([xy[i] for i in anchors])
    action = np.stack([xy[i + 1: i + 1 + horizon] for i in anchors])
    goal = np.stack([xy[i + goal_horizon] for i in anchors])

    state_t = torch.tensor(state, device=device)
    n_obs_steps = policy.config.n_obs_steps
    obs = {
        "observation.state": state_t.unsqueeze(1).expand(-1, n_obs_steps, -1).clone(),
        "observation.environment_state": state_t.unsqueeze(1).expand(-1, n_obs_steps, -1).clone(),
    }
    if policy.use_goal_cond:
        obs["episode_goal"] = torch.tensor(goal, device=device).unsqueeze(1)
    x0 = torch.tensor(action, device=device)
    return x0, obs


def eps_at(policy, x, t, global_cond):
    """eps(x, t). The UNet for a plain policy, grad of the energy for an EBM -- same call."""
    t_batch = torch.full(x.shape[:1], t, dtype=torch.long, device=x.device)
    out = policy.diffusion.model(x.clone().requires_grad_(False), t_batch, global_cond=global_cond)
    return out.detach()


def stats(ea, eb):
    a = ea.flatten(1)
    b = eb.flatten(1)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=1)
    na, nb = a.norm(dim=1), b.norm(dim=1)
    return cos.mean().item(), na.mean().item(), nb.mean().item()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help="path to policy A's pretrained_model dir")
    p.add_argument("--b", required=True, help="path to policy B's pretrained_model dir")
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.add_argument("--dataset", default=None,
                   help="hdf5 to draw REAL x0 windows from for the on-tube control (e.g. "
                        "data/maze2d-sparse-v3.hdf5). Without it the control uses generated samples, "
                        "which are not actually on-tube.")
    p.add_argument("--n-samples", type=int, default=8, help="samples per start observation")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    pa, cfg_a = load_policy(args.a, args.device)
    pb, _ = load_policy(args.b, args.device)

    # Comparing fields across models is meaningless if they normalize differently.
    for key in pa.normalize_inputs.state_dict():
        assert torch.allclose(pa.normalize_inputs.state_dict()[key],
                              pb.normalize_inputs.state_dict()[key]), f"input norm stats differ: {key}"
    for key in pa.unnormalize_outputs.state_dict():
        assert torch.allclose(pa.unnormalize_outputs.state_dict()[key],
                              pb.unnormalize_outputs.state_dict()[key]), f"output norm stats differ: {key}"
    assert pa.config.num_inference_steps == pb.config.num_inference_steps

    obs = build_obs(cfg_a, pa, args.device, args.n_samples)
    nobs_a = pa.normalize_inputs({k: v.clone() for k, v in obs.items()})
    nobs_b = pb.normalize_inputs({k: v.clone() for k, v in obs.items()})
    gc_a = pa.diffusion._prepare_global_conditioning(nobs_a)
    gc_b = pb.diffusion._prepare_global_conditioning(nobs_b)
    B = gc_a.shape[0]

    # --- sampling paths (each model's own) ---
    paths = {}
    for label, pol, gc in ((args.label_a, pa, gc_a), (args.label_b, pb, gc_b)):
        gen = torch.Generator(device=args.device).manual_seed(args.seed)
        x0, path = pol.diffusion.conditional_sample(B, global_cond=gc, generator=gen, return_path=True)
        paths[label] = (x0.detach(), path)

    # --- on-tube control: noise each model's own final samples back up, where eps is known ---
    sched = pa.diffusion.noise_scheduler
    timesteps = [int(t) for t in sched.timesteps]

    if args.dataset:
        x0_ref, ctrl_obs = real_tube_points(
            args.dataset, pa, args.device, B, pa.config.horizon,
            int(cfg_a.training.get("goal_horizon", pa.config.horizon)), args.seed,
        )
        x0_ref = pa.normalize_targets({"action": x0_ref})["action"]
        cgc_a = pa.diffusion._prepare_global_conditioning(pa.normalize_inputs({k: v.clone() for k, v in ctrl_obs.items()}))
        cgc_b = pb.diffusion._prepare_global_conditioning(pb.normalize_inputs({k: v.clone() for k, v in ctrl_obs.items()}))
        source = f"REAL trajectories from {args.dataset}"
    else:
        x0_ref, cgc_a, cgc_b = paths[args.label_b][0], gc_a, gc_b
        source = f"{args.label_b}'s GENERATED samples (not true on-tube -- pass --dataset)"

    print(f"\n### on-tube control: {source} ###")
    print("### ground truth = the drawn eps; mean L2 error per sample ###")
    print(f"{'t':>5}{args.label_a:>12}{args.label_b:>12}{'ratio':>9}")
    for t in timesteps:
        eps = torch.randn(x0_ref.shape, device=x0_ref.device)
        xt = sched.add_noise(x0_ref, eps, torch.full((B,), t, dtype=torch.long, device=x0_ref.device))
        ea = eps_at(pa, xt, t, cgc_a)
        eb = eps_at(pb, xt, t, cgc_b)
        err_a = (ea - eps).flatten(1).norm(dim=1).mean().item()
        err_b = (eb - eps).flatten(1).norm(dim=1).mean().item()
        print(f"{t:>5}{err_a:>12.3f}{err_b:>12.3f}{err_a / max(err_b, 1e-9):>9.3f}")

    # --- on each sampling path ---
    for owner in (args.label_a, args.label_b):
        print(f"\n### on {owner}'s own sampling path ###")
        print(f"{'t':>5}{'cos':>9}{'|eps| ' + args.label_a:>14}{'|eps| ' + args.label_b:>14}{'ratio':>9}")
        for t, xt in paths[owner][1]:
            ea = eps_at(pa, xt, t, gc_a)
            eb = eps_at(pb, xt, t, gc_b)
            cos, na, nb = stats(ea, eb)
            print(f"{t:>5}{cos:>9.4f}{na:>14.3f}{nb:>14.3f}{na / max(nb, 1e-9):>9.3f}")

    print(f"\nsqrt(D) for reference: {np.sqrt(np.prod(paths[args.label_a][0].shape[1:])):.3f}"
          f"   (expected ||eps|| if the model matched a true N(0,I) draw)")


if __name__ == "__main__":
    main()
