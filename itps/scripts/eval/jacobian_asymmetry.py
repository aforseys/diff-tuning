#!/usr/bin/env python
"""
How much of the DP's noise field CANNOT be a gradient, per timestep -- i.e. the floor on what
distilling it into an EBM can achieve.

The EBM predicts `eps = grad_x E`, so its Jacobian is a Hessian and is symmetric by construction.
The DP predicts `eps` directly, under no such constraint. Split its Jacobian

    J = S + A,   S = (J + J^T)/2,   A = (J - J^T)/2

The split is orthogonal in Frobenius inner product (tr(S^T A) = 0, since tr(SA) = -tr(SA)), so
||J||_F^2 = ||S||_F^2 + ||A||_F^2 and

    rho = ||A||_F / ||J||_F  in [0, 1]

is the fraction of the Jacobian no energy function can reproduce. S is also the best symmetric
approximation to J in Frobenius norm, so rho is exactly the relative Jacobian error a gradient
field is forced to incur there. rho = 0 means the field is locally a gradient; rho ~ 0.6 (a generic
net) means most of it is not.

WHY rho IS EXPECTED TO BE SMALL ON-TUBE: the true score `-grad log p_t` IS a gradient field, so a
denoiser is nearly curl-free wherever it is accurate. Nonzero rho is therefore a double-edged
measurement -- it bounds the EBM's floor AND says the DP has departed from any true score there.

The `forced` column puts rho in field units. A gradient field matching eps_DP exactly at a path
point still errs by ~A d over a displacement d; for d of norm r in a random direction,
E||A d||^2 = r^2 ||A||_F^2 / D, so the error relative to the field's own scale is

    forced = r * ||A||_F / (sqrt(D) * ||eps||),   r = the measured mean DDIM step ||x_{t-1} - x_t||

Compare `forced` against the ~31% transverse disagreement compare_fields.py reports at low t:
  forced << that   most of the collision gap is reducible -- distillation is worth the run.
  forced ~ that    the curl-free constraint is the binding constraint and distillation will
                   plateau. Known before spending the run.

Probe sets mirror compare_fields.py: the on-tube control (real hdf5 windows, where the denoising
loss supervises) and each model's own DDIM path (where it never looks). Pass --ebm to get the
harness control: its rho MUST come out ~1e-7 or below, since its field is a gradient by
construction. If it does not, the measurement is wrong, not the model.

Cost: D batched VJPs per (probe set, t), D = horizon * action_dim. Each EBM VJP is a
double-backward, so keep --n-points small.

Usage (from itps/):
  python scripts/eval/jacobian_asymmetry.py \
      --dp  data/.../23.08.49_maze2d_diffusion/checkpoints/last/pretrained_model \
      --ebm data/.../14.35.19_maze2d_ebm_diffusion/checkpoints/last/pretrained_model \
      --dataset data/maze2d-sparse-v3.hdf5 --n-points 8
"""

import argparse

import torch

# Sibling script; python puts this file's directory on sys.path.
from compare_fields import build_obs, load_policy, real_tube_points


def jacobian(field_fn, x):
    """
    J[b,i,j] = d field_fn(x)[b,i] / d x[b,j], flattening (horizon, action_dim) -> D on both sides.

    D batched VJPs. Valid batched because the 1D UNet's only norm is GroupNorm, which is per-sample,
    so no sample's output depends on another's input.
    """
    x = x.detach().clone().requires_grad_(True)
    y = field_fn(x).reshape(x.shape[0], -1)
    rows = []
    for i in range(y.shape[1]):
        g = torch.zeros_like(y)
        g[:, i] = 1.0
        (r,) = torch.autograd.grad(y, x, grad_outputs=g, retain_graph=True)
        rows.append(r.reshape(x.shape[0], -1))
    return y.detach(), torch.stack(rows, dim=1)


def asymmetry(J):
    """(rho, ||A||_F, ||J||_F), per sample."""
    A = 0.5 * (J - J.transpose(-1, -2))
    nA = A.flatten(1).norm(dim=1)
    nJ = J.flatten(1).norm(dim=1)
    return nA / nJ.clamp_min(1e-30), nA, nJ


def report(label, probes, field_fn, step_norms):
    """probes: list of (t, x_t). step_norms: {t: mean ||x_{t-1} - x_t||}, or {} to skip `forced`."""
    print(f"\n### {label} ###")
    head = f"{'t':>5}{'rho':>9}{'||A||_F':>11}{'||J||_F':>11}{'||eps||':>10}"
    print(head + (f"{'r':>9}{'forced':>9}" if step_norms else ""))
    for t, xt in probes:
        eps, J = jacobian(lambda z, t=t: field_fn(z, t), xt)
        rho, nA, nJ = asymmetry(J)
        D = J.shape[-1]
        n_eps = eps.flatten(1).norm(dim=1).mean()
        line = (f"{t:>5}{rho.mean():>9.4f}{nA.mean():>11.3f}{nJ.mean():>11.3f}{n_eps:>10.3f}")
        if step_norms:
            r = step_norms.get(t, float("nan"))
            line += f"{r:>9.3f}{r * nA.mean().item() / (D ** 0.5 * n_eps.item()):>9.4f}"
        print(line, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dp", required=True, help="value-parameterized policy to measure (pretrained_model dir)")
    p.add_argument("--ebm", default=None, help="EBM checkpoint; curl-free control, rho must be ~0")
    p.add_argument("--dataset", default=None,
                   help="hdf5 for the on-tube control (e.g. data/maze2d-sparse-v3.hdf5). Without it "
                        "only the sampling paths are probed, and there is no supervised-region baseline.")
    p.add_argument("--n-points", type=int, default=8, help="probe points per timestep; D VJPs each")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    dp, cfg = load_policy(args.dp, args.device)
    models = {"DP": dp}
    if args.ebm:
        ebm, _ = load_policy(args.ebm, args.device)
        # train() matters: EBMWrapper passes create_graph=self.training, so eval() would make its
        # output non-differentiable w.r.t. x and every VJP would be zero. Behaviour-identical here
        # because the 1D UNet's only norm is GroupNorm and there is no dropout.
        ebm.train()
        models["EBM"] = ebm

    def field_of(pol, gc):
        def f(z, t):
            tb = torch.full(z.shape[:1], t, dtype=torch.long, device=z.device)
            return pol.diffusion.model(z, tb, global_cond=gc)
        return f

    # One observation batch, shared by every probe set so conditioning is never a confound.
    obs = build_obs(cfg, dp, args.device, 1)
    obs = {k: v[: args.n_points] for k, v in obs.items()}
    n = next(iter(obs.values())).shape[0]
    gc = {k: m.diffusion._prepare_global_conditioning(m.normalize_inputs({kk: vv.clone() for kk, vv in obs.items()}))
          for k, m in models.items()}

    # --- each model's own DDIM path, and the step size along it ---
    paths, steps = {}, {}
    for k, m in models.items():
        gen = torch.Generator(device=args.device).manual_seed(args.seed)
        with torch.no_grad():
            _, path = m.diffusion.conditional_sample(n, global_cond=gc[k], generator=gen, return_path=True)
        paths[k] = path
        steps[k] = {t: (path[i + 1][1] - xt).flatten(1).norm(dim=1).mean().item() if i + 1 < len(path) else 0.0
                    for i, (t, xt) in enumerate(path)}

    # --- on-tube control: real windows, noised with a known eps ---
    if args.dataset:
        x0, ctrl_obs = real_tube_points(args.dataset, dp, args.device, n, dp.config.horizon,
                                        int(cfg.training.get("goal_horizon", dp.config.horizon)), args.seed)
        x0 = dp.normalize_targets({"action": x0})["action"]
        ctrl_gc = {k: m.diffusion._prepare_global_conditioning(
                       m.normalize_inputs({kk: vv.clone() for kk, vv in ctrl_obs.items()}))
                   for k, m in models.items()}
        sched = dp.diffusion.noise_scheduler
        tube = []
        for t in [int(t) for t in sched.timesteps]:
            eps = torch.randn(x0.shape, device=x0.device)
            tube.append((t, sched.add_noise(x0, eps, torch.full((n,), t, dtype=torch.long, device=x0.device))))

    for k, m in models.items():
        if args.dataset:
            report(f"{k}: ON-TUBE (real {args.dataset}) -- where the denoising loss supervises",
                   tube, field_of(m, ctrl_gc[k]), {})
        for owner in paths:
            report(f"{k}: on {owner}'s DDIM path -- noise draws denoised by {owner} to each t",
                   paths[owner], field_of(m, gc[k]), steps[owner])

    print(f"\nD = {dp.config.horizon} x {dp.config.output_shapes['action'][0]} = "
          f"{dp.config.horizon * dp.config.output_shapes['action'][0]}; sqrt(D) = "
          f"{(dp.config.horizon * dp.config.output_shapes['action'][0]) ** 0.5:.3f}")
    if args.ebm:
        print("EBM rho is the harness control: anything above ~1e-6 means the measurement is wrong.")


if __name__ == "__main__":
    main()
