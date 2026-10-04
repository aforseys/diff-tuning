#!/usr/bin/env python

# Alexandra Forsey-Smerek
"""Evaluation for the GMM environment: sampling, energy landscapes, KL divergence and
log-likelihood against the ground-truth mixture, and post-hoc preference metrics.

The ground-truth density is always the spec's own blend -- preference never enters it.
A finetuned policy is judged instead by the utility of the samples it draws and by
whether its energy ranks held-out points the way the utility does.
"""
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from scipy.special import logsumexp

from itps.common.policies.diffusion.modeling_diffusion import (
    DEFAULT_ENERGY_N_NOISE,
    DEFAULT_ENERGY_SEED,
)
from itps.common.utils.preference_scoring import pairwise_win_rate
from itps.common.utils.utils import seeded_context


## -- RUN INFERENCE --
def gen_obs(conditional, N, device, n_clusters):
    "Generates a batch object that matches same type as passed through model, only contains obs."
    observations=[]
    for i in range(n_clusters if conditional else 1):
        obs_tensor = torch.full((N, 1, 1), i, dtype=torch.float32, device=torch.device(device))
        obs_dict= {
            'observation.state':obs_tensor,
            'observation.environment_state':obs_tensor
        }
        observations.append(obs_dict)
    return observations

def run_inference(policy, N=100, conditional=False, methods=("ddim",), opt_params=None, n_clusters=1):
    if 'ired' in methods and opt_params is None:
        raise ValueError("IRED sampling requires `opt_params` (one dict per IRED variant).")

    device = next(policy.parameters()).device
    obs = gen_obs(conditional=conditional, N=N, device=device, n_clusters=n_clusters)

    IRED_inference_output = [[] for _ in opt_params] if 'ired' in methods else None
    DDIM_inference_output = [] if 'ddim' in methods else None

    for o in obs:
        actions = policy.run_inference(o, methods=methods, opt_params=opt_params)
        if 'ired' in methods:
            for i in range(len(opt_params)):
                IRED_inference_output[i].append(actions[i].detach().cpu().squeeze(1).numpy())
        if 'ddim' in methods:
            DDIM_inference_output.append(actions[-1].detach().cpu().squeeze(1).numpy())

    results = []
    if 'ired' in methods:
        results += IRED_inference_output
    if 'ddim' in methods:
        results += [DDIM_inference_output]

    return results

def run_inference_with_grad_steps(policy, N=50, conditional=False, opt_params=None, n_clusters=1):
    if opt_params is None:
        raise ValueError("IRED sampling requires `opt_params` (one dict per IRED variant).")

    device = next(policy.parameters()).device
    obs = gen_obs(conditional=conditional, N=N, device=device, n_clusters=n_clusters)

    grad_histories_per_opt = [[] for _ in opt_params]

    for o in obs:
        _, grad_histories = policy.run_inference(o, methods=['ired'], opt_params=opt_params, return_grad_steps=True)
        for i in range(len(opt_params)):
            grad_histories_per_opt[i].append(grad_histories[i])

    return grad_histories_per_opt


def method_labels(methods, opt_params, ired_prefix='IRED', ddim_label='DDIM'):
    """
    Names for each sample set returned by `run_inference`, in the same order: one per
    IRED variant, then DDIM last.
    """
    labels = []
    if 'ired' in methods:
        for params in opt_params:
            label = f'{ired_prefix}_{params["n_opt"]}steps'
            if params["t_subset"] is not None:
                label += f'_last{params["t_subset"]}'
            if params["denoise"]:
                label += '_denoise'
            labels.append(label)
    if 'ddim' in methods:
        labels.append(ddim_label)
    return labels


## -- CALCULATE ENERGY  --
def torchify(t, device):
    return torch.tensor(t, dtype=torch.float32, device=torch.device(device)).unsqueeze(dim=1)


def gen_xy_grid(x_range, y_range, device, return_tensor=True, n=200):
    xmin,xmax=x_range
    ymin,ymax=y_range

    xx, yy = np.meshgrid(
    np.linspace(xmin, xmax, n),
    np.linspace(ymin, ymax, n)
    )

    trajs = np.column_stack([xx.ravel(), yy.ravel()])

    if return_tensor:
        trajs = torchify(trajs, device)

    return trajs

def eval_energy(policy, trajs, t, conditional, n_clusters, batch_size=256,
                deterministic=True, n_noise=DEFAULT_ENERGY_N_NOISE, seed=DEFAULT_ENERGY_SEED):
    """
    Evaluate the policy's energy over `trajs` (a grid or sample set), one list
    entry per observation context.
    """
    device = next(policy.parameters()).device
    observations = gen_obs(conditional=conditional, N=len(trajs), device=device, n_clusters=n_clusters)
    energies = []
    for obs in observations:
        outputs=[]
        for i in range(0, trajs.size(0), batch_size):
            batch_traj = {'action': trajs[i:i+batch_size]}
            batch_obs = {k: v[i:i+batch_size] for k, v in obs.items()}
            out = policy.get_energy(action_batch=batch_traj, t=t, observation_batch=batch_obs,
                                    n_noise=n_noise, deterministic=deterministic, seed=seed)
            outputs.append(out.detach().cpu().numpy())
        energies.append(np.concatenate(outputs, axis=0))
    return energies


def eval_dpo_reward(policy, trajs, conditional, n_clusters, ref_policy=None, n_timesteps=10,
                    seed=DEFAULT_ENERGY_SEED, batch_size=256):
    """
    DPO's implicit reward over `trajs` (higher = more preferred), one list entry per
    observation context: -mse for forward-KL DPO, or -(mse - mse_ref) with `ref_policy`
    (traditional DPO), where mse is the denoising error averaged over `n_timesteps`
    uniformly drawn (t, eps).

    The draws are seeded and shared by every trajectory and by both policies, so the
    reward differences between trajectories don't depend on which draws came up.
    """
    device = next(policy.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    n_train_timesteps = policy.diffusion.noise_scheduler.config.num_train_timesteps
    timesteps = torch.randint(0, n_train_timesteps, (n_timesteps,), device=device, generator=generator)
    noise = torch.randn((n_timesteps, *trajs.shape[1:]), device=device, generator=generator)

    def denoising_error(p, traj, obs):
        batch = {**p.normalize_inputs(obs), **p.normalize_targets({'action': traj})}
        errors = [p.diffusion._compute_denoising_sq_error(batch, t.expand(len(traj)), eps.expand_as(traj))
                  for t, eps in zip(timesteps, noise)]
        return torch.stack(errors).mean(dim=0)  # (B, 1)

    observations = gen_obs(conditional=conditional, N=len(trajs), device=device, n_clusters=n_clusters)
    rewards = []
    for obs in observations:
        outputs = []
        for i in range(0, trajs.size(0), batch_size):
            batch_traj = trajs[i:i+batch_size]
            batch_obs = {k: v[i:i+batch_size] for k, v in obs.items()}
            mse = denoising_error(policy, batch_traj, batch_obs)
            if ref_policy is not None:
                mse = mse - denoising_error(ref_policy, batch_traj, batch_obs)
            outputs.append(-mse.detach().cpu().numpy())
        rewards.append(np.concatenate(outputs, axis=0))
    return rewards


def eval_noise_pred(policy, trajs, t, conditional, n_clusters, batch_size=256):
    """
    The model's noise prediction at each point of `trajs` and timestep t, one (N, action_dim) array per
    observation context. Works for any diffusion policy; for an EBM it is dE/dx_t. It is with respect to
    the rescaled, normalized trajectory, not the raw data-space point.
    """
    device = next(policy.parameters()).device
    observations = gen_obs(conditional=conditional, N=len(trajs), device=device, n_clusters=n_clusters)
    preds = []
    for obs in observations:
        outputs = []
        for i in range(0, trajs.size(0), batch_size):
            batch_traj = {'action': trajs[i:i+batch_size]}
            batch_obs = {k: v[i:i+batch_size] for k, v in obs.items()}
            out = policy.predict_noise(action_batch=batch_traj, t=t, observation_batch=batch_obs)
            outputs.append(out.detach().cpu().squeeze(1).numpy())
        preds.append(np.concatenate(outputs, axis=0))
    return preds


def eval_gt_pdf(trajs, spec, conditional, centers=None):
    """Ground-truth density, either of the full mixture or of each cluster on its own."""
    if not conditional:
        return [spec.pdf(trajs)]
    if centers is None:  # if not specified evaluate for each cluster
        centers = list(range(spec.n_clusters))
    return [spec.pdf(trajs, centers=[i]) for i in centers]


## -- METRICS --
def energy_log_density_scale(policy, t):
    """
    sqrt(1 - alpha_bar_t), the factor between the energy and a log-density.

    dE/dx is the model's noise prediction, and the optimal noise prediction is
    -sqrt(1 - alpha_bar_t) * grad log p_t (denoising score matching; Luo 2022,
    "Understanding Diffusion Models: A Unified Perspective", p. 17). Integrating,
    E = -sqrt(1 - alpha_bar_t) * log p_t + c, so log p_t = -E / sqrt(1 - alpha_bar_t).

    The factor is 39.8 at t=0, 5.5 at t=10, 1.4 at t=50, 1.0 at t=90, so reading -E
    directly as a log-density flattens the landscape toward uniform at low t. This
    rewards over-sharpness.
    """
    return float(torch.sqrt(1.0 - policy.diffusion.noise_scheduler.alphas_cumprod[t]))


def _normalize_log(log_u):
    """Normalize an unnormalized log-density over the grid."""
    return log_u - logsumexp(log_u)


def _kl(log_p, log_q):
    """KL(p || q) for two normalized log-densities on a shared grid."""
    return float(np.sum(np.exp(log_p) * (log_p - log_q)))


def kl_divergence(policy, spec, conditional, t=0, eps=1e-8, utility=None, betas=None,
                  calibrate=True, x_range=(-10, 10), y_range=(-10, 10)):
    """
    KL between the ground-truth density and the policy's energy landscape, over a grid
    covering the mixture's support. Returns a dict:

      kl_forward  KL(p || q) -- penalizes the policy missing mass the target has
      kl_reverse  KL(q || p) -- penalizes the policy putting mass where the target has none

    One value per observation context, suffixed `_obs{i}` when there is more than one
    (matching the win-rate keys), so a conditional policy shows which clusters moved.

    With `utility`, also compares against the preference-tilted target
    p*(x) ∝ p(x) * exp(u(x) / beta) at the best-fit beta over `betas`, adding
    kl_{forward,reverse}_tilted and beta_{forward,reverse}. 

    beta is the tilt strength in utility units: large beta means the landscape still matches
    the untilted ground truth (no preference absorbed), small beta means it has concentrated
    onto the high-utility modes. Fitted per direction, since the two disagree. Note the pair
    labels are deterministic (see gaussian_mm_pref_data), so the Bradley-Terry optimum is the
    beta -> 0 limit; beta measures how far along the tilt a run got, not distance from a
    known optimum.

    Computed entirely in log space, with the energy divided by `energy_log_density_scale` so
    it is a log-density rather than merely proportional to one. 
    """
    device = next(policy.parameters()).device

    # Generate grid over GT distribution support
    traj_grid = gen_xy_grid(x_range=x_range, y_range=y_range, device=device, return_tensor=False)

    p_x = eval_gt_pdf(traj_grid, spec, conditional=conditional)
    q_energy = eval_energy(policy, torchify(traj_grid, device=device), t=t,
                           conditional=conditional, n_clusters=spec.n_clusters)

    assert len(p_x) == len(q_energy), "Incorrect number of distributions"

    sigma = energy_log_density_scale(policy, t) if calibrate else 1.0

    log_ps, log_qs = [], []
    for p, energy in zip(p_x, q_energy):
        p = np.clip(p.flatten().astype(np.float64), eps, None)
        log_ps.append(_normalize_log(np.log(p)))
        log_qs.append(_normalize_log(-energy.flatten().astype(np.float64) / sigma))

    def key(name, i):
        return name if len(log_ps) == 1 else f"{name}_obs{i}"

    out = {}
    for i, (lp, lq) in enumerate(zip(log_ps, log_qs)):
        out[key("kl_forward", i)] = _kl(lp, lq)
        out[key("kl_reverse", i)] = _kl(lq, lp)
    if utility is None:
        return out

    # Sweep the tilt strength over the cached energy grid -- no extra policy forward passes.
    u = np.asarray(utility(traj_grid), dtype=np.float64).flatten()
    betas = np.logspace(-1, 2, 31) if betas is None else np.asarray(betas, dtype=np.float64)
    for i, (lp, lq) in enumerate(zip(log_ps, log_qs)):
        log_tilted = [_normalize_log(lp + u / beta) for beta in betas]
        fwd = [_kl(lt, lq) for lt in log_tilted]
        rev = [_kl(lq, lt) for lt in log_tilted]
        i_f, i_r = int(np.argmin(fwd)), int(np.argmin(rev))
        out.update({
            key("kl_forward_tilted", i): fwd[i_f], key("beta_forward", i): float(betas[i_f]),
            key("kl_reverse_tilted", i): rev[i_r], key("beta_reverse", i): float(betas[i_r]),
        })
    return out

def log_likelihood(policy, spec, conditional, N=100, samples=None, opt_params=None, methods=("ddim",)):
    """Mean log ground-truth density of the policy's samples, one value per sample set."""
    if samples is None:
        samples = run_inference(policy, N=N, conditional=conditional, methods=methods,
                                opt_params=opt_params, n_clusters=spec.n_clusters)
    else:
        samples = [samples]

    for s in samples:
        assert (conditional and (len(s) == spec.n_clusters)) or (len(s) == 1), "Incorrect number of sample sets"

    if conditional: # evaluate each sample set under corresponding gt pdf
        p_x = [[eval_gt_pdf(s[i], spec, conditional=True, centers=[i])[0] for i in range(len(s))] for s in samples]
    else: # evaluate sample set under the full mixture
        p_x = [eval_gt_pdf(s[0], spec, conditional=False) for s in samples]

    #get average log likelihood across all samples and distributions
    eps =1e-8
    ll = [np.mean(np.log(np.clip(np.concatenate(dist, axis=0),eps, None))) for dist in p_x]

    return samples, ll


def mean_utility(utility, samples):
    """Mean preference utility of each sample set (higher = the policy prefers what we do)."""
    return [float(utility(np.concatenate(s, axis=0)).mean()) for s in samples]


def preference_win_rate(policy, spec, utility, test_points, t=0, conditional=False,
                        n_noise=DEFAULT_ENERGY_N_NOISE, deterministic=True, seed=DEFAULT_ENERGY_SEED,
                        tie_tol=0.0, n_pairs=1000, pair_seed=0, score="energy", ref_policy=None):
    """
    Over a fixed held-out point set, how often does the policy order a pair the same way
    the utility does? Held-out and fixed so the number is comparable across policies.

    score: what the policy ranks the points by. "energy" uses the policy's energy (lower =
        preferred); "dpo" uses DPO's implicit reward (see eval_dpo_reward), relative to
        `ref_policy` when one is given.
    tie_tol: set this to the margin the training pairs were generated with, so the test
        comparisons are as hard as the ones the policy was trained on.
    n_pairs: score this many randomly chosen pairs rather than all N*(N-1)/2, drawn from
        `pair_seed` so every policy is scored on the same pairs.

    Returns one win rate per observation context.
    """
    device = next(policy.parameters()).device
    trajs = torchify(test_points, device=device)
    if score == "energy":
        energies = eval_energy(policy, trajs, t=t, conditional=conditional, n_clusters=spec.n_clusters,
                               n_noise=n_noise, deterministic=deterministic, seed=seed)
        # lower energy = preferred by the model, so rank against -energy
        values = [-energy for energy in energies]
    elif score == "dpo":
        values = eval_dpo_reward(policy, trajs, conditional=conditional, n_clusters=spec.n_clusters,
                                 ref_policy=ref_policy, seed=seed)
    else:
        raise ValueError(f"Unknown win rate score '{score}'. Expected 'energy' or 'dpo'.")
    gt_scores = utility(test_points)
    return [pairwise_win_rate(gt_scores, v.reshape(-1), tie_tol=tie_tol,
                              n_pairs=n_pairs, seed=pair_seed)[0] for v in values]


## -- VISUALIZATION FUNCTIONS --
def _show_or_save(fig, save_dir, name):
    """Show the figure, or save it as <save_dir>/<name>.png and close it."""
    if save_dir is None:
        plt.show()
        return
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f"{name}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def viz_inference(policy, samples, conditional, spec=None, learned_contour=True, t=0,
                  x_range=(-10, 10), y_range=(-10,10), save_dir=None, name="samples"):
    """Scatter samples over either the policy's energy landscape or the ground-truth density."""
    device = next(policy.parameters()).device
    #if plotting over learned energy contour
    if learned_contour:
        trajs = gen_xy_grid(x_range=x_range, y_range=y_range, device=device)
        print('Evaluating energy')
        energies = eval_energy(policy, trajs, t, conditional=conditional,
                               n_clusters=spec.n_clusters if spec is not None else 1)
        xx = trajs[:, 0, 0].cpu().numpy().reshape(200,200)
        yy = trajs[:, 0, 1].cpu().numpy().reshape(200,200)
        print('Energy evaluated, generating samples')

    #otherwise plot over gt pdf
    else:
        trajs = gen_xy_grid(x_range=x_range, y_range=y_range, device=device, return_tensor=False)
        energies = eval_gt_pdf(trajs, spec, conditional=conditional)
        xx = trajs[:,0].reshape(200,200)
        yy = trajs[:,1].reshape(200,200)

    #plot all landscapes in list given trajs (len(list) = n conditions)
    for i in range(len(energies)):
        #plot
        zz = energies[i].reshape(200,200)
        if conditional:
            title = f"{'Energy' if learned_contour else 'Density'} conditioned on cluster observation {i}"
        else:
            title = f"{'Energy' if learned_contour else 'Density'} (unconditional)"

        fig = plt.figure(i)
        # Energy is plotted raw (lower = more likely); exponentiating it crushes
        # everything outside the modes onto a flat floor.
        im = plt.imshow(zz, origin="lower",
                    extent=[xx.min(), xx.max(), yy.min(), yy.max()],
                    aspect="equal",
                    cmap="viridis_r" if learned_contour else "viridis",
                    )
        plt.colorbar(im, label="energy (lower = more likely)" if learned_contour else "density")
        # plot where sampled points are
        plt.scatter(samples[i][:,0], samples[i][:,1], s=8, alpha=0.6, edgecolor='none', c='red')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(title)
        _show_or_save(fig, save_dir, f"{name}_obs{i}")

def viz_energy_landscape(policy, conditional, spec=None, t=0, x_range=(-10, 10), y_range=(-10, 10),
                         save_dir=None, name="energy_surface"):
    """3D surface of the raw energy at denoising timestep t."""
    device = next(policy.parameters()).device
    trajs = gen_xy_grid(x_range=x_range, y_range=y_range, device=device)
    energies = eval_energy(policy, trajs, t, conditional=conditional,
                           n_clusters=spec.n_clusters if spec is not None else 1)

    xx = trajs[:, 0, 0].cpu().numpy().reshape(200,200)
    yy = trajs[:, 0, 1].cpu().numpy().reshape(200,200)

    #plot all energy landscapes in list given trajs (len(list) = n conditions)
    for i in range(len(energies)):
        zz = energies[i].reshape(200,200)
        if conditional:
            title = f"Energy landscape conditioned on cluster observation {i} (t={t})"
        else:
            title = f"Energy landscape (unconditional, t={t})"

        fig = plt.figure(i)
        ax = plt.axes(projection="3d")
        ax.plot_surface(xx, yy, zz, cmap="viridis_r", edgecolor="none")
        ax.view_init(elev=35, azim=-70)
        ax.set_zlabel("energy")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(title)
        _show_or_save(fig, save_dir, f"{name}_obs{i}")


def viz_denoising_mse(policy, conditional, spec=None, seed=DEFAULT_ENERGY_SEED, x_range=(-10, 10), y_range=(-10, 10),
                      save_dir=None, name="denoising_mse"):
    """
    Heatmap of the denoising MSE over the grid, averaged over the same seeded (t, eps) draws
    the DPO win rate uses (see eval_dpo_reward), so every grid point sees identical draws.
    """
    device = next(policy.parameters()).device
    trajs = gen_xy_grid(x_range=x_range, y_range=y_range, device=device)
    # Without a reference policy, eval_dpo_reward is -mse.
    rewards = eval_dpo_reward(policy, trajs, conditional=conditional,
                              n_clusters=spec.n_clusters if spec is not None else 1, seed=seed)

    xx = trajs[:, 0, 0].cpu().numpy().reshape(200,200)
    yy = trajs[:, 0, 1].cpu().numpy().reshape(200,200)

    for i in range(len(rewards)):
        zz = -rewards[i].reshape(200,200)
        if conditional:
            title = f"Denoising MSE conditioned on cluster observation {i}"
        else:
            title = "Denoising MSE (unconditional)"

        fig = plt.figure(i)
        # Log scale: the error far from the data is orders of magnitude above the error near the
        # modes, which a linear scale would flatten to one color.
        im = plt.imshow(zz, origin="lower",
                        extent=[xx.min(), xx.max(), yy.min(), yy.max()],
                        aspect="equal", cmap="viridis_r", norm=LogNorm())
        plt.colorbar(im, label="denoising MSE (lower = more likely)")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(title)
        _show_or_save(fig, save_dir, f"{name}_obs{i}")


def viz_gradient_field(policy, conditional, spec=None, t=0, x_range=(-10, 10), y_range=(-10, 10),
                       arrow_n=30, background="white", save_dir=None, name="grad"):
    """
    Quiver of the denoising direction -eps_hat over a grid at timestep t. Works for any diffusion policy;
    for an EBM, -eps_hat is -dE/dx_t.

    background: "white", or "energy" to draw the arrows over the learned energy (EBM only).
    """
    if background not in ("white", "energy"):
        raise ValueError(f"background must be 'white' or 'energy', got {background!r}")
    device = next(policy.parameters()).device
    n_clusters = spec.n_clusters if spec is not None else 1

    energies = None
    if background == "energy":
        grid = gen_xy_grid(x_range=x_range, y_range=y_range, device=device)
        energies = eval_energy(policy, grid, t, conditional=conditional, n_clusters=n_clusters)

    arrows = gen_xy_grid(x_range=x_range, y_range=y_range, device=device, n=arrow_n)
    preds = eval_noise_pred(policy, arrows, t, conditional=conditional, n_clusters=n_clusters)
    ax_pts = arrows[:, 0, 0].cpu().numpy()
    ay_pts = arrows[:, 0, 1].cpu().numpy()
    # Longest arrow spans one grid cell; relative lengths still carry magnitude.
    cell = min((x_range[1] - x_range[0]) / (arrow_n - 1),
               (y_range[1] - y_range[0]) / (arrow_n - 1))

    for i, pred in enumerate(preds):
        dx, dy = -pred[:, 0], -pred[:, 1]
        magnitudes = np.sqrt(dx**2 + dy**2)
        arrow_scale = cell / max(magnitudes.max(), 1e-12)

        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        if energies is not None:
            im = ax.imshow(energies[i].reshape(200, 200), origin="lower",
                           extent=[*x_range, *y_range], cmap="viridis_r")
            fig.colorbar(im, ax=ax, label="energy (lower = more likely)")
        q = ax.quiver(ax_pts, ay_pts, dx * arrow_scale, dy * arrow_scale, magnitudes,
                      cmap="cool", pivot="mid", angles="xy", scale_units="xy", scale=1)
        fig.colorbar(q, ax=ax, label="|eps_hat|")
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_aspect("equal")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        context = f"conditioned on cluster observation {i}" if conditional else "unconditional"
        over = " over energy" if energies is not None else ""
        ax.set_title(f"Gradient field{over}, {context} (t={t})")
        fig.tight_layout()
        _show_or_save(fig, save_dir, f"{name}_obs{i}")

def viz_sample_comparison(samples, train_data, save_dir=None, name="samples_vs_train"):

    for i in range(len(samples)):
        fig = plt.figure(i)
        plt.scatter(train_data[i][:,0], train_data[i][:,1], s=8, alpha=0.6, edgecolor='none')
        plt.scatter(samples[i][:,0], samples[i][:,1], s=8, alpha=0.6, edgecolor='none')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.gca().set_aspect("equal")
        plt.title(f"Samples against training data (Obs:{i})")
        _show_or_save(fig, save_dir, f"{name}_obs{i}")

def viz_ired_grad_steps(policy, grad_history, t, conditional, opt_vals, spec=None, context=0,
                        x_range=(-10, 10), y_range=(-10,10), save_dir=None, name="ired"):

    """
    Overlay IRED gradient step arrows on the learned energy landscape at denoising timestep t.

    grad_history: {t_int: [{'pos': Tensor(B,H,D), 'next_pos': Tensor(B,H,D)}]}
                for one observation context and one opt_step config, positions in data space.
    context: which observation context grad_history came from; its landscape is drawn underneath.
    """
    steps_at_t = grad_history.get(t, [])
    if not steps_at_t:
        print(f"No grad steps recorded for timestep {t}")
        return

    device = next(policy.parameters()).device
    trajs = gen_xy_grid(x_range=x_range, y_range=y_range, device=device)
    energies = eval_energy(policy, trajs, t=t, conditional=conditional,
                           n_clusters=spec.n_clusters if spec is not None else 1)
    xx = trajs[:, 0, 0].cpu().numpy().reshape(200, 200)
    yy = trajs[:, 0, 1].cpu().numpy().reshape(200, 200)

    zz = energies[context].reshape(200, 200)

    n_inner = len(steps_at_t)
    fig, axes = plt.subplots(1, n_inner, figsize=(5 * n_inner, 5), squeeze=False)
    axes = axes[0]

    for step_i, step_data in enumerate(steps_at_t):
        ax = axes[step_i]
        ax.imshow(zz, origin='lower', extent=[xx.min(), xx.max(), yy.min(), yy.max()], aspect='equal', cmap='viridis_r')

        pos = step_data['pos'].squeeze(1).cpu().numpy()   # (B, 2)
        nxt = step_data['next_pos'].squeeze(1).cpu().numpy()  # (B, 2)
        dx = nxt[:, 0] - pos[:, 0]
        dy = nxt[:, 1] - pos[:, 1]
        magnitudes = np.sqrt(dx**2 + dy**2)

        ax.scatter(pos[:, 0], pos[:, 1], s=10, c='red', alpha=0.6, zorder=3)
        q = ax.quiver(pos[:, 0], pos[:, 1], dx, dy, magnitudes,
                    cmap='cool', alpha=0.8, scale=1, scale_units='xy', angles='xy', zorder=4)
        plt.colorbar(q, ax=ax, label='step magnitude')

        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_title(f"inner step {step_i + 1}/{n_inner}\nmean={magnitudes.mean():.3f}, max={magnitudes.max():.3f}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        title = f"IRED gradient steps at denoising t={t}"
        if conditional:
            title += f", cluster observation {context}"
        n_inner_steps_label = opt_vals["n_opt"]
        t_time_steps_label = opt_vals["t_subset"]
        denoise = opt_vals["denoise"]

    title += f" ({n_inner_steps_label} inner steps/timestep, {t_time_steps_label} timesteps, denoise = {denoise})"

    plt.suptitle(title)
    plt.tight_layout()
    _show_or_save(fig, save_dir, name)

def filter_samples(samples, conditional, n_clusters):
    """Split a saved (N, 3) observation array into one point set per observation context."""
    samples_by_obs = [samples[samples[:, 0] == i, 1:] for i in range(n_clusters)]

    if conditional:
        return samples_by_obs #return list with all observations divided
    return [np.concatenate(samples_by_obs)] #return list with concatenated np array

def eval_GMM(policy, spec, condition_type, N, viz=False, training_samples=None, opt_params=None,
             methods=("ddim",), viz_opt=False, save_samples_path=None, seed=None,
             utility=None, pref_test_points=None, tie_tol=0.0,
             viz_timesteps=(90, 80, 70, 60, 50, 40, 30, 20, 10, 0), viz_dir=None,
             is_dpo=False, ref_policy=None, viz_dmse=False, dmse_seed=DEFAULT_ENERGY_SEED):
    if seed is None:
        return _eval_GMM(policy, spec, condition_type, N, viz, training_samples,
                         opt_params, methods, viz_opt, save_samples_path, utility,
                         pref_test_points, viz_timesteps, viz_dir, tie_tol,
                         is_dpo, ref_policy, viz_dmse, dmse_seed)
    # See eval_maze below: seeded_context restores the caller's RNG on exit, so an
    # in-training eval doesn't reseed training's noise stream.
    with seeded_context(seed):
        return _eval_GMM(policy, spec, condition_type, N, viz, training_samples,
                         opt_params, methods, viz_opt, save_samples_path, utility,
                         pref_test_points, viz_timesteps, viz_dir, tie_tol,
                         is_dpo, ref_policy, viz_dmse, dmse_seed)


def _eval_GMM(policy, spec, condition_type, N, viz, training_samples,
              opt_params, methods, viz_opt, save_samples_path, utility,
              pref_test_points, viz_timesteps, viz_dir, tie_tol=0.0,
              is_dpo=False, ref_policy=None, viz_dmse=False, dmse_seed=DEFAULT_ENERGY_SEED):
    if condition_type == "conditional":
        conditional=True
    elif condition_type == "unconditional":
        conditional=False
    else:
        raise NotImplementedError("Only 'unconditional' or 'conditional' condition_types supported for GMM")

    has_energy = hasattr(policy, "get_energy")

    # KL divergence compares the learned energy landscape against the ground-truth density, so it is only
    # available for policies that expose energies. With a utility it also reports the preference-tilted
    # comparison and its fitted tilt strength.
    kl = kl_divergence(policy, spec, conditional, utility=utility) if has_energy else None

    # Generate samples and calculate log likelihood
    samples, ll = log_likelihood(policy, spec, conditional, N, opt_params=opt_params, methods=methods)

    # One label per sample set, in run_inference's order (each IRED opt_params entry, then DDIM).
    # wandb_labels name the logged metrics, so keep them stable for logged history to line up;
    # file_labels name what is saved to disk (the .npz keys and figure files).
    wandb_labels = method_labels(methods, opt_params)
    file_labels = method_labels(methods, opt_params, ired_prefix='ired', ddim_label='ddim')

    if save_samples_path is not None:
        save_dict = {label: np.concatenate(s, axis=0) for label, s in zip(file_labels, samples)}
        np.savez(save_samples_path, **save_dict)
        print(f"Saved samples to {save_samples_path}.npz")

    info = {"aggregated": {}}
    if kl is not None:
        info["aggregated"].update(kl)

    for label, value in zip(wandb_labels, ll):
        info["aggregated"][f"{label}_log_likelihood"] = value

    # Mean denoising MSE on the held-out points, which are drawn from the original (untilted)
    # distribution: a cross-entropy proxy for how much of it finetuning preserved, available for
    # every policy (unlike the energy KL). Same seeded (t, eps) draws as the DPO win rate.
    if pref_test_points is not None:
        device = next(policy.parameters()).device
        # Deliberately no ref_policy, even for traditional DPO: without one, eval_dpo_reward is
        # -mse of this policy alone, not the reference-relative -(mse - mse_ref) its win rate uses.
        rewards = eval_dpo_reward(policy, torchify(pref_test_points, device=device), conditional=conditional,
                                  n_clusters=spec.n_clusters, seed=dmse_seed)
        for i, reward in enumerate(rewards):
            key = "dmse_test" if len(rewards) == 1 else f"dmse_test_obs{i}"
            info["aggregated"][key] = float(-reward.mean())

    # Post-hoc preference metrics.
    if utility is not None:
        for label, value in zip(wandb_labels, mean_utility(utility, samples)):
            info["aggregated"][f"mean_utility_{label}"] = value
        if pref_test_points is not None:
            # Rank the held-out points every way this policy supports: by its energy, and by
            # DPO's implicit reward if it was DPO-finetuned.
            scores = (["energy"] if has_energy else []) + (["dpo"] if is_dpo else [])
            for score in scores:
                win_rates = preference_win_rate(policy, spec, utility, pref_test_points,
                                                conditional=conditional, tie_tol=tie_tol,
                                                score=score, ref_policy=ref_policy)
                for i, win_rate in enumerate(win_rates):
                    key = f"win_rate_{score}" if len(win_rates) == 1 else f"win_rate_{score}_obs{i}"
                    info["aggregated"][key] = win_rate

    if training_samples is not None:
        train_data = np.load(training_samples)
        filtered_samples = filter_samples(train_data, conditional, spec.n_clusters)
        ll_training = log_likelihood(policy, spec, conditional, samples=filtered_samples)
        print('Log likelihood of training samples:', ll_training)

    if viz:
        # viz_dir=None shows each figure; otherwise every figure is saved there, named by type,
        # sampler-order index s and timestep t so each series sorts together.
        def sample_sets(samples):
            """(label, per-context samples) for each sampling method, DDIM first."""
            ired = list(zip(file_labels[0:len(opt_params)], samples[0:len(opt_params)])) \
                if 'ired' in methods else []
            return ([(file_labels[-1], samples[-1])] if 'ddim' in methods else []) + ired

        # Visualize training samples if passed in
        if training_samples is not None:
            train_data_raw = np.load(training_samples)
            train_data_split = filter_samples(train_data_raw, conditional, spec.n_clusters)
            N_per_obs = len(train_data_split[0])
            # Draw as many policy samples per context as there are training points, so the two
            # scatters' densities compare fairly; reuse the eval samples when the counts already
            # match. A separate name keeps the plots below on the samples the logged metrics used.
            if N_per_obs == N:
                train_cmp_samples = samples
            else:
                train_cmp_samples = run_inference(policy, N=N_per_obs, conditional=conditional, methods=methods,
                                                  opt_params=opt_params, n_clusters=spec.n_clusters)
            for label, s in sample_sets(train_cmp_samples):
                viz_sample_comparison(s, train_data_split, save_dir=viz_dir, name=f"samples_vs_train_{label}")

        if viz_dmse:
            # Denoising-MSE map only, for any diffusion policy.
            viz_denoising_mse(policy, conditional, spec=spec, seed=dmse_seed,
                              save_dir=viz_dir, name="denoising_mse")
        elif viz_opt:
            grad_N = min(50, N)
            grad_histories_per_opt = run_inference_with_grad_steps(
                policy, N=grad_N, conditional=conditional, opt_params=opt_params,
                n_clusters=spec.n_clusters,
            )
            # Runs IRED even when 'ired' isn't in `methods` (so file_labels may have no IRED
            # entries), hence labels straight from opt_params.
            ired_file_labels = method_labels(['ired'], opt_params, ired_prefix='ired')
            for step_i, opt_vals in enumerate(opt_params):
                label = ired_file_labels[step_i]
                # One history per observation context (a single one when unconditional).
                for context, grad_hist in enumerate(grad_histories_per_opt[step_i]):
                    optimized = [t for t in sorted(grad_hist, reverse=True) if grad_hist[t]]
                    for k, t in enumerate(optimized):
                        viz_ired_grad_steps(
                            policy, grad_hist, t=t, conditional=conditional,
                            opt_vals=opt_vals, spec=spec, context=context,
                            save_dir=viz_dir, name=f"{label}_s{k}_t{t:03d}_obs{context}",
                        )
        else:
            # Visualize inferred samples over gt distribution
            for label, s in sample_sets(samples):
                viz_inference(policy, samples=s, conditional=conditional, spec=spec, learned_contour=False,
                              save_dir=viz_dir, name=f"samples_true_density_{label}")
            if has_energy:
                # Visualize the learned energy landscape at each timestep
                for k, t in enumerate(viz_timesteps):
                    for label, s in sample_sets(samples):
                        viz_inference(policy, samples=s, conditional=conditional, spec=spec, learned_contour=True,
                                      t=t, save_dir=viz_dir, name=f"samples_energy_{label}_s{k}_t{t:03d}")
                    viz_energy_landscape(policy, conditional, spec=spec, t=t,
                                         save_dir=viz_dir, name=f"energy_surface_s{k}_t{t:03d}")
            # Gradient field at each timestep, for any diffusion policy
            for k, t in enumerate(viz_timesteps):
                viz_gradient_field(policy, conditional, spec=spec, t=t,
                                   save_dir=viz_dir, name=f"grad_white_s{k}_t{t:03d}")
            # When saving, also the gradient drawn over the energy
            if has_energy and viz_dir is not None:
                for k, t in enumerate(viz_timesteps):
                    viz_gradient_field(policy, conditional, spec=spec, t=t, background="energy",
                                       save_dir=viz_dir, name=f"grad_energy_s{k}_t{t:03d}")

    return info
