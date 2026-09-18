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
def kl_divergence(policy, spec, conditional, t=0, eps=1e-8):
    """
    KL(ground truth || policy) over a grid covering the mixture's support, averaged over
    observation contexts.

    Computed entirely in log space. The policy's energy is an unnormalized negative
    log-density, so the policy's distribution over the grid is softmax(-E): subtracting
    logsumexp makes the result invariant to the arbitrary additive constant in the
    energy, as it must be. Exponentiating first is not safe -- exp(-E) was previously
    clipped at 1e-8 *before* normalizing, so shifting every energy by a constant (which
    changes no distribution at all) drove the number to its uniform-q saturation value.
    """
    device = next(policy.parameters()).device

    # Generate grid over GT distribution support
    traj_grid = gen_xy_grid(x_range=(-10, 10), y_range=(-10,10), device=device, return_tensor=False)

    p_x = eval_gt_pdf(traj_grid, spec, conditional=conditional)
    q_energy = eval_energy(policy, torchify(traj_grid, device=device), t=t,
                           conditional=conditional, n_clusters=spec.n_clusters)

    assert len(p_x) == len(q_energy), "Incorrect number of distributions"

    kls = []
    for p, energy in zip(p_x, q_energy):
        p = np.clip(p.flatten().astype(np.float64), eps, None)
        p = p / p.sum()
        log_q = -energy.flatten().astype(np.float64)
        log_q = log_q - logsumexp(log_q)          # normalize the policy over the grid
        kls.append(float(np.sum(p * (np.log(p) - log_q))))

    return float(np.mean(kls))

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
                        n_noise=DEFAULT_ENERGY_N_NOISE, deterministic=True, seed=DEFAULT_ENERGY_SEED):
    """
    Over a fixed held-out point set, how often does the policy's energy order a pair the
    same way the utility does? Held-out and fixed so the number is comparable across
    policies -- unlike ranking a policy's own samples, where the point set moves too.

    Returns one win rate per observation context.
    """
    device = next(policy.parameters()).device
    energies = eval_energy(policy, torchify(test_points, device=device), t=t,
                           conditional=conditional, n_clusters=spec.n_clusters,
                           n_noise=n_noise, deterministic=deterministic, seed=seed)
    scores = utility(test_points)
    # lower energy = preferred by the model, so rank against -energy
    return [pairwise_win_rate(scores, -energy.reshape(-1))[0] for energy in energies]


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

    #plot all landscapes in list given trajs
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
                    aspect="auto",
                    cmap="viridis_r" if learned_contour else "viridis",
                    )
        plt.colorbar(im, label="energy (lower = more likely)" if learned_contour else "density")
        # plot where sampled points are with x's
        plt.scatter(samples[i][:,0], samples[i][:,1], s=8, alpha=0.6, edgecolor='none', c='red')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(title)
        _show_or_save(fig, save_dir, f"{name}_obs{i}")

def viz_energy_landscape(policy, conditional, spec=None, t=0, x_range=(-8, 8), y_range=(-8,8),
                         save_dir=None, name="energy_surface"):
    """3D surface of the raw energy at denoising timestep t."""
    device = next(policy.parameters()).device
    trajs = gen_xy_grid(x_range=x_range, y_range=y_range, device=device)
    energies = eval_energy(policy, trajs, t, conditional=conditional,
                           n_clusters=spec.n_clusters if spec is not None else 1)

    xx = trajs[:, 0, 0].cpu().numpy().reshape(200,200)
    yy = trajs[:, 0, 1].cpu().numpy().reshape(200,200)

    #plot all energy landscapes in list given trajs
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


def viz_gradient_field(policy, conditional, spec=None, t=0, x_range=(-8, 8), y_range=(-8, 8),
                       arrow_n=25, background="white", save_dir=None, name="grad"):
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
        plt.xlim(-8,8)
        plt.ylim(-8,8)
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
        ax.imshow(zz, origin='lower', extent=[xx.min(), xx.max(), yy.min(), yy.max()], aspect='auto', cmap='viridis_r')

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
             utility=None, pref_test_points=None,
             viz_timesteps=(90, 80, 70, 60, 50, 40, 30, 20, 10, 0), viz_dir=None):
    if seed is None:
        return _eval_GMM(policy, spec, condition_type, N, viz, training_samples,
                         opt_params, methods, viz_opt, save_samples_path, utility,
                         pref_test_points, viz_timesteps, viz_dir)
    # See eval_maze below: seeded_context restores the caller's RNG on exit, so an
    # in-training eval doesn't reseed training's noise stream.
    with seeded_context(seed):
        return _eval_GMM(policy, spec, condition_type, N, viz, training_samples,
                         opt_params, methods, viz_opt, save_samples_path, utility,
                         pref_test_points, viz_timesteps, viz_dir)


def _eval_GMM(policy, spec, condition_type, N, viz, training_samples,
              opt_params, methods, viz_opt, save_samples_path, utility,
              pref_test_points, viz_timesteps, viz_dir):
    if condition_type == "conditional":
        conditional=True
    elif condition_type == "unconditional":
        conditional=False
    else:
        raise NotImplementedError("Only 'unconditional' or 'conditional' condition_types supported for GMM")

    has_energy = hasattr(policy, "get_energy")

    # KL divergence compares the learned energy landscape against the ground-truth density, so it is only
    # available for policies that expose energies.
    kl_div = kl_divergence(policy, spec, conditional) if has_energy else None

    # Generate samples and calculate log likelihood
    samples, ll = log_likelihood(policy, spec, conditional, N, opt_params=opt_params, methods=methods)
    labels = method_labels(methods, opt_params)

    if save_samples_path is not None:
        save_labels = method_labels(methods, opt_params, ired_prefix='ired', ddim_label='ddim')
        save_dict = {label: np.concatenate(s, axis=0) for label, s in zip(save_labels, samples)}
        np.savez(save_samples_path, **save_dict)
        print(f"Saved samples to {save_samples_path}.npz")

    info = {"aggregated": {}}
    if kl_div is not None:
        info["aggregated"]["kl_div"] = kl_div

    for label, value in zip(labels, ll):
        info["aggregated"][f"{label}_log_likelihood"] = value

    # Post-hoc preference metrics. The ground-truth density above knows nothing about
    # preference, so these are what say whether finetuning moved the policy our way.
    if utility is not None:
        for label, value in zip(labels, mean_utility(utility, samples)):
            info["aggregated"][f"mean_utility_{label}"] = value
        if pref_test_points is not None and has_energy:
            win_rates = preference_win_rate(policy, spec, utility, pref_test_points,
                                            conditional=conditional)
            for i, win_rate in enumerate(win_rates):
                key = "win_rate" if len(win_rates) == 1 else f"win_rate_obs{i}"
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
            ired = list(zip(method_labels(['ired'], opt_params, ired_prefix='ired'), samples[0:len(opt_params)])) \
                if 'ired' in methods else []
            return ([("ddim", samples[-1])] if 'ddim' in methods else []) + ired

        # Visualize training samples if passed in
        if training_samples is not None:
            train_data_raw = np.load(training_samples)
            train_data_split = filter_samples(train_data_raw, conditional, spec.n_clusters)
            N_per_obs = len(train_data_split[0])
            samples = run_inference(policy, N=N_per_obs, conditional=conditional, methods=methods,
                                    opt_params=opt_params, n_clusters=spec.n_clusters)
            for label, s in sample_sets(samples):
                viz_sample_comparison(s, train_data_split, save_dir=viz_dir, name=f"samples_vs_train_{label}")

        if viz_opt:
            # IRED steps only, so different opt_params can be compared quickly.
            grad_N = min(50, N)
            grad_histories_per_opt = run_inference_with_grad_steps(
                policy, N=grad_N, conditional=conditional, opt_params=opt_params,
                n_clusters=spec.n_clusters,
            )
            for step_i, opt_vals in enumerate(opt_params):
                label = method_labels(['ired'], [opt_vals], ired_prefix='ired')[0]
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
