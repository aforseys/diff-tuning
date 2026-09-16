#!/usr/bin/env python

# Alexandra Forsey-Smerek
"""Evaluation for the GMM environment: sampling, energy landscapes, KL divergence and log-likelihood
against the ground-truth mixture."""
import numpy as np
import torch
from matplotlib import pyplot as plt

from itps.common.policies.diffusion.modeling_diffusion import (
    DEFAULT_ENERGY_N_NOISE,
    DEFAULT_ENERGY_SEED,
)
from itps.common.utils.utils import seeded_context
from itps.envs.gmm.gaussian_mm import get_weights, get_means, get_covs, mixture_pdf


## -- RUN INFERENCE --
def gen_obs(conditional, N, device):
    "Generates a batch object that matches same type as passed through model, only contains obs."
    observations=[]
    for i in range(1 if not conditional else 3):
        obs_tensor = torch.full((N, 1, 1), i, dtype=torch.float32, device=torch.device(device))
        obs_dict= {
            'observation.state':obs_tensor, 
            'observation.environment_state':obs_tensor
        }
        observations.append(obs_dict)
    return observations

def run_inference(policy, N=100, conditional=False, methods=['ired', 'ddim'], opt_params=None):
    if 'ired' in methods and opt_params is None:
        raise ValueError("IRED sampling requires `opt_params` (one dict per IRED variant).")

    device = next(policy.parameters()).device
    obs = gen_obs(conditional=conditional, N=N, device=device)

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

def run_inference_with_grad_steps(policy, N=50, conditional=False, opt_params=None):
    if opt_params is None:
        raise ValueError("IRED sampling requires `opt_params` (one dict per IRED variant).")

    device = next(policy.parameters()).device
    obs = gen_obs(conditional=conditional, N=N, device=device)

    grad_histories_per_opt = [[] for _ in opt_params]
    
    for o in obs:
        _, grad_histories = policy.run_inference(o, methods=['ired'], opt_params=opt_params, return_grad_steps=True)
        for i in range(len(opt_params)):
            grad_histories_per_opt[i].append(grad_histories[i])

    return grad_histories_per_opt

## -- CALCULATE ENERGY  -- 
def torchify(t, device):
    return torch.tensor(t, dtype=torch.float32, device=torch.device(device)).unsqueeze(dim=1)


def gen_xy_grid(x_range, y_range, device, return_tensor=True):
    xmin,xmax=x_range
    ymin,ymax=y_range
    
    xx, yy = np.meshgrid(
    np.linspace(xmin, xmax, 200),
    np.linspace(ymin, ymax, 200)
    )

    trajs = np.column_stack([xx.ravel(), yy.ravel()]) 

    if return_tensor:
        trajs = torchify(trajs, device)

    return trajs

def eval_energy(policy, trajs, t, conditional=False, batch_size=256,
                deterministic=True, n_noise=DEFAULT_ENERGY_N_NOISE, seed=DEFAULT_ENERGY_SEED):
    """
    Evaluate the policy's energy over `trajs` (a grid or sample set), one list
    entry per observation context.

    deterministic: defaults to True here, unlike the ranking scripts. These
        energies are used as a LANDSCAPE -- summed over a 200x200 grid by
        `kl_divergence`, or contoured for plots -- and get_traj_energies shares a
        single eps draw across the whole batch (common random numbers). That
        collapses the variance of *differences between* trajectories, which is
        what ranking needs, but it leaves an integral over the grid carrying only
        `n_noise` independent samples instead of n_points * n_noise, so the error
        no longer cancels across the sum. Deterministic scoring also keeps these
        values comparable with every GMM result produced before 2026-08-17.
        Pass deterministic=False to average over `n_noise` draws instead.
    """
    device = next(policy.parameters()).device
    observations = gen_obs(conditional=conditional, N=len(trajs), device=device)
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

def eval_gt_pdf(trajs, conditional, centers=None):
    if not conditional:
        energies = [mixture_pdf(trajs, get_weights(), get_means(), get_covs())]
    else:  #set weight to be nonzero for non conditional cluster
        if centers is None:  # if not specififed evaluate for each pdf 
            centers = list(range(3))
        energies = [mixture_pdf(trajs, np.eye(3, dtype=int)[i], get_means(), get_covs()) for i in centers]

    return energies


## -- METRICS -- 
def kl_divergence(policy, conditional, finetune, t=0, eps=1e-8):
    """
    Assumes only conditional or finetune is true. 
    """ 
    assert not (conditional and finetune), "Simultaneous conditional and finetune not supported"
    device = next(policy.parameters()).device

    # Generate grid over GT distribution support
    traj_grid = gen_xy_grid(x_range=(-10, 10), y_range=(-10,10), device=device, return_tensor=False)

    if conditional: # get gt and learned pdf for 3 separate observations
        p_x = eval_gt_pdf(traj_grid, conditional=True)
        q_x = eval_energy(policy, torchify(traj_grid, device=device), t=t, conditional=True)

    elif finetune: # get gt pdf for target distribution and general learned distribution 
        p_x = eval_gt_pdf(traj_grid, conditional=True, centers=[0]) 
        q_x = eval_energy(policy, torchify(traj_grid, device=device), t=t, conditional=False)

    else: # get gt pdf for mixture model and general learned distribution 
        p_x = eval_gt_pdf(traj_grid, conditional=False)
        q_x = eval_energy(policy, torchify(traj_grid, device=device), t=t, conditional=False)

    assert ((conditional or finetune) and (len(p_x)==len(q_x)==3)) or (len(p_x)==len(q_x)==1), "Incorrect number of distributions"

    # transform energy to pdf 
    q_x = [np.exp(-energy_dist) for energy_dist in q_x]
    # flatten, clip, and normalize within each distribution 
    eps = 1e-8  # good for float32

    p_x = [dist.flatten() for dist in p_x]
    q_x = [dist.flatten() for dist in q_x]
    p_x = [np.clip(dist, eps, None) for dist in p_x]
    q_x = [np.clip(dist, eps, None) for dist in q_x]
    p_x = [dist/dist.sum() for dist in p_x]
    q_x = [dist/dist.sum() for dist in q_x]

    # get average kl divergence across distributions
    kl_div = np.mean([np.sum(p_x[i]*np.log(p_x[i]/q_x[i])) for i in range(len(p_x))])

    return kl_div

def log_likelihood(policy, conditional, finetune, N=100, samples=None, opt_params=None, methods=['ired', 'ddim']):
    """
    Assumes only conditional or finetune is true. 
    """

    assert not (conditional and finetune), "Simultaneous conditional and finetune not supported"
    if samples is None:
        samples =run_inference(policy, N=N, conditional=conditional, methods=methods, opt_params=opt_params)
    else:
        samples = [samples]

    for s in samples:
        assert (conditional and (len(s)==3)) or (len(s)==1), "Incorrect number of sample sets"

    if conditional: # evaluate each sample set under corresponding gt pdf
        p_x = [[eval_gt_pdf(s[i], conditional=True, centers=[i])[0] for i in range(len(s))] for s in samples]
    elif finetune: # evaluate sample set under target distribution
        p_x = [eval_gt_pdf(s[0], conditional=True, centers=[0]) for s in samples]
    else: # evaluate sample set under mixture 
        p_x = [eval_gt_pdf(s[0], conditional=False) for s in samples]

    #get average log likelihood across all samples and distributions
    eps =1e-8
    ll = [np.mean(np.log(np.clip(np.concatenate(dist, axis=0),eps, None))) for dist in p_x]

    return samples, ll


## -- VISUALIZATION FUNCTIONS -- 
def viz_inference(policy, samples, conditional, learned_contour=True, t=0, x_range=(-10, 10), y_range=(-10,10)):

    device = next(policy.parameters()).device
    #if plotting over learned energy contour
    if learned_contour:
        trajs = gen_xy_grid(x_range=x_range, y_range=y_range, device=device)
        print('Evaluating energy')
        energies = eval_energy(policy, trajs, t, conditional=conditional)
        xx = trajs[:, 0, 0].cpu().numpy().reshape(200,200)
        yy = trajs[:, 0, 1].cpu().numpy().reshape(200,200)
        print('Energy evaluated, generating samples')

    #otherwise plot over gt pdf 
    else: 
        trajs = gen_xy_grid(x_range=x_range, y_range=y_range, device=device, return_tensor=False)
        energies = eval_gt_pdf(trajs, conditional=conditional)
        xx = trajs[:,0].reshape(200,200)
        yy = trajs[:,1].reshape(200,200)

    #plot all energy landscapes in list given trajs
    for i in range(len(energies)):
        #plot
        zz = energies[i].reshape(200,200)
        if conditional:
            title = f"Energy landscape conditioned on cluster observation {i}"
        else:
            title = "Energy landscape (unconditional)"

        plt.figure(i)
        if learned_contour:
            zz=np.exp(-zz)
        plt.imshow(zz, origin="lower",
                    extent=[xx.min(), xx.max(), yy.min(), yy.max()],
                    aspect="auto"
                    )
        # plot where sampled points are with x's 
        plt.scatter(samples[i][:,0], samples[i][:,1], s=8, alpha=0.6, edgecolor='none')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(title)
        plt.show()

def viz_energy_landscape(policy, conditional, t=0, x_range=(-8, 8), y_range=(-8,8)):

    device = next(policy.parameters()).device
    trajs = gen_xy_grid(x_range=x_range, y_range=y_range, device=device)
    energies = eval_energy(policy, trajs, t, conditional=conditional)

    print(trajs.shape)
    print(len(energies))
    print(energies[0].shape)

    xx = trajs[:, 0, 0].cpu().numpy().reshape(200,200)
    yy = trajs[:, 0, 1].cpu().numpy().reshape(200,200)

    #plot all energy landscapes in list given trajs
    for i in range(len(energies)):
        zz = np.exp(-energies[i].reshape(200,200))
        if conditional:
            title = f"Energy landscape conditioned on cluster observation {i}"
        else:
            title = "Energy landscape (unconditional)"

        plt.figure(i)
        ax = plt.axes(projection="3d")
        ax.plot_surface(xx, yy, zz, cmap="viridis", edgecolor="none")
        ax.view_init(elev=35, azim=-70)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(title)
        plt.show()

def viz_sample_comparison(samples, train_data):

    for i in range(len(samples)):
        plt.figure(i)
        plt.scatter(train_data[i][:,0], train_data[i][:,1], s=8, alpha=0.6, edgecolor='none')
        plt.scatter(samples[i][:,0], samples[i][:,1], s=8, alpha=0.6, edgecolor='none')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.xlim(-8,8)
        plt.ylim(-8,8)
        plt.title(f"Samples against training data (Obs:{i})")
        plt.show()

def viz_ired_grad_steps(policy, grad_history, t, conditional, opt_vals, x_range=(-10, 10), y_range=(-10,10)):
 
    """         
    Overlay IRED gradient step arrows on the learned energy landscape at denoising timestep t.                                                                         

    grad_history: {t_int: [{'pos': Tensor(B,H,D), 'next_pos': Tensor(B,H,D)}]}      
                for one obs and one opt_step config, positions in data space.     
    """                                           
    assert not conditional, "Conditional sampling not supported for multiple opt steps"

    steps_at_t = grad_history.get(t, [])                                            
    if not steps_at_t:                                                              
        print(f"No grad steps recorded for timestep {t}")
        return                                                                      
                  
    device = next(policy.parameters()).device                                       
    trajs = gen_xy_grid(x_range=x_range, y_range=y_range, device=device)
    energies = eval_energy(policy, trajs, t=t, conditional=conditional)             
    xx = trajs[:, 0, 0].cpu().numpy().reshape(200, 200)                             
    yy = trajs[:, 0, 1].cpu().numpy().reshape(200, 200)                             
                                                                                    
    energy = energies[0]              
    zz = np.exp(-energy.reshape(200, 200))
                                                                                    
    n_inner = len(steps_at_t)
    fig, axes = plt.subplots(1, n_inner, figsize=(5 * n_inner, 5), squeeze=False)   
    axes = axes[0]                                                                  

    for step_i, step_data in enumerate(steps_at_t):                                 
        ax = axes[step_i]
        ax.imshow(zz, origin='lower', extent=[xx.min(), xx.max(), yy.min(), yy.max()], aspect='auto', cmap='viridis')                                         

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
        n_inner_steps_label = opt_vals["n_opt"]
        t_time_steps_label = opt_vals["t_subset"]
        denoise = opt_vals["denoise"]

    title += f" ({n_inner_steps_label} inner steps/timestep, {t_time_steps_label} timesteps, denoise = {denoise})"      
                
    plt.suptitle(title)                                                             
    plt.tight_layout()
    plt.show()   

def filter_samples(samples, finetune, conditional):

    samples_by_obs = [samples[samples[:, 0] == i, 1:] for i in range(3)]

    if conditional: 
        return samples_by_obs #return list with all observations divided
    elif finetune: 
        return samples_by_obs[:1] #return list of just 0 observation
    else: 
        return [np.concatenate(samples_by_obs)] #return list with concatenated np array

def eval_GMM(policy, condition_type, finetune, N, viz=False, training_samples=None, opt_params=None, methods=['ired', 'ddim'], viz_opt=False, save_samples_path=None, seed=None):
    if seed is None:
        return _eval_GMM(policy, condition_type, finetune, N, viz, training_samples,
                         opt_params, methods, viz_opt, save_samples_path)
    # See eval_maze below: seeded_context restores the caller's RNG on exit, so an
    # in-training eval doesn't reseed training's noise stream.
    with seeded_context(seed):
        return _eval_GMM(policy, condition_type, finetune, N, viz, training_samples,
                         opt_params, methods, viz_opt, save_samples_path)


def _eval_GMM(policy, condition_type, finetune, N, viz, training_samples,
              opt_params, methods, viz_opt, save_samples_path):
    if condition_type == "conditional":
        conditional=True
    elif condition_type == "unconditional":
        conditional=False
    else: 
        raise NotImplementedError("Only 'unconditional' or 'conditional' condition_types supported for GMM")

    # KL divergence compares the learned energy landscape against the ground-truth density, so it is only
    # available for policies that expose energies.
    kl_div = kl_divergence(policy, conditional, finetune) if hasattr(policy, "get_energy") else None

    # Generate samples and calculate log likelihood
    samples, ll = log_likelihood(policy, conditional, finetune, N, opt_params=opt_params, methods=methods)

    if save_samples_path is not None:
        save_dict = {}
        n_ired = len(opt_params) if 'ired' in methods else 0
        for i, s in enumerate(samples):
            arr = np.concatenate(s, axis=0)  # stack across obs
            if 'ired' in methods and i < n_ired:
                label = f'ired_{opt_params[i]["n_opt"]}steps'
                if opt_params[i]["t_subset"] is not None:
                    label += f'_last{opt_params[i]["t_subset"]}'
                if opt_params[i]["denoise"]:
                    label += '_denoise'
            else:
                label = 'ddim'
            save_dict[label] = arr
        np.savez(save_samples_path, **save_dict)
        print(f"Saved samples to {save_samples_path}.npz")

    # DDIM samples are last set, all others are IRED sampling
    if 'ddim' in methods:
        DDIM_samples = samples[-1]
        DDIM_ll = ll[-1]
    if 'ired' in methods:
        IRED_samples = samples[0:len(opt_params)]
        IRED_ll = ll[0:len(opt_params)]

    info = {"aggregated": {}}
    if kl_div is not None:
        info["aggregated"]["kl_div"] = kl_div

    if 'ddim' in methods:
        info["aggregated"]["DDIM_log_likelihood"]=DDIM_ll

    if 'ired' in methods:
        for i in range(len(opt_params)):
            label = f'IRED_{opt_params[i]["n_opt"]}steps'
            if opt_params[i]["t_subset"] is not None:
                label+=f'_last{opt_params[i]["t_subset"]}'
            if opt_params[i]["denoise"]:
                label+='_denoise'

            info["aggregated"][label] = IRED_ll[i]

    if training_samples is not None:
        train_data = np.load(training_samples)
        filtered_samples = filter_samples(train_data, finetune, conditional)
        ll_training = log_likelihood(policy, conditional, finetune, samples=filtered_samples)
        print('Log likelihood of training samples:', ll_training)

    if viz:
        # Visualize training samples if passed in
        if training_samples is not None:
            train_data_raw = np.load(training_samples)
            train_data_split = filter_samples(train_data_raw, finetune=finetune, conditional=conditional)
            N_per_obs = len(train_data_split[0])
            samples = run_inference(policy, N=N_per_obs, conditional=conditional,  methods=methods, opt_params=opt_params)
            if 'ddim' in methods:
                DDIM_samples=samples[-1]
                viz_sample_comparison(DDIM_samples, train_data_split)
            if 'ired' in methods:
                IRED_samples=samples[0:len(opt_params)]
                for opt_step_samples in IRED_samples:
                    viz_sample_comparison(opt_step_samples, train_data_split)

        # Visualize inferred samples over gt distribution 
        if 'ddim' in methods:
            viz_inference(policy, samples=DDIM_samples, conditional=conditional, learned_contour=False)
        if 'ired' in methods:
            for opt_step_samples in IRED_samples:
                viz_inference(policy, samples=opt_step_samples, conditional=conditional, learned_contour=False)

        # If visualizing the gradient steps
        if viz_opt:
          assert not conditional, "Conditional sampling not supported for grad viz"
          grad_N = min(50, N)                                                         
          grad_histories_per_opt = run_inference_with_grad_steps(                     
              policy, N=grad_N, conditional=conditional, opt_params=opt_params          
          )                                                                           
          for step_i, opt_vals in enumerate(opt_params):
              grad_hist = grad_histories_per_opt[step_i][0]
              for t in sorted(grad_hist.keys(), reverse=True):
                    viz_ired_grad_steps(
                        policy, grad_hist, t=t, conditional=conditional,
                        opt_vals=opt_vals
                    )
        else:
            # Visualize learned distribution at different denoising steps
            for i in range(10):
                if 'ddim' in methods:
                    viz_inference(policy, samples=DDIM_samples, conditional=conditional, learned_contour=True, t=i)
                if 'ired' in methods:
                    for opt_step_samples in IRED_samples:
                        viz_inference(policy, samples=opt_step_samples, conditional=conditional, learned_contour=True, t=i)
                viz_energy_landscape(policy, conditional, t=i)

    return info
