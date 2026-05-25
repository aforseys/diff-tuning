#!/usr/bin/env python

# Alexandra Forsey-Smerek
import time
import json
import numpy as np
import torch
from matplotlib import pyplot as plt
from itps.scripts.gaussian_mm import get_weights, get_means, get_covs, mixture_pdf
from itps.common.utils.utils import set_global_seed

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

def run_inference(policy, N=100, conditional=False, methods=['ired', 'ddim'], opt_params=[{'n_opt': 1, 't_subset': None, 'denoise': False}]):

    device = next(policy.parameters()).device
    obs = gen_obs(conditional=conditional, N=N, device=device)

    IRED_inference_output = [[] for _ in opt_params] if 'ired' in methods else None
    DDIM_inference_output = [] if 'ddim' in methods else None
    #sample_times=[]

    for o in obs:
        #start = time.perf_counter()
        actions = policy.run_inference(o, methods=methods, opt_params=opt_params)
        #torch.cuda.synchronize()
        # elapsed = time.perf_counter() - start

        # sample_times.append(elapsed)
        # print(f"Sample time: {elapsed*1000:.1f}ms")
        if 'ired' in methods:
            for i in range(len(opt_params)): 
                IRED_inference_output[i].append(actions[i].detach().cpu().squeeze(1).numpy())
        if 'ddim' in methods:
            DDIM_inference_output.append(actions[-1].detach().cpu().squeeze(1).numpy())

    # print(f"Avg sample time over {len(sample_times)} obs: {np.mean(sample_times)*1000:.1f}ms")
    results = []
    if 'ired' in methods: 
        results += IRED_inference_output
    if 'ddim' in methods: 
        results += [DDIM_inference_output]

    return results

def run_inference_with_grad_steps(policy, N=50, conditional=False,opt_params=[{'n_opt': 1, 't_subset': None, 'denoise': False}]):

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

def eval_energy(policy, trajs, t, conditional=False, batch_size=256):
    device = next(policy.parameters()).device
    observations = gen_obs(conditional=conditional, N=len(trajs), device=device)
    energies = []
    for obs in observations:
        outputs=[]
        for i in range(0, trajs.size(0), batch_size):
            batch_traj = {'action': trajs[i:i+batch_size]}
            batch_obs = {k: v[i:i+batch_size] for k, v in obs.items()}
            out = policy.get_energy(action_batch=batch_traj, t=t, observation_batch=batch_obs)
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

def log_likelihood(policy, conditional, finetune, N=100, samples=None, opt_params=[{'n_opt': 1, 't_subset': None, 'denoise': False}], methods=['ired', 'ddim']):
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
        #plt.heatmap(xx, yy, zz)
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
                
    #   if conditional:
    #       title += f"\nConditioned on obs {obs_i}"                                    
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

def eval_GMM(policy, condition_type, finetune, N, viz=False, training_samples=None, opt_params=[{'n_opt': 1, 't_subset': None, 'denoise': False}], methods=['ired', 'ddim'], viz_opt=False, save_samples_path=None, seed=None):
    if seed is not None:
        set_global_seed(seed)

    if condition_type == "conditional":
        conditional=True
    elif condition_type == "unconditional":
        conditional=False
    else: 
        raise NotImplementedError("Only 'unconditional' or 'conditional' condition_types supported for GMM")

    #Calculate KL divergence between distributions 
    kl_div = kl_divergence(policy, conditional, finetune)

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

    info ={
        "aggregated":{
            "kl_div": kl_div, 
    }}

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


# -- MAZE2D EVALUATION --

from itps.common.utils.maze_maps import MAZE_MAPS


def check_maze_collision(xy_traj, maze):
    """
    xy_traj: (batch, steps, 2) numpy array in maze coordinate space
    maze: 2D boolean array where True = wall
    Returns: (batch,) boolean array, True if trajectory has any collision
    """
    batch_size, num_steps, _ = xy_traj.shape
    xy_flat = xy_traj.reshape(-1, 2)
    nan_mask = np.any(np.isnan(xy_flat), axis=1)
    xy_flat = np.nan_to_num(xy_flat, nan=0.0)
    xy_flat = np.clip(xy_flat, [0, 0], [maze.shape[0] - 1, maze.shape[1] - 1])
    mx = np.round(xy_flat[:, 0]).astype(int)
    my = np.round(xy_flat[:, 1]).astype(int)
    collisions = maze[mx, my].reshape(batch_size, num_steps)
    collisions |= nan_mask.reshape(batch_size, num_steps)
    return np.any(collisions, axis=1)


def eval_maze(policy, cfg, split='test', seed=None):
    """
    Offline trajectory evaluation for maze2d without running the gym env.

    For each starting obs in eval_cfg.{split}_obs, generates n_samples_per_obs
    trajectories via DDIM and computes the requested metrics.

    Config fields (under eval):
      maze_type:         "large" | "sparse" | "open"
      n_samples_per_obs: int
      metrics:           list of "collision_rate" | "path_length" | "goal_dist"
      train_obs / test_obs: list of [state_x, state_y, goal_x, goal_y]
    """
    if seed is not None:
        set_global_seed(seed)

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

        #TODO: check all this math with numpy and the batches etc. executes / lines up correctly
        for m in metrics:
            if m == 'collision_rate':
                collisions = check_maze_collision(traj, maze)
                per_obs[m] = collisions.reshape(N_obs, n_samples).mean(axis=1).tolist() 
            elif m == 'obs_goal_dist':
                assert policy.use_goal_cond, "obs_goal_dist requires a goal-conditioned policy"
                goals_repeated = np.repeat(goal_pos, n_samples, axis=0)
                dists = np.linalg.norm(traj[:, -1, :]- goals_repeated, axis=1)
                per_obs[m] = dists.reshape(N_obs, n_samples).mean(axis=1).tolist()
            elif m == 'finetune_goal_dist':
                assert cfg.eval.goal is not None, "finetune_goal_dist requires cfg.eval.goal to be set"
                goal = np.array(cfg.eval.goal, dtype=np.float32)
                dists = np.linalg.norm(traj[:, -1, :] - goal, axis=1)
                per_obs[m] = dists.reshape(N_obs, n_samples).mean(axis=1).tolist()
            #Center preference — 1 minus mean normalized distance from maze center over all trajectory steps:
            elif m == 'center_rate':
                rows, cols = maze.shape
                x_center = (rows - 1) / 2.0
                y_center = (cols - 1) / 2.0
                max_dist = np.sqrt(x_center**2 + y_center**2)
                dx = traj[:, :, 0] - x_center  # (N_obs*n_samples, horizon)
                dy = traj[:, :, 1] - y_center
                scores = 1 - np.sqrt(dx**2 + dy**2).mean(axis=1) / max_dist  # (N_obs*n_samples,)
                per_obs[m] = scores.reshape(N_obs, n_samples).mean(axis=1).tolist()
            #Bottom half preference — fraction of trajectory steps with x > x_mid (higher row index = bottom of maze):
            elif m == 'bottom_half_rate':
                rows, cols = maze.shape
                x_mid = (rows - 1) / 2.0
                scores = (traj[:, :, 0] > x_mid).astype(float).mean(axis=1)  # (N_obs*n_samples,)
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


# -- ROBOSUITE EVALUATION --

def _robosuite_conditions(is_goal_cond, train_prisms, test_prisms, train_bins, test_bins):
    """Return list of (label, prism_list, bin_list_or_None) eval conditions."""
    conditions = []
    if is_goal_cond:
        combos = [
            ('seen_start_seen_goal', train_prisms, train_bins),
            ('new_start_seen_goal',  test_prisms,  train_bins),
            ('seen_start_new_goal',  train_prisms, test_bins),
            ('new_start_new_goal',   test_prisms,  test_bins),
        ]
        for label, prisms, bins in combos:
            if prisms is not None and bins is not None:
                conditions.append((label, prisms, bins))
    else:
        if train_prisms is not None:
            conditions.append(('train', train_prisms, None))
        if test_prisms is not None:
            conditions.append(('test', test_prisms, None))
    return conditions


def _sample_episode_indices(obs_data, prisms, bins, n_episodes, is_goal_cond, rng):
    """Sample n_episodes row indices from obs_data matching the given prism/bin lists."""
    prism_idx = obs_data['prism_idx']
    bin_idx   = obs_data['bin_idx']

    mask = np.isin(prism_idx, prisms)
    if is_goal_cond:
        mask &= np.isin(bin_idx, bins)
    else:
        mask &= (bin_idx == 0)   # deduplicate: each start config appears once per bin

    candidates = np.where(mask)[0]
    if len(candidates) == 0:
        bin_desc = f"bin_idx==0" if bins is None else f"bins={bins}"
        raise ValueError(f"No observations found for prisms={prisms}, {bin_desc}")
    return rng.choice(candidates, size=n_episodes, replace=len(candidates) < n_episodes)


def _viz_sampled_trajectories(env, obs, policy, obs_batch, n_viz_samples, n_joints, chunk_size, start_idx, device):
    """Sample n_viz_samples trajectories, place sphere markers at EEF waypoints, render at high res."""
    import mujoco
    import matplotlib.pyplot as plt
    import colorsys

    # Batch obs n_viz_samples times → different noise initializations → different trajectories
    batched = {k: v.repeat(n_viz_samples, *([1] * (v.dim() - 1))) for k, v in obs_batch.items()}
    with torch.no_grad():
        _, full_trajs = policy.run_inference(batched, methods=['ddim'], return_full=True)
    trajs = full_trajs[0][:, start_idx:, :7].cpu().numpy()  # (N, T, 7) — full predicted future

    # FK: compute EEF path for each sample without stepping physics
    saved_qpos = env.sim.data.qpos.copy()
    start_joints = obs["robot0_joint_pos"].copy()
    all_eef = []
    for i in range(n_viz_samples):
        eef_path = []
        current = start_joints.copy()
        for t in range(trajs.shape[1]):
            current = current + trajs[i, t]
            env.sim.data.qpos[:n_joints] = current
            env.sim.forward()
            eef_path.append(env._eef_pos().copy())
        all_eef.append(np.array(eef_path))
    env.sim.data.qpos[:] = saved_qpos
    env.sim.forward()

    # Use the existing offscreen render context — inject sphere geoms before rendering
    VIZ_W, VIZ_H = 640, 480
    cam_id = env.sim.model.camera_name2id('agentview')
    ctx = env.sim._render_context_offscreen
    ctx.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    ctx.cam.fixedcamid = cam_id

    mujoco.mjv_updateScene(
        env.sim.model._model, env.sim.data._data,
        ctx.vopt, ctx.pert, ctx.cam, mujoco.mjtCatBit.mjCAT_ALL, ctx.scn
    )

    # Inject sphere geoms — subsample to ~10 waypoints per trajectory
    step = max(1, trajs.shape[1] // 15)
    for i, eef_path in enumerate(all_eef):
        h = i / n_viz_samples
        r, g, b = colorsys.hsv_to_rgb(h, 0.9, 0.95)
        rgba = np.array([r, g, b, 0.8], dtype=np.float32)
        for pos in eef_path[::step]:
            if ctx.scn.ngeom >= ctx.scn.maxgeom:
                break
            mujoco.mjv_initGeom(
                ctx.scn.geoms[ctx.scn.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.012, 0.012, 0.012]),
                pos.astype(np.float64),
                np.eye(3, dtype=np.float64).flatten(),
                rgba,
            )
            ctx.scn.ngeom += 1

    # Render directly — don't use ctx.render() which would call mjv_updateScene again and wipe our geoms
    ctx.update_offscreen_size(VIZ_W, VIZ_H)
    viewport = mujoco.MjrRect(0, 0, VIZ_W, VIZ_H)
    mujoco.mjr_render(viewport, ctx.scn, ctx.con)
    img = ctx.read_pixels(VIZ_W, VIZ_H)[::-1]  # flip bottom-up → top-down

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f'{n_viz_samples} sampled EEF trajectories', fontsize=13)
    plt.tight_layout()
    plt.show(block=True)


def eval_robosuite(policy, cfg, seed=None, render=False, n_viz_samples=0):
    """
    Evaluate the robosuite policy against fixed observations from cfg.eval.obs_file.

    Config fields (under eval):
      obs_file:      path to .npz with start_pos, goal_pos, bin_idx, prism_idx, start_joints
      train_prisms:  list of prism indices seen during finetuning
      test_prisms:   list of prism indices NOT seen during finetuning
      train_bins:    list of bin indices seen during finetuning (GC only; null for non-GC)
      test_bins:     list of bin indices NOT seen during finetuning (GC only; null for non-GC)
      n_episodes:    episodes sampled per condition
    """
    if seed is not None:
        set_global_seed(seed)
    rng = np.random.default_rng(seed)

    from collections import deque
    from itps.scripts.bin_placing import make_eval_env, OBJECT_MAP

    n_episodes       = cfg.eval.n_episodes
    n_bins           = 4
    n_obs_steps      = policy.config.n_obs_steps
    steps_per_action = cfg.eval.get('steps_per_action', 2)
    is_goal_cond     = 'episode_goal' in policy.config.input_shapes
    chunk_sizes      = list(cfg.eval.get('action_chunk_sizes', [policy.config.n_action_steps]))
    max_steps        = cfg.eval.get('max_episode_steps', 300)
    img_size         = cfg.eval.get('img_size', 84)
    obj              = OBJECT_MAP.get(cfg.eval.get('object', 'can'))
    device           = next(policy.parameters()).device

    obs_file     = cfg.eval.get('obs_file', None)
    train_prisms = list(cfg.eval.get('train_prisms')) if cfg.eval.get('train_prisms') else None
    test_prisms  = list(cfg.eval.get('test_prisms'))  if cfg.eval.get('test_prisms')  else None
    train_bins   = list(cfg.eval.get('train_bins'))   if cfg.eval.get('train_bins')   else None
    test_bins    = list(cfg.eval.get('test_bins'))    if cfg.eval.get('test_bins')    else None

    if obs_file is None:
        raise ValueError("cfg.eval.obs_file must be set for robosuite eval")

    obs_data = np.load(obs_file)
    n_joints = obs_data['start_joints'].shape[1]

    metrics      = list(cfg.eval.get('metrics', []))
    eval_methods = list(cfg.eval.get('methods', ['ddim']))
    opt_params   = list(cfg.eval.get('opt_params') or [])

    method_variants = []
    if 'ired' in eval_methods:
        for i, op in enumerate(opt_params):
            label = f"ired_{op['n_opt']}steps"
            if op.get('t_subset') is not None:
                label += f"_last{op['t_subset']}"
            if op.get('denoise'):
                label += '_denoise'
            method_variants.append((label, i))
    if 'ddim' in eval_methods:
        ddim_idx = len(opt_params) if 'ired' in eval_methods else 0
        method_variants.append(('ddim', ddim_idx))

    conditions = _robosuite_conditions(is_goal_cond, train_prisms, test_prisms, train_bins, test_bins)
    if not conditions:
        raise ValueError("No eval conditions found — set train_prisms/test_prisms (and train_bins/test_bins for GC)")

    state_dim = policy.config.input_shapes["observation.state"][0]
    past_action_visible = state_dim > 8

    def get_state(obs, gripper_cmd, prev_action=None):
        state = np.concatenate([obs["robot0_joint_pos"], [gripper_cmd]]).astype(np.float32)
        if past_action_visible:
            pa = prev_action if prev_action is not None else np.zeros(8, dtype=np.float32)
            state = np.concatenate([state, pa])
        return state

    def get_image(obs):
        img = obs["agentview_image"].astype(np.float32) / 255.0
        return img.transpose(2, 0, 1)  # (3, H, W)

    def get_placed_bin(env):
        result = {'location': None, 'placement': None}
        for b in range(n_bins):
            if env.location_success(b):
                result['location'] = b
                if env.placement_success(b):
                    result['placement'] = b
                break
        return result

    all_metrics = {}
    env = make_eval_env(img_size=img_size, mujoco_object=obj, render=render)

    for method_label, method_traj_idx in method_variants:
      for chunk_size in chunk_sizes:
        rng = np.random.default_rng(seed)
        for cond_label, prisms, bins in conditions:
            indices = _sample_episode_indices(obs_data, prisms, bins, n_episodes, is_goal_cond, rng)
            location_bins, placement_bins, target_bins = [], [], []
            feat_scores = {m: [] for m in metrics}

            for ep_i, idx in enumerate(indices):
                target_bin  = int(obs_data['bin_idx'][idx])
                joint_start = obs_data['start_joints'][idx]

                obs = env.reset()
                env.sim.data.qpos[:n_joints] = joint_start
                env.sim.data.qvel[:n_joints] = 0.0
                env.sim.forward()
                # Re-place object at new EEF so the robot is grasping it
                # (_reset_internal placed it at the default rest-pose EEF, which moved when we teleported)
                eef_pos = env._eef_pos()
                env.sim.data.set_joint_qpos(
                    env.grasp_obj.joints[0],
                    np.concatenate([eef_pos, np.array([1, 0, 0, 0])])
                )
                env.sim.forward()
                # Hold arm at joint_start via full controller while closing gripper to stabilize grasp
                hold_act = np.concatenate([joint_start, [1.0]])
                for _ in range(50):
                    env.step(hold_act)
                obs, _, _, _ = env.step(hold_act)

                gripper_cmd = 1.0
                prev_action = np.zeros(8, dtype=np.float32)
                state_buf = deque([get_state(obs, gripper_cmd, prev_action)] * n_obs_steps, maxlen=n_obs_steps)
                image_buf = deque([get_image(obs)]                            * n_obs_steps, maxlen=n_obs_steps)

                if is_goal_cond:
                    goal = np.zeros(n_bins, dtype=np.float32)
                    goal[target_bin] = 1.0
                    goal_tensor = torch.tensor(goal).unsqueeze(0).to(device)

                eef_pos_buf  = []
                eef_quat_buf = []

                step = 0
                while step < max_steps:
                    obs_batch = {
                        'observation.state':
                            torch.tensor(np.stack(state_buf), dtype=torch.float32).unsqueeze(0).to(device),
                    }
                    if 'observation.image.agentview' in policy.config.input_shapes:
                        obs_batch['observation.image.agentview'] = \
                            torch.tensor(np.stack(image_buf), dtype=torch.float32).unsqueeze(0).to(device)
                    if is_goal_cond:
                        obs_batch['episode_goal'] = goal_tensor

                    if ep_i == 0 and step == 0:
                        print(f"=== EVAL DEBUG (ep={ep_i}, step={step}) ===")
                        print("obs_batch shapes:", {k: v.shape for k, v in obs_batch.items()})
                        print("state sample (first obs step):", obs_batch["observation.state"][0, 0])
                        if "observation.image.agentview" in obs_batch:
                            img = obs_batch["observation.image.agentview"][0, 0]
                            print(f"image range: [{img.min():.3f}, {img.max():.3f}]")
                        print("=================================")

                    if n_viz_samples > 0 and step == 0:
                        _viz_sampled_trajectories(env, obs, policy, obs_batch, n_viz_samples, n_joints, chunk_size, policy.config.n_obs_steps - 1, device)

                    with torch.no_grad():
                        _, full_trajs = policy.run_inference(obs_batch, methods=eval_methods, opt_params=opt_params, return_full=True)
                    start = policy.config.n_obs_steps - 1
                    chunk = full_trajs[method_traj_idx][0][start:start + chunk_size].cpu().numpy()

                    if ep_i == 0 and step == 0:
                        print(f"=== ACTION DEBUG (chunk_size={chunk_size}) ===")
                        print(f"full_traj shape: {full_trajs[0][0].shape}")
                        print(f"chunk[0] (first action): {chunk[0]}")
                        print(f"chunk mean: {chunk.mean(axis=0).round(4)}")
                        print(f"chunk std:  {chunk.std(axis=0).round(4)}")
                        print(f"current joint_pos: {obs['robot0_joint_pos'].round(4)}")
                        print(f"target_joints[0]:  {(obs['robot0_joint_pos'] + chunk[0, :7]).round(4)}")
                        print("=========================================")

                    for t in range(chunk_size):
                        delta         = chunk[t]
                        target_joints = obs["robot0_joint_pos"] + delta[:7]
                        gripper_cmd   = float(delta[7])
                        action        = np.concatenate([target_joints, [gripper_cmd]])
                        for _ in range(steps_per_action):
                            obs, _, _, _ = env.step(action)
                            if render:
                                env.render()
                        prev_action = delta.astype(np.float32)
                        state_buf.append(get_state(obs, gripper_cmd, prev_action))
                        image_buf.append(get_image(obs))
                        eef_pos_buf.append(obs["robot0_eef_pos"].copy())
                        eef_quat_buf.append(obs["robot0_eef_quat"].copy())
                        step += 1
                        if step >= max_steps or any(env.placement_success(b) for b in range(n_bins)):
                            break

                result = get_placed_bin(env)
                location_bins.append(result['location'])
                placement_bins.append(result['placement'])
                target_bins.append(target_bin)

                if metrics and eef_pos_buf:
                    from itps.trajectory_opt.geometric_features import (
                        BinXAlignment, BinYAlignment, ZTableDistance, GoalProgress
                    )
                    from itps.scripts.bin_placing import BinTableArena
                    positions = np.array(eef_pos_buf)
                    quats     = np.array(eef_quat_buf)
                    bx, by    = BinTableArena.BIN_XY[target_bin]
                    goal_pos  = obs_data['goal_pos'][idx]
                    feat_map  = {
                        'x_alignment':   BinXAlignment(x_bin=bx),
                        'y_alignment':   BinYAlignment(y_bin=by),
                        'z_table_dist':  ZTableDistance(table_z=0.8),
                        'goal_progress': GoalProgress(goal_pos=goal_pos),
                    }
                    for m in metrics:
                        if m in feat_map:
                            scores = feat_map[m](positions, quats)
                            feat_scores[m].append({'mean': float(scores.mean()), 'sum': float(scores.sum())})

            p = f'{method_label}/chunk{chunk_size}/{cond_label}'
            for m, ep_list in feat_scores.items():
                if ep_list:
                    all_metrics[f'{p}/{m}_mean'] = float(np.mean([e['mean'] for e in ep_list]))
                    all_metrics[f'{p}/{m}_sum']  = float(np.mean([e['sum']  for e in ep_list]))
            all_metrics[f'{p}/location_rate']        = float(np.mean([b is not None for b in location_bins]))
            all_metrics[f'{p}/placement_rate']       = float(np.mean([b is not None for b in placement_bins]))
            for i in range(n_bins):
                all_metrics[f'{p}/location_bin_{i}']  = sum(b == i for b in location_bins  if b is not None)
                all_metrics[f'{p}/placement_bin_{i}'] = sum(b == i for b in placement_bins if b is not None)
            if is_goal_cond:
                all_metrics[f'{p}/correct_location_rate']  = float(np.mean([b == t for b, t in zip(location_bins, target_bins)]))
                all_metrics[f'{p}/correct_placement_rate'] = float(np.mean([b == t for b, t in zip(placement_bins, target_bins)]))

    env.close()
    return {'aggregated': all_metrics}
