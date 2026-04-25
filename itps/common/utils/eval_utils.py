#!/usr/bin/env python

# Alexandra Forsey-Smerek
import time
import json
import numpy as np
import torch
from matplotlib import pyplot as plt
from itps.scripts.gaussian_mm import get_weights, get_means, get_covs, mixture_pdf

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

def run_inference(policy, N=100, conditional=False, opt_params=[1]):

    assert not conditional, "Conditional sampling not supported for multiple opt steps"

    device = next(policy.parameters()).device
    obs = gen_obs(conditional=conditional, N=N, device=device)

    IRED_inference_output = [[] for _ in opt_params]
    DDIM_inference_output = []
    #sample_times = []
    o=obs[0]

    #for o in obs:
        #start = time.perf_counter()
    actions = policy.run_inference(o, both=True, opt_params=opt_params)
    #torch.cuda.synchronize()
    # elapsed = time.perf_counter() - start

    # sample_times.append(elapsed)
    # print(f"Sample time: {elapsed*1000:.1f}ms")
    DDIM_inference_output.append(actions[-1].detach().cpu().squeeze(1).numpy())
    for i in range(len(opt_params)):
        IRED_inference_output[i].append(actions[i].detach().cpu().squeeze(1).numpy())

    # print(f"Avg sample time over {len(sample_times)} obs: {np.mean(sample_times)*1000:.1f}ms")
    return IRED_inference_output + [DDIM_inference_output]

def run_inference_with_grad_steps(policy, N=50, conditional=False, opt_params=[1]):

    assert not conditional, "Conditional sampling not supported for multiple opt steps"

    device = next(policy.parameters()).device
    obs = gen_obs(conditional=conditional, N=N, device=device)

    grad_histories_per_opt = [[] for _ in opt_params]
    #sample_times = []
    o=obs[0]

    
    #start = time.perf_counter()
    _, grad_histories = policy.run_inference(o, both=False, opt_params=opt_params, return_grad_steps=True)
    #torch.cuda.synchronize()
    # elapsed = time.perf_counter() - start
    for i in range(len(opt_params)):
        grad_histories_per_opt[i].append(grad_histories[i])

    # sample_times.append(elapsed)
    # print(f"Sample time: {elapsed*1000:.1f}ms")
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

def log_likelihood(policy, conditional, finetune, N=100, samples=None, opt_params=[1]):
    """
    Assumes only conditional or finetune is true. 
    """

    assert not (conditional and finetune), "Simultaneous conditional and finetune not supported"
    if samples is None:
        samples =run_inference(policy, N=N, conditional=conditional, opt_params=opt_params)
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
def vis_inference(policy, samples, conditional, learned_contour=True, t=0, x_range=(-10, 10), y_range=(-10,10)):

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

def vis_energy_landscape(policy, conditional, t=0, x_range=(-8, 8), y_range=(-8,8)):

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

def vis_sample_comparison(samples, train_data):

    for i in range(len(samples)):
        plt.figure(i)
        plt.scatter(train_data[:,0], train_data[:,1], s=8, alpha=0.6, edgecolor='none')
        plt.scatter(samples[i][:,0], samples[i][:,1], s=8, alpha=0.6, edgecolor='none')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.xlim(-8,8)
        plt.ylim(-8,8)
        plt.title(f"Samples against training data (Obs:{i})")
        plt.show()

def vis_ired_grad_steps(policy, grad_history, t, conditional, n_inner_steps_label="", x_range=(-10, 10), y_range=(-10,10)):
 
      """         
      Overlay IRED gradient step arrows on the learned energy landscape at denoising  
  timestep t.                                                                         
   
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
                        cmap='cool', alpha=0.8, scale=1, scale_units='xy',            
  angles='xy', zorder=4)                                                              
          plt.colorbar(q, ax=ax, label='step magnitude')                              
                                                                                      
          ax.set_xlim(x_range)                                                        
          ax.set_ylim(y_range)
          ax.set_title(f"inner step {step_i + 1}/{n_inner}\nmean={magnitudes.mean():.3f}, max={magnitudes.max():.3f}")            
          ax.set_xlabel("X")
          ax.set_ylabel("Y")                                                          
                  
      title = f"IRED gradient steps at denoising t={t}"                               
      if n_inner_steps_label:
          title += f" ({n_inner_steps_label} inner steps/timestep)"                   
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

def eval_GMM(policy, condition_type, finetune, N, viz=False, training_samples=None, opt_params=[1], viz_opt=False):

    if condition_type == "conditional":
        conditional=True
    elif condition_type == "unconditional":
        conditional=False
    else: 
        raise NotImplementedError("Only 'unconditional' or 'conditional' condition_types supported for GMM")

    #Calculate KL divergence between distributions 
    kl_div = kl_divergence(policy, conditional, finetune)

    # Generate samples and calculate log likelihood
    samples, ll = log_likelihood(policy, conditional, finetune, N, opt_params=opt_params)

    # DDIM samples are last set, all others are IRED sampling 
    DDIM_samples = samples[-1]
    IRED_samples = samples[0:-1]

    DDIM_ll = ll[-1]
    IRED_ll = ll[0:-1]

    info ={
        "aggregated":{
            "kl_div": kl_div, 
            "DDIM_log_likelihood": DDIM_ll
            }
    }
    for i in range(len(opt_params)):

        if isinstance(opt_params[i], (list, tuple)):
            label = f'IRED_{opt_params[i][0]}steps_last{opt_params[i][1]}'                                                                                          
        else:                                     
            label = f'IRED_{opt_params[i]}_steps' 

        info["aggregated"][label] = IRED_ll[i]

    if training_samples is not None:
        train_data = np.load(training_samples)
        filtered_samples = filter_samples(train_data, finetune, conditional)
        ll_training = log_likelihood(policy, conditional, finetune, samples=filtered_samples)
        print('Log likelihood of training samples:', ll_training)

    if viz:
        # Visualize training samples if passed in
        if training_samples is not None:
            train_data = np.load(training_samples)[:, 1:] #remove conditional obs
            samples = run_inference(policy, N=np.shape(train_data)[0], conditional=conditional, opt_params=opt_params)
            DDIM_samples = samples[-1]
            IRED_samples = samples[0:-1]
            vis_sample_comparison(DDIM_samples, train_data)
            for opt_step_samples in IRED_samples:
                vis_sample_comparison(opt_step_samples, train_data)

            # Visualize inferred samples over gt distribution 
            vis_inference(policy, samples=DDIM_samples, conditional=conditional, learned_contour=False)
            for opt_step_samples in IRED_samples:
                vis_inference(policy, samples=opt_step_samples, conditional=conditional, learned_contour=False)

        # If visualizing the gradient steps
        if viz_opt:
          assert not conditional, "Conditional sampling not supported for grad viz"
          grad_N = min(50, N)                                                         
          grad_histories_per_opt = run_inference_with_grad_steps(                     
              policy, N=grad_N, conditional=conditional, opt_params=opt_params          
          )                                                                           
          for t in range(10):
              for step_i, n_steps in enumerate(opt_params):                            
                    grad_hist = grad_histories_per_opt[step_i][0]
                    vis_ired_grad_steps(                                            
                        policy, grad_hist, t=t, conditional=conditional,
                        n_inner_steps_label=str(n_steps)               
                    )
        else:
            # Visualize learned distribution at different denoising steps
            for i in range(10):
                vis_inference(policy, samples=DDIM_samples, conditional=conditional, learned_contour=True, t=i)
                for opt_step_samples in IRED_samples:
                    vis_inference(policy, samples=opt_step_samples, conditional=conditional, learned_contour=True, t=i)
                vis_energy_landscape(policy, conditional, t=i)

    return info


# -- MAZE2D EVALUATION --

MAZE_MAPS = {
    "open": np.array([[1, 1, 1, 1, 1, 1, 1],
                      [1, 0, 0, 0, 0, 0, 1],
                      [1, 0, 0, 0, 0, 0, 1],
                      [1, 0, 0, 0, 0, 0, 1],
                      [1, 1, 1, 1, 1, 1, 1]]).astype(bool),
    "sparse": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
                        [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                        [1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1],
                        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                        [1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).astype(bool),
    "large": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                       [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                       [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                       [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                       [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                       [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                       [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                       [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).astype(bool),
}


def check_maze_collision(xy_traj, maze):
    """
    xy_traj: (batch, steps, 2) numpy array in maze coordinate space
    maze: 2D boolean array where True = wall
    Returns: (batch,) boolean array, True if trajectory has any collision
    """
    batch_size, num_steps, _ = xy_traj.shape
    xy_flat = xy_traj.reshape(-1, 2)
    xy_flat = np.clip(xy_flat, [0, 0], [maze.shape[0] - 1, maze.shape[1] - 1])
    mx = np.round(xy_flat[:, 0]).astype(int)
    my = np.round(xy_flat[:, 1]).astype(int)
    collisions = maze[mx, my].reshape(batch_size, num_steps)
    return np.any(collisions, axis=1)


def eval_maze(policy, cfg, split='test'):
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
    states = np.repeat(obs_data, n_samples, axis=0)
    state_t = torch.tensor(states, dtype=torch.float32, device=device)  

    # (N_obs*n_samples, n_obs_steps, 2) — repeat starting pos for all history steps
    obs = {
        'observation.state':
            state_t.unsqueeze(1).expand(-1, n_obs_steps, -1).clone(),
        'observation.environment_state':
            state_t.unsqueeze(1).expand(-1, n_obs_steps, -1).clone(),
    }

    with torch.no_grad():
        actions_list = policy.run_inference(obs, both=True, opt_params=opt_params)

    # run infernece unnormalizes --> trajectories are in coordinate space
    # actions_list: [IRED_opt0, ..., DDIM] each (N_obs * n_samples, horizon, 2)
    trajs = [a.cpu().numpy() for a in actions_list]

    metrics_dict ={}
    for i, traj in enumerate(trajs):
        per_obs ={}

        #TODO: check all this math with numpy and the batches etc. executes / lines up correctly
        for m in metrics:
            if m == 'collision_rate':
                collisions = check_maze_collision(traj, maze)
                per_obs[m] = collisions.reshape(N_obs, n_samples).mean(axis=1).tolist() 
            elif m == 'goal_dist':
                dists = np.linalg.norm(traj[:, -1, :] - np.array(cfg.eval.goal), axis=1)
                per_obs[m] = dists.reshape(N_obs, n_samples).mean(axis=1).tolist()
            else:
                raise NotImplementedError(f"Metric '{m}' not implemented") 

        if isinstance(opt_params[i], (list, tuple)):
            label = f'IRED_{opt_params[i][0]}steps_last{opt_params[i][1]}'                                                                                          
        else:                                     
            label = f'IRED_{opt_params[i]}_steps'   

        metrics_dict[label] ={
            f"{split}_{m}": {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "per_obs": vals}
            for m, vals in per_obs.items()
        }
    
    return metrics_dict
