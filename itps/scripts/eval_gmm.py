#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import numpy as np
import torch
import logging 
from pathlib import Path
from matplotlib import pyplot as plt
from datetime import datetime as dt
from torch import Tensor, nn
from itps.common.policies.factory import make_policy
from itps.common.datasets.factory import make_dataset
from itps.common.envs.factory import make_env
from itps.common.utils.utils import get_safe_torch_device, init_hydra_config, init_logging, set_global_seed
from itps.common.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
)
from itps.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from itps.scripts.gaussian_mm import get_weights, get_means, get_covs, mixture_pdf

def gen_obs(conditional, N):
    "generates a batch object that matches same type as passed through model, only contains obs"

    observations=[]
    for i in range(1 if not conditional else 3):
        obs_tensor = torch.full((N, 1, 1), i, dtype=torch.float32, device=torch.device("cuda"))
        obs_dict= {
            'observation.state':obs_tensor, 
            'observation.environment_state':obs_tensor
        }
        observations.append(obs_dict)
    return observations


def gen_xy_grid(x_range, y_range, torchify=True):

    xmin,xmax=x_range
    ymin,ymax=y_range
    
    xx, yy = np.meshgrid(
    np.linspace(xmin, xmax, 200),
    np.linspace(ymin, ymax, 200)
    )

    trajs = np.column_stack([xx.ravel(), yy.ravel()]) 

    if torchify: 
        trajs = torch.tensor(trajs, dtype=torch.float32, device=torch.device("cuda")).unsqueeze(dim=1) 

    return trajs


def run_inference(policy, N=100, conditional=False): #, return_energy=False):

    obs = gen_obs(conditional=conditional, N=N)

    inference_output = []
    for o in obs:
        #print('obs shape:', o.shape) 
        # if return_energy: 
        #     actions, energy = policy.run_inference(o, return_energy=return_energy)
        #     inference_output.append((actions.detach().cpu().squeeze(1).numpy(), energy.detach().cpu().numpy())) 
        # else:
        actions = policy.run_inference(o) #, return_energy=return_energy)
        inference_output.append(actions.detach().cpu().squeeze(1).numpy())
    #print('actions output shape:', actions.shape)

    return inference_output

def eval_energy(policy, trajs, t, conditional=False, batch_size=256):

    observations = gen_obs(conditional=conditional, N=len(trajs))
    # global_conds = [policy.diffusion._prepare_global_conditioning(o) for o in obs]
    energies = []
    for obs in observations:
        #print(gc.dtype)
        #print(trajs.dtype)
        outputs=[]
        for i in range(0, trajs.size(0), batch_size):
            batch_traj = {'action': trajs[i:i+batch_size]}
            batch_obs = {k: v[i:i+batch_size] for k, v in obs.items()}
            out = policy.get_energy(action_batch=batch_traj, t=t, observation_batch=batch_obs)
            outputs.append(out.detach().cpu().numpy())
        energies.append(np.concatenate(outputs, axis=0))
    return energies

def vis_inference(policy, samples, conditional, learned_contour=True, t=0, x_range=(-10, 10), y_range=(-10,10)):

     #if plotting over learned energy contour
    if learned_contour:
        trajs = gen_xy_grid(x_range=x_range, y_range=y_range)
        print('Evaluating energy')
        energies = eval_energy(policy, trajs, t, conditional=conditional)
        xx = trajs[:, 0, 0].cpu().numpy().reshape(200,200)
        yy = trajs[:, 0, 1].cpu().numpy().reshape(200,200)
        print('Energy evaluated, generating samples')

    #otherwise plot over gt pdf 
    else: 
        trajs = gen_xy_grid(x_range=x_range, y_range=y_range, torchify=False)
        if not conditional:
            energies = [mixture_pdf(trajs, get_weights(), get_means(), get_covs())]
        else:  #set weight to be nonzero for non conditional cluster
            energies = [mixture_pdf(trajs, np.eye(3, dtype=int)[i], get_means(), get_covs()) for i in range(3)]

        xx = trajs[:,0].reshape(200,200)
        yy = trajs[:,1].reshape(200,200)

    #samples = run_inference(policy, N=N, conditional=conditional)

    # print('Samples collected')
    # print(trajs.shape)
    # print(energies[0].shape)
    # print(samples[0].shape)

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
            zz=-zz
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

    trajs = gen_xy_grid(x_range=x_range, y_range=y_range)
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

    #train_data = np.load(training_samples_path)[:, 1:] #remove conditional obs
    #N = np.shape(train_data)[0]
    #samples = run_inference(policy, N=N, conditional=conditional)

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


def main(
    pretrained_policy_path: Path | None = None,
    hydra_cfg_path: str | None = None,
    out_dir: str | None = None,
    config_overrides: list[str] | None = None,
    conditional: bool | None = False,
    seed: int | None = 0,
    training_samples = None,
):
    assert (pretrained_policy_path is None) ^ (hydra_cfg_path is None)
    if hydra_cfg_path is not None:
    #     hydra_cfg = init_hydra_config(str(pretrained_policy_path / "config.yaml"), config_overrides)
    # else:
        hydra_cfg = init_hydra_config(hydra_cfg_path, config_overrides)
    else: 
        hydra_cfg = None

    if out_dir is None:
        if hydra_cfg is not None:
            out_dir = f"outputs/eval/{dt.now().strftime('%Y-%m-%d/%H-%M-%S')}_{hydra_cfg.env.name}_{hydra_cfg.policy.name}"
        else:
            out_dir = f"outputs/eval/{dt.now().strftime('%Y-%m-%d/%H-%M-%S')}_{str(pretrained_policy_path).split('/')[-1]}"
    
    # Check device is available
    #device = get_safe_torch_device(hydra_cfg.device, log=True)

    logging.info("Making policy.")
    if hydra_cfg_path is None:
        #policy = make_policy(hydra_cfg=hydra_cfg, pretrained_policy_name_or_path=str(pretrained_policy_path))
        policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)
    else:
        # Note: We need the dataset stats to pass to the policy's normalization modules.
        policy = make_policy(hydra_cfg=hydra_cfg, dataset_stats=make_dataset(hydra_cfg).stats)
    assert isinstance(policy, nn.Module)
    policy.cuda()
    policy.eval()
    #print('MY OUTPUT ACTION SPACE', policy.config.output_shapes["action"])
    #print('MY CONFIG HORIZON SIZE', policy.config.horizon)
    # device = get_device_from_parameters(policy)
    set_global_seed(seed)
    
    if training_samples is not None:
        train_data = np.load(training_samples)[:, 1:] #remove conditional obs
        N = np.shape(train_data)[0]
        samples = run_inference(policy, N=N, conditional=conditional)
        vis_sample_comparison(samples, train_data)

    else:
        samples=run_inference(policy, N=100, conditional=conditional)
        for obs_samples in samples:
             print(np.mean(np.array(obs_samples),axis=0))

    vis_inference(policy, samples=samples, conditional=conditional, learned_contour=False)
    
    for i in range(10):
         vis_inference(policy, samples=samples, conditional=conditional, learned_contour=True, t=i*10)
         vis_energy_landscape(policy, conditional, t=i*10)

if __name__ == "__main__":
    init_logging()

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-p",
        "--pretrained-policy-name-or-path",
        help=(
            "Path to a directory containing weights saved using `Policy.save_pretrained`."
            "This argument is mutually exclusive with `--config`."
        ),
    )
    group.add_argument(
        "--config",
        help=(
            "Path to a yaml config you want to use for initializing a policy from scratch (useful for "
            "debugging). This argument is mutually exclusive with `--pretrained-policy-name-or-path` (`-p`)."
        ),
    )
    parser.add_argument("--revision", help="Optionally provide the Hugging Face Hub revision ID.")
    parser.add_argument(
        "--out-dir",
        help=(
            "Where to save the evaluation outputs. If not provided, outputs are saved in "
            "outputs/eval/{timestamp}_{env_name}_{policy_name}"
        ),
    )


    parser.add_argument("--training-samples", help = "Optionally provide path to original samples for visualiztion.")


    parser.add_argument(
        "overrides",
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
    )

    parser.add_argument(
        "--conditional",
        action="store_true",
        help="Conditional GMM",
    )

    parser.add_argument(
        "--seed",
        type=int,
        nargs="?",
        help="Inference seed",
        const=0
    )

    args = parser.parse_args()

    if args.pretrained_policy_name_or_path is None:
        main(hydra_cfg_path=args.config, out_dir=args.out_dir, config_overrides=args.overrides)
    else:
        pretrained_policy_path = Path(args.pretrained_policy_name_or_path)

        main(
            pretrained_policy_path=pretrained_policy_path,
            out_dir=args.out_dir,
            config_overrides=args.overrides,
            conditional=args.conditional,
            seed=args.seed,
            training_samples=args.training_samples
        )
