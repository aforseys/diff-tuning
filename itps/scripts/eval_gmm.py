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


def gen_xy_grid(x_range, y_range):

    xmin,xmax=x_range
    ymin,ymax=y_range
    
    xx, yy = np.meshgrid(
    np.linspace(xmin, xmax, 200),
    np.linspace(ymin, ymax, 200)
    )

    grid = np.column_stack([xx.ravel(), yy.ravel()]) 
    trajs = torch.tensor(grid, dtype=torch.float32, device=torch.device("cuda")).unsqueeze(dim=1) 
    return trajs


def run_inference(policy, N=100, conditional=False, return_energy=False):

    obs = gen_obs(conditional=conditional, N=N)

    inference_output = []
    for o in obs: 
        if return_energy: 
            actions, energy = policy.run_inference(o, return_energy=return_energy)
            inference_output.append((actions.detach().cpu().unsqueeze(1).numpy(), energy.detach().cpu().numpy())) 

        actions = policy.run_inference(o, return_energy=return_energy)
        inference_output.append(actions.detach().cpu().unsqueeze(1).numpy())

    return inference_output

def eval_energy(policy, trajs, conditional=False, batch_size=256):

    obs = gen_obs(conditional=conditional, N=len(trajs))
    global_conds = [policy.diffusion._prepare_global_conditioning(o) for o in obs]
    energies = []
    for gc in global_conds:
        #print(gc.dtype)
        #print(trajs.dtype)
        outputs=[]
        for i in range(0, trajs.size(0), batch_size):
            out = policy.diffusion.get_traj_energies(trajectories=trajs[i:i+batch_size], global_cond=gc[i:i+batch_size])
            outputs.append(out.detach().cpu().numpy())
        energies.append(np.concatenate(outputs, axis=0))
    return energies

def vis_inference(policy, conditional, N, x_range=(-8, 8), y_range=(-8,8)):

    trajs = gen_xy_grid(x_range=x_range, y_range=y_range)
    energies = eval_energy(policy, trajs, conditional=conditional)
    samples = run_inference(policy, N=N, conditional=conditional)

    print(trajs.shape)
    print(energies.shape)
    print(samples.shape)

    xx = trajs[:, 0, 0].cpu().numpy().reshape(200,200)
    yy = trajs[:, 0, 1].cpu().numpy().reshape(200,200)

    #plot all energy landscapes in list given trajs
    for i in range(len(energies)):
        #plot 
        zz = energies[i].reshape(200,200)
        if conditional:
            title = "Energy landscape conditioned on cluster observation {i}"
        else:
            title = "Energy landscape (unconditional)"

        plt.figure(i)
        plt.contourf(xx, yy, zz, levels=20)
        # plot where sampled points are with x's 
        plt.plot(samples[:,0], samples[:,1], marker='x', markersize=4)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(title)
  
        
def vis_energy_landscape(policy, conditional, x_range=(-8, 8), y_range=(-8,8)):

    trajs = gen_xy_grid(x_range=x_range, y_range=y_range)
    energies = eval_energy(policy, trajs, conditional=conditional)

    print(trajs.shape)
    print(len(energies))
    print(energies[0].shape)

    xx = trajs[:, 0, 0].cpu().numpy().reshape(200,200)
    yy = trajs[:, 0, 1].cpu().numpy().reshape(200,200)

    #plot all energy landscapes in list given trajs
    for i in range(len(energies)):
        zz = energies[i].reshape(200,200)
        if conditional:
            title = "Energy landscape conditioned on cluster observation {i}"
        else:
            title = "Energy landscape (unconditional)"

        plt.figure(i)
        plt.axes(projection="3d").plot_surface(xx, yy, zz, cmap="viridis", edgecolor="none")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(title)
        plt.show()

def main(
    pretrained_policy_path: Path | None = None,
    hydra_cfg_path: str | None = None,
    out_dir: str | None = None,
    config_overrides: list[str] | None = None,
    conditional: bool | None = False,
    seed: int | None = 0
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
    # device = get_device_from_parameters(policy)
    set_global_seed(seed)
    vis_energy_landscape(policy, conditional)
    vis_inference(policy, conditional, N=500)


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
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
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
        )
