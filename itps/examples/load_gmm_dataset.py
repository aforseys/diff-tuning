"""
This script demonstrates the use of `LeRobotDataset` class for handling and processing robotic datasets from Hugging Face.
It illustrates how to load datasets, manipulate them, and apply transformations suitable for machine learning tasks in PyTorch.

Features included in this script:
- Loading a dataset and accessing its properties.
- Filtering data by episode number.
- Converting tensor data for visualization.
- Saving video files from dataset frames.
- Using advanced dataset features like timestamp-based frame selection.
- Demonstrating compatibility with PyTorch DataLoader for batch processing.

The script ends with examples of how to batch process data using PyTorch's DataLoader.
"""

from pathlib import Path
from pprint import pprint

import imageio
import torch

from itps.common.datasets.lerobot_dataset import LeRobotDataset

# print("List of available datasets:")
# pprint(lerobot.available_datasets)

# Let's take one for this example
repo_id = "gmm"

delta_timestamps = {
    "observation.environment_state": [0],
    "observation.state": [0],
    "action": [0]
}

# You can easily load a dataset from a Hugging Face repository
dataset = LeRobotDataset(repo_id, 'data/gmm_conditional_1000_42_20251105_161707.npy', split=None, delta_timestamps=delta_timestamps)

# LeRobotDataset is actually a thin wrapper around an underlying Hugging Face dataset
# (see https://huggingface.co/docs/datasets/index for more information).
print(dataset)
print(dataset.hf_dataset)

# And provides additional utilities for robotics and compatibility with Pytorch
print(f"\naverage number of frames per episode: {dataset.num_samples / dataset.num_episodes:.3f}")
print(f"frames per second used during data collection: {dataset.fps=}")
print(f"keys to access images from cameras: {dataset.camera_keys=}\n")

print(f"\n{dataset[0]['observation.environment_state'].shape=}")  # (4,c,h,w)
print(f"{dataset[0]['observation.state'].shape=}")  # (8,c)
print(f"{dataset[0]['action'].shape=}\n")  # (64,c)

#TODO: PRINT AND TEST EPISODE INDICES

dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=0,
    batch_size=32,
    shuffle=True,
)
for batch in dataloader:
    print(f"{batch['observation.environment_state'].shape=}")  # (32,1,1)
    print(f"{batch['observation.state'].shape=}")  # (32,1,1)
    print(f"{batch['action'].shape=}")  # (32,1,2)
    if 'action_is_pad' in batch: 
        print('Includes padded actions')
        print('Pads:', batch['action_is_pad'])
        print('Pad counts:', batch['action_is_pad'].sum(dim=1))
    break