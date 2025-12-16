"""This scripts demonstrates how to train Diffusion Policy on the PushT environment.

Once you have trained a model with this script, you can try to evaluate it on
examples/2_evaluate_pretrained_policy.py
"""
from hydra import initialize, compose
from torch import nn
from omegaconf import OmegaConf
from pathlib import Path
import torch
from datetime import datetime
from itps.common.datasets.lerobot_dataset import LeRobotDataset
from itps.common.policies.factory import make_policy
from itps.common.datasets.factory import make_dataset
from itps.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from itps.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

class PreferencePairDataset(torch.utils.data.Dataset):
    def __init__(self, pos_ds, neg_ds):
        assert len(pos_ds) == len(neg_ds), \
            f"Positive ({len(pos_ds)}) and negative ({len(neg_ds)}) datasets must be same length."
        self.pos_ds = pos_ds
        self.neg_ds = neg_ds

    def __len__(self):
        # or `return min(len(self.pos_ds), len(self.neg_ds))`
        return len(self.pos_ds)

    def __getitem__(self, idx):
        pos_sample = self.pos_ds[idx]
        neg_sample = self.neg_ds[idx]
        return {
            "pos": pos_sample,
            "neg": neg_sample,
        }


def main():
    # Create a directory to store the training checkpoint.
    cond_type = 'unconditional_tuning'
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_directory = Path(f"outputs/tune/gmm/run_{cond_type}_{run_timestamp}")
    output_directory.mkdir(parents=True, exist_ok=True)

    # Number of offline training steps (we'll only do offline training for this example.)
    # Adjust as you prefer. 5000 steps are needed to get something worth evaluating.
    training_steps = 500
    device = torch.device("cuda:1")
    log_freq = 250

    # Make policy
    #pretrained_policy_path = "outputs/train/gmm/run_unconditional_2025-12-03_15-41-36/"
    pretrained_policy_path = "outputs/train/gmm/run_unconditional_2025-12-04_18-29-47/"
    policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)
    assert isinstance(policy, nn.Module)
    policy.train()
    policy.to(device)
    # Freeze all non-FiLM layers
    trainable_params = policy.freeze_nonFiLM()
    optimizer = torch.optim.Adam(trainable_params, lr=1e-4)

    # Emulate CLI arguments like:
    # python train.py env=sim policy=diffusion training.lr=1e-4
    initialize(config_path="../configs")
    cfg_main = compose(
        config_name="default",
        overrides=[ "env=gmm", "policy=gmm_dp_tune_base"]
    )
    dataset=make_dataset(cfg_main)

    cfg_pos = compose(
        config_name="default",
        overrides=[ "env=gmm", "policy=gmm_dp_tune_pos"],
    )
    cfg_neg = compose(
        config_name="default",
        overrides=[ "env=gmm", "policy=gmm_dp_tune_neg"]
    )
    dataset_pos = make_dataset(cfg_pos)
    dataset_neg = make_dataset(cfg_neg)
    pref_dataset = PreferencePairDataset(dataset_pos, dataset_neg)

    # Create dataloaders for offline training.
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=64,
        shuffle=True,
        pin_memory=device != torch.device("cpu"),
        drop_last=True,
    )

    pref_dataloader = torch.utils.data.DataLoader(
        pref_dataset,
        num_workers=4,
        batch_size=64,
        shuffle=True,
        pin_memory=device != torch.device("cpu"),
        drop_last=True,
    )

    # Run training loop.
    # step = 0
    # done = False
    # while not done:
    #     for batch in dataloader:

    main_iter = iter(dataloader)
    pref_iter = iter(pref_dataloader)


    #reset loss parameters
    # Loss computation
    policy.config.do_mask_loss_for_padding = cfg_main['policy']['do_mask_loss_for_padding']
    policy.config.gradient_loss_weight = cfg_main['policy']['gradient_loss_weight']
    policy.config.supervise_energy_landscape = cfg_main['policy']['supervise_energy_landscape']
    policy.config.energy_landscape_loss_weight = cfg_main['policy']['energy_landscape_loss_weight']
    policy.config.finetune_energy_landscape = cfg_main['policy']['finetune_energy_landscape']
    policy.config.finetune_loss_weight = cfg_main['policy']['finetune_loss_weight']
    #print(policy.diffusion.loss_weight)
    #policy.diffusion.loss_weight = torch.ones_like(policy.diffusion.loss_weight)
    #print(policy.diffusion.loss_weight)
    for step in range(training_steps):

        # --- Main dataset batch ---
        try:
            main_batch = next(main_iter)
        except StopIteration:
            main_iter = iter(dataloader)
            main_batch = next(main_iter)

        # --- Preference dataset batch ---
        try:
            pref_batch = next(pref_iter)
        except StopIteration:
            pref_iter = iter(pref_dataloader)
            pref_batch = next(pref_iter)

        main_batch = {k: v.to(device, non_blocking=True) for k, v in main_batch.items()}
        pos_batch = {k: v.to(device, non_blocking=True) for k, v in pref_batch["pos"].items()}
        neg_batch = {k: v.to(device, non_blocking=True) for k, v in pref_batch["neg"].items()}
        output_dict = policy.forward(main_batch, tune_batch=(pos_batch, neg_batch))
        loss = output_dict["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % log_freq == 0:
            print(f"step: {step} loss: {loss.item():.3f}")
        step += 1
        if step >= training_steps:
            done = True
            break

    # Save a policy checkpoint.
    policy.save_pretrained(output_directory)


if __name__ == "__main__":
    main()
