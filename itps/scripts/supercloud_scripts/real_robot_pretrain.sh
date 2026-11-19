#!/bin/bash
# real_robot_pretrain.sh
#
# Supercloud LLsub submission for the real-robot base-policy batch on cup_multi_shelf (297 demos, 400k steps):
# the 5 configs in configs/policy/real_robot/pretrain/ --
#   run_dp_visual (base DP), run_ebm_visual (base EBM),
#   run_dp_visual_encoder, run_dp_visual_h24, run_dp_visual_capacity (DP ablation arms, one factor each).
# The overfit configs are in a different folder and are NOT picked up (see real_robot_overfit.sh).
#
# Before submitting:
#   * check_dataset.py passed on cup_multi_shelf.hdf5 WITH images (rank 0 of real_robot_overfit.sh runs it);
#   * the EBM AMP check is done (use_amp: True in both base configs is pending it; on V100 AMP is fp16);
#   * for run_dp_visual_encoder: download the ImageNet ResNet-18 weights once on a LOGIN node (compute nodes are
#     offline); they land in ~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth, which compute nodes share:
#       python -c "import torchvision; torchvision.models.resnet18(weights='IMAGENET1K_V1')"
#
# Submit from the itps/ dir:
#   LLsub ./scripts/supercloud_scripts/real_robot_pretrain.sh [3,2,9] -g volta:2
#     3 nodes x 2 processes (one run per GPU) x 9 threads (1 main + 8 dataloader workers) = 6 slots for 5 runs.
# Memory: each run holds all images in RAM (~9 GB, ~2x that while loading, ~28 GB peak with overhead); at
# ~9.6 GB per core, NTPP 9 gives ~87 GB per process.
# Rough time per run (V100 + AMP, +-2x): DP ~8 h, EBM ~11 h; the capacity arm is the slowest DP run.

# Initialize the module command first
source /etc/profile

# Load the conda ML module ONLY -- no `module load cuda/...` (breaks cuDNN, see real_robot_overfit.sh).
module load conda/Python-ML-2026a-pytorch

export PYTHONPATH="/home/gridsan/aforsey/diff-tuning:$PYTHONPATH"
export WANDB_MODE=offline

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIGS_DIR="/home/gridsan/aforsey/diff-tuning/itps/configs/policy/real_robot/pretrain"
SCRIPT="scripts/train.py"
ENV_NAME="real_robot"
# ──────────────────────────────────────────────────────────────────────────────

echo "======================================"
echo "My task rank:    $LLSUB_RANK"
echo "Number of tasks: $LLSUB_SIZE"
echo "Configs dir:     $CONFIGS_DIR"
echo "======================================"
nvidia-smi --query-gpu=name,memory.total --format=csv

python scripts/supercloud_scripts/run_job.py $LLSUB_RANK $LLSUB_SIZE \
    --configs_dir $CONFIGS_DIR \
    --script $SCRIPT \
    --env $ENV_NAME
