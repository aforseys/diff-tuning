#!/bin/bash
# real_robot_cmd.sh
#
# Supercloud LLsub submission for the commanded-joint-label ablation on cup_multi_shelf (400k steps):
# the 2 configs in configs/policy/real_robot/cmd_labels/ --
#   rank 0: run_dp_visual_cmd (base DP), rank 1: run_ebm_visual_cmd (base EBM).
# Each is identical to its pretrain/ counterpart except dataset_root, action_labels: commanded and the job name.
#
# Before submitting:
#   * data/real_robot/cup_multi_shelf_cmd.hdf5 (convert_cup_demos.py + build_cmd_labels.py) is on Supercloud;
#   * check_dataset.py passed on it WITH images.
#
# Submit from the itps/ dir:
#   LLsub ./scripts/supercloud_scripts/real_robot_cmd.sh [1,2,9] -g volta:2
#     1 node x 2 processes (one run per GPU) x 9 threads (1 main + 8 dataloader workers).
# Memory: each run holds all images in RAM (~28 GB peak); NTPP 9 gives ~87 GB per process.
# Rough time per run (V100 + AMP, +-2x): DP ~8 h, EBM ~11 h.

# Initialize the module command first
source /etc/profile

# Load the conda ML module ONLY -- no `module load cuda/...` (breaks cuDNN, see real_robot_overfit.sh).
module load conda/Python-ML-2026a-pytorch

export PYTHONPATH="/home/gridsan/aforsey/diff-tuning:$PYTHONPATH"
export WANDB_MODE=offline

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIGS_DIR="/home/gridsan/aforsey/diff-tuning/itps/configs/policy/real_robot/cmd_labels"
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
