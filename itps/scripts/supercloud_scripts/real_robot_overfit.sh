#!/bin/bash
# real_robot_overfit.sh
#
# Supercloud LLsub submission for the two real-robot OVERFIT runs (DP + EBM, 10 demos of s1_disc5 -> e1,
# 30k steps each), used to develop/test inference. Separate from real_robot_pretrain.sh on purpose: the
# overfit configs live in configs/policy/real_robot/overfit/, which that script never globs.
#
# Before submitting:
#   * the repo code (with the real_robot loader, EMA, per-camera encoders) is on Supercloud;
#   * itps/data/real_robot/cup_overfit10.hdf5 AND cup_multi_shelf.hdf5 are copied over (rank 0 checks both).
#
# Submit from the itps/ dir:
#   LLsub ./scripts/supercloud_scripts/real_robot_overfit.sh [1,2,9] -g volta:2
#     1 node x 2 processes (one run per GPU) x 9 threads (1 main + 8 dataloader workers).
#
# Rank 0, before its training run, also runs envs/real_robot/check_dataset.py on both converted files WITH
# images (the full file needs ~28 GB RAM, too much for the laptop). Its output is in rank 0's log -- read it
# before submitting real_robot_pretrain.sh. A failed check does not stop the overfit training.

# Initialize the module command first
source /etc/profile

# Load the conda ML module ONLY -- no `module load cuda/...`: the module bundles its own CUDA/cuDNN, and a
# separate cuda module breaks conv1d ("FIND was unable to find an engine").
module load conda/Python-ML-2026a-pytorch

export PYTHONPATH="/home/gridsan/aforsey/diff-tuning:$PYTHONPATH"
export WANDB_MODE=offline

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIGS_DIR="/home/gridsan/aforsey/diff-tuning/itps/configs/policy/real_robot/overfit"
SCRIPT="scripts/train.py"
ENV_NAME="real_robot"
# ──────────────────────────────────────────────────────────────────────────────

echo "======================================"
echo "My task rank:    $LLSUB_RANK"
echo "Number of tasks: $LLSUB_SIZE"
echo "Configs dir:     $CONFIGS_DIR"
echo "======================================"
nvidia-smi --query-gpu=name,memory.total --format=csv

if [ "$LLSUB_RANK" = "0" ]; then
    echo "=== check_dataset: cup_overfit10.hdf5 (with images) ==="
    python envs/real_robot/check_dataset.py --root data/real_robot/cup_overfit10.hdf5 \
        --config real_robot/overfit/run_dp_visual_overfit10
    echo "=== check_dataset: cup_multi_shelf.hdf5 (with images) ==="
    python envs/real_robot/check_dataset.py --root data/real_robot/cup_multi_shelf.hdf5 \
        --config real_robot/pretrain/run_dp_visual --n_items 5000
    echo "=== checks done ==="
fi

python scripts/supercloud_scripts/run_job.py $LLSUB_RANK $LLSUB_SIZE \
    --configs_dir $CONFIGS_DIR \
    --script $SCRIPT \
    --env $ENV_NAME
