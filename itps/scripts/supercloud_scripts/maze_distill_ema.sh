#!/bin/bash
# maze_distill_ema.sh
#
# R1 with EMA weights, at 50k and 100k steps -- the 2 configs in configs/policy/maze/distill_ema/.
# Nothing differs from run_r1_product_dp_rollout except offline_steps and training.ema.
#
# R1 was the best arm (3-4% collisions, converged by 20-30k), so these ask two things:
#   * does EMA help on top of it;
#   * is 50k enough, given the cosine schedule over 100k spends most of the run at a learning rate
#     the result does not appear to need.
# Each run reports BOTH weight sets: the EMA metrics under the usual names (pretrained_model/ holds
# the EMA weights) and the online weights suffixed `_online`.
#
# Before submitting, check the teacher is present -- both configs inherit it from R1:
#   ls data/maze2d_dp/sparse/gc/train/dp_sparse_gc/pretrained_model
#
# Submit from the itps/ dir:
#   LLsub ./scripts/supercloud_scripts/maze_distill_ema.sh [1,2,9] -g volta:2
#     1 node x 2 processes (one run per GPU) x 9 threads (1 main + 8 dataloader workers).
#
# ~0.35 s/step measured for this arm, so ~5 h for the 50k run and ~10 h for the 100k one. Eval now
# runs twice per eval_freq (both weight sets), but eval is ~5 s once _ITEMS_CACHE is warm.

# Initialize the module command first
source /etc/profile

# Load the conda ML module ONLY -- no `module load cuda/...` (breaks cuDNN).
module load conda/Python-ML-2026a-pytorch

export PYTHONPATH="/home/gridsan/aforsey/diff-tuning:$PYTHONPATH"
export WANDB_MODE=offline

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIGS_DIR="/home/gridsan/aforsey/diff-tuning/itps/configs/policy/maze/distill_ema"
SCRIPT="scripts/train.py"
ENV_NAME="maze2d"
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
