#!/bin/bash
# maze_distill.sh
#
# Supercloud LLsub submission for the sparse-maze energy-distillation batch: the 6 configs in
# configs/policy/maze/distill/ --
#   R1 product/dp/teacher_rollout        R2 product/scratch/teacher_rollout
#   R3 product/scratch/ordinary-training R4 magnitude/scratch/teacher_rollout
#   R5 product/dp/demo_noise+ddim        R6 plain DP/scratch/teacher_rollout
# All six share maze_sparse(_ebm).yaml's training block: 100k steps, lr 1e-4 cosine, 500-step
# warmup, batch 256, save_freq/eval_freq 5000. Only the distillation knobs differ.
#
# Before submitting, confirm the teacher checkpoint exists on this filesystem -- every config but R3
# points at it, and make_policy fails late if it does not:
#   ls data/maze2d_dp/sparse/gc/train/dp_sparse_gc/pretrained_model
#
# Submit from the itps/ dir:
#   LLsub ./scripts/supercloud_scripts/maze_distill.sh [3,2,9] -g volta:2
#     3 nodes x 2 processes (one run per GPU) x 9 threads (1 main + 8 dataloader workers) = 6 slots
#     for 6 runs. These are pretrain-shaped runs with a SINGLE loader, hence 1 + 8 rather than
#     1 + 2*num_workers. NPPN=2 matches the node's two GPUs, so the GPUs pack evenly.
#
# Rough time per run on a V100: the four teacher_rollout arms spend ~5.5x batch_size extra no-grad
# teacher forward-rows per step (see DiffusionModel.teacher_rollout_points), putting them at ~2-3x
# the baseline step time, so ~3-5 h against ~1.5-2 h for R3 and R5.
#
# LLsub stripes ranks round-robin across nodes, not in blocks, so ranks 0,3 land together, 1,4
# together, 2,5 together. Nothing here depends on that, but it matters when reading step timings.

# Initialize the module command first
source /etc/profile

# Load the conda ML module ONLY -- no `module load cuda/...` (breaks cuDNN).
module load conda/Python-ML-2026a-pytorch

export PYTHONPATH="/home/gridsan/aforsey/diff-tuning:$PYTHONPATH"
export WANDB_MODE=offline

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIGS_DIR="/home/gridsan/aforsey/diff-tuning/itps/configs/policy/maze/distill"
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
