#!/bin/bash
# seeded_distilled_maze.sh
#
# Energy-landscape preference finetuning on the LARGE maze, re-run from the DISTILLED base policy
# (product energy, DP-initialized) instead of the original EBM base.
# 60 runs = seed(10) x timestep_support(all|ddim) x finetune_loss_weight(2|5|10).
#
# STEP 1 -- generate the run configs ONCE (from the itps/ dir). The path must match CONFIGS_DIR below:
#   python scripts/generate_configs.py \
#       --config configs/policy/maze_seeded/maze_energy_ebm_pref_only_distilled_seeded.yaml \
#       --out_dir configs/policy/maze_seeded/maze_energy_ebm_pref_only_distilled_seeded/
#
# STEP 2 -- inspect that dir (expect 60 run_*.yaml) and confirm the distilled checkpoint is there:
#   ls data/maze2d_dp/large/general/distill/2026.11.25/20.00.02_maze2d_large_distill_product_dp/checkpoints/last/pretrained_model
#
# STEP 3 -- submit. NTPP only needs 1 main + cfg.training.num_workers (4 in this config): the PREF
# loader runs at num_workers=0 since its windows are preloaded, so a pref-only finetune costs
# 1 + n threads, NOT the 1 + 2n a DEMO finetune costs.
#   LLsub ./scripts/supercloud_scripts/seeded_distilled_maze.sh [8,2,5] -g volta:2   # 16 procs, ~4 runs each
# Runs are only 500 steps, but eval_freq 50 means 10 evals each at n_samples 32 over 2 obs splits
# with methods ['ired','ddim'] -- the eval dominates the wall time, so size from the previous
# maze_energy_ebm_pref_only_seeded sweep rather than from the step count.

# Initialize the module command first
source /etc/profile

# Load the conda ML module ONLY -- no `module load cuda/...` (breaks cuDNN).
module load conda/Python-ML-2026a-pytorch

export PYTHONPATH="/home/gridsan/aforsey/diff-tuning:$PYTHONPATH"
export WANDB_MODE=offline

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIGS_DIR="/home/gridsan/aforsey/diff-tuning/itps/configs/policy/maze_seeded/maze_energy_ebm_pref_only_distilled_seeded"
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
