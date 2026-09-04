#!/bin/bash
# submit.sh
#
# Supercloud LLsub triples submission script.
# No SBATCH flags -- LLsub handles resource allocation via the triple [NODES,NPPN,NTPP].
#
# Currently pointed at the DPO param sweep on the NEW-PREF comparison set (pairs
# selected by finetune_goal_dist_relative -- the unclipped, batch-normalized rule --
# rather than the clipped finetune_goal_dist). Same grid as param_tuning_DPO/sweep_2,
# which is its A/B partner on the clipped set; only dataset_root.pos/neg differ, and
# the `new_pref` tag in the hydra dir and job name keeps the cohorts apart.
# 144 runs = offline_steps(4) x lr(3) x batch_size(2) x train_only_FiLM(2) x mu(3).
#
# BEFORE submitting, generate your run configs once (from the itps/ dir). The path
# must match CONFIGS_DIR further down, which is what run_job.py actually globs:
#   python scripts/data_generation/generate_configs.py \
#       --config configs/policy/ICRA/maze/large_maze/param_tuning_DPO_new_pref/DPO_param_tune.yaml \
#       --out_dir configs/policy/ICRA/maze/large_maze/param_tuning_DPO_new_pref/runs/
#
# Then inspect that dir (expect 144 run_*.yaml) to verify the generated configs look correct.
#
# Submit with:
#   LLsub ./submit.sh [NODES,NPPN,NTPP]
#
# Example (2 nodes, 4 processes per node, 1 thread per process = 8 total processes):
#   LLsub ./submit.sh [2,4,1]
#
# LLSUB_RANK: this process's index (0 to NODES*NPPN - 1)
# LLSUB_SIZE: total number of processes (NODES * NPPN)
#
# Each process is assigned a roughly equal slice of configs/runs/ and
# runs them sequentially.

# Initialize the module command first source
source /etc/profile

# Load Anaconda Module
module load conda/Python-ML-2026a-pytorch

export PYTHONPATH="/home/gridsan/aforsey/diff-tuning:$PYTHONPATH"
export WANDB_MODE=offline
#export WANDB_DIR=/home/gridsan/aforsey/wandb_logs/gmm/conditional/fine_tuning/param_tuning

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIGS_DIR="/home/gridsan/aforsey/diff-tuning/itps/configs/policy/ICRA/maze/large_maze/param_tuning_DPO_new_pref/runs_sweep_3"    # Directory containing generated run_*.yaml files
SCRIPT="scripts/train.py"             # The python training script
ENV_NAME="maze2d"                # env={ENV_NAME} passed to the script
# ──────────────────────────────────────────────────────────────────────────────

echo "======================================"
echo "My task rank:    $LLSUB_RANK"
echo "Number of tasks: $LLSUB_SIZE"
echo "Configs dir:     $CONFIGS_DIR"
echo "Script:          $SCRIPT"
echo "Env:             $ENV_NAME"
echo "======================================"

python scripts/supercloud_scripts/run_job.py $LLSUB_RANK $LLSUB_SIZE \
    --configs_dir $CONFIGS_DIR \
    --script $SCRIPT \
    --env $ENV_NAME
