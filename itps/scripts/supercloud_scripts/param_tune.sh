#!/bin/bash
# submit.sh
#
# Supercloud LLsub triples submission script.
# No SBATCH flags -- LLsub handles resource allocation via the triple [NODES,NPPN,NTPP].
#
# BEFORE submitting, generate your run configs once:
#   python generate_configs.py --config configs/joint_config.yaml --out_dir configs/runs/
#
# Then inspect configs/runs/ to verify the generated configs look correct.
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
module load cuda/13.1

export PYTHONPATH="/home/gridsan/aforsey/diff-tuning:$PYTHONPATH"
export WANDB_MODE=offline
#export WANDB_DIR=/home/gridsan/aforsey/wandb_logs/gmm/conditional/fine_tuning/param_tuning

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIGS_DIR="/home/gridsan/aforsey/diff-tuning/itps/configs/policy/gmm_param_tuning/DPO_finetuning_tuning/sweep_1"    # Directory containing generated run_*.yaml files
SCRIPT="scripts/train.py"             # The python training script
ENV_NAME="gmm"                # env={ENV_NAME} passed to the script
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
