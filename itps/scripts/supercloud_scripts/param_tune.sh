#!/bin/bash
# submit.sh
#
# Supercloud LLsub triples submission script.
# No SBATCH flags -- LLsub handles resource allocation via the triple [NODES,NPPN,NTPP].
#
# Currently pointed at the DPO_forward param sweep on the GMM EBM, with the preference
# pairs pooled as the base dataset (dataset_root has only pos/neg, and concat_pairs_as_base
# builds the denoising anchor from them -- no separate base samples).
# 320 runs = offline_steps(4) x lr(4) x batch_size(2) x train_only_FiLM(2) x mu(5),
# with rho=500 and b=0 held fixed this round.
#
# itps/data/ is gitignored, so copy the data this sweep reads before submitting:
#   data/gmm_obs/gmm_cluster_line_diagonal_any_100_1_unconditional_{pos,neg}.npy
#   data/gmm_obs/gmm_cluster_line_unconditional_200_2_20260930_094128.npy  (win-rate test set)
#   data/gmm/general/train/2026.09.18/gmm_2026.09.18.13.50.05_gmm_ebm_diffusion/checkpoints/last/pretrained_model/
#
# BEFORE submitting, generate your run configs once (from the itps/ dir). The path
# must match CONFIGS_DIR further down, which is what run_job.py actually globs:
#   python scripts/generate_configs.py \
#       --config configs/policy/gmm_param_tuning/DPO_forward_ebm_pref_only/sweep_1.yaml \
#       --out_dir configs/policy/gmm_param_tuning/DPO_forward_ebm_pref_only/sweep_1/
#
# Then inspect that dir (expect 320 run_*.yaml) to verify the generated configs look correct.
#
# Submit with:
#   LLsub ./param_tune.sh [NODES,NPPN,NTPP]
#
# GPU nodes have 2 GPUs, so keep NPPN at 2 (one run per GPU) or 4 (two per GPU); an odd
# NPPN packs the GPUs unevenly. NTPP must cover 1 main process + the dataloader workers,
# and a FINETUNING run holds TWO loaders at once (pooled base + pref), each with
# cfg.training.num_workers workers -- so 1 + 2*num_workers = 7 at the default
# num_workers: 3 (configs/default.yaml). NTPP=1 starves the loaders.
#
#   LLsub ./param_tune.sh [5,4,8]    # 20 procs, 2/GPU, 16 of the 320 runs each
#   LLsub ./param_tune.sh [10,2,7]   # 20 procs, 1/GPU, same 16 runs each
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
CONFIGS_DIR="/home/gridsan/aforsey/diff-tuning/itps/configs/policy/gmm_param_tuning/DPO_original_ebm_pref_only/sweep_1"    # Directory containing generated run_*.yaml files
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
