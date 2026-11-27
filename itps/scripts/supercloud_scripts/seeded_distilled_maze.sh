#!/bin/bash
# seeded_distilled_maze.sh
#
# Energy-landscape preference finetuning on the LARGE maze, re-run from the DISTILLED base policy
# (product energy, DP-initialized) instead of the original EBM base.
# 40 runs = seed(10) x timestep_support(all|ddim) x finetune_loss_weight(1|0.5).
#
# This is the SECOND pass. The first (weights 2/5/10, dir .../maze_energy_ebm_pref_only_distilled_seeded)
# showed train/loss_mse climbing from ~0.05 to 0.25-0.65 on every run: all three weights were too
# large and the denoising anchor collapsed. Point CONFIGS_DIR back at that dir to re-run the first pass.
#
# STEP 1 -- generate the run configs ONCE (from the itps/ dir). The path must match CONFIGS_DIR below:
#   python scripts/generate_configs.py \
#       --config configs/policy/maze_seeded/maze_energy_ebm_pref_only_distilled_lowweight_seeded.yaml \
#       --out_dir configs/policy/maze_seeded/maze_energy_ebm_pref_only_distilled_lowweight_seeded/
#
# STEP 2 -- inspect that dir (expect 40 run_*.yaml) and confirm the distilled checkpoint is there:
#   ls data/maze2d_dp/large/general/distill/2026.11.25/20.00.02_maze2d_large_distill_product_dp/checkpoints/last/pretrained_model
#
# STEP 3 -- submit. NTPP only needs 1 main + cfg.training.num_workers (4 in this config): the PREF
# loader runs at num_workers=0 since its windows are preloaded, so a pref-only finetune costs
# 1 + n threads, NOT the 1 + 2n a DEMO finetune costs.
#   LLsub ./scripts/supercloud_scripts/seeded_distilled_maze.sh [2,2,5] -g volta:2   # 4 procs, 10 runs each
# NPPN=2 = one run per GPU. The EBM needs create_graph=True for the energy gradient, so its memory
# footprint is higher than a plain DP's -- check nvidia-smi and go to [2,4,5] if there is headroom.
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
CONFIGS_DIR="/home/gridsan/aforsey/diff-tuning/itps/configs/policy/maze_seeded/maze_energy_ebm_pref_only_distilled_lowweight_seeded"
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
