#!/bin/bash
# pt_distilled_maze.sh
#
# Parameter tuning (sweep_4) for energy-landscape preference finetuning on the LARGE maze, from the
# DISTILLED base policy.
# 72 runs = seed(0,1,2) x offline_steps(100,250,500) x timestep_support(all|ddim)
#                       x finetune_loss_weight(0.5, 0.25, 0.1, 0.05).
#
# WHY a fourth sweep: the distilled passes at weights 2/5/10 and then 1/0.5 all drove train/loss_mse
# up while loss_energy_finetune fell -- the contrastive term overwhelms the denoising anchor. The old
# sweeps bracket this hole without covering it: sweep_2 reached weights 0.05/0.1 but only at >= 500
# steps, and sweep_3 went below 500 steps but only at weights 1/2. This pass moves both at once.
# lr 1e-5, batch_size 32 and train_only_FiLM false stay fixed at sweep_2's best cell.
#
# STEP 1 -- generate the run configs ONCE (from the itps/ dir). Already expanded locally, so this is
# only needed if the sweep yaml changes. The path must match CONFIGS_DIR below:
#   python scripts/generate_configs.py \
#       --config configs/policy/maze_param_tuning/energy_ebm_pref_only/sweep_4.yaml \
#       --out_dir configs/policy/maze_param_tuning/energy_ebm_pref_only/sweep_4/
#
# STEP 2 -- inspect that dir (expect 72 run_*.yaml) and confirm the distilled checkpoint is there:
#   ls data/maze2d_dp/large/general/distill/2026.11.25/20.00.02_maze2d_large_distill_product_dp/checkpoints/last/pretrained_model
#
# STEP 3 -- submit. NTPP only needs 1 main + cfg.training.num_workers (4 here): the PREF loader runs
# at num_workers=0 since its windows are preloaded, so a pref-only finetune costs 1 + n threads, NOT
# the 1 + 2n a DEMO finetune costs.
#   LLsub ./scripts/supercloud_scripts/pt_distilled_maze.sh [2,2,5] -g volta:2   # 4 procs, 18 runs each
# NPPN=2 = one run per GPU. The EBM needs create_graph=True for the energy gradient, so its memory
# footprint is higher than a plain DP's -- check nvidia-smi and go to [2,4,5] if there is headroom.
# Runs are 100-500 steps, but eval_freq 50 means 2-10 evals each at n_samples 32 over 2 obs splits
# with methods ['ired','ddim'], so EVAL dominates the wall time and the short arms are not 5x faster
# than the long ones -- roughly 2/5/10 evals, i.e. about half the previous sweep's cost per run on
# average, for 72 runs instead of 40.
#
# save_freq is ${training.offline_steps}, so each arm writes exactly one checkpoint, at its own end.

# Initialize the module command first
source /etc/profile

# Load the conda ML module ONLY -- no `module load cuda/...` (breaks cuDNN).
module load conda/Python-ML-2026a-pytorch

export PYTHONPATH="/home/gridsan/aforsey/diff-tuning:$PYTHONPATH"
export WANDB_MODE=offline

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIGS_DIR="/home/gridsan/aforsey/diff-tuning/itps/configs/policy/maze_param_tuning/energy_ebm_pref_only/sweep_4"
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
