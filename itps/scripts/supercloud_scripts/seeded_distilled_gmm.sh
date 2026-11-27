#!/bin/bash
# seeded_distilled_gmm.sh
#
# Energy-based preference finetuning on the GMM, re-run from the DISTILLED base policy
# (product energy, DP-initialized) instead of the original EBM base.
# 60 runs = seed(10) x timestep_support(all|ddim) x finetune_loss_weight(5|10|20),
# with the LANGEVIN IRED sampler (eval.opt_params langevin: True) -- the scheme that was live
# 2026-09-30..2026-10-23 and produced the original gmm seeded numbers. The first distilled pass
# (dir .../gmm_ebm_pref_only_distilled_seeded) used today's no-noise sampler; point CONFIGS_DIR
# back at it to re-run that.
#
# STEP 1 -- generate the run configs ONCE (from the itps/ dir). The path must match CONFIGS_DIR below:
#   python scripts/generate_configs.py \
#       --config configs/policy/gmm_seeded/gmm_ebm_pref_only_distilled_langevin_seeded.yaml \
#       --out_dir configs/policy/gmm_seeded/gmm_ebm_pref_only_distilled_langevin_seeded/
#
# STEP 2 -- inspect that dir (expect 60 run_*.yaml) and confirm the distilled checkpoint is there:
#   ls data/gmm/general/distill/2026.11.25/gmm_2026.11.25.20.00.02_gmm_distill_product_dp/checkpoints/last/pretrained_model
#
# STEP 3 -- submit. NTPP only needs 1 main + cfg.training.num_workers (3, the default): the PREF
# loader runs at num_workers=0 since its windows are preloaded, so a pref-only finetune costs
# 1 + n threads, NOT the 1 + 2n a DEMO finetune costs.
#   LLsub ./scripts/supercloud_scripts/seeded_distilled_gmm.sh [1,4,5] -g volta:2   # 4 procs, 15 runs each
# NPPN=4 packs two runs per GPU. A horizon-1, 2-dim model nowhere near saturates a V100, so this
# is close to free; check nvidia-smi and drop to [1,2,5] if util or memory is above half.
# Runs are ~1000 steps on a horizon-1 model, so these are minutes each.

# Initialize the module command first
source /etc/profile

# Load the conda ML module ONLY -- no `module load cuda/...` (breaks cuDNN).
module load conda/Python-ML-2026a-pytorch

export PYTHONPATH="/home/gridsan/aforsey/diff-tuning:$PYTHONPATH"
export WANDB_MODE=offline

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIGS_DIR="/home/gridsan/aforsey/diff-tuning/itps/configs/policy/gmm_seeded/gmm_ebm_pref_only_distilled_langevin_seeded"
SCRIPT="scripts/train.py"
ENV_NAME="gmm"
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
