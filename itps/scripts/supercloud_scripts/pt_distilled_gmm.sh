#!/bin/bash
# pt_distilled_gmm.sh
#
# Parameter tuning (sweep_2) for energy-based preference finetuning on the GMM, from the DISTILLED
# base policy, with the LANGEVIN IRED sampler.
# 72 runs = seed(0,1,2) x offline_steps(250,500,1000) x timestep_support(all|ddim)
#                       x finetune_loss_weight(5, 1, 0.5, 0.1).
#
# WHY: sweep_1's carried-forward cell was lr 1e-3 / bs 64 / FiLM false / 1000 steps / weight 5, and
# every distilled GMM run since has sat at weights 5/10/20 -- the top of the range sweep_1 explored.
# This pass walks both the weight and the step count down from there, the same correction the maze
# needed. lr, batch_size and train_only_FiLM stay fixed at sweep_1's cell. Weight 5 is kept as the
# anchor so the new grid ties to known behaviour.
#
# SAMPLER: eval.opt_params keeps langevin: True -- the scheme live 2026-09-30..2026-10-23 that
# produced the ORIGINAL gmm seeded numbers, restored in full (single-strength gradient step,
# + sqrt(2*beta_t)*eps, and NEITHER clamp). The maze sweep omits the key and keeps today's scheme.
#
# STEP 1 -- generate the run configs ONCE (from the itps/ dir). Already expanded locally, so this is
# only needed if the sweep yaml changes. The path must match CONFIGS_DIR below:
#   python scripts/generate_configs.py \
#       --config configs/policy/gmm_param_tuning/energy_ebm_pref_only/sweep_2.yaml \
#       --out_dir configs/policy/gmm_param_tuning/energy_ebm_pref_only/sweep_2/
#
# STEP 2 -- inspect that dir (expect 72 run_*.yaml) and confirm the distilled checkpoint is there:
#   ls data/gmm/general/distill/2026.11.25/gmm_2026.11.25.20.00.02_gmm_distill_product_dp/checkpoints/last/pretrained_model
#
# STEP 3 -- submit. This config sets no training.num_workers, so it takes the default 3 -> NTPP 4.
#   LLsub ./scripts/supercloud_scripts/pt_distilled_gmm.sh [1,4,5] -g volta:2   # 4 procs, 18 runs each
# NPPN=4 packs two runs per GPU: the model is horizon 1 and will not saturate a V100. If nvidia-smi
# shows either GPU above half its memory or near 100% util, drop to [1,2,5].
#
# eval_freq is 50 here, not sweep_1's 100 -- it must divide every swept offline_steps and 250 is not
# a multiple of 100. That doubles the number of evals per run against the previous GMM passes.
#
# save_freq is ${training.offline_steps}, so each arm writes exactly one checkpoint, at its own end.

# Initialize the module command first
source /etc/profile

# Load the conda ML module ONLY -- no `module load cuda/...` (breaks cuDNN).
module load conda/Python-ML-2026a-pytorch

export PYTHONPATH="/home/gridsan/aforsey/diff-tuning:$PYTHONPATH"
export WANDB_MODE=offline

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIGS_DIR="/home/gridsan/aforsey/diff-tuning/itps/configs/policy/gmm_param_tuning/energy_ebm_pref_only/sweep_2"
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
