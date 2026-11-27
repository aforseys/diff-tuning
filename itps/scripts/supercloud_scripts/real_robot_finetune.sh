#!/bin/bash
# real_robot_finetune.sh
#
# Supercloud LLsub submission for the 120-perturbed-demo fine-tune of the two pretrained
# cup_multi_shelf DPs: the 2 configs in configs/policy/real_robot/finetune/ --
#   rank 0: run_dp_visual_demos_cmd       (commanded joint labels, from 2026.11.23/04.24.25_...)
#   rank 1: run_dp_visual_demos_measured  (measured joint labels,  from 2026.11.19/11.40.41_...)
# Both are 100k steps, lr 1e-5, two loaders at batch 64 mixed 50/50, full finetuning (no FiLM freeze).
#
# One node is enough: each run is one GPU and ~38 GB of RAM, so the two fit side by side.
#
# Before submitting:
#   * all four hdf5 files are on Supercloud under data/real_robot/ -- the two bases
#     cup_multi_shelf{,_cmd}.hdf5 and the two 120-demo files cup_shifted_middle{,_cmd}.hdf5:
#       grep -n 'base:\|demo:' configs/policy/real_robot/finetune/*.yaml
#   * the cmd run's demo file has been through build_cmd_labels.py -- `action_labels: commanded`
#     is a GLOBAL key, so it is applied to the demo dataset too and nothing warns if that file
#     still holds measured labels;
#   * the new demos' per-dim action min/max are inside the CHECKPOINT's
#     normalize_targets.buffer_action.{min,max}. The normalizer is loaded from the checkpoint, not
#     refit (train.py passes dataset_stats=None when finetuning), so anything outside the original
#     range maps beyond +-1 and clip_sample (range 1.0) clamps it away.
#
# Submit from the itps/ dir:
#   LLsub ./scripts/supercloud_scripts/real_robot_finetune.sh [1,2,17] -g volta:2
#     1 node x 2 processes (one run per GPU) x 17 threads = 1 main + 2 loaders x 8 workers.
#     NTPP is 17, not pretraining's 9: a demo-finetune run holds TWO loaders at once, so the
#     thread budget is 1 + 2*num_workers. 2 x 17 = 34 of the node's 40 cores.
# Memory: each run holds BOTH datasets' images in RAM (297 + 119 demos, ~13 GB, ~38 GB peak with
# loading overhead). NTPP 17 GRANTS ~163 GB per process -- far more than needed, but the grant is
# a ceiling, not a cost, and 2 x 163 still fits the node.
# Rough time per run (V100 + AMP, +-2x): ~2-3 h. Pretraining was 400k steps at batch 128 in ~8 h,
# and this is 100k steps at the same 128 samples/step plus the second loader's overhead.

# Initialize the module command first
source /etc/profile

# Load the conda ML module ONLY -- no `module load cuda/...` (breaks cuDNN, see real_robot_overfit.sh).
module load conda/Python-ML-2026a-pytorch

export PYTHONPATH="/home/gridsan/aforsey/diff-tuning:$PYTHONPATH"
export WANDB_MODE=offline

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIGS_DIR="/home/gridsan/aforsey/diff-tuning/itps/configs/policy/real_robot/finetune"
SCRIPT="scripts/train.py"
ENV_NAME="real_robot"
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
