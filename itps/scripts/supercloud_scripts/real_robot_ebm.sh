#!/bin/bash
# real_robot_ebm.sh
#
# Supercloud LLsub resubmission of the two real-robot EBM runs only:
#   rank 0: overfit/run_ebm_visual_overfit10 (30k steps), rank 1: pretrain/run_ebm_visual (400k steps).
# Both crashed at step 1 in real_robot_overfit.sh / real_robot_pretrain.sh on the comparison-conditioning
# check, before it was skipped for random crops (EBMDiffusionModel._random_crop_in_use).
#
# Submit from the itps/ dir:
#   LLsub ./scripts/supercloud_scripts/real_robot_ebm.sh [1,2,9] -g volta:2
#     1 node x 2 processes (one run per GPU) x 9 threads (1 main + 8 dataloader workers).

# Initialize the module command first
source /etc/profile

# Load the conda ML module ONLY -- no `module load cuda/...` (breaks cuDNN, see real_robot_overfit.sh).
module load conda/Python-ML-2026a-pytorch

export PYTHONPATH="/home/gridsan/aforsey/diff-tuning:$PYTHONPATH"
export WANDB_MODE=offline

case "$LLSUB_RANK" in
    0) CONFIG="real_robot/overfit/run_ebm_visual_overfit10" ;;
    1) CONFIG="real_robot/pretrain/run_ebm_visual" ;;
    *) echo "rank $LLSUB_RANK: nothing to run"; exit 0 ;;
esac

echo "======================================"
echo "My task rank: $LLSUB_RANK  config: $CONFIG"
echo "======================================"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Same invocation as run_job.py.
python scripts/train.py env=real_robot policy=$CONFIG.yaml config_name=$(basename $CONFIG)
