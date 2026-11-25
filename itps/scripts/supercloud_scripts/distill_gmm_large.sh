#!/bin/bash
# distill_gmm_large.sh
#
# The R1 distillation recipe applied to the other two domains: the base GMM and the base LARGE maze.
# Two runs, one per GPU. run_job.py is not used because the two need different `env=` values.
#
#   rank 0: gmm/distill/run_gmm_distill_product_dp                  env=gmm      2500 steps
#   rank 1: maze/distill_large/run_maze_large_distill_product_dp    env=maze2d  50000 steps
#
# Each inherits its own base EBM config wholesale (gmm_ebm.yaml / maze_large_ebm.yaml) and changes
# only: teacher_policy_path, init_from_teacher, the distill knobs, energy_mode, half the steps, and
# supervise_energy_landscape off. No EMA.
#
# Before submitting, FILL IN teacher_policy_path in both configs and check both resolve:
#   grep -r teacher_policy_path configs/policy/gmm/distill configs/policy/maze/distill_large
#   ls <that path>            # must contain config.yaml + model.safetensors
#
# Submit from the itps/ dir:
#   LLsub ./scripts/supercloud_scripts/distill_gmm_large.sh [1,2,9] -g volta:2
#     1 node x 2 processes (one run per GPU) x 9 threads (1 main + up to 8 dataloader workers).
#
# Expect ~5 h for the large maze (0.35 s/step measured on the sparse maze, same architecture) and
# minutes for the GMM (horizon 1).
#
# NOTE neither domain shows the sparse maze's collision gap, so success here is "matches the
# existing EBM", not "improves on it". GMM reports KL and sample log-likelihood rather than the
# field match (eval_field_match is wired into MazeEnv only).

# Initialize the module command first
source /etc/profile

# Load the conda ML module ONLY -- no `module load cuda/...` (breaks cuDNN).
module load conda/Python-ML-2026a-pytorch

export PYTHONPATH="/home/gridsan/aforsey/diff-tuning:$PYTHONPATH"
export WANDB_MODE=offline
cd /home/gridsan/aforsey/diff-tuning/itps || exit 1

POLICIES=(
  "gmm/distill/run_gmm_distill_product_dp gmm"
  "maze/distill_large/run_maze_large_distill_product_dp maze2d"
)
ASSIGNED=${POLICIES[$LLSUB_RANK]}
if [ -z "$ASSIGNED" ]; then
  echo "[rank $LLSUB_RANK/$LLSUB_SIZE] no run assigned (${#POLICIES[@]} configs); exiting."
  exit 0
fi
set -- $ASSIGNED
POLICY=$1
ENV_NAME=$2

echo "======================================"
echo "My task rank:    $LLSUB_RANK"
echo "Number of tasks: $LLSUB_SIZE"
echo "Policy:          $POLICY"
echo "Env:             $ENV_NAME"
echo "======================================"
nvidia-smi --query-gpu=name,memory.total --format=csv

python scripts/train.py \
    env="$ENV_NAME" \
    policy="$POLICY" \
    config_name="$(basename "$POLICY")"
