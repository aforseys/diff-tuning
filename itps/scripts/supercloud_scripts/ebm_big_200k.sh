#!/bin/bash
# ebm_big_200k.sh
#
# Two sparse-maze EBM PRETRAIN runs at down_dims [256,512,1024] and offline_steps 200000,
# differing only in the negative construction (hard_negative + neg_eps_weight). One process
# per GPU, one run each.
#
#   LLsub ./scripts/supercloud_scripts/ebm_big_200k.sh [1,2,9]
#
# The triple: 1 node, NPPN=2 (the node's two GPUs, one run per GPU), NTPP=9.
# NTPP must cover 1 main process + training.num_workers dataloader workers. These are
# pretrain runs with a SINGLE loader, so 1 + 8 = 9. (A finetune run holds two loaders and
# needs 1 + 2*num_workers.) NPPN=2 x NTPP=9 = 18 threads, inside a volta node's 40.
#
# Nothing here pins a device: LLsub assigns GPUs from the triple and the configs leave
# `device: cuda`. Do not add `device: cuda:N` to a config.
#
# num_workers is overridden here rather than in the YAMLs so the two configs stay
# byte-comparable with their 100k parents -- it does not affect results, only throughput.

source /etc/profile
module load conda/Python-ML-2026a-pytorch

export PYTHONPATH="/home/gridsan/aforsey/diff-tuning:$PYTHONPATH"
export WANDB_MODE=offline

cd /home/gridsan/aforsey/diff-tuning/itps || exit 1

POLICIES=(
  maze/pretrain/maze_sparse_ebm_big_200k
  maze/pretrain/maze_sparse_ebm_hardneg_big_200k
)

POLICY=${POLICIES[$LLSUB_RANK]}
if [ -z "$POLICY" ]; then
  # More processes than runs: idle ranks exit cleanly instead of launching a bare default.
  echo "[rank $LLSUB_RANK/$LLSUB_SIZE] no run assigned (${#POLICIES[@]} configs); exiting."
  exit 0
fi

echo "======================================"
echo "My task rank:    $LLSUB_RANK"
echo "Number of tasks: $LLSUB_SIZE"
echo "Policy:          $POLICY"
echo "======================================"

python scripts/train.py \
    env=maze2d \
    policy="$POLICY" \
    training.num_workers=8
