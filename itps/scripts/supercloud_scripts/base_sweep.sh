#!/bin/bash
# base_sweep.sh
#
# Supercloud LLsub submission for the base-method n_queries sweep.
# Sweeps n_queries=[50,100,200,500] across 3 experiments
# (maze2d_large, sparse_collision, sparse_down) = 12 runs total, striped
# across all allocated processes so they run in parallel on different nodes.
#
# STEP 1 — generate the run configs ONCE (from the itps/ dir):
#   python scripts/generate_configs.py \
#       --config configs/policy/CoRL/maze2d_large/maze2d_dp_tune_base_IRED_q_range_sweep.yaml \
#       --out_dir configs/policy/CoRL/runs/base_large/
#   python scripts/generate_configs.py \
#       --config configs/policy/CoRL/maze2d_sparse_collision/maze2d_gc_dp_sparse_tune_base_collision_IRED_q_range_sweep.yaml \
#       --out_dir configs/policy/CoRL/runs/base_collision/
#   python scripts/generate_configs.py \
#       --config configs/policy/CoRL/maze2d_sparse_down/maze2d_gc_dp_sparse_tune_base_down_IRED_q_range_sweep.yaml \
#       --out_dir configs/policy/CoRL/runs/base_down/
#
# STEP 2 — inspect configs/policy/CoRL/runs/ (expect 12 run_*.yaml across 3 subdirs).
#
# STEP 3 — submit. LLSUB_SIZE = NODES*NPPN should be >= 12 to run all at once.
# Supercloud assigns processes to the node's GPUs from the triple itself -- nothing
# here pins devices, and configs must NOT hardcode `device: cuda:N`.
# GPU nodes have 2 GPUs, so NPPN=2 puts one run per GPU and NPPN=4 puts two per GPU
# (per Supercloud's docs: only worth doubling up if GPU util/memory are both <50%).
# NTPP must cover 1 main process + cfg.training.num_workers dataloader workers per
# run (4 by default), so NTPP=1 starves the loader -- use >= 5.
#   LLsub ./base_sweep.sh [6,2,5]     # 6 nodes x 2 procs x 5 threads = 12 runs, 1/GPU
#   LLsub ./base_sweep.sh [3,4,5]     # 3 nodes x 4 procs x 5 threads = 12 runs, 2/GPU
#
# LLSUB_RANK: this process's index (0 .. LLSUB_SIZE-1)
# LLSUB_SIZE: total number of processes (NODES * NPPN)
# run_job.py assigns each process a strided slice of the 12 configs.

# Initialize the module command first
source /etc/profile

# Load Anaconda + CUDA modules
module load conda/Python-ML-2026a-pytorch
module load cuda/13.1

export PYTHONPATH="/home/gridsan/aforsey/diff-tuning:$PYTHONPATH"
export WANDB_MODE=offline

# ── Configuration ─────────────────────────────────────────────────────────────
# Parent of the 3 per-experiment subdirs; run_job.py globs it recursively.
CONFIGS_DIR="/home/gridsan/aforsey/diff-tuning/itps/configs/policy/CoRL/runs_query_range"
SCRIPT="scripts/train.py"        # training script
ENV_NAME="maze2d"                # env={ENV_NAME} passed to the script
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
