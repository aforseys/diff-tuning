#!/bin/bash
# base_sweep.sh
#
# Supercloud LLsub submission for the base-method n_queries sweep on the
# large maze, using the NEW-PREF comparison set (pairs selected by
# finetune_goal_dist_relative -- the unclipped, batch-normalized rule -- rather
# than the clipped finetune_goal_dist). Sweeps n_queries=[50,100,200,500,750,1000]
# = 6 runs total, striped across all allocated processes.
#
# This is the A/B partner of the existing n_query_runs_base/ cohort, which is the
# same sweep against the clipped comparison set. Everything but dataset_root.pos/neg
# is identical between the two, so results are directly comparable; the runs are
# kept apart by the `new_pref` tag in the hydra dir and job name.
#
# STEP 1 — generate the run configs ONCE (from the itps/ dir). The path below must match
# CONFIGS_DIR further down, which is what run_job.py actually globs:
#   python scripts/data_generation/generate_configs.py \
#       --config configs/policy/ICRA/maze/large_maze/large_maze_n_query_sweep_base_new_pref.yaml \
#       --out_dir configs/policy/ICRA/maze/large_maze/n_query_runs_base_new_pref/
#
# STEP 2 — inspect configs/policy/ICRA/maze/large_maze/n_query_runs_base_new_pref/
# (expect 6 run_*.yaml).
#
# STEP 3 — submit. LLSUB_SIZE = NODES*NPPN should be >= 6 to run all at once.
# Supercloud assigns processes to the node's GPUs from the triple itself -- nothing
# here pins devices, and configs must NOT hardcode `device: cuda:N`.
# GPU nodes have 2 GPUs, so NPPN=2 puts one run per GPU and NPPN=4 puts two per GPU
# (per Supercloud's docs: only worth doubling up if GPU util/memory are both <50%).
# NTPP must cover 1 main process + the dataloader workers. A FINETUNING run holds TWO
# loaders at once (base + pref/demo), each with cfg.training.num_workers workers, so
# that is 1 + 2*num_workers threads -- 7 at the default num_workers: 3 (configs/
# default.yaml). NTPP=1 starves the loader; use >= 7 for finetuning sweeps like this
# one (>= 4 would only cover a single-loader base-training run).
#   LLsub ./base_sweep.sh [3,2,7]     # 3 nodes x 2 procs x 7 threads = 6 runs, 1/GPU
#   LLsub ./base_sweep.sh [2,4,7]     # 2 nodes x 4 procs x 7 threads = 8 slots, 2/GPU
#
# LLSUB_RANK: this process's index (0 .. LLSUB_SIZE-1)
# LLSUB_SIZE: total number of processes (NODES * NPPN)
# run_job.py assigns each process a strided slice of the 6 configs.

# Initialize the module command first
source /etc/profile

# Load Anaconda + CUDA modules
module load conda/Python-ML-2026a-pytorch

export PYTHONPATH="/home/gridsan/aforsey/diff-tuning:$PYTHONPATH"
export WANDB_MODE=offline

# ── Configuration ─────────────────────────────────────────────────────────────
# run_job.py globs this recursively for run_*.yaml.
CONFIGS_DIR="/home/gridsan/aforsey/diff-tuning/itps/configs/policy/ICRA/maze/large_maze/n_query_runs_base_new_pref"
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
