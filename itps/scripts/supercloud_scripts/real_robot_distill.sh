#!/bin/bash
# real_robot_distill.sh
#
# Supercloud LLsub submission for the real-robot energy-distillation pair: the 2 configs in
# configs/policy/real_robot/distill/ --
#   run_distill_product_measured, run_distill_product_cmd
# Both are the R1 recipe -- product energy, teacher-weight init, teacher_rollout states,
# teacher_eps targets -- at 200k steps, half the teachers' 400k pretrain. Everything else is
# inherited from the matching base EBM config, so student and teacher share the architecture.
#
# The teachers are the DEMO-FINETUNED DPs, not the 400k base policies. Note they are at DIFFERENT
# finetune depths -- measured at `last` (100k), cmd at step 020000 -- so a cmd-vs-measured read
# downstream confounds the label type with how long the teacher was finetuned.
#
# Before submitting:
#   * both teacher checkpoints are on THIS filesystem -- make_policy fails late if not:
#       ls data/real_robot/finetune/2026.11.26/20.29.06_real_robot_diffusion_demos120_measured/checkpoints/last/pretrained_model
#       ls data/real_robot/finetune/2026.11.26/20.29.06_real_robot_diffusion_demos120_cmd/checkpoints/020000/pretrained_model
#   * all four demo files are at data/real_robot/ (both configs now read TWO each).
#
# Submit from the itps/ dir:
#   LLsub ./scripts/supercloud_scripts/real_robot_distill.sh [1,2,9] -g volta:2
#     1 node x 2 processes (one run per GPU) x 9 threads (1 main + 8 dataloader workers).
#     Pretrain-shaped runs with a SINGLE loader, hence 1 + 8, not the finetune runs' 1 + 2*8.
#
# Memory: each run holds all images in RAM, and these runs load BOTH demo sets -- 111282 rows x 2
# cameras x 3x120x160 uint8 = ~12.8 GB, roughly 2x that while loading, so ~35 GB peak with overhead
# against a base run's ~28 GB. At ~9.6 GB per core, NTPP 9 gives ~87 GB per process, so it fits.
# On the GPU each process holds BOTH policies -- the frozen teacher (no-grad) and the student
# (double backward) -- so headroom is tighter than a base run.
#
# Rough time per run: teacher_rollout runs the teacher's whole 10-step DDIM grid at full batch every
# step (teacher_rollout_points keeps the batch shape constant on purpose -- see its docstring), plus
# one no-grad pass through the teacher's image encoders. Against the base EBM's ~11 h for 400k,
# expect ~8-12 h for 200k.
#
# OBSERVATION COVERAGE: teacher_rollout generates x_t from noise and takes only the CONDITIONING
# from each batch, so `dataset_root` is exactly where the student is supervised. Both configs
# therefore give it as a LIST of both demo sets -- base plus shifted -- since the teachers were
# finetuned on both and a base-only student would have no supervision at shifted-cup observations.
# _load_real_robot_hdf5 concatenates the list, renumbering episodes and rows to run continuously
# (416 episodes, 111282 rows, the shifted set 28.6% of them: proportional, not the finetune's
# 50/50, which is deliberate -- distil evenly over the teacher's states rather than reweight them).
# A DICT dataset_root would instead flip train.py's `finetune` switch and silently disable
# distillation, so the list form is load-bearing, not cosmetic.
#
# No in-training eval, and eval_freq MUST stay 0: envs/__init__.py's get_env registers gmm, maze2d
# and robosuite only, so a positive eval_freq raises NotImplementedError at the first eval step.
# `loss_distill` is therefore the only number -- eval_field_match is wired into MazeEnv.evaluate
# only. `loss_mse` logs its -1 sentinel because the anchor is off (distill_anchor_weight is 0).

# Initialize the module command first
source /etc/profile

# Load the conda ML module ONLY -- no `module load cuda/...` (breaks cuDNN, see real_robot_overfit.sh).
module load conda/Python-ML-2026a-pytorch

export PYTHONPATH="/home/gridsan/aforsey/diff-tuning:$PYTHONPATH"
export WANDB_MODE=offline

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIGS_DIR="/home/gridsan/aforsey/diff-tuning/itps/configs/policy/real_robot/distill"
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
