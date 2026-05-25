# Initialize the module command first source
source /etc/profile

# Load Anaconda Module
module load conda/Python-ML-2026a-pytorch
module load cuda/13.1

export PYTHONPATH="/home/gridsan/aforsey/diff-tuning:$PYTHONPATH"
export WANDB_MODE=offline

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIGS_DIR="/home/gridsan/aforsey/diff-tuning/itps/configs/policy/robosuite/robosuite_strategy_gc_dp_visual"    # Directory containing generated run_*.yaml files
SCRIPT="scripts/train.py"             # The python training script
ENV_NAME="robosuite"                # env={ENV_NAME} passed to the script
# ──────────────────────────────────────────────────────────────────────────────

python scripts/train.py \
    --configs_dir $CONFIGS_DIR \
    --env $ENV_NAME