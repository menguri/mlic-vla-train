#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/mlic/mingukang/lerobot}"
cd "$REPO_ROOT"
source .venv/bin/activate
source ./env_libero.sh

# Full fine-tuning SmolVLA base on lerobot/libero with W&B logging.
#
# W&B API key is read from:
#   ./wandb/wandb_id
#
# GPU examples:
#   ./sh_scripts/train_smolvla_libero_full_wandb.sh
#   NUM_GPUS=2 ./sh_scripts/train_smolvla_libero_full_wandb.sh
#   NUM_GPUS=2 GPU_IDS=2,3 ./sh_scripts/train_smolvla_libero_full_wandb.sh
#
# Default keeps the released setup's effective batch size at 32:
#   NUM_GPUS=1 -> BATCH_SIZE=32 per GPU
#   NUM_GPUS=2 -> BATCH_SIZE=16 per GPU

WANDB_KEY_FILE="${WANDB_KEY_FILE:-./wandb/wandb_id}"
NUM_GPUS="${NUM_GPUS:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-no}"

if [ "$NUM_GPUS" -ne 1 ] && [ "$NUM_GPUS" -ne 2 ]; then
  echo "NUM_GPUS must be 1 or 2. Got: $NUM_GPUS" >&2
  exit 1
fi

if [ -z "${GPU_IDS:-}" ]; then
  if [ "$NUM_GPUS" -eq 1 ]; then
    GPU_IDS="0"
  else
    GPU_IDS="0,1"
  fi
fi

if [ -z "${WANDB_API_KEY:-}" ]; then
  if [ ! -f "$WANDB_KEY_FILE" ]; then
    echo "Missing W&B key file: $WANDB_KEY_FILE" >&2
    echo "Create it, then put your W&B API key on the first line." >&2
    exit 1
  fi
  WANDB_API_KEY="$(tr -d '[:space:]' < "$WANDB_KEY_FILE")"
fi

BASE_POLICY="${BASE_POLICY:-./models/lerobot/smolvla_base}"
DATASET_REPO="${DATASET_REPO:-lerobot/libero}"
RUN_ID="${RUN_ID:-$(date +"%Y%m%d_%H%M%S")}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/smolvla_libero_full_wandb_${RUN_ID}}"
JOB_NAME="${JOB_NAME:-smolvla_libero_full_wandb_${RUN_ID}}"

WANDB_ENTITY="${WANDB_ENTITY:-m-personal-experiment}"
WANDB_PROJECT="${WANDB_PROJECT:-vla-lerobot-experiment}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_DISABLE_ARTIFACT="${WANDB_DISABLE_ARTIFACT:-true}"

if [ -z "${BATCH_SIZE:-}" ]; then
  if [ "$NUM_GPUS" -eq 1 ]; then
    BATCH_SIZE=32
  else
    BATCH_SIZE=16
  fi
fi

STEPS="${STEPS:-25000}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
LOG_FREQ="${LOG_FREQ:-200}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-1000}"

# Keep released smolvla_libero behavior by default: no train-time sim eval.
# Set EVAL_FREQ=5000, for example, to log eval/pc_success and eval/avg_sum_reward to W&B.
EVAL_FREQ="${EVAL_FREQ:-0}"
EVAL_TASK="${EVAL_TASK:-libero_spatial}"
EVAL_N_EPISODES="${EVAL_N_EPISODES:-10}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_USE_ASYNC_ENVS="${EVAL_USE_ASYNC_ENVS:-false}"

export WANDB_API_KEY
export WANDB_MODE
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

common_args=(
  --dataset.repo_id="$DATASET_REPO"
  --dataset.use_imagenet_stats=false
  --policy.path="$BASE_POLICY"
  --policy.empty_cameras=0
  --policy.use_peft=false
  --policy.push_to_hub=false
  --policy.freeze_vision_encoder=false
  --policy.train_expert_only=false
  --policy.train_state_proj=true
  --policy.optimizer_lr=1e-4
  --policy.optimizer_weight_decay=1e-10
  --policy.optimizer_grad_clip_norm=10
  --policy.scheduler_warmup_steps=1000
  --policy.scheduler_decay_steps="$STEPS"
  --policy.scheduler_decay_lr=2.5e-6
  --batch_size="$BATCH_SIZE"
  --steps="$STEPS"
  --save_freq="$SAVE_FREQ"
  --eval_freq="$EVAL_FREQ"
  --log_freq="$LOG_FREQ"
  --num_workers="$NUM_WORKERS"
  --seed="$SEED"
  --output_dir="$OUTPUT_DIR"
  --job_name="$JOB_NAME"
  --wandb.enable=true
  --wandb.project="$WANDB_PROJECT"
  --wandb.entity="$WANDB_ENTITY"
  --wandb.mode="$WANDB_MODE"
  --wandb.disable_artifact="$WANDB_DISABLE_ARTIFACT"
  --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}'
)

if [ "$EVAL_FREQ" -gt 0 ]; then
  common_args+=(
    --env.type=libero
    --env.task="$EVAL_TASK"
    --eval.n_episodes="$EVAL_N_EPISODES"
    --eval.batch_size="$EVAL_BATCH_SIZE"
    --eval.use_async_envs="$EVAL_USE_ASYNC_ENVS"
  )
fi

echo "SmolVLA LIBERO full fine-tuning"
echo "  GPU_IDS=$GPU_IDS"
echo "  NUM_GPUS=$NUM_GPUS"
echo "  BATCH_SIZE=$BATCH_SIZE per process"
echo "  EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))"
echo "  STEPS=$STEPS"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo "  WANDB_ENTITY=$WANDB_ENTITY"
echo "  WANDB_PROJECT=$WANDB_PROJECT"
echo "  WANDB_MODE=$WANDB_MODE"
echo "  WANDB_KEY_FILE=$WANDB_KEY_FILE"
echo "  EVAL_FREQ=$EVAL_FREQ"

if [ "$NUM_GPUS" -gt 1 ]; then
  accelerate launch     --multi_gpu     --num_processes="$NUM_GPUS"     --mixed_precision="$MIXED_PRECISION"     "$(which lerobot-train)"     "${common_args[@]}"
else
  lerobot-train "${common_args[@]}"
fi
