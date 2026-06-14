#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/mlic/mingukang/lerobot}"
cd "$REPO_ROOT"
source .venv/bin/activate
source ./env_libero.sh

# Full fine-tuning SmolVLA base on our collected real demonstrations with W&B logging.
#
# The default dataset is a merged local LeRobot dataset built from:
#   collected_demo/data/TASK1
#   collected_demo/data/TASK2
#   collected_demo/data/TASK3
#   collected_demo/data/TASK4
#
# If the merged dataset does not exist, this script builds it first. Set
# MERGE_DATASET=0 to require an existing DATASET_ROOT.
#
# GPU examples:
#   ./sh_scripts/train_smolvla_real_full_wandb.sh
#   NUM_GPUS=2 ./sh_scripts/train_smolvla_real_full_wandb.sh
#   DATASET_ROOT=./collected_demo/merged/my_run ./sh_scripts/train_smolvla_real_full_wandb.sh

WANDB_KEY_FILE="${WANDB_KEY_FILE:-./wandb/wandb_id}"
NUM_GPUS="${NUM_GPUS:-2}"
MIXED_PRECISION="${MIXED_PRECISION:-no}"
SMOKE_RUN="${SMOKE_RUN:-0}"
REAL_IMAGE_AUGMENT="${REAL_IMAGE_AUGMENT:-0}"
IMAGE_AUG_MAX_NUM_TRANSFORMS="${IMAGE_AUG_MAX_NUM_TRANSFORMS:-3}"
IMAGE_AUG_RANDOM_ORDER="${IMAGE_AUG_RANDOM_ORDER:-false}"
DEFAULT_REAL_IMAGE_TRANSFORMS_TFS='{"identity":{"weight":3.0,"type":"Identity","kwargs":{}},"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.92,1.08]}},"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.92,1.08]}},"sharpness":{"weight":0.6,"type":"SharpnessJitter","kwargs":{"sharpness":[0.85,1.15]}},"blur":{"weight":0.25,"type":"GaussianBlur","kwargs":{"kernel_size":3,"sigma":[0.1,0.5]}},"gamma":{"weight":0.7,"type":"GammaJitter","kwargs":{"gamma":[0.92,1.08]}},"compression":{"weight":0.35,"type":"CompressionJitter","kwargs":{"levels":[48,128]}},"noise":{"weight":0.25,"type":"GaussianNoise","kwargs":{"std":[0.0,0.01]}},"affine":{"weight":0.35,"type":"RandomAffine","kwargs":{"degrees":[-1.0,1.0],"translate":[0.015,0.015],"scale":[0.98,1.02]}}}'
REAL_IMAGE_TRANSFORMS_TFS="${REAL_IMAGE_TRANSFORMS_TFS:-$DEFAULT_REAL_IMAGE_TRANSFORMS_TFS}"

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

if [ "$SMOKE_RUN" != "1" ] && [ -z "${WANDB_API_KEY:-}" ]; then
  if [ ! -f "$WANDB_KEY_FILE" ]; then
    echo "Missing W&B key file: $WANDB_KEY_FILE" >&2
    echo "Create it, then put your W&B API key on the first line." >&2
    exit 1
  fi
  WANDB_API_KEY="$(tr -d '[:space:]' < "$WANDB_KEY_FILE")"
fi

BASE_POLICY="${BASE_POLICY:-./models/lerobot/smolvla_base}"
TASK_DATA_ROOT="${TASK_DATA_ROOT:-./collected_demo/data}"
MERGED_DATASET_ROOT="${MERGED_DATASET_ROOT:-./collected_demo/merged/task1_2_3_4}"
DELTA_DATASET_ROOT="${DELTA_DATASET_ROOT:-./collected_demo/merged/task1_2_3_4_delta_action}"
AUGMENT_INSTRUCTIONS="${AUGMENT_INSTRUCTIONS:-1}"
AUGMENTED_DATASET_ROOT="${AUGMENTED_DATASET_ROOT:-./collected_demo/merged/task1_2_3_4_instruction_aug2x}"
AUGMENTED_DELTA_DATASET_ROOT="${AUGMENTED_DELTA_DATASET_ROOT:-./collected_demo/merged/task1_2_3_4_delta_action_instruction_aug2x}"
ACTION_MODE="${ACTION_MODE:-delta}"
if [ "$ACTION_MODE" = "delta" ]; then
  BASE_DATASET_ROOT="$DELTA_DATASET_ROOT"
  BASE_DATASET_REPO="local/collected_demo_task1_2_3_4_delta_action"
  DEFAULT_DATASET_ROOT="$AUGMENTED_DELTA_DATASET_ROOT"
  DEFAULT_DATASET_REPO="local/collected_demo_task1_2_3_4_delta_action_instruction_aug2x"
else
  BASE_DATASET_ROOT="$MERGED_DATASET_ROOT"
  BASE_DATASET_REPO="local/collected_demo_task1_2_3_4"
  DEFAULT_DATASET_ROOT="$MERGED_DATASET_ROOT"
  DEFAULT_DATASET_REPO="local/collected_demo_task1_2_3_4"
fi
if [ "$AUGMENT_INSTRUCTIONS" != "1" ]; then
  DEFAULT_DATASET_ROOT="$BASE_DATASET_ROOT"
  DEFAULT_DATASET_REPO="$BASE_DATASET_REPO"
fi
DATASET_ROOT="${DATASET_ROOT:-$DEFAULT_DATASET_ROOT}"
DATASET_REPO="${DATASET_REPO:-$DEFAULT_DATASET_REPO}"
MERGE_DATASET="${MERGE_DATASET:-1}"
MAKE_DELTA_DATASET="${MAKE_DELTA_DATASET:-1}"
FORCE_MERGE="${FORCE_MERGE:-0}"
FORCE_DELTA="${FORCE_DELTA:-0}"
FORCE_INSTRUCTION_AUGMENT="${FORCE_INSTRUCTION_AUGMENT:-0}"

RUN_ID="${RUN_ID:-$(date +"%Y%m%d_%H%M%S")}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/smolvla_real_full_wandb_${RUN_ID}}"
JOB_NAME="${JOB_NAME:-smolvla_real_full_wandb_${RUN_ID}}"

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
EVAL_FREQ="${EVAL_FREQ:-0}"
EMPTY_CAMERAS="${EMPTY_CAMERAS:-0}"
DEFAULT_RENAME_MAP='{"observation.images.front":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}'
RENAME_MAP="${RENAME_MAP:-$DEFAULT_RENAME_MAP}"
DEFAULT_INPUT_FEATURES='{"observation.state":{"type":"STATE","shape":[7]},"observation.images.camera1":{"type":"VISUAL","shape":[3,256,256]},"observation.images.camera2":{"type":"VISUAL","shape":[3,256,256]}}'
POLICY_INPUT_FEATURES="${POLICY_INPUT_FEATURES:-$DEFAULT_INPUT_FEATURES}"

if [ "$ACTION_MODE" != "absolute" ] && [ "$ACTION_MODE" != "delta" ]; then
  echo "ACTION_MODE must be absolute or delta. Got: $ACTION_MODE" >&2
  exit 1
fi

if [ "$MERGE_DATASET" = "1" ]; then
  if [ "$FORCE_MERGE" = "1" ] || [ ! -f "$MERGED_DATASET_ROOT/meta/info.json" ]; then
    merge_args=(
      --task-data-root "$TASK_DATA_ROOT"
      --output-root "$MERGED_DATASET_ROOT"
      --repo-id "local/collected_demo_task1_2_3_4"
    )
    if [ "$FORCE_MERGE" = "1" ]; then
      merge_args+=(--force)
    fi
    python scripts/merge_collected_demo_tasks.py "${merge_args[@]}"
  fi
fi

if [ "$ACTION_MODE" = "delta" ] && [ "$MAKE_DELTA_DATASET" = "1" ]; then
  if [ "$FORCE_DELTA" = "1" ] || [ ! -f "$DELTA_DATASET_ROOT/meta/info.json" ]; then
    delta_args=(
      --source-root "$MERGED_DATASET_ROOT"
      --output-root "$DELTA_DATASET_ROOT"
    )
    if [ "$FORCE_DELTA" = "1" ]; then
      delta_args+=(--force)
    fi
    python scripts/make_delta_action_dataset.py "${delta_args[@]}"
  fi
fi

if [ "$ACTION_MODE" = "delta" ] && [ "$AUGMENT_INSTRUCTIONS" = "1" ]; then
  if [ "$FORCE_INSTRUCTION_AUGMENT" = "1" ] || [ ! -f "$DEFAULT_DATASET_ROOT/meta/info.json" ]; then
    aug_args=(
      --source-root "$BASE_DATASET_ROOT"
      --output-root "$DEFAULT_DATASET_ROOT"
      --source-repo-id "$BASE_DATASET_REPO"
      --aug-repo-id "${BASE_DATASET_REPO}_instruction_paraphrase"
      --output-repo-id "$DEFAULT_DATASET_REPO"
    )
    if [ "$FORCE_INSTRUCTION_AUGMENT" = "1" ]; then
      aug_args+=(--force)
    fi
    python scripts/augment_task_instructions.py "${aug_args[@]}"
  fi
fi

if [ ! -f "$DATASET_ROOT/meta/info.json" ]; then
  echo "Missing merged dataset at: $DATASET_ROOT" >&2
  echo "Set MERGE_DATASET=1 or DATASET_ROOT to a valid LeRobot dataset root." >&2
  exit 1
fi

if [ -n "${WANDB_API_KEY:-}" ]; then
  export WANDB_API_KEY
fi
export WANDB_MODE
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

common_args=(
  --dataset.repo_id="$DATASET_REPO"
  --dataset.root="$DATASET_ROOT"
  --dataset.use_imagenet_stats=false
  --policy.path="$BASE_POLICY"
  --policy.empty_cameras="$EMPTY_CAMERAS"
  --policy.input_features="$POLICY_INPUT_FEATURES"
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
)

common_args+=(--rename_map="$RENAME_MAP")

if [ "$REAL_IMAGE_AUGMENT" = "1" ]; then
  common_args+=(
    --dataset.image_transforms.enable=true
    --dataset.image_transforms.max_num_transforms="$IMAGE_AUG_MAX_NUM_TRANSFORMS"
    --dataset.image_transforms.random_order="$IMAGE_AUG_RANDOM_ORDER"
    --dataset.image_transforms.tfs="$REAL_IMAGE_TRANSFORMS_TFS"
  )
fi

echo "SmolVLA real-demo full fine-tuning"
echo "  GPU_IDS=$GPU_IDS"
echo "  NUM_GPUS=$NUM_GPUS"
echo "  BATCH_SIZE=$BATCH_SIZE per process"
echo "  EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))"
echo "  STEPS=$STEPS"
echo "  BASE_POLICY=$BASE_POLICY"
echo "  ACTION_MODE=$ACTION_MODE"
echo "  AUGMENT_INSTRUCTIONS=$AUGMENT_INSTRUCTIONS"
echo "  DATASET_REPO=$DATASET_REPO"
echo "  DATASET_ROOT=$DATASET_ROOT"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo "  WANDB_ENTITY=$WANDB_ENTITY"
echo "  WANDB_PROJECT=$WANDB_PROJECT"
echo "  WANDB_MODE=$WANDB_MODE"
echo "  WANDB_KEY_FILE=$WANDB_KEY_FILE"
echo "  EMPTY_CAMERAS=$EMPTY_CAMERAS"
echo "  RENAME_MAP=$RENAME_MAP"
echo "  POLICY_INPUT_FEATURES=$POLICY_INPUT_FEATURES"
echo "  REAL_IMAGE_AUGMENT=$REAL_IMAGE_AUGMENT"
if [ "$REAL_IMAGE_AUGMENT" = "1" ]; then
  echo "  IMAGE_AUG_MAX_NUM_TRANSFORMS=$IMAGE_AUG_MAX_NUM_TRANSFORMS"
  echo "  IMAGE_AUG_RANDOM_ORDER=$IMAGE_AUG_RANDOM_ORDER"
  echo "  REAL_IMAGE_TRANSFORMS_TFS=$REAL_IMAGE_TRANSFORMS_TFS"
fi
echo "  SMOKE_RUN=$SMOKE_RUN"

python - "$DATASET_REPO" "$DATASET_ROOT" "$BASE_POLICY" <<'PYCHECK'
import sys
from pathlib import Path
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata

repo_id, dataset_root, base_policy = sys.argv[1:4]
dataset_root = Path(dataset_root)
base_policy = Path(base_policy)

if not base_policy.exists():
    raise FileNotFoundError(f"BASE_POLICY does not exist: {base_policy}")

meta = LeRobotDatasetMetadata(repo_id, root=dataset_root)
dataset = LeRobotDataset(repo_id, root=dataset_root)
print("Dataset smoke check")
print(f"  root:      {dataset_root}")
print(f"  frames:    {dataset.num_frames}")
print(f"  episodes:  {dataset.num_episodes}")
print(f"  fps:       {meta.fps}")
print(f"  cameras:   {meta.camera_keys}")
print(f"  features:  {list(meta.features)}")
PYCHECK

if [ "$NUM_GPUS" -gt 1 ]; then
  train_cmd=(
    accelerate launch
    --multi_gpu
    --num_processes="$NUM_GPUS"
    --mixed_precision="$MIXED_PRECISION"
    "$(which lerobot-train)"
    "${common_args[@]}"
  )
else
  train_cmd=(lerobot-train "${common_args[@]}")
fi

if [ "$SMOKE_RUN" = "1" ]; then
  echo "Smoke run complete. Training command that would run next:"
  printf '  %q' "${train_cmd[@]}"
  printf '
'
  exit 0
fi

"${train_cmd[@]}"
