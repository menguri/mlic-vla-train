#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/mlic/mingukang/lerobot}"
cd "$REPO_ROOT"
source .venv/bin/activate

# Generic real-robot remote inference helper.
#
# Examples:
#   POLICY_PATH=outputs/train/pi05_real_full_wandb_20260612_160420/checkpoints/025000/pretrained_model ./sh_scripts/run_real_eval.sh
#   POLICY_TYPE=pi05 REAL_RUN_DIR=outputs/train/pi05_real_full_wandb_20260612_160420 CHECKPOINT_STEP=025000 ./sh_scripts/run_real_eval.sh
#   POLICY_TYPE=smolvla REAL_RUN_DIR=latest CHECKPOINT_STEP=latest ./sh_scripts/run_real_eval.sh
#   POLICY_TYPE=pi0 REAL_RUN_DIR=latest CHECKPOINT_STEP=latest ./sh_scripts/run_real_eval.sh

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export CUDA_VISIBLE_DEVICES

POLICY_TYPE="${POLICY_TYPE:-auto}"
REAL_RUN_DIR="${REAL_RUN_DIR:-latest}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-latest}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
FPS="${FPS:-10}"
INFERENCE_LATENCY="${INFERENCE_LATENCY:-0.1}"
OBS_QUEUE_TIMEOUT="${OBS_QUEUE_TIMEOUT:-1}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
CLIENT_DEVICE="${CLIENT_DEVICE:-cpu}"
ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-20}"
CHUNK_SIZE_THRESHOLD="${CHUNK_SIZE_THRESHOLD:-0.5}"
AGGREGATE_FN_NAME="${AGGREGATE_FN_NAME:-weighted_average}"
TASK="${TASK:-real robot task}"
START_SERVER="${START_SERVER:-1}"
DEFAULT_RENAME_MAP='{"observation.images.front":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}'
RENAME_MAP="${RENAME_MAP:-$DEFAULT_RENAME_MAP}"

run_glob_for_policy() {
  local policy_type="$1"
  case "$policy_type" in
    smolvla)
      printf '%s\n' 'smolvla_real_*'
      ;;
    pi05|pi0.5|pi0_5)
      printf '%s\n' 'pi05_real_*'
      ;;
    pi0|pizero|pi_zero)
      printf '%s\n' 'pi0_real_*'
      ;;
    auto)
      printf '%s\n' '*_real_*'
      ;;
    *)
      printf '%s\n' "${policy_type}_real_*"
      ;;
  esac
}

find_latest_run_dir() {
  local policy_type="$1"
  local pattern
  pattern="$(run_glob_for_policy "$policy_type")"
  find outputs/train -maxdepth 1 -type d -name "$pattern" \
    | sort \
    | tail -n 1
}

find_latest_step_in_run() {
  local run_dir="$1"
  find "$run_dir/checkpoints" -mindepth 1 -maxdepth 1 -type d 2>/dev/null \
    | sort \
    | tail -n 1 \
    | xargs -r basename
}

infer_policy_type_from_config() {
  local policy_path="$1"
  python - "$policy_path/config.json" <<'PY'
import json
import sys
from pathlib import Path
cfg = json.loads(Path(sys.argv[1]).read_text())
print(cfg.get("type", ""))
PY
}

normalize_policy_type() {
  local policy_type="$1"
  case "$policy_type" in
    pi0.5|pi0_5)
      printf '%s\n' 'pi05'
      ;;
    pizero|pi_zero)
      printf '%s\n' 'pi0'
      ;;
    *)
      printf '%s\n' "$policy_type"
      ;;
  esac
}

if [ -z "${POLICY_PATH:-}" ]; then
  if [ "$REAL_RUN_DIR" = "latest" ]; then
    REAL_RUN_DIR="$(find_latest_run_dir "$POLICY_TYPE")"
  fi

  if [ -z "$REAL_RUN_DIR" ]; then
    echo "No real run found under outputs/train for POLICY_TYPE=$POLICY_TYPE." >&2
    echo "Set POLICY_PATH=.../pretrained_model or REAL_RUN_DIR=outputs/train/<run>." >&2
    exit 1
  fi

  if [ "$CHECKPOINT_STEP" = "latest" ]; then
    CHECKPOINT_STEP="$(find_latest_step_in_run "$REAL_RUN_DIR")"
  fi

  POLICY_PATH="$REAL_RUN_DIR/checkpoints/$CHECKPOINT_STEP/pretrained_model"
fi

if [ ! -f "$POLICY_PATH/config.json" ]; then
  echo "POLICY_PATH does not look like a LeRobot pretrained_model directory: $POLICY_PATH" >&2
  echo "Expected: $POLICY_PATH/config.json" >&2
  exit 1
fi

POLICY_PATH_ABS="$(cd "$(dirname "$POLICY_PATH")" && pwd)/$(basename "$POLICY_PATH")"
CONFIG_POLICY_TYPE="$(infer_policy_type_from_config "$POLICY_PATH_ABS")"
if [ "$POLICY_TYPE" = "auto" ]; then
  POLICY_TYPE="$CONFIG_POLICY_TYPE"
else
  POLICY_TYPE="$(normalize_policy_type "$POLICY_TYPE")"
  if [ -n "$CONFIG_POLICY_TYPE" ] && [ "$CONFIG_POLICY_TYPE" != "$POLICY_TYPE" ]; then
    echo "Warning: POLICY_TYPE=$POLICY_TYPE but checkpoint config type=$CONFIG_POLICY_TYPE" >&2
  fi
fi

cat <<EOM
$POLICY_TYPE real remote inference server
  HOST=$HOST
  PORT=$PORT
  FPS=$FPS
  REAL_RUN_DIR=$REAL_RUN_DIR
  CHECKPOINT_STEP=$CHECKPOINT_STEP
  POLICY_PATH=$POLICY_PATH_ABS
  CONFIG_POLICY_TYPE=$CONFIG_POLICY_TYPE
  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
  POLICY_DEVICE=$POLICY_DEVICE
  ACTIONS_PER_CHUNK=$ACTIONS_PER_CHUNK
  RENAME_MAP=$RENAME_MAP

Local robot machine setup:
  1) Open an SSH tunnel from the local robot machine:
     ssh -N -L ${PORT}:127.0.0.1:${PORT} 10server

  2) In another local terminal, run your LeRobot robot client with your robot config.
     The checkpoint path below is a REMOTE SERVER path; the client sends it to the server.

     python -m lerobot.async_inference.robot_client \\
       --server_address=127.0.0.1:${PORT} \\
       --policy_type=$POLICY_TYPE \\
       --pretrained_name_or_path=$POLICY_PATH_ABS \\
       --policy_device=$POLICY_DEVICE \\
       --client_device=$CLIENT_DEVICE \\
       --actions_per_chunk=$ACTIONS_PER_CHUNK \\
       --chunk_size_threshold=$CHUNK_SIZE_THRESHOLD \\
       --aggregate_fn_name=$AGGREGATE_FN_NAME \\
       --fps=$FPS \\
       --task="$TASK" \\
       --rename_map='$RENAME_MAP' \\
       --robot.type=<your_robot_type> \\
       ... robot/camera args ...

Starting policy server now. Keep this process running.
EOM

if [ "$START_SERVER" != "1" ]; then
  echo "START_SERVER=$START_SERVER, not starting policy_server."
  exit 0
fi

python -m lerobot.async_inference.policy_server \
  --host="$HOST" \
  --port="$PORT" \
  --fps="$FPS" \
  --inference_latency="$INFERENCE_LATENCY" \
  --obs_queue_timeout="$OBS_QUEUE_TIMEOUT"
