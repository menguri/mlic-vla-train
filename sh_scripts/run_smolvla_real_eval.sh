#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/mlic/mingukang/lerobot}"
cd "$REPO_ROOT"
source .venv/bin/activate

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

# Edit these two lines for ordinary real-robot testing.
# CHECKPOINT_STEP can be 005000, 010000, 015000, 020000, 025000, or latest.
REAL_RUN_DIR="${REAL_RUN_DIR:-outputs/train/smolvla_real_full_wandb_20260612_101124}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-025000}"

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
DEFAULT_RENAME_MAP='{"observation.images.front":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}'
RENAME_MAP="${RENAME_MAP:-$DEFAULT_RENAME_MAP}"

find_latest_run_dir() {
  find outputs/train -maxdepth 1 -type d -name 'smolvla_real_full_wandb_*' \
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

if [ "$REAL_RUN_DIR" = "latest" ]; then
  REAL_RUN_DIR="$(find_latest_run_dir)"
fi

if [ -z "$REAL_RUN_DIR" ]; then
  echo "No real SmolVLA run found under outputs/train." >&2
  echo "Set REAL_RUN_DIR=outputs/train/<run> or finish a real training run first." >&2
  exit 1
fi

if [ "$CHECKPOINT_STEP" = "latest" ]; then
  CHECKPOINT_STEP="$(find_latest_step_in_run "$REAL_RUN_DIR")"
fi

POLICY_PATH="${POLICY_PATH:-$REAL_RUN_DIR/checkpoints/$CHECKPOINT_STEP/pretrained_model}"

if [ -z "$POLICY_PATH" ]; then
  echo "No real SmolVLA checkpoint found under outputs/train." >&2
  echo "Set POLICY_PATH=/absolute/path/to/pretrained_model or finish a real training run first." >&2
  exit 1
fi

if [ ! -f "$POLICY_PATH/config.json" ]; then
  echo "POLICY_PATH does not look like a LeRobot pretrained_model directory: $POLICY_PATH" >&2
  echo "Expected: $POLICY_PATH/config.json" >&2
  exit 1
fi

POLICY_PATH_ABS="$(cd "$(dirname "$POLICY_PATH")" && pwd)/$(basename "$POLICY_PATH")"

cat <<EOF
SmolVLA real remote inference server
  HOST=$HOST
  PORT=$PORT
  FPS=$FPS
  REAL_RUN_DIR=$REAL_RUN_DIR
  CHECKPOINT_STEP=$CHECKPOINT_STEP
  POLICY_PATH=$POLICY_PATH_ABS
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
       --policy_type=smolvla \\
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
EOF

python -m lerobot.async_inference.policy_server \
  --host="$HOST" \
  --port="$PORT" \
  --fps="$FPS" \
  --inference_latency="$INFERENCE_LATENCY" \
  --obs_queue_timeout="$OBS_QUEUE_TIMEOUT"
