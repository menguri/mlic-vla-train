#!/usr/bin/env bash
set -o pipefail

REPO_ROOT="${REPO_ROOT:-/home/mlic/mingukang/lerobot}"
cd "$REPO_ROOT"
source .venv/bin/activate
source ./env_libero.sh

# Full LIBERO eval for the raw SmolVLA base checkpoint.
#
# The base checkpoint is 6D in its saved action normalizer, while LIBERO is 7D.
# This script keeps the base weights, but rebuilds eval processors with
# lerobot/libero dataset stats so the policy emits/unnormalizes LIBERO 7D actions.
#
# Override examples:
#   N_EPISODES=3 ./sh_scripts/run_libero_eval_smolvla_base.sh
#   LIBERO_TASKS=libero_spatial ./sh_scripts/run_libero_eval_smolvla_base.sh
#   BASE_POLICY=./models/lerobot/smolvla_base ./sh_scripts/run_libero_eval_smolvla_base.sh

RUN_ID=$(date +"%Y%m%d_%H%M%S")
MODEL_NAME="${MODEL_NAME:-smolvla_base}"
BASE_POLICY="${BASE_POLICY:-./models/lerobot/smolvla_base}"
DATASET_REPO="${DATASET_REPO:-lerobot/libero}"
LIBERO_TASKS="${LIBERO_TASKS:-libero_spatial,libero_object,libero_goal,libero_10}"
N_EPISODES="${N_EPISODES:-10}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
MAX_PARALLEL_TASKS="${MAX_PARALLEL_TASKS:-1}"
USE_ASYNC_ENVS="${USE_ASYNC_ENVS:-true}"
SEED="${SEED:-1000}"
BASE_OUT="${BASE_OUT:-./eval_logs/libero_eval_smolvla_base_${RUN_ID}}"
OUT_DIR="${BASE_OUT}/${MODEL_NAME}"

mkdir -p "$OUT_DIR"

echo "LIBERO SmolVLA-base eval output dir: $BASE_OUT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "MUJOCO_GL=$MUJOCO_GL"
echo "MODEL_NAME=$MODEL_NAME"
echo "BASE_POLICY=$BASE_POLICY"
echo "DATASET_REPO=$DATASET_REPO"
echo "LIBERO_TASKS=$LIBERO_TASKS"
echo "N_EPISODES=$N_EPISODES"
echo "EVAL_BATCH_SIZE=$EVAL_BATCH_SIZE"
echo "MAX_PARALLEL_TASKS=$MAX_PARALLEL_TASKS"
echo "USE_ASYNC_ENVS=$USE_ASYNC_ENVS"
echo "SEED=$SEED"

python - "$OUT_DIR" "$BASE_POLICY" "$DATASET_REPO" "$LIBERO_TASKS" "$N_EPISODES" "$EVAL_BATCH_SIZE" "$MAX_PARALLEL_TASKS" "$USE_ASYNC_ENVS" "$SEED" <<'PY' 2>&1 | tee "${OUT_DIR}/cli.log"
import json
import sys
from pathlib import Path

import torch

from lerobot.configs import PreTrainedConfig
from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.envs import close_envs, make_env, make_env_pre_post_processors
from lerobot.envs.configs import LiberoEnv
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging


out_dir = Path(sys.argv[1])
base_policy = Path(sys.argv[2])
dataset_repo = sys.argv[3]
libero_tasks = sys.argv[4]
n_episodes = int(sys.argv[5])
eval_batch_size = int(sys.argv[6])
max_parallel_tasks = int(sys.argv[7])
use_async_envs = sys.argv[8].lower() in {"1", "true", "yes", "on"}
seed = int(sys.argv[9])

rename_map = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
}

init_logging()
set_seed(seed)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

out_dir.mkdir(parents=True, exist_ok=True)

env_cfg = LiberoEnv(
    task=libero_tasks,
    max_parallel_tasks=max_parallel_tasks,
)

policy_cfg = PreTrainedConfig.from_pretrained(str(base_policy))
policy_cfg.pretrained_path = str(base_policy)
policy_cfg.empty_cameras = 1

dataset_meta = LeRobotDatasetMetadata(dataset_repo)

envs = None
try:
    envs = make_env(env_cfg, n_envs=eval_batch_size, use_async_envs=use_async_envs)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)

    policy = make_policy(policy_cfg, env_cfg=env_cfg, rename_map=rename_map)

    features_for_normalization = {
        **policy.config.input_features,
        **policy.config.output_features,
    }
    preprocessor_overrides = {
        "rename_observations_processor": {"rename_map": rename_map},
        "device_processor": {"device": str(policy.config.device)},
        "normalizer_processor": {
            "stats": dataset_meta.stats,
            "features": features_for_normalization,
            "norm_map": policy.config.normalization_mapping,
        },
    }
    postprocessor_overrides = {
        "unnormalizer_processor": {
            "stats": dataset_meta.stats,
            "features": policy.config.output_features,
            "norm_map": policy.config.normalization_mapping,
        },
    }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(base_policy),
        preprocessor_overrides=preprocessor_overrides,
        postprocessor_overrides=postprocessor_overrides,
    )

    info = eval_policy_all(
        envs=envs,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=0,
        videos_dir=out_dir / "videos",
        return_episode_data=False,
        start_seed=seed,
        max_parallel_tasks=max_parallel_tasks,
    )

    with (out_dir / "eval_info.json").open("w") as f:
        json.dump(info, f, indent=2)
finally:
    if envs is not None:
        close_envs(envs)
PY

code=${PIPESTATUS[0]}
echo "$code" > "${OUT_DIR}/exit_code.txt"

write_csv_summary () {
  local summary_csv="${BASE_OUT}/libero_success_summary.csv"
  local task_csv="${BASE_OUT}/libero_success_by_task.csv"

  python - "$BASE_OUT" "$LIBERO_TASKS" "$summary_csv" "$task_csv" <<'PYCSV'
import csv
import json
import math
import sys
from pathlib import Path

base_out = Path(sys.argv[1])
requested_suites = [x.strip() for x in sys.argv[2].split(",") if x.strip()]
summary_csv = Path(sys.argv[3])
task_csv = Path(sys.argv[4])

def fmt(value):
    if value is None:
        return ""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(value):
        return ""
    return f"{value:.2f}"

def read_exit_code(model_dir):
    path = model_dir / "exit_code.txt"
    if not path.exists():
        return ""
    return path.read_text().strip()

model_dirs = sorted([p for p in base_out.iterdir() if p.is_dir()])

summary_fields = ["model", *requested_suites, "overall", "n_episodes", "eval_s", "exit_code"]
task_fields = ["model", "suite", "task_id", "n_episodes", "success_rate", "exit_code"]

with summary_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary_fields)
    writer.writeheader()

    for model_dir in model_dirs:
        exit_code = read_exit_code(model_dir)
        info_path = model_dir / "eval_info.json"
        row = {"model": model_dir.name, "exit_code": exit_code}

        if info_path.exists():
            with info_path.open() as info_file:
                info = json.load(info_file)
            per_group = info.get("per_group", {})
            for suite in requested_suites:
                row[suite] = fmt(per_group.get(suite, {}).get("pc_success"))
            overall = info.get("overall", {})
            row["overall"] = fmt(overall.get("pc_success"))
            row["n_episodes"] = overall.get("n_episodes", "")
            row["eval_s"] = fmt(overall.get("eval_s"))
        else:
            for suite in requested_suites:
                row[suite] = ""
            row["overall"] = ""
            row["n_episodes"] = ""
            row["eval_s"] = ""

        writer.writerow(row)

with task_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=task_fields)
    writer.writeheader()

    for model_dir in model_dirs:
        exit_code = read_exit_code(model_dir)
        info_path = model_dir / "eval_info.json"
        if not info_path.exists():
            writer.writerow(
                {
                    "model": model_dir.name,
                    "suite": "",
                    "task_id": "",
                    "n_episodes": "",
                    "success_rate": "",
                    "exit_code": exit_code,
                }
            )
            continue

        with info_path.open() as info_file:
            info = json.load(info_file)

        for task_info in info.get("per_task", []):
            metrics = task_info.get("metrics", {})
            successes = metrics.get("successes", [])
            n_episodes = len(successes)
            success_rate = None
            if n_episodes:
                success_rate = 100.0 * sum(bool(x) for x in successes) / n_episodes
            writer.writerow(
                {
                    "model": model_dir.name,
                    "suite": task_info.get("task_group", ""),
                    "task_id": task_info.get("task_id", ""),
                    "n_episodes": n_episodes,
                    "success_rate": fmt(success_rate),
                    "exit_code": exit_code,
                }
            )

print(f"Wrote {summary_csv}")
print(f"Wrote {task_csv}")
PYCSV
}

write_csv_summary

echo ""
echo "============================================================"
if [ "$code" -eq 0 ]; then
  echo "[OK] ${MODEL_NAME}"
else
  echo "[FAIL] ${MODEL_NAME} exit code=${code}"
fi
echo "CSV summary: ${BASE_OUT}/libero_success_summary.csv"
echo "Task CSV: ${BASE_OUT}/libero_success_by_task.csv"
echo "Logs saved in: $BASE_OUT"
echo "============================================================"

exit "$code"
