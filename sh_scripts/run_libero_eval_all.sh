#!/usr/bin/env bash
set -o pipefail

REPO_ROOT="${REPO_ROOT:-/home/mlic/mingukang/lerobot}"
cd "$REPO_ROOT"
source .venv/bin/activate
source ./env_libero.sh

# LIBERO published-style evaluation:
#   4 standard suites x 10 tasks/suite x 10 episodes/task = 400 episodes/model.
#
# Override examples:
#   N_EPISODES=5 ./sh_scripts/run_libero_eval_all.sh
#   LIBERO_TASKS=libero_object ./sh_scripts/run_libero_eval_all.sh
#   MAX_PARALLEL_TASKS=2 ./sh_scripts/run_libero_eval_all.sh

RUN_ID=$(date +"%Y%m%d_%H%M%S")
LIBERO_TASKS="${LIBERO_TASKS:-libero_spatial,libero_object,libero_goal,libero_10}"
N_EPISODES="${N_EPISODES:-10}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
MAX_PARALLEL_TASKS="${MAX_PARALLEL_TASKS:-1}"
BASE_OUT="${BASE_OUT:-./eval_logs/libero_eval_${RUN_ID}}"

mkdir -p "$BASE_OUT"

echo "LIBERO eval output dir: $BASE_OUT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "MUJOCO_GL=$MUJOCO_GL"
echo "LIBERO_TASKS=$LIBERO_TASKS"
echo "N_EPISODES=$N_EPISODES"
echo "EVAL_BATCH_SIZE=$EVAL_BATCH_SIZE"
echo "MAX_PARALLEL_TASKS=$MAX_PARALLEL_TASKS"

run_eval () {
  local name="$1"
  local policy_path="$2"
  shift 2

  local out_dir="${BASE_OUT}/${name}"
  mkdir -p "$out_dir"

  echo ""
  echo "============================================================"
  echo "[START] ${name}"
  echo "Policy: ${policy_path}"
  echo "Output: ${out_dir}"
  echo "Tasks: ${LIBERO_TASKS}"
  echo "Episodes per task: ${N_EPISODES}"
  echo "============================================================"

  lerobot-eval \
    --output_dir="${out_dir}" \
    --policy.path="${policy_path}" \
    --env.type=libero \
    --env.task="${LIBERO_TASKS}" \
    --eval.batch_size="${EVAL_BATCH_SIZE}" \
    --eval.n_episodes="${N_EPISODES}" \
    --env.max_parallel_tasks="${MAX_PARALLEL_TASKS}" \
    "$@" 2>&1 | tee "${out_dir}/cli.log"

  local code=${PIPESTATUS[0]}
  echo "$code" > "${out_dir}/exit_code.txt"

  if [ "$code" -eq 0 ]; then
    echo "[OK] ${name}"
  else
    echo "[FAIL] ${name} exit code=${code}"
  fi
}

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

# 1) pi0.5 LIBERO
# LeRobot reproduces published pi0.5 LIBERO results with n_action_steps=10.
run_eval \
  "pi05_libero_finetuned" \
  "./models/lerobot/pi05_libero_finetuned" \
  --policy.n_action_steps=10

# 2) pi0 LIBERO
run_eval \
  "pi0_libero_finetuned_v044" \
  "./models/lerobot/pi0_libero_finetuned_v044"

# 3) SmolVLA LIBERO
run_eval \
  "smolvla_libero" \
  "./models/lerobot/smolvla_libero" \
  --policy.empty_cameras=1 \
  --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'

# 4) X-VLA LIBERO
# X-VLA LIBERO examples use absolute control mode and a longer episode horizon.
run_eval \
  "xvla-libero" \
  "./models/lerobot/xvla-libero" \
  --env.control_mode=absolute \
  --env.episode_length=800 \
  --seed=142

echo ""
echo "============================================================"
echo "LIBERO eval finished."
echo "Summary:"
for f in "${BASE_OUT}"/*/exit_code.txt; do
  d=$(dirname "$f")
  name=$(basename "$d")
  code=$(cat "$f")
  if [ "$code" -eq 0 ]; then
    echo "  OK   $name"
  else
    echo "  FAIL $name exit_code=$code"
  fi
done
write_csv_summary
echo "CSV summary: ${BASE_OUT}/libero_success_summary.csv"
echo "Task CSV: ${BASE_OUT}/libero_success_by_task.csv"
echo "Logs saved in: $BASE_OUT"
echo "============================================================"
