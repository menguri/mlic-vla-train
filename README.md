<p align="center">
  <img alt="LeRobot, Hugging Face Robotics Library" src="./media/readme/lerobot-logo-thumbnail.png" width="100%">
</p>

# MLIC VLA Training

이 레포는 연구실에서 VLA 모델을 LIBERO와 real robot 데이터로 실험하기 위한 LeRobot 기반 작업 공간입니다. Docker는 사용하지 않고, `uv`로 `.venv` 가상환경을 만들어 실행합니다.

모델 checkpoint, dataset, 학습 output, eval log는 용량이 커서 git에 올리지 않습니다. 필요한 사람은 각자 Hugging Face에서 모델을 받고, 로컬/로봇 PC에서 dataset을 서버로 복사해서 사용합니다.

## 한눈에 보기

| 할 일 | 명령/위치 |
| --- | --- |
| 가상환경 세팅 | `uv venv --python 3.12`, `uv sync --locked ...` |
| VLA 모델 다운로드 | `./sh_scripts/download_lerobot_models.sh --set libero_eval` |
| LIBERO 평가 | `./sh_scripts/run_libero_eval_all.sh` |
| LIBERO fine-tuning | `./sh_scripts/train_smolvla_libero_full_wandb.sh` |
| real dataset 위치 | `collected_demo/data/TASK1` ... `TASK4` |
| real-data fine-tuning | `./sh_scripts/train_smolvla_real_full_wandb.sh`, `./sh_scripts/train_pi05_real_full_wandb.sh` |
| real robot eval 서버 | `./sh_scripts/run_real_eval.sh` |
| 로봇 PC client | `vla_client.py` 또는 `python -m lerobot.async_inference.robot_client` |

## 디렉터리 규칙

큰 파일은 git에 올리지 않습니다. 아래 위치에 로컬로만 둡니다.

```text
models/          # Hugging Face에서 받은 VLA checkpoint
collected_demo/  # 로컬/로봇 PC에서 가져온 real robot dataset
eval_logs/       # LIBERO eval 결과, CSV, rollout mp4
outputs/         # fine-tuning output과 checkpoint
wandb/           # W&B key 같은 개인 설정
```

## 1. 처음 환경 세팅

`uv`가 없다면 먼저 설치합니다.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
exec $SHELL -l
```

repo root에서 Python 3.12 기반 `.venv`를 만들고 dependency를 설치합니다.

```bash
cd /home/mlic/mingukang/lerobot
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
uv sync --locked --extra libero --extra smolvla --extra pi --extra xvla --extra async --extra training --extra peft
source ./env_libero.sh
```

이미 `.venv`가 만들어진 뒤에는 매번 다음만 실행하면 됩니다.

```bash
cd /home/mlic/mingukang/lerobot
source .venv/bin/activate
source ./env_libero.sh
```

W&B를 온라인으로 쓰려면 key를 파일로 저장합니다. `SMOKE_RUN=1`, `WANDB_MODE=disabled`, `WANDB_MODE=offline`으로 돌릴 때는 생략할 수 있습니다.

```bash
mkdir -p wandb
printf '%s\n' '<your-wandb-api-key>' > wandb/wandb_id
```

## 2. VLA 모델 다운로드

대부분의 VLA 모델은 Hugging Face Hub에서 받습니다. 먼저 본인 Hugging Face ID로 로그인하고, 원하는 모델이 gated model이면 해당 모델 페이지에서 access request를 승인받아야 합니다.

```bash
huggingface-cli login
```

미리 정의된 모델 묶음 다운로드:

```bash
./sh_scripts/download_lerobot_models.sh --set smolvla
./sh_scripts/download_lerobot_models.sh --set pi05
./sh_scripts/download_lerobot_models.sh --set libero_eval
```

원하는 모델 repo id를 직접 추가해서 받을 수도 있습니다.

```bash
./sh_scripts/download_lerobot_models.sh --set pi05 --model lerobot/pi05_libero_finetuned
./sh_scripts/download_lerobot_models.sh --models-dir ./models/lerobot --model <org-or-user>/<model-id>
```

기본적으로 다음 경로들이 만들어집니다.

```text
models/lerobot/smolvla_base
models/lerobot/smolvla_libero
models/lerobot/pi05_base
models/lerobot/pi05_libero_finetuned
models/lerobot/pi0_libero_finetuned_v044
models/lerobot/xvla-libero
```

## 3. LIBERO에서 VLA 동작 확인

다운로드한 VLA checkpoint를 LIBERO benchmark에서 평가합니다.

```bash
./sh_scripts/run_libero_eval_all.sh
```

자주 쓰는 옵션:

```bash
N_EPISODES=3 LIBERO_TASKS=libero_spatial ./sh_scripts/run_libero_eval_all.sh
N_EPISODES=10 LIBERO_TASKS=libero_spatial,libero_object ./sh_scripts/run_libero_eval_all.sh
MAX_PARALLEL_TASKS=2 EVAL_BATCH_SIZE=2 ./sh_scripts/run_libero_eval_all.sh
```

결과는 `eval_logs/` 아래에 저장됩니다.

```text
eval_logs/<run>/libero_success_summary.csv     # 모델별 성공률 요약
eval_logs/<run>/libero_success_by_task.csv     # task별 결과
eval_logs/**/videos/*.mp4                      # rollout 영상
```

README에 포함된 예시 mp4로 LIBERO에서 정책이 어떻게 움직이는지 바로 확인할 수 있습니다.

<video src="./media/readme/eval_episode_4.mp4" controls width="420"></video>
<video src="./media/readme/eval_episode_6.mp4" controls width="420"></video>

SmolVLA base checkpoint를 LIBERO dataset normalization으로 따로 평가하려면 다음을 실행합니다.

```bash
./sh_scripts/run_libero_eval_smolvla_base.sh
```

## 4. LIBERO로 fine-tuning

SmolVLA는 LIBERO full fine-tuning 스크립트를 제공합니다.

```bash
./sh_scripts/train_smolvla_libero_full_wandb.sh
```

자주 쓰는 옵션:

```bash
NUM_GPUS=2 GPU_IDS=0,1 ./sh_scripts/train_smolvla_libero_full_wandb.sh
STEPS=10000 WANDB_MODE=offline ./sh_scripts/train_smolvla_libero_full_wandb.sh
EVAL_FREQ=5000 EVAL_TASK=libero_spatial ./sh_scripts/train_smolvla_libero_full_wandb.sh
BASE_POLICY=./models/lerobot/smolvla_base DATASET_REPO=lerobot/libero ./sh_scripts/train_smolvla_libero_full_wandb.sh
```

모델별 사용 방식은 다음처럼 보면 됩니다.

| 모델 | 이 레포에서의 기본 사용 |
| --- | --- |
| SmolVLA | LIBERO/real-data full fine-tuning 스크립트 제공 |
| pi0.5 | real-data에서 `full`, `expert`, `lora` 모드 선택 가능 |
| pi0 | LeRobot policy/eval stack에서 checkpoint와 policy type 지정 후 서빙/평가 |
| XVLA | 현재는 LIBERO evaluation용 checkpoint 중심 |

## 5. real robot dataset 가져오기

로봇 또는 로컬 PC의 demonstration dataset을 서버의 `collected_demo/data/` 아래로 복사합니다. 기본 스크립트는 `TASK1`부터 `TASK4`까지를 합치는 흐름을 가정합니다.

```text
collected_demo/data/TASK1
collected_demo/data/TASK2
collected_demo/data/TASK3
collected_demo/data/TASK4
```

로컬 PC에서 서버로 올리는 예시:

```bash
LOCAL_SRC=/path/on/local/machine/TASK1
REMOTE=10server
REMOTE_DST=/home/mlic/mingukang/lerobot/collected_demo/data/TASK1
ssh "$REMOTE" "mkdir -p /home/mlic/mingukang/lerobot/collected_demo/data"
rsync -az --human-readable --info=progress2 --partial "$LOCAL_SRC"/ "$REMOTE:$REMOTE_DST"/
rsync -azc --dry-run --delete --itemize-changes "$LOCAL_SRC"/ "$REMOTE:$REMOTE_DST"/
```

LeRobot dataset은 보통 아래 구조를 가져야 합니다.

```text
meta/info.json
meta/tasks.parquet
data/
videos/ 또는 images/
```

## 6. real dataset을 학습용으로 변환

real-data fine-tuning 스크립트는 기본적으로 merge, delta-action 변환, instruction augmentation을 자동으로 수행합니다. 수동으로 확인하고 싶으면 아래 순서로 실행합니다.

```bash
python scripts/merge_collected_demo_tasks.py --dry-run
python scripts/merge_collected_demo_tasks.py --force
python scripts/make_delta_action_dataset.py \
  --source-root ./collected_demo/merged/task1_2_3_4 \
  --output-root ./collected_demo/merged/task1_2_3_4_delta_action
python scripts/augment_task_instructions.py \
  --source-root ./collected_demo/merged/task1_2_3_4_delta_action \
  --output-root ./collected_demo/merged/task1_2_3_4_delta_action_instruction_aug2x \
  --source-repo-id local/collected_demo_task1_2_3_4_delta_action \
  --aug-repo-id local/collected_demo_task1_2_3_4_delta_action_instruction_paraphrase \
  --output-repo-id local/collected_demo_task1_2_3_4_delta_action_instruction_aug2x
```

기본 학습 dataset은 다음 경로입니다.

```text
collected_demo/merged/task1_2_3_4_delta_action_instruction_aug2x
```

LeRobot video dataset에서는 `videos/observation.images.front/chunk-000/file-000.mp4` 같은 하나의 mp4에 여러 episode frame이 들어갈 수 있습니다. Episode boundary는 `meta/episodes/`와 timestamp metadata로 관리되므로 camera별 mp4가 하나로 보이는 것은 정상입니다.

## 7. real data로 VLA fine-tuning

SmolVLA real-data full fine-tuning:

```bash
./sh_scripts/train_smolvla_real_full_wandb.sh
```

pi0.5 real-data fine-tuning은 `FINETUNE_MODE`로 방식을 고릅니다.

```bash
# 전체 fine-tuning
FINETUNE_MODE=full NUM_GPUS=2 GPU_IDS=0,1 ./sh_scripts/train_pi05_real_full_wandb.sh

# action expert/projection 위주 fine-tuning
FINETUNE_MODE=expert NUM_GPUS=2 GPU_IDS=2,3 ./sh_scripts/train_pi05_real_full_wandb.sh

# LoRA/PEFT fine-tuning
FINETUNE_MODE=lora NUM_GPUS=2 GPU_IDS=4,5 ./sh_scripts/train_pi05_real_full_wandb.sh
```

자주 쓰는 옵션:

```bash
SMOKE_RUN=1 WANDB_MODE=disabled NUM_GPUS=1 ./sh_scripts/train_pi05_real_full_wandb.sh
REAL_IMAGE_AUGMENT=1 FINETUNE_MODE=full ./sh_scripts/train_pi05_real_full_wandb.sh
ACTION_MODE=absolute AUGMENT_INSTRUCTIONS=0 ./sh_scripts/train_pi05_real_full_wandb.sh
DATASET_ROOT=./collected_demo/merged/my_dataset DATASET_REPO=local/my_dataset ./sh_scripts/train_pi05_real_full_wandb.sh
PEFT_R=32 PEFT_LORA_ALPHA=64 FINETUNE_MODE=lora ./sh_scripts/train_pi05_real_full_wandb.sh
BASE_POLICY=./models/lerobot/pi05_base STEPS=25000 SAVE_FREQ=5000 ./sh_scripts/train_pi05_real_full_wandb.sh
```

기본 camera rename mapping은 다음과 같습니다. 수집 데이터의 camera key가 다르면 `RENAME_MAP` 또는 `POLICY_INPUT_FEATURES`를 맞춰야 합니다.

```json
{"observation.images.front":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}
```

`REAL_IMAGE_AUGMENT=1`은 원본 이미지를 섞으면서 약한 brightness/contrast, sharpness/blur, gamma, compression, Gaussian noise, 작은 affine 변환을 추가합니다. 필요하면 `REAL_IMAGE_TRANSFORMS_TFS='<json>'`로 직접 지정합니다.

## 8. real robot VLA 실험

GPU 서버에서 remote policy server를 켜고 그대로 기다립니다. 아래 명령은 checkpoint를 찾은 뒤 `policy_server`를 실행하고, 로봇 PC에서 실행할 tunnel/client 명령도 같이 출력합니다.

```bash
cd /home/mlic/mingukang/lerobot
POLICY_PATH=outputs/train/<run>/checkpoints/<step>/pretrained_model ./sh_scripts/run_real_eval.sh
```

예시:

```bash
POLICY_TYPE=smolvla REAL_RUN_DIR=latest CHECKPOINT_STEP=latest ./sh_scripts/run_real_eval.sh
POLICY_TYPE=pi05 REAL_RUN_DIR=latest CHECKPOINT_STEP=latest ./sh_scripts/run_real_eval.sh
POLICY_PATH=outputs/train/pi05_real_full_wandb_20260612_160420/checkpoints/025000/pretrained_model ./sh_scripts/run_real_eval.sh
CUDA_VISIBLE_DEVICES=6 POLICY_PATH=outputs/train/<run>/checkpoints/<step>/pretrained_model ./sh_scripts/run_real_eval.sh
```

기본 port는 `8080`입니다. 이미 사용 중이면 새 port를 지정합니다.

```bash
PORT=8090 POLICY_PATH=outputs/train/<run>/checkpoints/<step>/pretrained_model ./sh_scripts/run_real_eval.sh
```

로봇 PC에서는 먼저 서버 port를 열어둡니다.

```bash
ssh -N -L 8080:127.0.0.1:8080 10server
# 서버에서 PORT=8090을 썼다면:
ssh -N -L 8090:127.0.0.1:8090 10server
```

그 다음 로봇 PC의 다른 터미널에서 `vla_client.py`를 실행합니다. `--pretrained_name_or_path`는 로봇 PC 로컬 경로가 아니라 GPU 서버에 존재하는 checkpoint 경로여야 합니다.

```bash
python vla_client.py \
  --server_address=127.0.0.1:8080 \
  --policy_type=pi05 \
  --pretrained_name_or_path=/home/mlic/mingukang/lerobot/outputs/train/<run>/checkpoints/<step>/pretrained_model \
  --policy_device=cuda \
  --client_device=cpu \
  --actions_per_chunk=20 \
  --chunk_size_threshold=0.5 \
  --aggregate_fn_name=weighted_average \
  --fps=10 \
  --task="your instruction" \
  --rename_map='{"observation.images.front":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}'
```

LeRobot 기본 client를 직접 쓸 경우에는 robot/camera argument를 뒤에 추가합니다.

```bash
python -m lerobot.async_inference.robot_client \
  --server_address=127.0.0.1:8080 \
  --policy_type=pi05 \
  --pretrained_name_or_path=/home/mlic/mingukang/lerobot/outputs/train/<run>/checkpoints/<step>/pretrained_model \
  --policy_device=cuda \
  --client_device=cpu \
  --actions_per_chunk=20 \
  --chunk_size_threshold=0.5 \
  --aggregate_fn_name=weighted_average \
  --fps=10 \
  --task="your instruction" \
  --rename_map='{"observation.images.front":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}' \
  --robot.type=<your_robot_type> \
  ... robot/camera args ...
```

정리하면 서버는 `run_real_eval.sh`를 켜고 기다리고, 로봇 PC는 SSH tunnel을 연 뒤 `vla_client.py` 또는 `lerobot.async_inference.robot_client`를 실행합니다. 추론은 서버 GPU에서 돌고, 로봇 PC는 camera/robot I/O를 담당합니다.

## 9. GitHub push 메모

GitHub SSH key와 `origin` 설정은 repo 바깥의 다음 문서를 참고합니다.

```text
/home/mlic/mingukang/GITHUB_SSH_PUSH_GUIDE.md
```
