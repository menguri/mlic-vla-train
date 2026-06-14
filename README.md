<p align="center">
  <img alt="LeRobot, Hugging Face Robotics Library" src="./media/readme/lerobot-logo-thumbnail.png" width="100%">
</p>

<div align="center">

[![Python versions](https://img.shields.io/pypi/pyversions/lerobot)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/huggingface/lerobot/blob/main/LICENSE)
[![Status](https://img.shields.io/pypi/status/lerobot)](https://pypi.org/project/lerobot/)
[![Version](https://img.shields.io/pypi/v/lerobot)](https://pypi.org/project/lerobot/)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.1-ff69b4.svg)](https://github.com/huggingface/lerobot/blob/main/CODE_OF_CONDUCT.md)
[![Discord](https://img.shields.io/badge/Discord-Join_Us-5865F2?style=flat&logo=discord&logoColor=white)](https://discord.gg/q8Dzzpym3f)

</div>

**LeRobot** aims to provide models, datasets, and tools for real-world robotics in PyTorch. The goal is to lower the barrier to entry so that everyone can contribute to and benefit from shared datasets and pretrained models.

🤗 A hardware-agnostic, Python-native interface that standardizes control across diverse platforms, from low-cost arms (SO-100) to humanoids.

🤗 A standardized, scalable LeRobotDataset format (Parquet + MP4 or images) hosted on the Hugging Face Hub, enabling efficient storage, streaming and visualization of massive robotic datasets.

🤗 State-of-the-art policies that have been shown to transfer to the real-world ready for training and deployment.

🤗 Comprehensive support for the open-source ecosystem to democratize physical AI.


## Lab VLA Workflow

이 레포는 연구실 내부에서 VLA 모델을 LIBERO와 real robot 데이터로 실험하기 위한 LeRobot 기반 작업 공간입니다. 모델 checkpoint, dataset, 학습 output, eval log는 용량이 커서 git에 올리지 않습니다. 아래 명령은 특별히 적지 않으면 repo root에서 실행합니다. 이 레포는 Docker를 쓰지 않고 `uv`로 `.venv` 가상환경을 만든 뒤 실행합니다.

처음 세팅할 때 `uv`가 없다면 먼저 설치합니다.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
exec $SHELL -l
```

그 다음 repo root에서 가상환경을 만듭니다.

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

로컬 자산 위치는 다음처럼 맞춥니다.

```text
models/          # Hugging Face에서 받은 VLA checkpoint
collected_demo/  # 로컬/로봇 PC에서 가져온 real robot dataset
eval_logs/       # LIBERO eval 결과, CSV, rollout mp4
outputs/         # fine-tuning output과 checkpoint
wandb/           # W&B key 같은 개인 설정
```

W&B를 온라인으로 쓰려면 다음 파일에 key를 넣습니다. `SMOKE_RUN=1` 또는 `WANDB_MODE=disabled/offline`으로 돌릴 때는 생략할 수 있습니다.

```bash
mkdir -p wandb
printf '%s\n' '<your-wandb-api-key>' > wandb/wandb_id
```

### 1. VLA 모델 다운로드

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

### 2. LIBERO에서 VLA 동작 확인

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

결과는 `eval_logs/` 아래에 저장됩니다. 성공률 요약은 `libero_success_summary.csv`, task별 결과는 `libero_success_by_task.csv`를 보면 됩니다. 실제 rollout 영상은 `eval_logs/**/videos/*.mp4`에 생성되며, README에 포함된 아래 예시 mp4로 LIBERO에서 정책이 어떻게 움직이는지 바로 확인할 수 있습니다.

<video src="./media/readme/eval_episode_4.mp4" controls width="420"></video>
<video src="./media/readme/eval_episode_6.mp4" controls width="420"></video>

SmolVLA base checkpoint를 LIBERO dataset normalization으로 따로 평가하려면:

```bash
./sh_scripts/run_libero_eval_smolvla_base.sh
```

### 3. LIBERO로 VLA fine-tuning

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

모델별 학습 옵션은 아주 간단히 이렇게 보면 됩니다.

```text
SmolVLA: LIBERO/real-data full fine-tuning 스크립트 제공. vision encoder는 유지하고 state/action 쪽까지 학습하는 기본 설정.
pi0.5: real-data 스크립트에서 full, expert(action expert/projection only), lora(PEFT) 선택 가능.
pi0: LeRobot policy/eval stack에서 사용 가능. pi0 checkpoint와 policy type을 지정해서 같은 방식으로 서빙/평가.
XVLA: 현재 이 레포에서는 LIBERO evaluation용 checkpoint 중심으로 사용.
```

### 4. 로컬 PC에서 real robot dataset 가져오기

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

### 5. 학습용 real dataset으로 변환

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

### 6. real data로 VLA fine-tuning

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

### 7. real robot VLA 실험

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


## Quick Start

LeRobot can be installed directly from PyPI.

```bash
pip install lerobot
lerobot-info
```

> [!IMPORTANT]
> For detailed installation guide, please see the [Installation Documentation](https://huggingface.co/docs/lerobot/installation).

## Robots & Control

<div align="center">
  <img src="./media/readme/robots_control_video.webp" width="640px" alt="Reachy 2 Demo">
</div>

LeRobot provides a unified `Robot` class interface that decouples control logic from hardware specifics. It supports a wide range of robots and teleoperation devices.

```python
from lerobot.robots.myrobot import MyRobot

# Connect to a robot
robot = MyRobot(config=...)
robot.connect()

# Read observation and send action
obs = robot.get_observation()
action = model.select_action(obs)
robot.send_action(action)
```

**Supported Hardware:** SO100, LeKiwi, Koch, HopeJR, OMX, EarthRover, Reachy2, Gamepads, Keyboards, Phones, OpenARM, Unitree G1.

While these devices are natively integrated into the LeRobot codebase, the library is designed to be extensible. You can easily implement the Robot interface to utilize LeRobot's data collection, training, and visualization tools for your own custom robot.

For detailed hardware setup guides, see the [Hardware Documentation](https://huggingface.co/docs/lerobot/integrate_hardware).

## LeRobot Dataset

To solve the data fragmentation problem in robotics, we utilize the **LeRobotDataset** format.

- **Structure:** Synchronized MP4 videos (or images) for vision and Parquet files for state/action data.
- **HF Hub Integration:** Explore thousands of robotics datasets on the [Hugging Face Hub](https://huggingface.co/lerobot).
- **Tools:** Seamlessly delete episodes, split by indices/fractions, add/remove features, and merge multiple datasets.

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Load a dataset from the Hub
dataset = LeRobotDataset("lerobot/aloha_mobile_cabinet")

# Access data (automatically handles video decoding)
episode_index=0
print(f"{dataset[episode_index]['action'].shape=}\n")
```

Learn more about it in the [LeRobotDataset Documentation](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)

## SoTA Models

LeRobot implements state-of-the-art policies in pure PyTorch, covering Imitation Learning, Reinforcement Learning, and Vision-Language-Action (VLA) models, with more coming soon. It also provides you with the tools to instrument and inspect your training process.

<p align="center">
  <img alt="Gr00t Architecture" src="./media/readme/VLA_architecture.jpg" width="640px">
</p>

Training a policy is as simple as running a script configuration:

```bash
lerobot-train \
  --policy=act \
  --dataset.repo_id=lerobot/aloha_mobile_cabinet
```

| Category                   | Models                                                                                                                                                                                                                  |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Imitation Learning**     | [ACT](./docs/source/policy_act_README.md), [Diffusion](./docs/source/policy_diffusion_README.md), [VQ-BeT](./docs/source/policy_vqbet_README.md), [Multitask DiT Policy](./docs/source/policy_multi_task_dit_README.md) |
| **Reinforcement Learning** | [HIL-SERL](./docs/source/hilserl.mdx), [TDMPC](./docs/source/policy_tdmpc_README.md) & QC-FQL (coming soon)                                                                                                             |
| **VLAs Models**            | [Pi0Fast](./docs/source/pi0fast.mdx), [Pi0.5](./docs/source/pi05.mdx), [GR00T N1.5](./docs/source/policy_groot_README.md), [SmolVLA](./docs/source/policy_smolvla_README.md), [XVLA](./docs/source/xvla.mdx)            |

Similarly to the hardware, you can easily implement your own policy & leverage LeRobot's data collection, training, and visualization tools, and share your model to the HF Hub

For detailed policy setup guides, see the [Policy Documentation](https://huggingface.co/docs/lerobot/bring_your_own_policies). For GPU/RAM requirements and expected training time per policy, see the [Compute Hardware Guide](https://huggingface.co/docs/lerobot/hardware_guide).

## Inference & Evaluation

Evaluate your policies in simulation or on real hardware using the unified evaluation script. LeRobot supports standard benchmarks like **LIBERO**, **MetaWorld** and more to come.

```bash
# Evaluate a policy on the LIBERO benchmark
lerobot-eval \
  --policy.path=lerobot/pi0_libero_finetuned \
  --env.type=libero \
  --env.task=libero_object \
  --eval.n_episodes=10
```

Learn how to implement your own simulation environment or benchmark and distribute it from the HF Hub by following the [EnvHub Documentation](https://huggingface.co/docs/lerobot/envhub)

## Resources

- **[Documentation](https://huggingface.co/docs/lerobot/index):** The complete guide to tutorials & API.
- **[Chinese Tutorials: LeRobot+SO-ARM101中文教程-同济子豪兄](https://zihao-ai.feishu.cn/wiki/space/7589642043471924447)** Detailed doc for assembling, teleoperate, dataset, train, deploy. Verified by Seed Studio and 5 global hackathon players.
- **[Discord](https://discord.gg/q8Dzzpym3f):** Join the `LeRobot` server to discuss with the community.
- **[X](https://x.com/LeRobotHF):** Follow us on X to stay up-to-date with the latest developments.
- **[Robot Learning Tutorial](https://huggingface.co/spaces/lerobot/robot-learning-tutorial):** A free, hands-on course to learn robot learning using LeRobot.

## Citation

If you use LeRobot in your project, please cite the GitHub repository to acknowledge the ongoing development and contributors:

```bibtex
@misc{cadene2024lerobot,
    author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Palma, Steven and Kooijmans, Pepijn and Aractingi, Michel and Shukor, Mustafa and Aubakirova, Dana and Russi, Martino and Capuano, Francesco and Pascal, Caroline and Choghari, Jade and Moss, Jess and Wolf, Thomas},
    title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
    howpublished = "\url{https://github.com/huggingface/lerobot}",
    year = {2024}
}
```

If you are referencing our research or the academic paper, please also cite our ICLR publication:

<details>
<summary><b>ICLR 2026 Paper</b></summary>

```bibtex
@inproceedings{cadenelerobot,
  title={LeRobot: An Open-Source Library for End-to-End Robot Learning},
  author={Cadene, Remi and Alibert, Simon and Capuano, Francesco and Aractingi, Michel and Zouitine, Adil and Kooijmans, Pepijn and Choghari, Jade and Russi, Martino and Pascal, Caroline and Palma, Steven and Shukor, Mustafa and Moss, Jess and Soare, Alexander and Aubakirova, Dana and Lhoest, Quentin and Gallou\'edec, Quentin and Wolf, Thomas},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://arxiv.org/abs/2602.22818}
}
```

</details>

## Contribute

We welcome contributions from everyone in the community! To get started, please read our [CONTRIBUTING.md](https://github.com/huggingface/lerobot/blob/main/CONTRIBUTING.md) guide. Whether you're adding a new feature, improving documentation, or fixing a bug, your help and feedback are invaluable. We're incredibly excited about the future of open-source robotics and can't wait to work with you on what's next—thank you for your support!

<p align="center">
  <img alt="SO101 Video" src="./media/readme/so100_video.webp" width="640px">
</p>

<div align="center">
<sub>Built by the <a href="https://huggingface.co/lerobot">LeRobot</a> team at <a href="https://huggingface.co">Hugging Face</a> with ❤️</sub>
</div>
