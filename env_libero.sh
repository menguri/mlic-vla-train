#!/usr/bin/env bash

export LEROBOT_ROOT="$HOME/mingukang/lerobot"

# 앞으로 Hugging Face에서 자동 다운로드되는 모델 cache
export HF_HUB_CACHE="$LEROBOT_ROOT/.hf_cache/hub"

# 앞으로 Hugging Face datasets cache
export HF_DATASETS_CACHE="$LEROBOT_ROOT/.hf_cache/datasets"

# LeRobot 자체 cache / data 위치
export HF_LEROBOT_HOME="$LEROBOT_ROOT/.hf_cache/lerobot"

# MuJoCo / LIBERO rendering
export MUJOCO_GL=egl

# 처음에는 GPU 0번만 사용
export CUDA_VISIBLE_DEVICES=0

# Timeout 설정
export HF_HUB_DOWNLOAD_TIMEOUT=1200
export HF_HUB_ETAG_TIMEOUT=1200
export HF_HUB_ENABLE_HF_TRANSFER=0

mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$HF_LEROBOT_HOME"
mkdir -p "$LEROBOT_ROOT/models/lerobot"
mkdir -p "$LEROBOT_ROOT/models/openvla"
mkdir -p "$LEROBOT_ROOT/models/openvla_oft"
mkdir -p "$LEROBOT_ROOT/eval_logs"
mkdir -p "$LEROBOT_ROOT/outputs"
