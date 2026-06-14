#!/usr/bin/env python
"""Download LeRobot model snapshots into ./models/lerobot."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_SETS = {
    "pi05": ["lerobot/pi05_base"],
    "smolvla": ["lerobot/smolvla_base", "lerobot/smolvla_libero"],
    "libero_eval": [
        "lerobot/smolvla_base",
        "lerobot/smolvla_libero",
        "lerobot/pi05_base",
        "lerobot/pi05_libero_finetuned",
        "lerobot/pi0_libero_finetuned_v044",
        "lerobot/xvla-libero",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models/lerobot"),
        help="Directory where model snapshots are materialized.",
    )
    parser.add_argument(
        "--set",
        choices=sorted(MODEL_SETS),
        default="smolvla",
        help="Predefined model group to download.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Extra model repo id to download. Can be passed multiple times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models_dir = args.models_dir.resolve()
    model_ids = [*MODEL_SETS[args.set], *args.model]

    for model_id in model_ids:
        local_dir = models_dir / model_id.split("/", 1)[1]
        print(f"Downloading {model_id} -> {local_dir}")
        snapshot_download(repo_id=model_id, repo_type="model", local_dir=local_dir)

    print("Done.")


if __name__ == "__main__":
    main()
