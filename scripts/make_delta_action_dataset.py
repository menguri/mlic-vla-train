#!/usr/bin/env python
"""Create a derived LeRobot dataset whose action is full 7D delta.

The source real dataset stores 7D absolute target actions:
    [target_x, target_y, target_z, target_roll, target_pitch, target_yaw, target_gripper]

This script creates a copy where all 7 action dimensions become:
    target_action - observation.state

That includes the gripper dimension:
    delta_gripper = target_gripper_pos - gripper_pos

The source folder is never modified.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ACTION_NAMES = [
    "delta_tcp_x_mm",
    "delta_tcp_y_mm",
    "delta_tcp_z_mm",
    "delta_tcp_roll_rad",
    "delta_tcp_pitch_rad",
    "delta_tcp_yaw_rad",
    "delta_gripper_pos",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("collected_demo/merged/task1_2_3_4"),
        help="Source LeRobot dataset with absolute target actions.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("collected_demo/merged/task1_2_3_4_delta_action"),
        help="Output derived dataset root.",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing output dataset.")
    return parser.parse_args()


def as_vector_array(series: pd.Series) -> np.ndarray:
    return np.stack(series.to_numpy()).astype(np.float32)


def vector_stats(values: np.ndarray) -> dict[str, list[float]]:
    return {
        "min": values.min(axis=0).astype(float).tolist(),
        "max": values.max(axis=0).astype(float).tolist(),
        "mean": values.mean(axis=0).astype(float).tolist(),
        "std": values.std(axis=0).astype(float).tolist(),
        "count": [int(values.shape[0])],
        "q01": np.quantile(values, 0.01, axis=0).astype(float).tolist(),
        "q10": np.quantile(values, 0.10, axis=0).astype(float).tolist(),
        "q50": np.quantile(values, 0.50, axis=0).astype(float).tolist(),
        "q90": np.quantile(values, 0.90, axis=0).astype(float).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).astype(float).tolist(),
    }


def convert_data_files(root: Path) -> np.ndarray:
    all_actions: list[np.ndarray] = []
    for parquet_path in sorted((root / "data").glob("chunk-*/*.parquet")):
        df = pd.read_parquet(parquet_path)
        state = as_vector_array(df["observation.state"])
        action = as_vector_array(df["action"])
        if state.shape[1] < 7 or action.shape[1] != 7:
            raise ValueError(
                f"Expected state >=7D and action 7D in {parquet_path}, "
                f"got state={state.shape}, action={action.shape}"
            )
        delta_action = action.copy()
        delta_action[:, :7] = action[:, :7] - state[:, :7]
        df["action"] = list(delta_action.astype(np.float32))
        df.to_parquet(parquet_path, index=False)
        all_actions.append(delta_action)
    if not all_actions:
        raise FileNotFoundError(f"No parquet files found under {root / 'data'}")
    return np.concatenate(all_actions, axis=0)


def update_info(root: Path) -> None:
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["features"]["action"]["names"] = ACTION_NAMES
    info_path.write_text(json.dumps(info, indent=4) + "\n")


def update_global_stats(root: Path, actions: np.ndarray) -> None:
    stats_path = root / "meta/stats.json"
    stats = json.loads(stats_path.read_text())
    stats["action"] = vector_stats(actions)
    stats_path.write_text(json.dumps(stats, indent=4) + "\n")


def update_episode_stats(root: Path) -> None:
    episodes_root = root / "meta/episodes"
    for parquet_path in sorted(episodes_root.glob("chunk-*/*.parquet")):
        df = pd.read_parquet(parquet_path)
        for _, row in df.iterrows():
            data_path = root / "data" / f"chunk-{int(row['data/chunk_index']):03d}" / f"file-{int(row['data/file_index']):03d}.parquet"
            data = pd.read_parquet(data_path)
            start = int(row["dataset_from_index"])
            end = int(row["dataset_to_index"])
            episode_actions = as_vector_array(data[(data["index"] >= start) & (data["index"] < end)]["action"])
            if episode_actions.size == 0:
                continue
            ep_stats = vector_stats(episode_actions)
            for stat_name, stat_value in ep_stats.items():
                df.at[row.name, f"stats/action/{stat_name}"] = stat_value
        df.to_parquet(parquet_path, index=False)


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if not (source_root / "meta/info.json").exists():
        raise FileNotFoundError(f"Source does not look like a LeRobot dataset: {source_root}")
    if output_root.exists():
        if not args.force:
            raise FileExistsError(f"Output already exists: {output_root}. Pass --force to replace it.")
        shutil.rmtree(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_root, output_root, symlinks=True)

    actions = convert_data_files(output_root)
    update_info(output_root)
    update_global_stats(output_root, actions)
    update_episode_stats(output_root)

    print("Delta-action dataset ready")
    print(f"  source: {source_root}")
    print(f"  output: {output_root}")
    print(f"  frames: {actions.shape[0]}")
    print(f"  action mean: {np.round(actions.mean(axis=0), 4).tolist()}")
    print(f"  action std:  {np.round(actions.std(axis=0), 4).tolist()}")
    print(f"  action min:  {np.round(actions.min(axis=0), 4).tolist()}")
    print(f"  action max:  {np.round(actions.max(axis=0), 4).tolist()}")


if __name__ == "__main__":
    main()
