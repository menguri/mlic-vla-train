#!/usr/bin/env python
"""Merge collected TASK datasets into one local LeRobot dataset."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from lerobot.datasets import LeRobotDataset, aggregate_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-data-root",
        type=Path,
        default=Path("collected_demo/data"),
        help="Directory containing TASK1, TASK2, TASK3, TASK4 LeRobot datasets.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["TASK1", "TASK2", "TASK3", "TASK4"],
        help="Task dataset directory names under --task-data-root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("collected_demo/merged/task1_2_3_4"),
        help="Destination LeRobot dataset root.",
    )
    parser.add_argument(
        "--repo-id",
        default="local/collected_demo_task1_2_3_4",
        help="Repo id stored in the merged dataset metadata.",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing output dataset.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print the plan only.")
    return parser.parse_args()


def require_dataset(path: Path) -> None:
    missing = [rel for rel in ("meta/info.json", "meta/tasks.parquet", "data") if not (path / rel).exists()]
    if missing:
        raise FileNotFoundError(f"{path} is not a complete LeRobot dataset; missing: {', '.join(missing)}")


def main() -> None:
    args = parse_args()
    task_data_root = args.task_data_root.resolve()
    output_root = args.output_root.resolve()
    roots = [(task_data_root / task).resolve() for task in args.tasks]
    repo_ids = [f"local/{task.lower()}" for task in args.tasks]

    for root in roots:
        require_dataset(root)

    print("Collected demo merge plan")
    for repo_id, root in zip(repo_ids, roots, strict=True):
        print(f"  {repo_id}: {root}")
    print(f"  output repo_id: {args.repo_id}")
    print(f"  output root:    {output_root}")

    if args.dry_run:
        return

    if output_root.exists():
        if not args.force:
            raise FileExistsError(f"Output already exists: {output_root}. Pass --force to replace it.")
        shutil.rmtree(output_root)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    aggregate_datasets(repo_ids=repo_ids, roots=roots, aggr_repo_id=args.repo_id, aggr_root=output_root)

    dataset = LeRobotDataset(args.repo_id, root=output_root)
    print("Merged dataset ready")
    print(f"  root:      {output_root}")
    print(f"  frames:    {dataset.num_frames}")
    print(f"  episodes:  {dataset.num_episodes}")
    print(f"  cameras:   {dataset.meta.camera_keys}")


if __name__ == "__main__":
    main()
