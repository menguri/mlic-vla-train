#!/usr/bin/env python
"""Double a LeRobot dataset by duplicating episodes with paraphrased task instructions."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from lerobot.datasets import LeRobotDataset, aggregate_datasets

DEFAULT_CANONICAL = {
    "pick up the white bottle and place it in the dark brown box":
        "pick up the bottle with the white cap and place it in the dark brown box",
    "pick up the white cup and place it in the dark brown box":
        "pick up the cup and place it in the dark brown box",
    "pick up the white square box and place it in the dark brown box":
        "pick up the light brown square box and place it in the dark brown box",
    "pick up the yellow tape measure and place it in the dark brown box":
        "pick up the yellow tape measure and place it in the dark brown box",
}

DEFAULT_PARAPHRASES = {
    "pick up the white bottle and place it in the dark brown box":
        "pick up the bottle with the pale cap and place it in the deep brown box",
    "pick up the white cup and place it in the dark brown box":
        "pick up the cup and place it in the deep brown box",
    "pick up the white square box and place it in the dark brown box":
        "pick up the tan square box and place it in the deep brown box",
    "pick up the yellow tape measure and place it in the dark brown box":
        "pick up the golden tape measure and place it in the deep brown box",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="Source LeRobot dataset root.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output doubled dataset root.")
    parser.add_argument("--source-repo-id", default="local/source", help="Temporary repo id for source dataset.")
    parser.add_argument("--aug-repo-id", default="local/source_instruction_paraphrase", help="Temporary repo id for paraphrased copy.")
    parser.add_argument("--output-repo-id", default="local/source_instruction_aug2x", help="Repo id stored in output metadata.")
    parser.add_argument("--force", action="store_true", help="Replace an existing output dataset.")
    parser.add_argument("--dry-run", action="store_true", help="Print task mapping without writing output.")
    return parser.parse_args()


def require_dataset(root: Path) -> None:
    missing = [rel for rel in ("meta/info.json", "meta/tasks.parquet", "meta/episodes", "data") if not (root / rel).exists()]
    if missing:
        raise FileNotFoundError(f"{root} is not a complete LeRobot dataset; missing: {', '.join(missing)}")


def normalize_episode_meta_file_indices(root: Path) -> None:
    """Keep episode metadata self-contained when source metadata was merged into one file."""
    for ep_path in sorted((root / "meta/episodes").glob("chunk-*/*.parquet")):
        eps = pd.read_parquet(ep_path)
        chunk = int(ep_path.parent.name.removeprefix("chunk-"))
        file_idx = int(ep_path.stem.removeprefix("file-"))
        eps["meta/episodes/chunk_index"] = chunk
        eps["meta/episodes/file_index"] = file_idx
        eps.to_parquet(ep_path, index=False)


def rewrite_tasks(root: Path, mapping: dict[str, str]) -> None:
    tasks_path = root / "meta/tasks.parquet"
    tasks = pd.read_parquet(tasks_path)
    missing = sorted(set(tasks.index) - set(mapping))
    if missing:
        raise ValueError(f"Missing paraphrases for tasks: {missing}")
    tasks.index = pd.Index([mapping[task] for task in tasks.index], name="task")
    tasks.to_parquet(tasks_path)

    for ep_path in sorted((root / "meta/episodes").glob("chunk-*/*.parquet")):
        eps = pd.read_parquet(ep_path)

        def map_task_array(value):
            return [mapping[str(task)] for task in value]

        eps["tasks"] = eps["tasks"].apply(map_task_array)
        eps.to_parquet(ep_path, index=False)


def make_dataset_view(source_root: Path, tmp_root: Path, name: str) -> Path:
    view_root = tmp_root / name
    view_root.mkdir(parents=True)
    shutil.copytree(source_root / "meta", view_root / "meta")
    # Symlink large immutable payloads; aggregate_datasets follows these paths while writing the real output.
    (view_root / "data").symlink_to(source_root / "data", target_is_directory=True)
    if (source_root / "videos").exists():
        (view_root / "videos").symlink_to(source_root / "videos", target_is_directory=True)
    if (source_root / "images").exists():
        (view_root / "images").symlink_to(source_root / "images", target_is_directory=True)
    normalize_episode_meta_file_indices(view_root)
    return view_root


def make_rewritten_view(source_root: Path, tmp_root: Path, name: str, mapping: dict[str, str]) -> Path:
    view_root = make_dataset_view(source_root, tmp_root, name)
    rewrite_tasks(view_root, mapping)
    return view_root


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    require_dataset(source_root)

    tasks = pd.read_parquet(source_root / "meta/tasks.parquet")
    canonical_mapping = {task: DEFAULT_CANONICAL[task] for task in tasks.index}
    paraphrase_mapping = {task: DEFAULT_PARAPHRASES[task] for task in tasks.index}

    print("Instruction augmentation mapping")
    for old in tasks.index:
        print(f"  canonical:  {old!r} -> {canonical_mapping[old]!r}")
        print(f"  paraphrase: {old!r} -> {paraphrase_mapping[old]!r}")
    print(f"source: {source_root}")
    print(f"output: {output_root}")

    if args.dry_run:
        return

    if output_root.exists():
        if not args.force:
            raise FileExistsError(f"Output already exists: {output_root}. Pass --force to replace it.")
        shutil.rmtree(output_root)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lerobot_instruction_aug_") as tmp:
        tmp_root = Path(tmp)
        source_view = make_rewritten_view(source_root, tmp_root, "source", canonical_mapping)
        aug_root = make_rewritten_view(source_root, tmp_root, "paraphrased", paraphrase_mapping)
        aggregate_datasets(
            repo_ids=[args.source_repo_id, args.aug_repo_id],
            roots=[source_view, aug_root],
            aggr_repo_id=args.output_repo_id,
            aggr_root=output_root,
        )

    dataset = LeRobotDataset(args.output_repo_id, root=output_root)
    out_tasks = pd.read_parquet(output_root / "meta/tasks.parquet")
    print("Instruction-augmented dataset ready")
    print(f"  root:      {output_root}")
    print(f"  frames:    {dataset.num_frames}")
    print(f"  episodes:  {dataset.num_episodes}")
    print(f"  tasks:     {len(out_tasks)}")


if __name__ == "__main__":
    main()
