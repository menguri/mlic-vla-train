# Collected Demonstration Data

This directory is for local LeRobot demonstrations collected with `lerobot-record`.

LeRobot v3 datasets are directory datasets. A healthy recorded dataset usually looks like:

```text
<dataset_root>/
  data/
    chunk-000/
      file-000.parquet
      ...
  meta/
    info.json
    stats.json
    tasks.parquet
    episodes/
      chunk-000/
        file-000.parquet
  videos/
    observation.images.<camera_name>/
      chunk-000/
        file-000.mp4
```

## Safe Move

Prefer `rsync` over `mv` when moving demonstrations across directories or disks. Copy first, verify, then delete the original only after verification passes.

### From Your Local Machine To This SSH Server

If the demonstration folder is on your local machine and this repository is on the remote SSH server, run these commands from a terminal on your local machine, not from inside the SSH session.

Set the local source and remote destination:

```bash
LOCAL_SRC=/path/on/your/local/machine/to/your_recorded_dataset
REMOTE=10server
REMOTE_BASE=/home/mlic/mingukang/lerobot/collected_demo
REMOTE_DST="$REMOTE_BASE/$(basename "$LOCAL_SRC")"
```

This assumes your local `~/.ssh/config` contains something like:

```sshconfig
Host 10server
    HostName 10.0.12.149
    User mlic
    Port 4010
```

Create the destination directory on the remote server:

```bash
ssh "$REMOTE" "mkdir -p \"$REMOTE_BASE\""
```

Preview the upload:

```bash
rsync -az --dry-run --itemize-changes "$LOCAL_SRC"/ "$REMOTE:$REMOTE_DST"/
```

Upload the dataset:

```bash
rsync -az --human-readable --info=progress2 --partial "$LOCAL_SRC"/ "$REMOTE:$REMOTE_DST"/
```

Verify the remote copy with checksums:

```bash
rsync -azc --dry-run --delete --itemize-changes "$LOCAL_SRC"/ "$REMOTE:$REMOTE_DST"/
```

If the checksum verification prints no file changes, the remote copy matches your local dataset.

Quick remote structural check:

```bash
ssh "$REMOTE" "test -f \"$REMOTE_DST/meta/info.json\" && test -f \"$REMOTE_DST/meta/stats.json\" && test -f \"$REMOTE_DST/meta/tasks.parquet\" && test -d \"$REMOTE_DST/data\""
```

Optional remote size/count checks:

```bash
du -sh "$LOCAL_SRC"
find "$LOCAL_SRC" -type f | wc -l
ssh "$REMOTE" "du -sh \"$REMOTE_DST\" && find \"$REMOTE_DST\" -type f | wc -l"
```

Only delete the local original after the checksum verification passes and you are sure you no longer need the local copy.

### Within This SSH Server

Use this when the source demonstration folder is already somewhere on this remote server.

Set the source dataset path:

```bash
SRC=/absolute/path/to/your_recorded_dataset
DST=/home/mlic/mingukang/lerobot/collected_demo/$(basename "$SRC")
```

Preview what will be copied:

```bash
rsync -a --dry-run --itemize-changes "$SRC"/ "$DST"/
```

Copy the dataset:

```bash
mkdir -p "$(dirname "$DST")"
rsync -a --human-readable --info=progress2 --partial "$SRC"/ "$DST"/
```

Verify the copy with checksums:

```bash
rsync -a --checksum --dry-run --delete --itemize-changes "$SRC"/ "$DST"/
```

If the verification command prints no file changes, the copied directory matches the source.

Quick structural check:

```bash
test -f "$DST/meta/info.json"
test -f "$DST/meta/stats.json"
test -f "$DST/meta/tasks.parquet"
test -d "$DST/data"
find "$DST/data" -name "*.parquet" | head
```

Optional size/count checks:

```bash
du -sh "$SRC" "$DST"
find "$SRC" -type f | wc -l
find "$DST" -type f | wc -l
```

After the checksum verification passes, remove the original only if you really want to:

```bash
# Be careful: this deletes the original source directory.
# rm -rf "$SRC"
```

## Notes

- Keep the trailing slashes in `"$SRC"/ "$DST"/`. This copies the contents of the dataset directory into the destination dataset directory.
- Do not edit files inside `data/`, `meta/`, or `videos/` by hand unless you know the LeRobot dataset schema.
- If your dataset came from a Hugging Face cache snapshot with symlinks and you want a fully materialized copy, use `rsync -aL` instead of `rsync -a`. For normal locally recorded demos, `rsync -a` is the safer default.
- If a copy is interrupted, rerun the same `rsync` command. It will resume and correct the destination.

## Using A Copied Dataset

Point LeRobot tools at the copied dataset root:

```bash
DATASET_ROOT=/home/mlic/mingukang/lerobot/collected_demo/<your_dataset_name>
```

You can inspect it with:

```bash
python -m json.tool "$DATASET_ROOT/meta/info.json" | head -80
```
