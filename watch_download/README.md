# watch_download: EgoVerse episodes for WATCH training

The step-by-step guide (download, assembly, randref mid-training and post-training) is
**`docs/egoverse_randref_from_scratch.md`** in the `jeongeun980906/watch` repository (also in `jeongeun980906/cosmos_policy`).
This directory holds the data-side tools it uses:

| file | purpose |
|---|---|
| `all_episodes.csv` | the 13,449 human v10 episodes (MECKA 6,718 / scale 6,074 / aria 657), columns `ep, emb, cls` |
| `fetch_raw.py` | resolves each episode's zarr path in the EgoVerse SQL table and downloads it from R2 (`--full` includes video, ~3.9 TB; `--arrays images.front_1 --repack` keeps it to a few files per episode) |
| `repack.py` | rewrites a one-file-per-frame `images.front_1` into one shard (same bytes, fewer files); used by `fetch_raw.py --repack`, or standalone with `--verify` |
| `assemble.py` | links the v10 `state57`/`action57`/`meta` (HF `Jeongeun/egoverse-v10-state`) into each video episode and checks that every episode a run reads is complete |

Quick reference (details and expected outputs are in the guide):

```bash
venv/bin/python watch_download/fetch_raw.py --list watch_download/all_episodes.csv --dry-run   # all must resolve
venv/bin/python watch_download/fetch_raw.py --list watch_download/all_episodes.csv --full \
    --arrays images.front_1 --repack --out <video root> --workers 128          # ~0.05M files instead of ~16M
python3 watch_download/assemble.py --video-root <video root> --state-root <egoverse-v10-state>/episodes \
    --manifest <egoverse-v10-state>/manifest --link --check
```

## Why `--arrays images.front_1 --repack`

About 40% of the scale episodes store the video one file per frame (~6,300 files) and their tracking arrays
in ~450 more files. That is ~14-17M files for the full set. Training reads only `images.front_1` from R2
(`state57`/`action57` come from HF), so `--arrays images.front_1` skips the rest. `--repack` concatenates the
per-frame files into one shard with a trailing index, the layout aria episodes already use. Each per-frame
file is already an encoded inner chunk of that layout, so nothing is re-encoded. On a 15,919-frame episode the
training reader and zarr-python return byte-identical JPEGs before and after, and the episode goes from 15,920
files to 2. Re-runs resume and skip repacked episodes. To repack video downloaded earlier:

```bash
python watch_download/repack.py <video root>/<episode> ... --verify
```
