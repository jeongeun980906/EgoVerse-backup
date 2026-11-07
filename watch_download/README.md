# watch_download: EgoVerse episodes for WATCH training

The step-by-step guide (download, assembly, randref mid-training and post-training) is
**`cosmos_policy/docs/egoverse_randref_from_scratch.md`** in the `jeongeun980906/cosmos_policy` repository.
This directory holds the data-side tools it uses:

| file | purpose |
|---|---|
| `all_episodes.csv` | the 13,449 human v10 episodes (MECKA 6,718 / scale 6,074 / aria 657), columns `ep, emb, cls` |
| `fetch_raw.py` | resolves each episode's zarr path in the EgoVerse SQL table and downloads it from R2 (`--full` includes video, ~3.9 TB) |
| `assemble.py` | links the v10 `state57`/`action57`/`meta` (HF `Jeongeun/egoverse-v10-state`) into each video episode and checks that every episode a run reads is complete |

Quick reference (details and expected outputs are in the guide):

```bash
venv/bin/python watch_download/fetch_raw.py --list watch_download/all_episodes.csv --dry-run   # all must resolve
venv/bin/python watch_download/fetch_raw.py --list watch_download/all_episodes.csv --full --out <video root> --workers 128
python3 watch_download/assemble.py --video-root <video root> --state-root <egoverse-v10-state>/episodes \
    --manifest <egoverse-v10-state>/manifest --link --check
```
