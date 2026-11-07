# WATCH: download the EgoVerse human episodes used for pre-training

`all_episodes.csv`: the 13,449 human training episodes (MECKA 6,718 / scale 6,074 / aria 657), columns `ep, emb, cls`.
`fetch_raw.py`: resolves each episode's current zarr path from the EgoVerse SQL table and downloads it from R2.

## Steps

```bash
git clone -b watch-download https://github.com/jeongeun980906/EgoVerse-backup.git
cd EgoVerse-backup

# AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip
unzip awscliv2.zip && ./aws/install -i ~/aws-cli -b ~/bin

# public EgoVerse keys from ../README.md ("AWS Configure"), region us-east-2
~/bin/aws configure
PATH=~/bin:$PATH ./egomimic/utils/aws/setup_secret.sh      # writes ~/.egoverse_env (R2 + DB, read-only)

# minimal Python env (3.11+)
python3.11 -m venv venv
venv/bin/pip install pandas boto3 sqlalchemy "psycopg[binary]" cloudpathlib

# 1) check: all 13,449 should resolve, "unresolved: {}"
venv/bin/python watch_download/fetch_raw.py --list watch_download/all_episodes.csv --dry-run

# 2) full download incl. video (~3.8 TB, ~0.28 GB per episode)
PATH=~/bin:$PATH nohup venv/bin/python watch_download/fetch_raw.py --list watch_download/all_episodes.csv \
    --full --out <dest> --workers 128 > fetch_all.log 2>&1 &
```

- Re-running skips files that already exist; boto3 writes each file atomically, so an interrupted run leaves no partial files.
- Without `--full` only the raw tracking arrays are fetched (a few MB per episode).
- The log prints one line per 500 episodes; use `du -sh <dest>` for progress.
- Video (`images.front_1`) is either one chunk per frame or a single shard `c/0`; an episode is complete when
  `ceil(shape / chunk_shape)` chunk files exist.
