"""Download EgoVerse episodes (zarr v3) from the EgoVerse R2 bucket, resolving each episode's current path
through the EgoVerse SQL episode table.

Two modes:
  --full   the WHOLE episode, including the video (images.front_1): ~0.28 GB per episode on average,
           ~3.9 TB for watch_download/all_episodes.csv. This is what WATCH training needs.
  default  only the small raw tracking arrays (a few MB per episode):
           left/right.obs_ee_pose, left/right.obs_keypoints, left/right.obs_wrist_pose, obs_head_pose, zarr.json

The R2 episodes do NOT contain state57/action57 (WATCH's 57-d state/action). Those come from the HF dataset
Jeongeun/egoverse-v10-state and are linked into each episode by assemble.py (see README.md).

Prerequisites (run by the user, credentials are never printed here):
    aws configure                                  # keys from EgoVerse-backup/README.md
    EgoVerse-backup/egomimic/utils/aws/setup_secret.sh   # writes ~/.egoverse_env (R2 + SECRETS_ARN)

Usage:
    python fetch_raw.py --list all_episodes.csv --dry-run          # DB lookup only, no download
    python fetch_raw.py --list all_episodes.csv --full --out DEST  # download into DEST/<episode>/
    python fetch_raw.py --list all_episodes.csv --full --arrays images.front_1 --repack --out DEST
        # training-only: video only, one-file-per-frame episodes repacked into one shard (~50K files in total
        # instead of ~16M); re-running resumes and skips repacked episodes
"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import repack as RP  # noqa: E402
from pathlib import Path

import pandas as pd

EGOVERSE = Path(__file__).resolve().parent.parent  # repo root (EgoVerse-backup)
HERE = Path(__file__).resolve().parent
ARRAYS = ["left.obs_ee_pose", "right.obs_ee_pose", "left.obs_keypoints", "right.obs_keypoints",
          "left.obs_wrist_pose", "right.obs_wrist_pose", "obs_head_pose"]




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(HERE / "raw"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--list", default=str(HERE / "sample_episodes.csv"))
    ap.add_argument("--full", action="store_true", help="download the WHOLE episode (incl. images)")
    ap.add_argument("--resolved", default=None, help="output csv of resolved paths")
    ap.add_argument("--arrays", default="",
                    help="with --full: comma-separated arrays to fetch instead of the whole episode, e.g. "
                         "images.front_1 (all WATCH training reads from R2; state57/action57 come from HF)")
    ap.add_argument("--inner-workers", type=int, default=16,
                    help="parallel downloads inside one episode with many small objects (per-frame video)")
    ap.add_argument("--repack", action="store_true",
                    help="with --full: repack a one-file-per-frame images.front_1 into a single shard right after "
                         "it is downloaded (repack.py). Cuts ~6,000 files per affected scale episode to 1")
    a = ap.parse_args()

    sys.path.insert(0, str(EGOVERSE))
    from egomimic.utils.aws import aws_data_utils as adu, aws_sql as sql
    adu.load_env()  # ~/.egoverse_env -> os.environ (values are not printed)
    engine = sql.create_default_engine()
    db = sql.episode_table_to_df(engine)
    print("DB rows:", len(db), "| columns:", list(db.columns))

    s = pd.read_csv(a.list, dtype=str).fillna("")
    if "scale_task_id" not in s: s["scale_task_id"] = ""
    db["episode_hash"] = db["episode_hash"].astype(str)
    hit = s.merge(db[["episode_hash", "zarr_processed_path"]], left_on="ep", right_on="episode_hash", how="left")
    # Scale was re-hashed on 2026-05-03: fall back to scale_task_id if the DB carries it
    miss = hit.zarr_processed_path.isna() & (hit.emb == "scale")
    tid_col = next((c for c in db.columns if "scale_task_id" in c or c == "task_id"), None)
    if miss.any() and tid_col:
        m2 = db[[tid_col, "zarr_processed_path"]].dropna().drop_duplicates(tid_col).set_index(tid_col).zarr_processed_path
        hit.loc[miss, "zarr_processed_path"] = hit.loc[miss, "scale_task_id"].map(m2)
    ok = hit.zarr_processed_path.fillna("").str.strip() != ""
    print("resolved by source:", hit[ok].groupby("emb").size().to_dict(),
          "| unresolved:", hit[~ok].groupby("emb").size().to_dict(),
          "| scale fallback column:", tid_col)
    hit.to_csv(a.resolved or (HERE / (Path(a.list).stem + "_resolved.csv")), index=False)
    if a.dry_run:
        return

    import boto3
    r2 = boto3.client("s3", endpoint_url=os.environ["R2_ENDPOINT_URL"],
                      aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
                      aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"], region_name="auto")

    def fetch(row):
        p = row.zarr_processed_path
        bucket, key = (p[5:].split("/", 1) if p.startswith("s3://") else ("rldb", p.lstrip("/")))
        key = key.rstrip("/") + "/"
        dst = Path(a.out) / row.ep
        n = 0
        if a.full:
            arrs = [x for x in a.arrays.split(",") if x]
            prefixes = [key + "zarr.json"] + [key + x + "/" for x in arrs] if arrs else [key]
        else:
            prefixes = [key + "zarr.json"] + [key + arr + "/" for arr in ARRAYS]
        listed = {}
        for prefix in prefixes:
            if a.repack and prefix.endswith("images.front_1/") and RP.is_repacked(dst / "images.front_1"):
                continue                      # already repacked: the per-frame keys are intentionally absent
            todo = []
            for page in r2.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    rel = obj["Key"][len(key):]
                    if rel.startswith("images.front_1/c/"):
                        listed["images.front_1"] = listed.get("images.front_1", 0) + 1
                    f = dst / rel
                    if not f.exists():
                        todo.append((obj["Key"], f))
            for d in {f.parent for _, f in todo}:
                d.mkdir(parents=True, exist_ok=True)
            if len(todo) > 64:   # one-file-per-frame arrays: thousands of small objects, fetch them in parallel
                with ThreadPoolExecutor(a.inner_workers) as inner:
                    list(inner.map(lambda kf: r2.download_file(bucket, kf[0], str(kf[1])), todo))
            else:
                for k, f in todo:
                    r2.download_file(bucket, k, str(f))
            n += len(todo)
        if a.repack and (dst / "images.front_1" / "zarr.json").exists() and not RP.is_repacked(dst / "images.front_1"):
            # repack only a COMPLETE download: every chunk object listed on R2 must be present locally
            have = len(os.listdir(dst / "images.front_1" / "c")) if (dst / "images.front_1" / "c").is_dir() else 0
            if have < listed.get("images.front_1", 0):
                raise RuntimeError(f"{row.ep}: {have} of {listed['images.front_1']} video chunks present, not repacking")
            RP.repack(str(dst / "images.front_1"), listed=listed.get("images.front_1"))
        return row.ep, n

    rows = list(hit[ok].itertuples())
    with ThreadPoolExecutor(a.workers) as ex:
        for i, (ep, n) in enumerate(ex.map(fetch, rows)):
            if i % 500 == 0:
                print(f"  {i}/{len(rows)} {ep}: {n} objects", flush=True)
    print("done ->", a.out)


if __name__ == "__main__":
    main()
