"""Fetch the RAW (pre-smoothing) pose/keypoint arrays of the sampled episodes from the EgoVerse R2 bucket.

Only the small tracking arrays are downloaded (no images):
    left/right.obs_ee_pose, left/right.obs_keypoints, left/right.obs_wrist_pose, obs_head_pose, zarr.json
-> ~a few MB per episode instead of ~352 MB.

Prerequisites (run by the user, credentials are never printed here):
    aws configure                                  # keys from EgoVerse-backup/README.md
    EgoVerse-backup/egomimic/utils/aws/setup_secret.sh   # writes ~/.egoverse_env (R2 + SECRETS_ARN)

Usage:
    python fetch_raw.py --dry-run        # DB lookup only: how many sampled episodes resolve to a zarr path
    python fetch_raw.py                  # download into ./raw/<episode>/
"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
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
        for prefix in ([key] if a.full else [key + "zarr.json"] + [key + arr + "/" for arr in ARRAYS]):
            for page in r2.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    f = dst / obj["Key"][len(key):]
                    f.parent.mkdir(parents=True, exist_ok=True)
                    if not f.exists():
                        r2.download_file(bucket, obj["Key"], str(f)); n += 1
        return row.ep, n

    rows = list(hit[ok].itertuples())
    with ThreadPoolExecutor(a.workers) as ex:
        for i, (ep, n) in enumerate(ex.map(fetch, rows)):
            if i % 500 == 0:
                print(f"  {i}/{len(rows)} {ep}: {n} objects", flush=True)
    print("done ->", a.out)


if __name__ == "__main__":
    main()
