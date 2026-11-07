"""Assemble the WATCH training episode root and check that it is complete.

The training loader (cosmos_policy EgoZarrEpisode) opens ONE directory per episode and reads, side by side,
    images.front_1/            <- video, downloaded from the EgoVerse R2 bucket by fetch_raw.py --full
    state57/{state,valid}      <- from the HF dataset Jeongeun/egoverse-v10-state
    action57/{action,valid}    <- from the HF dataset Jeongeun/egoverse-v10-state
so the v10-state arrays have to be placed inside each downloaded video episode. This script symlinks them in
(state57, action57, meta) and then checks every episode the training run will read.

The reader does NOT fail on missing video: a missing chunk/shard file is returned as an empty frame. An
incomplete download therefore trains silently on blank frames, which is why --check is mandatory before launch.

usage:
  python assemble.py --video-root <dest of fetch_raw.py> --state-root <egoverse-v10-state>/episodes \
                     --manifest <manifest dir> [--pairs pairs_randref.npz] [--query-set query_eps_nomd.json] \
                     [--link] [--check] [--workers 32]

  --link   create the state57/action57/meta symlinks (idempotent; existing entries are left alone)
  --check  verify, for every query and reference episode of the training rows:
             * images.front_1/zarr.json exists and all ceil(shape/chunk_shape) chunk files are present
             * state57/state and action57/action exist (directly or via the symlink)
             * the video has at least as many frames as state57 has rows
           exits 1 and writes <video-root>/assemble_missing.txt if anything is wrong.
"""
import argparse
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

LINKED = ("state57", "action57", "meta")


def needed_episodes(manifest: str, pairs: str, query_set: str | None) -> list[str]:
    """Query + reference episodes of the rows the training run keeps (same rule as EgoVerseRetDataset)."""
    W = np.load(os.path.join(manifest, "windows.npz"), allow_pickle=True)
    ep = W["ep"].astype(str)
    P = np.load(os.path.join(manifest, pairs))
    q, m = P["query_idx"], P["match_idx"]
    keep = np.ones(len(q), bool)
    if query_set:
        want = set(json.load(open(os.path.join(manifest, query_set)))["episodes"])
        keep = np.isin(ep[q], np.asarray(sorted(want)))
    refs = m[keep]
    refs = refs[refs >= 0]
    return sorted(set(ep[q[keep]].tolist()) | set(ep[refs].tolist()))


def link(ep: str, video_root: str, state_root: str) -> tuple[int, bool]:
    d = os.path.join(video_root, ep)
    if not os.path.isdir(d):
        return 0, False
    n = 0
    for sub in LINKED:
        src = os.path.join(state_root, ep, sub)
        dst = os.path.join(d, sub)
        if os.path.exists(src) and not os.path.lexists(dst):
            os.symlink(os.path.abspath(src), dst)
            n += 1
    return n, True


def rows(array_dir: str) -> int:
    return int(json.load(open(os.path.join(array_dir, "zarr.json")))["shape"][0])


def check(ep: str, video_root: str) -> str | None:
    d = os.path.join(video_root, ep)
    img = os.path.join(d, "images.front_1")
    if not os.path.isdir(d):
        return "episode directory missing"
    if not os.path.exists(os.path.join(img, "zarr.json")):
        return "images.front_1/zarr.json missing"
    meta = json.load(open(os.path.join(img, "zarr.json")))
    T = int(meta["shape"][0])
    ocs = int(meta["chunk_grid"]["configuration"]["chunk_shape"][0])
    have = set(os.listdir(os.path.join(img, "c"))) if os.path.isdir(os.path.join(img, "c")) else set()
    want = {str(i) for i in range(math.ceil(T / ocs))}
    if want - have:
        return f"video incomplete: {len(want - have)}/{len(want)} chunk files missing"
    for g, a in (("state57", "state"), ("action57", "action")):
        if not os.path.exists(os.path.join(d, g, a, "zarr.json")):
            return f"{g}/{a} missing (run --link, or the episode is absent from egoverse-v10-state)"
    if T < rows(os.path.join(d, "state57", "state")):
        return f"video has {T} frames < state57 rows {rows(os.path.join(d, 'state57', 'state'))}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--state-root", required=True, help="<egoverse-v10-state>/episodes")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pairs", default="pairs_randref.npz")
    ap.add_argument("--query-set", default="query_eps_nomd.json", help="'' = all query episodes")
    ap.add_argument("--link", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    if not (a.link or a.check):
        ap.error("pass --link and/or --check")

    eps = needed_episodes(a.manifest, a.pairs, a.query_set or None)
    print(f"{len(eps)} episodes used by {a.pairs}" + (f" restricted to {a.query_set}" if a.query_set else ""))

    with ThreadPoolExecutor(a.workers) as ex:
        if a.link:
            res = list(ex.map(lambda e: link(e, a.video_root, a.state_root), eps))
            print(f"link: {sum(n for n, _ in res)} symlinks created; "
                  f"{sum(not ok for _, ok in res)} episodes have no video directory yet")
        if a.check:
            bad = [(e, r) for e, r in zip(eps, ex.map(lambda e: check(e, a.video_root), eps)) if r]
            if bad:
                out = os.path.join(a.video_root, "assemble_missing.txt")
                with open(out, "w") as fp:
                    fp.writelines(f"{e}\t{r}\n" for e, r in bad)
                kinds = {}
                for _, r in bad:
                    kinds[r.split(":")[0]] = kinds.get(r.split(":")[0], 0) + 1
                print(f"check: FAILED for {len(bad)}/{len(eps)} episodes {kinds} -> {out}")
                sys.exit(1)
            print(f"check: OK, all {len(eps)} episodes complete (video chunks, state57, action57)")


if __name__ == "__main__":
    main()
