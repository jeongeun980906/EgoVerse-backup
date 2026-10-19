"""EgoVerse human-data filtering + smoothing pipeline.

- filter: drop outlier episodes (should_drop_episode) -> manifest, non-destructive
- transform: copy episode, overwrite head/hand pose & keypoint arrays with smoothed
  values (images/annotations/timestamps copied verbatim).

Usage:
  python process_egoverse.py --scan                    # build drop manifest only
  python process_egoverse.py --process <hash> [<hash>...] --out <dir>   # clean specific eps
"""
import os, sys, json, shutil, argparse
import numpy as np, zarr
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, os.path.dirname(__file__))
from smooth_poses import (smooth_pose, smooth_keypoints, should_drop_episode,
                          episode_bad_frac, POSE_KEYS, KEYPOINT_KEYS)

SRC = "/mnt/ddn/dataset/egoverse"

# ---- head-frame transformation (express hands/keypoints relative to head) ----
def _safe_R(q_wxyz):             # wxyz (T,4) -> (rotmat (T,3,3), invalid mask)
    q = np.asarray(q_wxyz, dtype=np.float64).copy()
    norm = np.linalg.norm(q, axis=1)
    bad = ~np.isfinite(norm) | (norm < 1e-6)
    q[bad] = [1.0, 0.0, 0.0, 0.0]
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    Rm = R.from_quat(np.c_[q[:, 1], q[:, 2], q[:, 3], q[:, 0]]).as_matrix()
    return Rm, bad

def _pose_to_T(pose):            # (T,7) xyz+wxyz -> (T,4,4), invalid mask
    Rm, bad = _safe_R(pose[:, 3:7])
    M = np.tile(np.eye(4), (len(pose), 1, 1))
    M[:, :3, :3] = Rm; M[:, :3, 3] = pose[:, :3]
    return M, bad

def _T_to_pose(M):               # (T,4,4) -> (T,7) xyz+wxyz
    q = R.from_matrix(M[:, :3, :3]).as_quat()    # xyzw
    return np.c_[M[:, :3, 3], q[:, 3], q[:, 0], q[:, 1], q[:, 2]]

def headframe_pose(pose_world, head_world):      # both (T,7)
    Th, hbad = _pose_to_T(head_world)
    Tp, pbad = _pose_to_T(pose_world)
    out = _T_to_pose(np.linalg.inv(Th) @ Tp)
    out[hbad | pbad] = 0.0       # absent hand / padded frame -> zeros
    return out

def headframe_keypoints(kp_world, head_world):   # kp (T,63), head (T,7)
    n = len(kp_world); kp = kp_world.reshape(n, -1, 3)
    Rm, hbad = _safe_R(head_world[:, 3:7])
    rel = np.einsum('tij,tpj->tpi', np.transpose(Rm, (0, 2, 1)), kp - head_world[:, None, :3])
    kpbad = ~np.isfinite(kp).all(axis=(1, 2)) | (np.abs(kp) < 1e-9).all(axis=(1, 2))
    rel[hbad | kpbad] = 0.0
    return rel.reshape(n, -1)

HEADFRAME_POSE_KEYS = ["left.obs_ee_pose", "right.obs_ee_pose",
                       "left.obs_wrist_pose", "right.obs_wrist_pose"]
HEADFRAME_KP_KEYS = ["left.obs_keypoints", "right.obs_keypoints"]

def load_arrays(g):
    ks = dict(g.arrays()); n = int(g.attrs.get("total_frames", 0)) or None
    out = {}
    for k in POSE_KEYS + KEYPOINT_KEYS:
        if k in ks:
            out[k] = np.asarray(g[k][:]) if n is None else np.asarray(g[k][:n])
    return out

def process_episode_dir(h, out_dir, inplace=False):
    src = os.path.join(SRC, h)
    g = zarr.open_group(src, mode="r")
    arrays = load_arrays(g)
    if should_drop_episode(arrays):
        return "drop"
    if inplace:
        dst = src
    else:
        dst = os.path.join(out_dir, h)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)                   # verbatim copy (images etc.)
    gd = zarr.open_group(dst, mode="r+")
    if gd.attrs.get("egoverse_processed"):
        return "ok"
    ks = dict(gd.arrays())
    if "obs_head_pose" not in ks:
        return "ok"
    # per-embodiment smoothing strength: aria = strong(7,15,3 "green"), others = weak(5,9,2)
    emb = str(gd.attrs.get("embodiment", "")).split("_")[0]
    sp = dict(med_k=7, sg_win=15, sg_poly=3) if emb == "aria" else dict(med_k=5, sg_win=9, sg_poly=2)
    # SMOOTH first, then head-frame transform. head_pose smoothed in-memory only
    # (reference frame; NOT saved). Originals untouched.
    hp_s = smooth_pose(np.asarray(gd["obs_head_pose"][:]), **sp)
    def _add(name, data):                            # add NEW key (originals untouched)
        gd.create_array(name=name, shape=data.shape, dtype="float64",
                        chunks=(min(100, data.shape[0]),) + data.shape[1:],
                        overwrite=True)[:] = data
    for k in HEADFRAME_POSE_KEYS:                    # ee_pose, wrist_pose  (L/R)
        if k in ks:
            _add(k + "_transformed", headframe_pose(smooth_pose(np.asarray(gd[k][:]), **sp), hp_s))
    for k in HEADFRAME_KP_KEYS:                      # keypoints (L/R)
        if k in ks:
            _add(k + "_transformed", headframe_keypoints(smooth_keypoints(np.asarray(gd[k][:]), **sp), hp_s))
    gd.attrs["egoverse_processed"] = True            # idempotent marker (resume-safe)
    return "ok"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--inplace", action="store_true")
    ap.add_argument("--process", nargs="*")
    ap.add_argument("--out", default="/mnt/ddn/dataset/egoverse_clean")
    ap.add_argument("--manifest", default="/mnt/ddn/dataset/egoverse_drop_manifest.json")
    a = ap.parse_args()
    if a.scan:
        eps = sorted(d for d in os.listdir(SRC) if os.path.isdir(os.path.join(SRC, d)))
        drop = []; kept = 0
        for i, h in enumerate(eps):
            try:
                g = zarr.open_group(os.path.join(SRC, h), mode="r")
                if should_drop_episode(load_arrays(g)):
                    drop.append(h)
                else:
                    kept += 1
            except Exception:
                drop.append(h)
            if i % 2000 == 0:
                print(f"  scanned {i}/{len(eps)}  drop={len(drop)}", flush=True)
        json.dump({"drop": drop, "n_total": len(eps), "n_drop": len(drop), "n_kept": kept},
                  open(a.manifest, "w"))
        print(f"DROP MANIFEST: {len(drop)}/{len(eps)} dropped -> {a.manifest}")
    elif a.all:
        if not a.inplace:
            os.makedirs(a.out, exist_ok=True)
        eps = sorted(d for d in os.listdir(SRC) if os.path.isdir(os.path.join(SRC, d)))
        ok = drop = err = 0
        for i, h in enumerate(eps):
            if not a.inplace and os.path.isdir(os.path.join(a.out, h)):
                ok += 1; continue
            try:
                r = process_episode_dir(h, a.out, inplace=a.inplace)
                ok += (r == "ok"); drop += (r == "drop")
            except Exception as e:
                err += 1; print(f"  ERR {h}: {type(e).__name__}: {str(e)[:60]}", flush=True)
            if i % 500 == 0:
                print(f"  {i}/{len(eps)}  ok={ok} drop={drop} err={err}", flush=True)
        print(f"DONE: ok={ok} drop={drop} err={err}  inplace={a.inplace}")
    elif a.process:
        os.makedirs(a.out, exist_ok=True)
        for h in a.process:
            print(h, "->", process_episode_dir(h, a.out), flush=True)

if __name__ == "__main__":
    main()
