"""Render keypoint-overlay / multi-cam MP4s for a few local episodes.

Reuses the embodiment dataset pipeline (keymap + transforms) and
`viz_transformed_batch`, which projects 3D world-frame keypoints into the
egocentric image using per-embodiment intrinsics. Outputs MP4 videos.

Usage:
    python egomimic/scripts/tutorials/save_keypoint_overlays.py \
        --local-dir /mnt/ddn/dataset/egoverse --out-dir /mnt/ddn/dataset/_viz_samples
"""

import argparse
import io
import os

import imageio
import imageio_ffmpeg  # noqa: F401  (ensures ffmpeg is available)
import numpy as np
import torch
import zarr
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from egomimic.rldb.embodiment.human import Aria, Mecka, Scale
from egomimic.rldb.filters import DatasetFilter
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset, S3EpisodeResolver
from egomimic.utils.aws.aws_data_utils import load_env

# Human episodes -> keypoint overlay video.
HUMAN_TARGETS = [
    (Aria, "2026-01-21-17-35-13-109000"),
    (Mecka, "69b1efd791d453bbb325f8bc"),
]
# Robot episode -> 3-camera montage video.
EVA_TARGET = "2026-01-11-18-15-27-885000"

KEYPOINT_TRANSFORM_MODE = "keypoints_headframe_ypr"
FPS = 30
MAX_FRAMES = 1200  # cap (~40s) to keep render time/file size reasonable


def _build_keymap(cls):
    try:
        key_map = cls.get_keymap(mode="keypoints")
    except TypeError:
        key_map = cls.get_keymap(keymap_mode="keypoints")
    # Some keypoint keymaps (e.g. Scale) omit obs_head_pose, which the
    # head-frame transform needs. Inject it so projection has the pivot pose.
    if "obs_head_pose" not in key_map:
        key_map["obs_head_pose"] = {
            "key_type": "proprio_keys",
            "zarr_key": "obs_head_pose",
        }
    return key_map


def render_keypoint_video(cls, episode_hash, local_dir, out_dir):
    name = cls.__name__.lower()
    key_map = _build_keymap(cls)
    transform_list = cls.get_transform_list(mode=KEYPOINT_TRANSFORM_MODE)
    resolver = S3EpisodeResolver(
        str(local_dir), key_map=key_map, transform_list=transform_list
    )
    filters = DatasetFilter(
        filter_lambdas=[f"lambda row: row['episode_hash'] == {episode_hash!r}"]
    )
    dataset = MultiDataset._from_resolver(
        resolver, filters=filters, sync_from_s3=True, mode="total"
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    out_path = os.path.join(out_dir, f"{name}_{episode_hash}_keypoints.mp4")
    writer = imageio.get_writer(out_path, fps=FPS, macro_block_size=None)
    n = 0
    for batch in loader:
        vis = cls.viz_transformed_batch(
            batch, mode="keypoints", viz_batch_key="actions_keypoints"
        )
        writer.append_data(np.asarray(vis))
        n += 1
        if n % 200 == 0:
            print(f"  [{name}] {n} frames")
        if n >= MAX_FRAMES:
            break
    writer.close()
    print(f"  [{name}] wrote {out_path} ({n} frames)")
    return out_path


def _unwrap(x):
    while not isinstance(x, (bytes, bytearray)):
        x = x.item() if hasattr(x, "item") else x[()]
    return bytes(x)


def render_eva_montage(episode_hash, local_dir, out_dir):
    g = zarr.open_group(os.path.join(local_dir, episode_hash), mode="r")
    cams = sorted(k for k in dict(g.arrays()) if k.startswith("images."))
    total = int(g.attrs.get("total_frames", g["images.front_1"].shape[0]))
    total = min(total, MAX_FRAMES)
    out_path = os.path.join(out_dir, f"eva_{episode_hash}_3cam.mp4")
    writer = imageio.get_writer(out_path, fps=FPS, macro_block_size=None)
    for f in range(total):
        tiles = []
        for cam in cams:
            im = Image.open(io.BytesIO(_unwrap(g[cam][f]))).convert("RGB")
            tiles.append((cam.split(".")[1], im))
        W = sum(im.width for _, im in tiles)
        H = max(im.height for _, im in tiles)
        canvas = Image.new("RGB", (W, H + 18), (20, 20, 20))
        d = ImageDraw.Draw(canvas)
        x = 0
        for label, im in tiles:
            canvas.paste(im, (x, 18))
            d.text((x + 4, 2), label, fill=(255, 255, 0))
            x += im.width
        writer.append_data(np.asarray(canvas))
        if f and f % 200 == 0:
            print(f"  [eva] {f} frames")
    writer.close()
    print(f"  [eva] wrote {out_path} ({total} frames)")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-dir", default="/mnt/ddn/dataset/egoverse")
    ap.add_argument("--out-dir", default="/mnt/ddn/dataset/_viz_samples")
    args = ap.parse_args()
    load_env()
    os.makedirs(args.out_dir, exist_ok=True)
    out = []
    for cls, h in HUMAN_TARGETS:
        try:
            out.append(render_keypoint_video(cls, h, args.local_dir, args.out_dir))
        except Exception as exc:
            print(f"  [{cls.__name__}] FAILED: {type(exc).__name__}: {exc}")
    try:
        out.append(render_eva_montage(EVA_TARGET, args.local_dir, args.out_dir))
    except Exception as exc:
        print(f"  [eva] FAILED: {type(exc).__name__}: {exc}")
    print(f"\nWrote {len(out)} MP4s to {args.out_dir}")


if __name__ == "__main__":
    main()
