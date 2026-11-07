"""Repack a one-file-per-frame zarr v3 image array into a single shard (fewer files, same bytes).

Some scale episodes store `images.front_1` as one file per frame (`c/0` ... `c/T-1`), which puts thousands of
files in every such episode. Each of those files already holds exactly one encoded inner chunk of the
`sharding_indexed` layout that the other episodes use (vlen-bytes -> zstd), so repacking only concatenates
them, appends the shard index ((offset, nbytes) little-endian u64 per frame, index_location=end), and rewrites
`zarr.json`. Nothing is re-encoded. A missing frame file becomes an empty inner chunk (the fill value), which
is what the reader returned for it before.

usage:
  repack.py <episode_dir> [<episode_dir> ...] [--array images.front_1] [--verify] [--keep]
  --verify : decode every frame before and after with the training reader logic and compare the bytes
  --keep   : keep the per-frame files (default: delete them after a successful repack)
Episodes whose array is already sharded are skipped.
"""
import argparse
import json
import os
import shutil
import struct
import sys

EMPTY = (1 << 64) - 1


def is_per_frame(meta):
    return meta["codecs"][0]["name"] != "sharding_indexed" and int(meta["chunk_grid"]["configuration"]["chunk_shape"][0]) == 1


def is_repacked(arr_dir):
    """True if arr_dir holds a single-shard array (repacked here, or sharded at the source)."""
    z = os.path.join(str(arr_dir), "zarr.json")
    if not os.path.exists(z) or not os.path.exists(os.path.join(str(arr_dir), "c", "0")):
        return False
    m = json.load(open(z))
    return m["codecs"][0]["name"] == "sharding_indexed" and \
        int(m["chunk_grid"]["configuration"]["chunk_shape"][0]) >= int(m["shape"][0])


def repack(arr_dir, keep=False):
    meta = json.load(open(os.path.join(arr_dir, "zarr.json")))
    if not is_per_frame(meta):
        return "skip (not one-file-per-frame)"
    n = int(meta["shape"][0])
    cdir = os.path.join(arr_dir, "c")
    tmp = os.path.join(arr_dir, "c_repack.tmp")
    idx = []
    with open(tmp, "wb") as out:
        off = 0
        for i in range(n):
            p = os.path.join(cdir, str(i))
            if os.path.exists(p):
                b = open(p, "rb").read()
                out.write(b); idx.append((off, len(b))); off += len(b)
            else:
                idx.append((EMPTY, EMPTY))
        out.write(b"".join(struct.pack("<QQ", o, l) for o, l in idx))
    inner_codecs = meta["codecs"]                       # [vlen-bytes, zstd] of the per-frame layout
    new = dict(meta)
    new["chunk_grid"] = {"name": "regular", "configuration": {"chunk_shape": [n]}}
    new["codecs"] = [{"name": "sharding_indexed", "configuration": {
        "chunk_shape": [1], "codecs": inner_codecs,
        "index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "index_location": "end"}}]
    # swap in atomically-ish: new shard as c/0 in a fresh dir, then metadata
    newc = os.path.join(arr_dir, "c_new")
    os.makedirs(newc, exist_ok=True)
    os.replace(tmp, os.path.join(newc, "0"))
    json.dump(new, open(os.path.join(arr_dir, "zarr.json.new"), "w"), indent=2)
    old = os.path.join(arr_dir, "c_old")
    os.replace(cdir, old)
    os.replace(newc, cdir)
    os.replace(os.path.join(arr_dir, "zarr.json.new"), os.path.join(arr_dir, "zarr.json"))
    if not keep:
        shutil.rmtree(old)
    return f"repacked {n} frames ({sum(1 for o, _ in idx if o != EMPTY)} present) -> 1 file"


def read_all(arr_dir):
    """Frame bytes as the training reader (cosmos_policy/datasets/egoverse_zread.py) sees them, still encoded."""
    meta = json.load(open(os.path.join(arr_dir, "zarr.json")))
    n = int(meta["shape"][0]); cdir = os.path.join(arr_dir, "c")
    if not meta["codecs"][0]["name"] == "sharding_indexed":
        return [open(os.path.join(cdir, str(i)), "rb").read() if os.path.exists(os.path.join(cdir, str(i))) else b""
                for i in range(n)]
    cfg = meta["codecs"][0]["configuration"]; ocs = int(meta["chunk_grid"]["configuration"]["chunk_shape"][0])
    crc = any(c["name"] == "crc32c" for c in cfg.get("index_codecs", []))
    out = []
    for s in range((n + ocs - 1) // ocs):
        p = os.path.join(cdir, str(s)); k = ocs // int(cfg["chunk_shape"][0])
        if not os.path.exists(p):
            out += [b""] * k; continue
        data = open(p, "rb").read(); ib = k * 16 + (4 if crc else 0)
        raw = data[len(data) - ib:len(data) - (4 if crc else 0)]
        for j in range(k):
            o, l = struct.unpack_from("<QQ", raw, 16 * j)
            out.append(b"" if o == EMPTY or l == 0 else data[o:o + l])
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes", nargs="+"); ap.add_argument("--array", default="images.front_1")
    ap.add_argument("--verify", action="store_true"); ap.add_argument("--keep", action="store_true")
    a = ap.parse_args(); bad = 0
    for e in a.episodes:
        d = os.path.join(e, a.array)
        if not os.path.exists(os.path.join(d, "zarr.json")):
            print(f"{e}: no {a.array}"); bad += 1; continue
        before = read_all(d) if a.verify else None
        msg = repack(d, keep=a.keep)
        if a.verify and not msg.startswith("skip"):
            same = before == read_all(d)
            msg += f", verify {'OK' if same else 'MISMATCH'}"; bad += not same
        print(f"{e}: {msg}", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
