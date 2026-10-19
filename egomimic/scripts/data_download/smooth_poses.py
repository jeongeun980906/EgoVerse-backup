"""Smoothing filters for EgoVerse head/hand trajectories.

- positions / keypoints: median filter (spike removal) + Savitzky-Golay (de-jitter)
- pose rotation (quaternion wxyz): hemisphere-align, smooth components, renormalize

Pose arrays are XYZWXYZ (xyz + wxyz). Keypoints are (T, 63). All operate along time axis 0.
"""
import numpy as np
from scipy.signal import medfilt, savgol_filter


def reject_outliers_xyz(xyz, max_step=0.3):
    """Detect frames whose position jumps > max_step (m) from BOTH neighbors
    (nonphysical spikes) and replace by linear interpolation. xyz: (T,3).
    Returns (clean_xyz, n_replaced)."""
    T = len(xyz)
    if T < 3:
        return xyz, 0
    bad = np.zeros(T, bool)
    d_prev = np.linalg.norm(np.diff(xyz, axis=0), axis=1)  # step i-1 -> i
    # a frame is a spike if it jumps far from prev AND next jumps far back
    for i in range(1, T - 1):
        if d_prev[i - 1] > max_step and np.linalg.norm(xyz[i] - xyz[i + 1]) > max_step:
            bad[i] = True
    # also non-finite / placeholder
    bad |= ~np.isfinite(xyz).all(1)
    bad |= (np.abs(xyz) > 1e8).any(1)
    if bad.any():
        good = ~bad
        idx = np.arange(T)
        xyz = xyz.copy()
        for c in range(3):
            xyz[bad, c] = np.interp(idx[bad], idx[good], xyz[good, c])
    return xyz, int(bad.sum())


def _smooth_channels(x, med_k=7, sg_win=15, sg_poly=3):
    """Median filter then Savitzky-Golay along time, per channel. x: (T, C)."""
    T = x.shape[0]
    if T < 5:
        return x
    mk = min(med_k, T if T % 2 == 1 else T - 1)
    if mk % 2 == 0:
        mk -= 1
    out = np.empty_like(x)
    for c in range(x.shape[1]):
        v = medfilt(x[:, c], kernel_size=max(mk, 1))
        w = min(sg_win, T if T % 2 == 1 else T - 1)
        if w % 2 == 0:
            w -= 1
        if w >= sg_poly + 2:
            v = savgol_filter(v, w, sg_poly)
        out[:, c] = v
    return out


def _hemisphere_align(q):
    """Flip quaternion signs so consecutive quats are in the same hemisphere. q: (T,4) wxyz."""
    q = q.copy()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q


def smooth_pose(pose, med_k=7, sg_win=15, sg_poly=3):
    """Smooth an XYZWXYZ pose array (T,7): xyz position + wxyz quaternion."""
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape[0] < 5:
        return pose
    out = pose.copy()
    xyz, _ = reject_outliers_xyz(pose[:, :3])
    out[:, :3] = _smooth_channels(xyz, med_k, sg_win, sg_poly)
    q = _hemisphere_align(pose[:, 3:7])
    q = _smooth_channels(q, med_k, sg_win, sg_poly)
    q /= (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    out[:, 3:7] = q
    return out


def smooth_keypoints(kp, med_k=7, sg_win=15, sg_poly=3):
    """Smooth a (T, 63) keypoints array (21 x 3)."""
    kp = np.asarray(kp, dtype=np.float64)
    return _smooth_channels(kp, med_k, sg_win, sg_poly) if kp.shape[0] >= 5 else kp


# keys to smooth in an episode
POSE_KEYS = ["obs_head_pose",
             "left.obs_ee_pose", "right.obs_ee_pose",
             "left.obs_wrist_pose", "right.obs_wrist_pose"]
KEYPOINT_KEYS = ["left.obs_keypoints", "right.obs_keypoints",
                 "left.obs_aria_keypoints", "right.obs_aria_keypoints"]


def smooth_episode(arrays: dict, **kw) -> dict:
    """Return a new dict with head/hand pose & keypoint arrays smoothed."""
    out = dict(arrays)
    for k in POSE_KEYS:
        if k in arrays:
            out[k] = smooth_pose(arrays[k], **kw)
    for k in KEYPOINT_KEYS:
        if k in arrays:
            out[k] = smooth_keypoints(arrays[k], **kw)
    return out


# ---- outlier-episode drop filter -------------------------------------------

def episode_bad_frac(arrays: dict, max_dist: float = 2.0) -> float:
    """Fraction of frames where head<->hand distance exceeds max_dist (m), or
    where head/hand poses are non-finite / placeholder. Proxy for a corrupt
    episode (e.g. inconsistent SLAM origin). Returns 0.0 if head pose missing."""
    hp = arrays.get("obs_head_pose")
    if hp is None:
        return 0.0
    hp = np.asarray(hp, dtype=np.float64)
    n = len(hp)
    if n == 0:
        return 0.0
    hpos = hp[:, :3]
    bad = ~np.isfinite(hp).all(1) | (np.abs(hp) > 1e8).any(1)
    for s in ("left", "right"):
        a = arrays.get(f"{s}.obs_ee_pose")
        if a is None:
            continue
        a = np.asarray(a, dtype=np.float64)[:, :3]
        d = np.linalg.norm(a - hpos, axis=1)
        present = (a != 0).any(1) & np.isfinite(d) & np.isfinite(hpos).all(1)
        bad |= present & (d > max_dist)
    return float(bad.mean())


def should_drop_episode(arrays: dict, max_dist: float = 2.0,
                        max_bad_frac: float = 0.01) -> bool:
    """Drop episodes whose corrupt-frame fraction exceeds max_bad_frac.
    (Rare single spikes are tolerated and handled by smoothing; sustained
    bad segments => drop. ~0.5-0.8% of episodes, scale only.)"""
    return episode_bad_frac(arrays, max_dist) > max_bad_frac


def process_episode(arrays: dict, drop: bool = True, smooth: bool = True,
                    max_dist: float = 2.0, max_bad_frac: float = 0.01, **kw):
    """Conversion-pipeline entry point.
    Returns smoothed arrays, or None if the episode should be dropped."""
    if drop and should_drop_episode(arrays, max_dist, max_bad_frac):
        return None
    return smooth_episode(arrays, **kw) if smooth else dict(arrays)
