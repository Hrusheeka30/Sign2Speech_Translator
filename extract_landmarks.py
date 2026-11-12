"""
Extract hand landmarks from every video listed in manifest.csv using MediaPipe,
turn each clip into a fixed-length landmark sequence, and save the features.

Output:
    features/<category>/<class_name>/<clip>.npy   -> array of shape (30, 63)
    data/landmarks.npz                            -> X, y, split, paths

Each frame's 21 hand landmarks (x, y, z) are normalised to be translation- and
scale-invariant (centred on the wrist, scaled by the largest landmark distance),
then every clip is resampled to a fixed number of frames (default 30) so the
LSTM always receives the same input length.

Extraction is parallelised across processes (--workers) because a single
process spends most of its time decoding video / waiting on disk.

Usage:
    python extract_landmarks.py --limit 20     # quick smoke test
    python extract_landmarks.py                # full run, auto worker count
    python extract_landmarks.py --workers 8
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from hand_detector import HandDetector

FIXED_FRAMES = 30
NUM_LANDMARKS = 21
FEATURES_PER_FRAME = NUM_LANDMARKS * 3  # 63

# one detector per worker process (MediaPipe objects are not picklable)
_WORKER = {}


def normalize_landmarks(seq):
    """seq: (T,21,3). Centre on wrist, scale by max distance. Returns (T,63)."""
    out = np.zeros((seq.shape[0], FEATURES_PER_FRAME), dtype=np.float32)
    for t in range(seq.shape[0]):
        pts = seq[t]
        if not pts.any():
            continue
        pts = pts - pts[0]                       # centre on wrist
        scale = np.max(np.linalg.norm(pts, axis=1))
        if scale > 1e-6:
            pts = pts / scale
        out[t] = pts.reshape(-1)
    return out


def resample(seq, target=FIXED_FRAMES):
    """seq: (T,63) -> (target,63) via linear interpolation along time."""
    T = seq.shape[0]
    if T == target:
        return seq
    if T == 0:
        return np.zeros((target, FEATURES_PER_FRAME), dtype=np.float32)
    src = np.linspace(0, 1, T)
    dst = np.linspace(0, 1, target)
    out = np.empty((target, seq.shape[1]), dtype=np.float32)
    for f in range(seq.shape[1]):
        out[:, f] = np.interp(dst, src, seq[:, f])
    return out


def fill_missing(seq, valid):
    """Linearly interpolate frames where the hand was not detected.

    seq: (T,21,3)  valid: (T,) bool. Frames with no detection are replaced by
    interpolating their neighbours, which is far better than feeding the LSTM a
    hand sitting at the origin.
    """
    if valid.all() or not valid.any():
        return seq
    T = seq.shape[0]
    idx = np.arange(T)
    good = idx[valid]
    flat = seq.reshape(T, -1)
    for f in range(flat.shape[1]):
        flat[:, f] = np.interp(idx, good, flat[good, f])
    return flat.reshape(seq.shape)


def extract_clip(path, detector, max_frames=0):
    cap = cv2.VideoCapture(path)
    lms, valid = [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        lm = detector.landmarks(frame)
        if lm is None:
            lms.append(np.zeros((NUM_LANDMARKS, 3), np.float32))
            valid.append(False)
        else:
            lms.append(lm)
            valid.append(True)
        if max_frames and len(lms) >= max_frames:
            break
    cap.release()
    if not lms:
        return None, 0.0
    seq = np.stack(lms)                          # (T,21,3)
    valid = np.asarray(valid)
    rate = float(valid.mean())
    seq = fill_missing(seq, valid)
    seq = normalize_landmarks(seq)
    seq = resample(seq)
    return seq, rate


def _init_worker():
    _WORKER["det"] = HandDetector(mode="image")


def _process_one(job):
    """Worker entry point: extract one clip and save its .npy. Returns (index, rel_path, seq or None, rate)."""
    index, rel_path, root, features_dir, max_frames = job
    full = os.path.join(root, rel_path)
    seq, rate = extract_clip(full, _WORKER["det"], max_frames)
    if seq is not None:
        parts = rel_path.split(os.sep)
        category = parts[0] if len(parts) > 2 else "(root)"
        class_name = parts[1] if len(parts) > 2 else parts[-2]
        out_dir = os.path.join(features_dir, category, class_name)
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, os.path.splitext(os.path.basename(rel_path))[0] + ".npy"), seq)
    return index, rel_path, seq, rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="signlang_dataset")
    ap.add_argument("--manifest", default="manifest.csv")
    ap.add_argument("--features-dir", default="features")
    ap.add_argument("--out", default=os.path.join("data", "landmarks.npz"))
    ap.add_argument("--limit", type=int, default=0, help="only first N clips (smoke test)")
    ap.add_argument("--max-frames", type=int, default=0, help="cap frames per clip")
    ap.add_argument("--workers", type=int, default=min(8, (os.cpu_count() or 4) - 1))
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_csv(args.manifest)
    if args.limit:
        df = df.head(args.limit)

    jobs = [(i, row["rel_path"], args.root, args.features_dir, args.max_frames)
            for i, row in df.iterrows()]

    results = [None] * len(df)
    det_rates, failed = [], []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as ex:
        done = 0
        for index, rel_path, seq, rate in ex.map(_process_one, jobs, chunksize=1):
            done += 1
            if seq is None:
                failed.append(rel_path)
            else:
                results[index] = seq
                det_rates.append(rate)
            if done % 25 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} processed  (mean hand-detection {np.mean(det_rates):.1%})",
                      flush=True)

    # keep only successfully extracted clips, preserving manifest order
    keep = [i for i in range(len(df)) if results[i] is not None]
    X = np.stack([results[i] for i in keep]).astype(np.float32)
    y = np.array([int(df.loc[i, "label_id"]) for i in keep], dtype=np.int64)
    split = np.array([df.loc[i, "split"] for i in keep])
    paths = np.array([df.loc[i, "rel_path"] for i in keep])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y, split=split, paths=paths)

    print(f"\nSaved {len(X)} clips -> {args.out}")
    print(f"X shape: {X.shape}")
    print(f"Mean hand-detection rate: {np.mean(det_rates):.1%}")
    low = int((np.array(det_rates) < 0.3).sum())
    print(f"Clips with <30% hand detection: {low}")
    if failed:
        print(f"Failed to read: {len(failed)} clips -> first few: {failed[:5]}")


if __name__ == "__main__":
    main()
