"""
Extract MediaPipe hand landmarks from the ISL alphabet image dataset so the
letters can be RECOGNISED from the live camera (not just displayed).

The word model (29 signs) is trained on 30-frame landmark sequences from the
INCLUDE videos. Letters are static hand poses, so here we extract a single
normalised 63-value landmark vector per image (21 landmarks x, y, z), using
exactly the same normalisation as the word pipeline so both models see features
in the same space.

Output:
    data/letters.npz   -> X (N,63), y (N,), letters (list of 26 class names)

Usage:
    python extract_letter_landmarks.py --limit 40     # quick smoke test
    python extract_letter_landmarks.py                # full run
    python extract_letter_landmarks.py --per-class 600
"""

import argparse
import os
import random
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from extract_landmarks import normalize_landmarks
from hand_detector import HandDetector

DEFAULT_ROOT = r"C:\Users\hrush\Downloads\ISL_Alphabet\extracted\dataset - Gesture Speech"
SEED = 42

# one detector per worker process (MediaPipe objects are not picklable)
_WORKER = {}


def _init_worker():
    _WORKER["det"] = HandDetector(mode="image")


def read_image(path):
    """cv2.imread can fail on odd Windows paths; imdecode is safer."""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _process_one(job):
    """Return the normalised 63-value feature vector, or None if no hand found."""
    path = job
    img = read_image(path)
    if img is None:
        return None
    lm = _WORKER["det"].landmarks(img)
    if lm is None:
        return None
    # (1,21,3) -> centre on wrist + scale, identical to the training pipeline
    return normalize_landmarks(lm[None, ...])[0].astype(np.float32)


def collect_files(root, letters, per_class, rng):
    """One list of image paths per letter (sub-sampled for balance)."""
    jobs, labels = [], []
    for label, ch in enumerate(letters):
        folder = os.path.join(root, ch)
        if not os.path.isdir(folder):
            print(f"  !! missing folder for '{ch}': {folder}")
            continue
        files = [os.path.join(folder, f) for f in os.listdir(folder)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        files.sort()
        if per_class and len(files) > per_class:
            files = rng.sample(files, per_class)
        jobs.extend(files)
        labels.extend([label] * len(files))
        print(f"  {ch}: {len(files)} images")
    return jobs, np.array(labels, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out", default=os.path.join("data", "letters.npz"))
    ap.add_argument("--per-class", type=int, default=300, help="images per letter (0 = all)")
    ap.add_argument("--limit", type=int, default=0, help="only N images per letter (smoke test)")
    ap.add_argument("--workers", type=int, default=min(6, (os.cpu_count() or 4) - 1))
    args = ap.parse_args()

    per_class = args.limit or args.per_class
    letters = [chr(c) for c in range(ord("a"), ord("z") + 1)]

    if not os.path.isdir(args.root):
        raise SystemExit(f"Alphabet dataset not found: {args.root}")

    print(f"Scanning {args.root}")
    rng = random.Random(SEED)
    jobs, labels = collect_files(args.root, letters, per_class, rng)
    total = len(jobs)
    print(f"\n{total} images queued ({per_class} per letter), {len(letters)} classes\n")

    feats, kept_labels = [], []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as ex:
        done = 0
        for feat, label in zip(ex.map(_process_one, jobs, chunksize=8), labels):
            done += 1
            if feat is not None:
                feats.append(feat)
                kept_labels.append(label)
            if done % 500 == 0 or done == total:
                rate = len(feats) / max(done, 1)
                print(f"  {done}/{total} processed  (hand detected in {rate:.1%})", flush=True)

    X = np.stack(feats).astype(np.float32)
    y = np.array(kept_labels, dtype=np.int64)
    detect_rate = len(X) / max(total, 1)
    print(f"\nHand detected in {len(X)}/{total} images ({detect_rate:.1%})")

    # per-letter counts, so we can see which letters MediaPipe struggles with
    print("\nKept per letter:")
    for label, ch in enumerate(letters):
        n = int((y == label).sum())
        print(f"  {ch}: {n}" + ("   <-- LOW" if n < per_class * 0.5 else ""))

    # stratified 70/15/15 split (fixed seed)
    rng2 = np.random.default_rng(SEED)
    split = np.array(["train"] * len(X), dtype=object)
    for label in range(len(letters)):
        idx = np.where(y == label)[0]
        rng2.shuffle(idx)
        n = len(idx)
        n_val = int(round(n * 0.15))
        n_test = int(round(n * 0.15))
        split[idx[:n_val]] = "val"
        split[idx[n_val:n_val + n_test]] = "test"

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y, split=split.astype(str),
                        letters=np.array(letters))
    print(f"\nSaved {len(X)} samples -> {args.out}")
    print(f"X shape: {X.shape}   "
          f"train/val/test = {(split == 'train').sum()}/{(split == 'val').sum()}/{(split == 'test').sum()}")


if __name__ == "__main__":
    main()
