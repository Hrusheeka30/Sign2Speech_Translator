"""
Build a manifest.csv for the sign-language dataset.

- Recursively scans a dataset root.
- A "class" is the folder directly under a category folder
  (any deeper sub-folder, e.g. "21. Train Station/Extra", still belongs to its class).
- Assigns a stable integer label_id to every class.
- Creates a stratified train/val/test split with a fixed seed.
- Writes manifest.csv + labels.json. Never modifies the original videos.

Usage:
    python build_manifest.py --root signlang_dataset --out manifest.csv
"""

import argparse
import json
import os
from collections import defaultdict

from sklearn.model_selection import train_test_split

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def scan(root):
    rows = []
    for dirpath, _dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        parts = [] if rel == "." else rel.split(os.sep)
        if len(parts) >= 2:
            category, class_name = parts[0], parts[1]
        elif len(parts) == 1:
            category, class_name = "(root)", parts[0]
        else:
            category, class_name = "(root)", "(root)"
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in VIDEO_EXTS:
                rows.append({
                    "rel_path": os.path.relpath(os.path.join(dirpath, fn), root),
                    "category": category,
                    "class_name": class_name,
                    "size_bytes": os.path.getsize(os.path.join(dirpath, fn)),
                })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="signlang_dataset")
    ap.add_argument("--out", default="manifest.csv")
    ap.add_argument("--labels-out", default="labels.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    args = ap.parse_args()

    rows = scan(args.root)
    rows.sort(key=lambda r: (r["category"], r["class_name"], r["rel_path"]))

    classes = sorted({(r["category"], r["class_name"]) for r in rows},
                     key=lambda c: (c[0], c[1]))
    label_of = {c: i for i, c in enumerate(classes)}
    for r in rows:
        r["label_id"] = label_of[(r["category"], r["class_name"])]

    idx = list(range(len(rows)))
    labels = [rows[i]["label_id"] for i in idx]

    # stratified train (70) / val (15) / test (15)
    train_idx, temp_idx = train_test_split(
        idx, test_size=args.val + args.test, random_state=args.seed, stratify=labels)
    temp_labels = [rows[i]["label_id"] for i in temp_idx]
    rel_test = args.test / (args.val + args.test)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=rel_test, random_state=args.seed, stratify=temp_labels)

    split_of = {}
    for i in train_idx:
        split_of[i] = "train"
    for i in val_idx:
        split_of[i] = "val"
    for i in test_idx:
        split_of[i] = "test"
    for i, r in enumerate(rows):
        r["split"] = split_of[i]

    # write manifest.csv
    cols = ["rel_path", "category", "class_name", "label_id", "split", "size_bytes"]
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")

    labels_map = {str(i): {"category": c[0], "class_name": c[1]} for c, i in label_of.items()}
    with open(args.labels_out, "w", encoding="utf-8") as f:
        json.dump({"num_classes": len(classes), "id_to_class": labels_map}, f, indent=2)

    # console summary
    per_split = defaultdict(int)
    for r in rows:
        per_split[r["split"]] += 1
    print(f"Scanned {len(rows)} videos, {len(classes)} classes.")
    print("Split sizes:", dict(per_split))
    print("Per-class counts (train/val/test):")
    tally = defaultdict(lambda: defaultdict(int))
    for r in rows:
        tally[f'{r["category"]}/{r["class_name"]}'][r["split"]] += 1
    for cls in sorted(tally, key=lambda k: label_of[(k.split('/', 1)[0], k.split('/', 1)[1])]):
        t = tally[cls]
        print(f'  {cls:<34} train={t["train"]:<3} val={t["val"]:<3} test={t["test"]:<3}')
    print(f"\nWrote {args.out} and {args.labels_out}")


if __name__ == "__main__":
    main()
