"""
Read-only dataset inspection for signlang_dataset/.

Scanner: recursively walks signlang_dataset/, groups videos by their immediate
parent folder (a "sign class"), and their grandparent (a "category").
It never writes to, renames, or deletes any original file.

Usage:
    python analyze_dataset.py                 # full scan (metadata + integrity + duplicates)
    python analyze_dataset.py --limit 50      # only probe the first N files (fast smoke test)
    python analyze_dataset.py --skip-integrity  # skip cv2 open/decode checks
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict

# Local dependency folder (installed with pip --target)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".video_tools"))

try:
    import cv2
    if hasattr(cv2, "setLogLevel"):
        try:
            cv2.setLogLevel(0)  # silence ffmpeg/openCV noise
        except Exception:
            pass
    HAVE_CV2 = True
except Exception as e:  # pragma: no cover
    HAVE_CV2 = False
    print("WARNING: opencv not available ->", e)

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def human(n_bytes):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} PB"


def probe_video(path):
    """Return dict with metadata + integrity flag using OpenCV."""
    info = {
        "readable": False,
        "error": None,
        "width": None,
        "height": None,
        "fps": None,
        "frame_count": None,
        "duration_sec": None,
        "codec_fourcc": None,
        "decoded_first_frame": False,
    }
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            info["error"] = "cannot open (unsupported/corrupt header)"
            return info
        info["readable"] = True
        info["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        info["fps"] = round(fps, 3) if fps and fps > 0 else None
        fc = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        info["frame_count"] = int(fc) if fc and fc > 0 else None
        if info["fps"] and info["frame_count"]:
            info["duration_sec"] = round(info["frame_count"] / info["fps"], 2)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        if fourcc:
            info["codec_fourcc"] = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        ok, frame = cap.read()
        info["decoded_first_frame"] = bool(ok and frame is not None)
        if info["readable"] and not info["decoded_first_frame"]:
            info["error"] = "opens but no decodable frame"
    except Exception as e:
        info["error"] = f"exception: {e}"
    finally:
        cap.release()
    return info


def md5_of_file(path, chunk=1024 * 1024):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "signlang_dataset"))
    ap.add_argument("--limit", type=int, default=0, help="probe only first N videos (0 = all)")
    ap.add_argument("--skip-integrity", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_scan.json"))
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    print(f"Scanning root: {root}")
    t0 = time.time()

    videos = []  # (class_dir, category, filename, full_path)
    dir_has_video = defaultdict(int)   # dir -> count of videos directly inside
    all_dirs = set()

    for dirpath, dirnames, filenames in os.walk(root):
        all_dirs.add(dirpath)
        rel = os.path.relpath(dirpath, root)
        parts = [] if rel == "." else rel.split(os.sep)
        category = parts[0] if len(parts) >= 1 else "(root)"
        # A gesture CLASS is the folder directly under the category.
        # Any deeper sub-folder (e.g. "21. Train Station/Extra") still belongs
        # to that class, so we key on the first two path parts.
        if len(parts) >= 2:
            class_name = parts[1]
            class_dir = os.path.join(root, parts[0], parts[1])
        else:
            class_name = parts[0] if parts else "(root)"
            class_dir = dirpath
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in VIDEO_EXTS:
                videos.append({
                    "category": category,
                    "class_name": class_name,
                    "class_dir": class_dir,
                    "subfolder": parts[2] if len(parts) >= 3 else "",
                    "filename": fn,
                    "path": os.path.join(dirpath, fn),
                    "ext": ext,
                    "size": os.path.getsize(os.path.join(dirpath, fn)),
                })
                dir_has_video[dirpath] += 1

    total_videos = len(videos)
    total_size = sum(v["size"] for v in videos)

    # classes = directories that directly contain videos
    classes = defaultdict(list)
    for v in videos:
        classes[(v["category"], v["class_dir"])].append(v)

    # empty folders: dirs with no videos directly and none in descendants
    def subtree_has_video(d):
        for dd in all_dirs:
            if dd == d or dd.startswith(d + os.sep):
                if dir_has_video.get(dd, 0) > 0:
                    return True
        return False

    empty_dirs = []
    for d in sorted(all_dirs):
        if d == root:
            continue
        rel = os.path.relpath(d, root)
        # skip the category-level dirs that only hold subfolders with videos
        if dir_has_video.get(d, 0) == 0 and not subtree_has_video(d):
            empty_dirs.append(rel)

    print(f"Found {total_videos} videos in {len(classes)} class folders, "
          f"{len(all_dirs)} directories, total {human(total_size)}")

    # ---- integrity + metadata ----
    metas = []
    unreadable = []
    if HAVE_CV2 and not args.skip_integrity:
        to_probe = videos[:args.limit] if args.limit else videos
        print(f"Probing {len(to_probe)} videos (metadata + integrity)...")
        for i, v in enumerate(to_probe, 1):
            if i % 200 == 0 or i == len(to_probe):
                print(f"  {i}/{len(to_probe)} ...")
            info = probe_video(v["path"])
            v.update(info)
            metas.append(v)
            if not info["readable"] or not info["decoded_first_frame"]:
                unreadable.append(v)
        # files not probed get None metadata
        for v in videos[len(to_probe):]:
            v.update({"readable": None, "width": None, "height": None, "fps": None,
                      "frame_count": None, "duration_sec": None, "error": "not probed"})
    else:
        for v in videos:
            v.update({"readable": None, "width": None, "height": None, "fps": None,
                      "frame_count": None, "duration_sec": None, "error": "not probed"})

    # ---- duplicate detection (exact byte duplicates by size then md5) ----
    print("Checking duplicates (size buckets -> md5)...")
    by_size = defaultdict(list)
    for v in videos:
        by_size[v["size"]].append(v)
    dup_groups = []
    for size, group in by_size.items():
        if len(group) < 2:
            continue
        by_hash = defaultdict(list)
        for v in group:
            try:
                by_hash[md5_of_file(v["path"])].append(v)
            except Exception as e:
                v["hash_error"] = str(e)
        for h, g in by_hash.items():
            if len(g) >= 2:
                dup_groups.append({"md5": h, "size": size, "count": len(g),
                                   "files": [{"class": x["class_name"], "category": x["category"],
                                              "filename": x["filename"]} for x in g]})

    # duplicate filenames across (different) classes
    by_name = defaultdict(list)
    for v in videos:
        by_name[v["filename"].lower()].append(v)
    dup_names = {n: g for n, g in by_name.items() if len(g) >= 2}

    # For same-named files that live in >1 distinct class, check byte identity to
    # flag possible label leakage (same clip labelled as two different gestures).
    dup_name_detail = []
    for n, g in sorted(dup_names.items()):
        classes_in_group = {x["class_dir"] for x in g}
        if len(classes_in_group) < 2:
            continue
        hashed = {}
        for x in g:
            try:
                hashed.setdefault(md5_of_file(x["path"]), []).append(x)
            except Exception as e:
                x["hash_error"] = str(e)
        groups = [{"md5": h, "files": [
            {"category": y["category"], "class": y["class_name"],
             "subfolder": y.get("subfolder", ""), "filename": y["filename"]} for y in gg]}
            for h, gg in hashed.items()]
        identical = any(len(gg) >= 2 for gg in hashed.values())
        dup_name_detail.append({"filename": n, "classes": sorted(
            f'{y["category"]}::{y["class_name"]}' for y in g),
            "byte_identical_somewhere": identical, "hash_groups": groups})

    # ---- aggregates ----
    per_class = []
    for (cat, cdir), vids in sorted(classes.items(), key=lambda kv: os.path.basename(kv[0][1]).lower()):
        durations = [v["duration_sec"] for v in vids if v.get("duration_sec")]
        res = {(v["width"], v["height"]) for v in vids if v.get("width")}
        fps_list = [v["fps"] for v in vids if v.get("fps")]
        per_class.append({
            "category": cat,
            "class": os.path.basename(cdir),
            "count": len(vids),
            "total_size": sum(x["size"] for x in vids),
            "extensions": sorted({x["ext"].lstrip(".").upper() for x in vids}),
            "durations": sorted(durations),
            "resolutions": sorted([r for r in res if r[0]]),
            "fps": sorted(set(fps_list)),
            "unreadable": sum(1 for x in vids if x.get("readable") is False or x.get("decoded_first_frame") is False),
        })

    counts = [c["count"] for c in per_class]
    categories = defaultdict(lambda: {"classes": 0, "videos": 0})
    for c in per_class:
        categories[c["category"]]["classes"] += 1
        categories[c["category"]]["videos"] += c["count"]

    all_res = defaultdict(int)
    all_fps = defaultdict(int)
    all_dur = []
    for v in videos:
        if v.get("width") and v.get("height"):
            all_res[f'{v["width"]}x{v["height"]}'] += 1
        if v.get("fps"):
            all_fps[v["fps"]] += 1
        if v.get("duration_sec"):
            all_dur.append(v["duration_sec"])

    summary = {
        "root": root,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scan_seconds": round(time.time() - t0, 1),
        "num_categories": len(categories),
        "num_classes": len(classes),
        "total_videos": total_videos,
        "total_size_bytes": total_size,
        "total_size_human": human(total_size),
        "num_directories": len(all_dirs),
        "empty_directories": empty_dirs,
        "extensions": dict(sorted({e: sum(1 for v in videos if v["ext"] == e) for e in VIDEO_EXTS if any(v["ext"] == e for v in videos)}.items())),
        "class_count_min": min(counts) if counts else 0,
        "class_count_max": max(counts) if counts else 0,
        "class_count_mean": round(sum(counts) / len(counts), 2) if counts else 0,
        "class_count_median": sorted(counts)[len(counts) // 2] if counts else 0,
        "imbalance_ratio_max_over_min": round(max(counts) / min(counts), 2) if counts and min(counts) > 0 else None,
        "unreadable_count": len(unreadable),
        "unreadable": [{"path": os.path.relpath(x["path"], root), "error": x.get("error")} for x in unreadable],
        "duplicate_groups": len(dup_groups),
        "duplicate_files_total": sum(g["count"] for g in dup_groups),
        "duplicate_groups_detail": dup_groups,
        "duplicate_filename_count": len(dup_names),
        "duplicate_filename_across_classes": len(dup_name_detail),
        "duplicate_filename_detail": dup_name_detail,
        "resolutions": dict(sorted(all_res.items(), key=lambda kv: -kv[1])),
        "fps_values": dict(sorted(all_fps.items())),
        "duration_sec": {
            "min": min(all_dur) if all_dur else None,
            "max": max(all_dur) if all_dur else None,
            "mean": round(sum(all_dur) / len(all_dur), 2) if all_dur else None,
            "total_hours": round(sum(all_dur) / 3600, 2) if all_dur else None,
        },
        "categories": {k: dict(v) for k, v in categories.items()},
        "per_class": per_class,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nDone in {summary['scan_seconds']}s -> {args.out}")
    print(json.dumps({k: summary[k] for k in [
        "num_categories", "num_classes", "total_videos", "total_size_human",
        "class_count_min", "class_count_max", "class_count_mean",
        "imbalance_ratio_max_over_min", "unreadable_count", "duplicate_groups",
        "duplicate_files_total", "empty_directories", "resolutions", "fps_values",
        "duration_sec"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
