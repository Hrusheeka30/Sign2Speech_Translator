# Dataset Analysis Report — `signlang_dataset/`

**Project:** AI Sign-to-Speech Translator
**Generated:** 2026-11-08 (from `analyze_dataset.py`, read-only scan)
**Root scanned:** `signlang_dataset/`
**Original files:** untouched — nothing renamed, moved, modified, or deleted.

---

## 1. Executive summary

| Metric | Value |
|---|---|
| Categories (grouping folders) | **4** |
| Sign-language classes (gestures) | **29** |
| Total videos | **503** |
| Total size | **6.3 GB** |
| Video extensions | `.MOV` (430), `.MP4` (73) |
| Resolution | **1920×1080** for 100% of videos |
| Frame rate | **25 fps** for 100% of videos |
| Clip duration | min **1.64 s**, mean **2.58 s**, max **4.36 s** |
| Total footage | **≈ 0.36 h (~21.6 min)** |
| Corrupted / unreadable videos | **0** |
| Byte-identical duplicate videos | **0** |
| Empty folders | **0** |
| Class imbalance (max/min) | **1.57×** (14 → 22 clips/class) — mild |

**Overall:** the dataset is small but clean and highly consistent (single resolution + single frame-rate). The main risks are **dataset size** and **train/test separation**, not corruption.

---

## 2. Folder structure

```
signlang_dataset/
├── Days_and_Time/   (8 classes, 112 videos)
│    ├── 66. Sunday … 73. Today
├── Greetings/       (9 classes, 190 videos)
│    ├── 48. Hello … 56. Pleased
├── Home/            (7 classes,  98 videos)
│    ├── 24. Table … 30. Bedroom
└── Places/          (5 classes, 103 videos)
     ├── 18. City … 22. Restaurant
          └── 21. Train Station/Extra/   ← nested sub-folder (2 clips)
```

- The **gesture class** is the folder directly under a category (the `NN. Name` folders).
- Class numbering runs to **73**, but only **29** numbers are present → this is a **subset** of a larger source dataset (gaps: 1–17, 23, 31–47, 57–65).

---

## 3. Per-class video counts

| Category | Class | Clips | Ext | Unreadable | Size (MB) |
|---|---|---:|---|---:|---:|
| Days_and_Time | 66. Sunday | 14 | MOV,MP4 | 0 | 171.3 |
| Days_and_Time | 67. Monday | 14 | MOV,MP4 | 0 | 158.1 |
| Days_and_Time | 68. Tuesday | 14 | MOV,MP4 | 0 | 163.6 |
| Days_and_Time | 69. Wednesday | 14 | MOV,MP4 | 0 | 152.9 |
| Days_and_Time | 70. Thursday | 14 | MOV,MP4 | 0 | 161.6 |
| Days_and_Time | 71. Friday | 14 | MOV,MP4 | 0 | 156.3 |
| Days_and_Time | 72. Saturday | 14 | MOV,MP4 | 0 | 158.7 |
| Days_and_Time | 73. Today | 14 | MOV,MP4 | 0 | 144.3 |
| Greetings | 48. Hello | 21 | MOV | 0 | 286.5 |
| Greetings | 49. How are you | 21 | MOV | 0 | 371.7 |
| Greetings | 50. Alright | 21 | MOV | 0 | 313.2 |
| Greetings | 51. Good Morning | 21 | MOV | 0 | 309.4 |
| Greetings | 52. Good afternoon | 22 | MOV | 0 | 328.8 |
| Greetings | 53. Good evening | 21 | MOV | 0 | 325.5 |
| Greetings | 54. Good night | 21 | MOV | 0 | 323.4 |
| Greetings | 55. Thank you | 21 | MOV | 0 | 296.9 |
| Greetings | 56. Pleased | 21 | MOV | 0 | 291.0 |
| Home | 24. Table | 14 | MOV,MP4 | 0 | 144.0 |
| Home | 25. Chair | 14 | MOV,MP4 | 0 | 136.2 |
| Home | 26. Bed | 14 | MOV,MP4 | 0 | 167.0 |
| Home | 27. Dream | 14 | MOV,MP4 | 0 | 210.8 |
| Home | 28. Window | 14 | MOV,MP4 | 0 | 213.2 |
| Home | 29. Door | 14 | MOV,MP4 | 0 | 158.3 |
| Home | 30. Bedroom | 14 | MOV,MP4 | 0 | 232.5 |
| Places | 18. City | 20 | MOV | 0 | 304.7 |
| Places | 19. House | 21 | MOV | 0 | 276.9 |
| Places | 20. Street or Road | 20 | MOV | 0 | 258.0 |
| Places | 21. Train Station | 22 | MOV | 0 | 314.0 |
| Places | 22. Restaurant | 20 | MOV | 0 | 260.0 |

Per-category totals: **Greetings 190**, **Days_and_Time 112**, **Places 103**, **Home 98**.

---

## 4. Video properties (all clips)

- **Resolution:** 1920×1080 — 503/503 (100%).
- **Frame rate:** 25.0 fps — 503/503 (100%).
- **Duration:** 1.64–4.36 s (mean 2.58 s) → roughly **41–109 frames per clip** at 25 fps.
- **Formats:** `.MOV` 430 + `.MP4` 73. Mixed extensions but uniform codec family (H.264).

Because resolution and fps are already uniform, **no rescaling of the source is required** — the only temporal issue is variable clip length.

---

## 5. Integrity check

Every one of the 503 files was opened with OpenCV and its first frame decoded.

- **Unreadable / corrupt:** 0
- **Opens but no decodable frame:** 0

No repair or re-encoding is needed for readability.

---

## 6. Duplicate analysis

- **Byte-identical duplicates (exact MD5 match): 0.** No file is an exact copy of another.
- **Filename collisions (same name in 2 different classes): 29**, all inside `Greetings/`.
  - e.g. `MVI_0037.MOV` exists in both `48. Hello` and `50. Alright`;
    `MVI_0048.MOV` in `51. Good Morning` and `52. Good afternoon`.
  - Verification showed the two files **differ in size by 20–52%** → they are **different clips**, not copies.
  - **Action:** manually spot-check ~5 of these pairs to confirm the labels are correct; the shared camera file numbering is confusing and worth cleaning.

No sign of true duplicate/leakage at the byte level, but the naming collisions should be reviewed for **label correctness**.

---

## 7. Empty folders & class imbalance

- **Empty folders:** none. Every category contains class folders, and every class folder contains videos.
- **Class balance** (excluding the nested `Extra` sub-folder, which belongs to *Train Station*):

```
14 clips ── Home (7 classes) + Days_and_Time (8 classes)
20–22 clips ── Places (5) + Greetings (9)
```

  Imbalance ratio **22 / 14 = 1.57×** — **mild**, safely handled with class weights or light augmentation.

---

## 8. Anomalies / things to fix

1. **Nested `Extra` folder** — `Places/21. Train Station/Extra/` holds 2 clips that belong to *Train Station*. Structure is inconsistent with the rest (has been merged into Train Station's count = 22). Decide whether to keep it nested or flatten.
2. **29 filename collisions across classes** — review for mislabeling (see §6).
3. **Very small dataset** — 503 clips for 29 classes = ~17 clips/class. Not enough for a robust general-purpose translator on its own.
4. **Tiny total footage** — ~21.6 minutes. Expect heavy reliance on transfer learning / augmentation.
5. **Mixed containers** (`.MOV` + `.MP4`) — harmless, but normalize for a clean pipeline.

---

## 9. Recommended next preprocessing steps

**Do not train yet — do the following first (in order):**

### A. Build a manifest (do this first)
1. Generate a `manifest.csv` / `labels.json`: `relative_path, category, class_name, label_id, fps, width, height, duration, num_frames`. (I can generate this — see §10.)
2. Fix the `Extra` sub-folder and confirm the 29 class → integer label mapping.

### B. Normalize media
3. Transcode everything to a **single container/codec** (e.g., MP4/H.264, yuv420p) at the native **1080p/25 fps** so the pipeline never deals with mixed formats.
4. Decide a working resolution for training (keep 1080p masters; train at **224×224** or **256×256**).

### C. Temporal standardization
5. Because clips vary 1.64–4.36 s, adopt **fixed-length temporal sampling**:
   - Sample **32 frames** uniformly per clip (or 16 for speed), OR
   - Trim/pad to a fixed window (e.g. 2.5 s) with padding.
6. Optionally down-sample to **12–15 fps** for training to cut compute without losing sign dynamics.

### D. Choose the modeling pipeline (recommended for sign language)
7. **Pose-first pipeline (recommended):** extract keypoints with MediaPipe Holistic / Hands → normalize (center + scale by shoulder width) → sequence model (GRU/Transformer). Benefits: signer-invariant, far less data-hungry, real-time friendly.
8. **RGB pipeline (fallback):** uniform frame sampling → 2D CNN + LSTM, or 3D CNN / VideoMAE with transfer learning.

### E. Splitting (critical)
9. Use a **stratified split** (e.g., 70/15/15) with a **fixed random seed**.
10. Since the MVI numbers imply session/clip grouping, **split by recording session/group where possible** to avoid near-duplicate clips leaking across train and test sets.
11. Re-check the 29 colliding filenames are not split across train/test.

### F. Augmentation (sign-language-aware)
12. Safe: temporal jitter, random crop/scale, brightness/contrast, small rotation.
13. **Avoid horizontal flip** — it swaps handedness and corrupts sign meaning.

### G. Address the small size
14. Use class weights / oversampling for the 14-clip classes.
15. Consider supplementing with a larger public sign dataset (e.g., WLASL, or an ISL dataset) for robustness, or collect more takes per class (target ≥ 30–50 clips/class).

### H. Speech output stage
16. Text → speech: use a lightweight local TTS (e.g. `pyttsx3` offline, or `gTTS` online). Wire it after the classifier outputs a class label / sentence.

---

## 10. Artifacts produced

| File | Purpose |
|---|---|
| `analyze_dataset.py` | Read-only inspection script (re-runnable after any preprocessing). |
| `dataset_scan.json` | Full machine-readable scan results (per-class, metadata, integrity, duplicates). |
| `.video_tools/` | Locally installed `opencv-python-headless`, `numpy`, `imageio-ffmpeg` used by the script (safe to delete; reinstall with `pip install --target .video_tools opencv-python-headless numpy imageio-ffmpeg`). |

**Reproduce:** `python analyze_dataset.py`

---

## 11. Suggested immediate next action

Ready to proceed with **Step A — generate `manifest.csv` (path → class → label_id → metadata)** as the foundation for all preprocessing, without touching the original videos. Awaiting your go-ahead.
