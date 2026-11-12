"""
Sign-to-Speech Translator - Flask backend.

The browser sends a short batch of webcam frames; the server runs MediaPipe
hand-landmark detection on every frame, builds a fixed-length sequence and lets
the trained LSTM classify the sign. The recognised English word can then be
translated into five languages and spoken aloud.

Endpoints
---------
GET  /                     recognition + translation + speech UI
GET  /dictionary           redirects to the dashboard (dictionary is a section)
GET  /login                name-based login / greeting page
POST /api/login            set the current user's name
POST /api/logout           clear the current user
GET  /api/signs            the user's own personal signs with notes/favourites
POST /api/signs/<id>       save a personal note / favourite for one sign
POST /api/custom-signs     add a sign to the user's own dictionary
POST /api/custom-signs/<id> edit one of the user's own signs
DELETE /api/custom-signs/<id> delete one of the user's own signs
GET  /api/export           download the user's dictionary as a JSON backup
POST /api/import           merge a JSON backup back into the user's dictionary
POST /api/reset            clear the user's notes / favourites / own signs
POST /api/predict          recognise a sign from webcam frames
POST /api/predict-letter   recognise an alphabet letter (A-Z) from webcam frames
GET  /api/sign-vocab       words available as reference videos (word -> sign)
GET  /api/sign-video       stream a reference sign video from the dataset
POST /api/translate        translate English text into one of 5 languages
POST /api/tts              text -> speech audio (mp3)
GET  /api/backend          which translation / speech backends are active
"""

import base64
import csv
import glob
import json
import os
import queue
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from flask import (
    Flask, Response, jsonify, redirect, render_template, request, send_file,
    session, url_for,
)

import translator
from extract_landmarks import FIXED_FRAMES, fill_missing, normalize_landmarks, resample
from hand_detector import HandDetector

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "sign_lstm.keras")
# The alphabet is a separate model from the word ensemble: static hand poses
# (one 63-value landmark frame) vs. dynamic signs (30-frame sequences).
LETTER_MODEL_DIR = os.path.join(BASE_DIR, "models", "letters")
SIGNS_PATH = os.path.join(BASE_DIR, "signs.json")
USER_DATA_DIR = os.path.join(BASE_DIR, "user_data")  # every user lives in one file here
USER_DB_PATH = os.path.join(USER_DATA_DIR, "users.json")

# Reference sign videos. Override with SIGNS_DATASET_ROOT if the dataset lives elsewhere.
DATASET_ROOT = os.getenv("SIGNS_DATASET_ROOT", os.path.join(BASE_DIR, "signlang_dataset"))
MANIFEST_PATH = os.path.join(BASE_DIR, "manifest.csv")
VIDEO_MIME = {
    ".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime",
    ".webm": "video/webm", ".avi": "video/x-msvideo", ".mkv": "video/x-matroska",
}

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "sign2speech-dev-secret")

# MediaPipe landmarker + Keras model are expensive to create, so build them once.
_models = None
_letter_model = None
_letter_labels = None

# Ensemble members live here (produced by train_lstm.py). Predictions from every
# member are averaged, which is far more stable than a single model.
ENSEMBLE_DIR = os.path.join(BASE_DIR, "models", "ensemble")

# MediaPipe cost scales with frame COUNT, and the sequence is resampled to
# FIXED_FRAMES afterwards anyway, so cap how many frames it has to process.
# Keeps a live sign responsive without losing the gesture's shape.
MAX_LANDMARK_FRAMES = 18

# Parallel MediaPipe workers (one detector each). Capped so we don't
# oversubscribe the CPU - MediaPipe already uses several threads internally.
MAX_DETECTORS = min(4, os.cpu_count() or 1)
_detector_pool = None       # queue.Queue holding one HandDetector per worker
_detector_exec = None       # thread pool, one worker per detector
_pool_size = 0
_ensemble_exec = None       # runs the ensemble members concurrently

# Every user's name + saved data lives in one file (user_data/users.json).
_user_db = None
_user_db_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# lazy singletons
# --------------------------------------------------------------------------- #
def get_models():
    """Load the ensemble members once (falls back to the single model file)."""
    global _models
    if _models is None:
        import tensorflow as tf

        paths = sorted(glob.glob(os.path.join(ENSEMBLE_DIR, "*.keras")))
        if not paths:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(
                    "Trained model not found. Run:  python extract_landmarks.py  then  "
                    "python train_lstm.py")
            paths = [MODEL_PATH]
        _models = [tf.keras.models.load_model(p) for p in paths]
    return _models


def _init_detector_pool():
    """Build the pooled MediaPipe instances once.

    One MediaPipe landmarker is not thread-safe, but the frames of a sign are
    independent, so running a few instances in parallel is what removes the
    "keeps scanning" delay. IMAGE mode matches how the training features were
    extracted and scores far better here than VIDEO tracking mode.
    """
    global _detector_pool, _detector_exec, _pool_size
    if _detector_pool is None:
        _pool_size = max(1, MAX_DETECTORS)
        _detector_pool = queue.Queue()
        for _ in range(_pool_size):
            _detector_pool.put(HandDetector(mode="image"))
        _detector_exec = ThreadPoolExecutor(max_workers=_pool_size)
    return _detector_pool


def _track(img):
    """Landmark one frame on a free pooled detector, then hand it back."""
    det = _detector_pool.get()
    try:
        return det.landmarks(img)
    finally:
        _detector_pool.put(det)


def _landmarks_batch(frames):
    """[(21, 3) or None, ...] for frames, landmarked in parallel."""
    _init_detector_pool()
    return list(_detector_exec.map(_track, frames))


def get_letter_model():
    """Load the A-Z classifier + its label list once."""
    global _letter_model, _letter_labels
    if _letter_model is None:
        import tensorflow as tf

        model_file = os.path.join(LETTER_MODEL_DIR, "letter_model.keras")
        map_file = os.path.join(LETTER_MODEL_DIR, "label_map.json")
        if not (os.path.exists(model_file) and os.path.exists(map_file)):
            raise FileNotFoundError(
                "Alphabet model not found. Run:  python extract_letter_landmarks.py  "
                "then  python train_letter_model.py")
        with open(map_file, encoding="utf-8") as f:
            _letter_labels = json.load(f)["letters"]
        _letter_model = tf.keras.models.load_model(model_file)
    return _letter_model, _letter_labels


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def load_signs():
    """The built-in signs (no personalisation applied)."""
    with open(SIGNS_PATH, encoding="utf-8") as f:
        return json.load(f)["signs"]


_label_names = None


def get_label_names():
    """Model class index -> display name.

    The LSTM ensemble emits a class index ordered by models/label_map.json
    (alphabetical by category then class_name). That order is NOT the id order
    used in signs.json, so an index has to be resolved through the shared
    ``class_name`` - otherwise e.g. index 0 ("1. Dog") is shown as "Sunday".
    """
    global _label_names
    if _label_names is None:
        by_class = {s["class_name"]: s["name"] for s in load_signs()}
        names = []
        map_path = os.path.join(BASE_DIR, "models", "label_map.json")
        if os.path.exists(map_path):
            with open(map_path, encoding="utf-8") as f:
                id_to_class = json.load(f).get("id_to_class", {})
            for i in range(len(id_to_class)):
                class_name = id_to_class.get(str(i), {}).get("class_name", "")
                names.append(by_class.get(class_name, class_name or str(i)))
        _label_names = names
    return _label_names


_clip_index = None


def get_clip_index():
    """class_name -> list of video paths relative to DATASET_ROOT (MP4 clips first)."""
    global _clip_index
    if _clip_index is None:
        idx = {}
        if os.path.exists(MANIFEST_PATH):
            with open(MANIFEST_PATH, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    idx.setdefault(row["class_name"], []).append(row["rel_path"])
        for name in idx:
            idx[name].sort(key=lambda p: (0 if p.lower().endswith((".mp4", ".m4v")) else 1, p))
        _clip_index = idx
    return _clip_index


def class_clips(class_name):
    """Existing reference clips for a sign class, as absolute paths."""
    clips = []
    for rel in get_clip_index().get(class_name, []):
        path = os.path.join(DATASET_ROOT, rel)
        if os.path.exists(path):
            clips.append(path)
    return clips


def current_user():
    return session.get("username")


def _user_key(username):
    return re.sub(r"[^A-Za-z0-9_-]", "_", (username or "").lower())[:40] or "guest"


def _load_user_db():
    """The whole user directory: {user_key: {name, notes, favorites, custom}}.

    One file for everybody (user_data/users.json) instead of a separate
    ``<name>.json`` per user, so a name and everything the user saved sit
    together. The first time it runs it folds any old per-user files in.
    """
    global _user_db
    if _user_db is None:
        db = {}
        if os.path.exists(USER_DB_PATH):
            with open(USER_DB_PATH, encoding="utf-8") as f:
                db = json.load(f)
        else:                                   # one-time migration of the old layout
            for path in glob.glob(os.path.join(USER_DATA_DIR, "*.json")):
                key = os.path.splitext(os.path.basename(path))[0]
                try:
                    with open(path, encoding="utf-8") as f:
                        db[key] = json.load(f)
                except Exception:
                    continue
        _user_db = db
    return _user_db


def _save_user_db():
    os.makedirs(USER_DATA_DIR, exist_ok=True)
    with open(USER_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(_user_db, f, indent=2, ensure_ascii=False)


def load_user_data(username):
    """Personal data for one user: notes, favourites and their own signs."""
    with _user_db_lock:
        entry = _load_user_db().get(_user_key(username))
        if entry is None:
            return {"notes": {}, "favorites": [], "custom": []}
        return entry


def save_user_data(username, data):
    with _user_db_lock:
        db = _load_user_db()
        entry = db.setdefault(_user_key(username), {})
        entry["name"] = username               # remember the entered name
        entry["notes"] = data.get("notes", {})
        entry["favorites"] = data.get("favorites", [])
        entry["custom"] = data.get("custom", [])
        _save_user_db()


def ensure_user(username):
    """Record the user (and the name they typed) in the shared file."""
    with _user_db_lock:
        db = _load_user_db()
        entry = db.setdefault(_user_key(username),
                              {"notes": {}, "favorites": [], "custom": []})
        entry["name"] = username
        _save_user_db()


def frames_to_sequence(frames):
    """Decode webcam frames -> (FIXED_FRAMES, 63) landmark sequence."""
    if len(frames) > MAX_LANDMARK_FRAMES:     # evenly subsample - keeps MediaPipe fast
        idx = np.linspace(0, len(frames) - 1, MAX_LANDMARK_FRAMES).round().astype(int)
        frames = [frames[i] for i in idx]
    results = _landmarks_batch(frames)        # landmarked in parallel
    pts, valid = [], []
    for lm in results:
        if lm is None:
            pts.append(np.zeros((21, 3), np.float32))
            valid.append(False)
        else:
            pts.append(lm)
            valid.append(True)
    seq = np.stack(pts)              # (T, 21, 3)
    valid = np.asarray(valid)
    rate = float(valid.mean())
    seq = fill_missing(seq, valid)   # interpolate undetected frames (as in training)
    seq = normalize_landmarks(seq)
    seq = resample(seq, FIXED_FRAMES)
    return seq, rate


def _ensemble_predict(models, batch):
    """Mean softmax over the ensemble members, evaluated concurrently.

    Each member is an independent model, so running them in parallel hides the
    per-call TensorFlow overhead (~0.15s each) and roughly halves this step.
    """
    global _ensemble_exec
    if len(models) == 1:
        return models[0].predict(batch, verbose=0)
    if _ensemble_exec is None:
        _ensemble_exec = ThreadPoolExecutor(max_workers=len(models))
    probs = list(_ensemble_exec.map(lambda m: m.predict(batch, verbose=0), models))
    return sum(probs) / len(probs)


def predict_probs(models, seq, copies=6):
    """Softmax averaged over the ensemble and lightly-jittered copies.

    Two layers of averaging make the live sign much steadier:
      * the ensemble - every trained model votes, so one over-confident member
        cannot flip the answer;
      * test-time augmentation (TTA) - each model also sees a few scaled /
        rotated / jittered variants of the same sequence (the training-time
        transforms), so a single unlucky frame batch cannot flip the answer.
    Together this is what stops "Good Morning" from silently becoming
    "Good Afternoon" on one noisy webcam sequence.
    """
    seq = seq.astype(np.float32)
    batch = [seq]
    rng = np.random.default_rng(0)
    for _ in range(copies):
        x = seq.copy() * rng.uniform(0.95, 1.05)
        pts = x.reshape(-1, 21, 3)
        ang = float(rng.uniform(-0.10, 0.10))
        c, s = np.cos(ang), np.sin(ang)
        xx, yy = pts[..., 0].copy(), pts[..., 1].copy()
        pts[..., 0] = c * xx - s * yy
        pts[..., 1] = s * xx + c * yy
        x = pts.reshape(seq.shape) + rng.normal(0, 0.01, seq.shape).astype(np.float32)
        batch.append(x.astype(np.float32))
    stacked = np.stack(batch)              # (copies+1, T, 63)
    probs = _ensemble_predict(models, stacked)
    return probs.mean(axis=0)


def predict_letter_probs(frames, max_frames=24):
    """Webcam frames -> letter probabilities.

    Every frame gives one normalised 63-value landmark vector, which the letter
    model classifies on its own. The per-frame softmax outputs are then averaged,
    so a single shaky frame cannot decide the answer. Returns
    (probs, letters, detection_rate) or (None, None, 0.0) if no hand was found.
    """
    model, letters = get_letter_model()
    kept = frames[:max_frames]
    feats = [normalize_landmarks(lm[None, ...])[0]
             for lm in _landmarks_batch(kept) if lm is not None]
    if not feats:
        return None, None, 0.0
    X = np.stack(feats).astype(np.float32)
    probs = model.predict(X, verbose=0)          # (frames, 26)
    return probs.mean(axis=0), letters, len(feats) / max(len(kept), 1)


def decode_frames(b64_frames):
    frames = []
    for b64 in b64_frames:
        try:
            if "," in b64:           # strip a "data:image/jpeg;base64," prefix
                b64 = b64.split(",", 1)[1]
            raw = base64.b64decode(b64)
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            continue
        if img is not None:
            frames.append(img)
    return frames


# --------------------------------------------------------------------------- #
# pages
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    if not current_user():
        return redirect(url_for("login"))
    return render_template("index.html", username=current_user())


@app.route("/dictionary")
def dictionary():
    # The sign dictionary now lives inside the dashboard as a sidebar section.
    if not current_user():
        return redirect(url_for("login"))
    return redirect(url_for("index"))


@app.route("/login")
def login():
    return render_template("login.html")


@app.post("/api/login")
def api_login():
    name = ((request.get_json(silent=True) or {}).get("username") or "").strip()
    if not name:
        return jsonify(error="Please enter your name."), 400
    session["username"] = name[:40]
    ensure_user(current_user())                # remember their name in users.json
    return jsonify(username=session["username"])


@app.post("/api/logout")
def api_logout():
    session.pop("username", None)
    return jsonify(ok=True)


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
@app.get("/api/backend")
def api_backend():
    return jsonify(translator.backend_info())


@app.get("/api/signs")
def api_signs():
    """The signed-in user's personal dictionary. Nothing is pre-loaded: every
    entry here is a sign the user added themselves."""
    data = load_user_data(current_user())
    favorites = {str(f) for f in data.get("favorites", [])}
    signs = []
    for c in data.get("custom", []):
        c = dict(c)
        c["custom"] = True
        c["favorite"] = str(c["id"]) in favorites
        signs.append(c)
    return jsonify(signs)


@app.post("/api/signs/<sign_id>")
def api_update_sign(sign_id):
    """Save a personal note and/or favourite flag for any sign."""
    payload = request.get_json(silent=True) or {}
    data = load_user_data(current_user())
    if "note" in payload:
        data.setdefault("notes", {})[sign_id] = payload["note"]
        for c in data.get("custom", []):
            if c["id"] == sign_id:
                c["note"] = payload["note"]
    if "favorite" in payload:
        fav = {str(f) for f in data.get("favorites", [])}
        if payload["favorite"]:
            fav.add(str(sign_id))
        else:
            fav.discard(str(sign_id))
        data["favorites"] = sorted(fav)
    save_user_data(current_user(), data)
    return jsonify(ok=True)


@app.post("/api/custom-signs")
def api_add_custom_sign():
    payload = request.get_json(silent=True) or {}
    name = (payload.get("name") or "").strip()
    if not name:
        return jsonify(error="A sign name is required."), 400
    data = load_user_data(current_user())
    sign = {
        "id": f"c_{int(time.time() * 1000)}",
        "name": name,
        "english": (payload.get("english") or name).strip(),
        "meaning": (payload.get("meaning") or "").strip(),
        "category": (payload.get("category") or "My Signs").strip(),
        "note": (payload.get("note") or "").strip(),
    }
    data.setdefault("custom", []).append(sign)
    save_user_data(current_user(), data)
    return jsonify(sign=sign)


@app.post("/api/custom-signs/<sign_id>")
def api_edit_custom_sign(sign_id):
    payload = request.get_json(silent=True) or {}
    data = load_user_data(current_user())
    for c in data.get("custom", []):
        if c["id"] == sign_id:
            for field in ("name", "english", "meaning", "category", "note"):
                if field in payload:
                    c[field] = (payload[field] or "").strip()
            save_user_data(current_user(), data)
            return jsonify(sign=c)
    return jsonify(error="Sign not found."), 404


@app.delete("/api/custom-signs/<sign_id>")
def api_delete_custom_sign(sign_id):
    data = load_user_data(current_user())
    data["custom"] = [c for c in data.get("custom", []) if c["id"] != sign_id]
    data["favorites"] = [f for f in data.get("favorites", []) if str(f) != sign_id]
    if isinstance(data.get("notes"), dict):
        data["notes"].pop(sign_id, None)
    save_user_data(current_user(), data)
    return jsonify(ok=True)


@app.get("/api/export")
def api_export():
    """Download the signed-in user's personal dictionary as a JSON backup."""
    data = load_user_data(current_user())
    return jsonify(
        app="Sign2Speech",
        username=current_user(),
        exported_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        notes=data.get("notes", {}),
        favorites=data.get("favorites", []),
        custom=data.get("custom", []),
    )


@app.post("/api/import")
def api_import():
    """Merge a previously exported backup into the user's own dictionary."""
    payload = request.get_json(silent=True) or {}
    data = load_user_data(current_user())

    notes = payload.get("notes")
    if isinstance(notes, dict):
        data.setdefault("notes", {}).update({str(k): str(v) for k, v in notes.items()})

    fav = {str(f) for f in data.get("favorites", [])}
    fav |= {str(f) for f in (payload.get("favorites") or [])}
    data["favorites"] = sorted(fav)

    existing = {c.get("id") for c in data.get("custom", [])}
    added = 0
    for c in payload.get("custom") or []:
        if not isinstance(c, dict) or not (c.get("name") or "").strip():
            continue
        new_id = str(c.get("id") or f"c_{int(time.time() * 1000)}")
        if new_id in existing:
            new_id = f"c_{int(time.time() * 1000)}"
        existing.add(new_id)
        data.setdefault("custom", []).append({
            "id": new_id,
            "name": str(c["name"]).strip(),
            "english": str(c.get("english") or c["name"]).strip(),
            "meaning": str(c.get("meaning") or "").strip(),
            "category": str(c.get("category") or "My Signs").strip(),
            "note": str(c.get("note") or "").strip(),
        })
        added += 1

    save_user_data(current_user(), data)
    return jsonify(ok=True, custom_added=added, total_custom=len(data["custom"]))


@app.post("/api/reset")
def api_reset():
    """Clear the user's notes, favourites and own signs (built-in signs stay)."""
    save_user_data(current_user(), {"notes": {}, "favorites": [], "custom": []})
    return jsonify(ok=True)


@app.get("/api/sign-vocab")
def api_sign_vocab():
    """Words that have a reference sign video (used by the Word -> Sign page)."""
    available = os.path.isdir(DATASET_ROOT)
    vocab = []
    for s in load_signs():
        clips = class_clips(s["class_name"]) if available else []
        vocab.append({
            "id": s["id"],
            "name": s["name"],
            "english": s.get("english", s["name"]),
            "meaning": s.get("meaning", ""),
            "category": s["category"],
            "class_name": s["class_name"],
            "clips": len(clips),
        })
    return jsonify(dataset_available=available, dataset_root=DATASET_ROOT, signs=vocab)


@app.get("/api/sign-video")
def api_sign_video():
    """Stream one reference sign clip (supports HTTP range requests for seeking)."""
    class_name = (request.args.get("class") or "").strip()
    if not class_name:
        return jsonify(error="Missing 'class' parameter."), 400
    clips = class_clips(class_name)
    if not clips:
        return jsonify(error="No reference video found for this sign."), 404
    try:
        index = int(request.args.get("i", 0))
    except ValueError:
        index = 0
    index %= len(clips)
    path = clips[index]
    if not os.path.abspath(path).startswith(os.path.abspath(DATASET_ROOT)):
        return jsonify(error="Invalid video path."), 400
    mime = VIDEO_MIME.get(os.path.splitext(path)[1].lower(), "application/octet-stream")
    resp = send_file(path, mimetype=mime, conditional=True, etag=True,
                     last_modified=os.path.getmtime(path))
    resp.headers["Accept-Ranges"] = "bytes"
    resp.headers["Cache-Control"] = "private, max-age=3600"
    resp.headers["X-Clip-Count"] = str(len(clips))
    resp.headers["X-Clip-Index"] = str(index)
    return resp


@app.post("/api/predict")
def api_predict():
    data = request.get_json(silent=True) or {}
    b64_frames = data.get("frames", [])
    if not b64_frames:
        return jsonify(error="No frames received."), 400

    frames = decode_frames(b64_frames)
    if not frames:
        return jsonify(error="Could not decode any frame."), 400

    try:
        models = get_models()
    except FileNotFoundError as e:
        return jsonify(error=str(e)), 503

    try:
        seq, rate = frames_to_sequence(frames)
    except OSError as e:
        # Most commonly Windows Smart App Control / WDAC blocking MediaPipe's
        # unsigned native library, which surfaces as "[WinError 4551]".
        return jsonify(error="Hand-tracking engine (MediaPipe) could not start: "
                             f"{e}"), 503
    if rate < 0.1:
        return jsonify(error="No hand detected. Please sign in front of the camera."), 200

    probs = predict_probs(models, seq, copies=2)   # fewer TTA passes -> lower latency
    names = get_label_names()
    order = np.argsort(probs)[::-1][:3]
    topk = [{"sign": names[int(i)] if int(i) < len(names) else str(i),
             "confidence": float(probs[i])}
            for i in order]
    return jsonify(sign=topk[0]["sign"], confidence=topk[0]["confidence"],
                   detection_rate=rate, topk=topk)


@app.post("/api/predict-letter")
def api_predict_letter():
    """Recognise one alphabet letter (A-Z) from a batch of webcam frames."""
    data = request.get_json(silent=True) or {}
    b64_frames = data.get("frames", [])
    if not b64_frames:
        return jsonify(error="No frames received."), 400

    frames = decode_frames(b64_frames)
    if not frames:
        return jsonify(error="Could not decode any frame."), 400

    try:
        probs, letters, rate = predict_letter_probs(frames)
    except FileNotFoundError as e:
        return jsonify(error=str(e)), 503

    if probs is None:
        return jsonify(error="No hand detected. Please sign in front of the camera."), 200

    order = np.argsort(probs)[::-1][:3]
    topk = [{"letter": letters[int(i)], "confidence": float(probs[i])} for i in order]
    return jsonify(letter=topk[0]["letter"], confidence=topk[0]["confidence"],
                   detection_rate=rate, topk=topk)


@app.post("/api/translate")
def api_translate():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    langs = data.get("langs") or list(translator.LANGS)
    if not text:
        return jsonify(error="No text provided."), 400
    out = {}
    for lang in langs:
        try:
            out[lang] = translator.translate(text, lang)
        except Exception:                            # keep going on one failure
            out[lang] = text                         # fall back to English — never show error text
    return jsonify(translations=out)


@app.post("/api/tts")
def api_tts():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    lang = data.get("lang", "en")
    if not text:
        return jsonify(error="No text provided."), 400
    if lang not in translator.LANGS:
        return jsonify(error=f"Unsupported language: {lang}"), 400
    try:
        audio = translator.speak_bytes(text, lang)
    except Exception as e:
        return jsonify(error=f"Speech synthesis failed: {e}"), 500
    return Response(audio, mimetype="audio/mpeg")


def warm_up():
    """Build the detector pool and trace the ensemble once at startup.

    The very first prediction otherwise pays the MediaPipe graph-build and Keras
    tracing cost (several seconds), which made the first live sign feel stuck.
    """
    try:
        _init_detector_pool()
        blank = np.zeros((480, 640, 3), np.uint8)
        _landmarks_batch([blank] * _pool_size)   # trace every pooled detector
        predict_probs(get_models(), np.zeros((FIXED_FRAMES, 63), np.float32), copies=2)
        print(f"Hand detector pool ({_pool_size}) + sign model ready.")
    except Exception as e:                       # never block startup
        print(f"Warm-up skipped: {e}")


if __name__ == "__main__":
    print("Translation / speech backends:", translator.backend_info())
    warm_up()
    app.run(host="127.0.0.1", port=5000, debug=False)
