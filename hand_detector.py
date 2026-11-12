"""
Shared MediaPipe hand-landmark detector (Tasks API).

MediaPipe >= 1.0 removed the legacy `mp.solutions.hands` API, so we use the
Tasks `HandLandmarker` with the `hand_landmarker.task` model bundle.

Used by both extract_landmarks.py (offline, image mode) and the Flask app
(real-time, video mode).
"""

import os

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

_DEFAULT_MODEL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "mediapipe", "hand_landmarker.task")


class HandDetector:
    def __init__(self, mode="image", num_hands=2, model_path=_DEFAULT_MODEL,
                 min_detection_confidence=0.3, min_presence_confidence=0.3,
                 min_tracking_confidence=0.3):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"MediaPipe model not found: {model_path}\n"
                "Download it with:\n"
                "  Invoke-WebRequest -Uri "
                "\"https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                "hand_landmarker/float16/1/hand_landmarker.task\" "
                "-OutFile models\\mediapipe\\hand_landmarker.task"
            )
        running = vision.RunningMode.VIDEO if mode == "video" else vision.RunningMode.IMAGE
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=running,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self.mode = mode

    def detect(self, frame_bgr, timestamp_ms=0):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=np.ascontiguousarray(rgb))
        if self.mode == "video":
            return self._landmarker.detect_for_video(mp_image, int(timestamp_ms))
        return self._landmarker.detect(mp_image)

    def landmarks(self, frame_bgr, timestamp_ms=0):
        """Return the first hand's (21,3) landmarks or None."""
        res = self.detect(frame_bgr, timestamp_ms)
        if not res.hand_landmarks:
            return None
        return np.array([[p.x, p.y, p.z] for p in res.hand_landmarks[0]], dtype=np.float32)

    def close(self):
        self._landmarker.close()
