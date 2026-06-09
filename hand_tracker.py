import os
import urllib.request
import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_FILENAME = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

FINGERTIP_INDICES = {4, 8, 12, 16, 20}

JOINT_COLORS = {
    "bone": (0, 220, 255),
    "fingertip_fill": (255, 0, 200),
    "fingertip_outline": (255, 255, 255),
    "joint_fill": (0, 255, 255),
}


def _ensure_model():
    if os.path.exists(MODEL_FILENAME):
        return
    print(f"[INFO] Downloading MediaPipe hand model -> {MODEL_FILENAME}")
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_FILENAME)
            print("[INFO] Model downloaded successfully.")
            return
        except Exception as e:
            if attempt == max_retries:
                print(f"[ERROR] Could not download model after {max_retries} attempts: {e}")
                print(f"        Download manually from:\n        {MODEL_URL}")
                raise
            print(f"[WARN] Download attempt {attempt} failed, retrying...")


class HandTracker:
    def __init__(self, max_hands=2, detection_confidence=0.7, tracking_confidence=0.7):
        _ensure_model()
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_FILENAME),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.detector = mp_vision.HandLandmarker.create_from_options(options)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def find_hands(self, frame, draw=True):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_img)

        all_hands = []
        if result and result.hand_landmarks:
            for hand_lms in result.hand_landmarks:
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_lms]
                all_hands.append(landmarks)
                if draw:
                    self._draw_hand(frame, landmarks)

        return all_hands

    def _draw_hand(self, frame, landmarks):
        h, w = frame.shape[:2]
        pts = [(int(x * w), int(y * h)) for (x, y, _) in landmarks]

        for (a, b) in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], JOINT_COLORS["bone"], 2, cv2.LINE_AA)

        for idx, pt in enumerate(pts):
            if idx in FINGERTIP_INDICES:
                cv2.circle(frame, pt, 7, JOINT_COLORS["fingertip_fill"], -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 7, JOINT_COLORS["fingertip_outline"], 1, cv2.LINE_AA)
            else:
                cv2.circle(frame, pt, 4, JOINT_COLORS["joint_fill"], -1, cv2.LINE_AA)

    @staticmethod
    def get_pixel_landmarks(landmarks, frame_shape):
        h, w = frame_shape[:2]
        return [(int(x * w), int(y * h)) for (x, y, _) in landmarks]

    @staticmethod
    def get_hand_center(pixel_landmarks):
        xs = [p[0] for p in pixel_landmarks]
        ys = [p[1] for p in pixel_landmarks]
        return (int(np.mean(xs)), int(np.mean(ys)))

    def close(self):
        self.detector.close()