import math
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

from .state import SharedState

_LANDMARK_COLORS = {
    "line":   (0, 180, 255),
    "point":  (255, 255, 255),
}

_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (0, 11), (0, 12),
]


class PostureDetector:
    BAD_NECK_THRESHOLD = 40.0
    BAD_TORSO_THRESHOLD = 10.0
    CONSECUTIVE_BAD_FRAMES = 5

    def __init__(self, model_path: str):
        options = PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(
                model_asset_path=model_path,
                delegate=mp_tasks.BaseOptions.Delegate.CPU,
            ),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)
        self._bad_count = 0
        self._start_ms = time.time() * 1000

    def process_frame(self, frame: np.ndarray, state: SharedState) -> None:
        h_orig, w_orig = frame.shape[:2]

        # Reduz resolução só para a inferência (menos CPU), display fica no original
        small = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000 - self._start_ms)

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        annotated = frame.copy()

        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            # Landmarks são normalizados (0-1) — escala para o frame original
            h, w = h_orig, w_orig

            def pt(idx):
                l = lm[idx]
                return l.x * w, l.y * h

            for (a, b) in _CONNECTIONS:
                p1 = pt(a)
                p2 = pt(b)
                cv2.line(annotated,
                         (int(p1[0]), int(p1[1])),
                         (int(p2[0]), int(p2[1])),
                         _LANDMARK_COLORS["line"], 2)

            for idx in range(len(lm)):
                px = pt(idx)
                cv2.circle(annotated, (int(px[0]), int(px[1])), 4,
                           _LANDMARK_COLORS["point"], -1)

            nose = pt(0)
            l_shoulder = pt(11)
            r_shoulder = pt(12)
            l_hip = pt(23)
            r_hip = pt(24)

            shoulder_mid = (
                (l_shoulder[0] + r_shoulder[0]) / 2,
                (l_shoulder[1] + r_shoulder[1]) / 2,
            )
            hip_mid = (
                (l_hip[0] + r_hip[0]) / 2,
                (l_hip[1] + r_hip[1]) / 2,
            )

            neck_angle = self._vertical_angle(shoulder_mid, nose)
            torso_angle = self._vertical_angle(hip_mid, shoulder_mid)

            is_bad = (
                neck_angle > self.BAD_NECK_THRESHOLD
                and torso_angle > self.BAD_TORSO_THRESHOLD
            )

            if is_bad:
                self._bad_count += 1
            else:
                self._bad_count = 0

            status = "Ruim" if self._bad_count >= self.CONSECUTIVE_BAD_FRAMES else "Boa"

            color_status = (0, 80, 255) if status == "Ruim" else (0, 220, 100)
            color_bg = (20, 20, 40)

            cv2.rectangle(annotated, (0, 0), (300, 90), color_bg, -1)
            cv2.putText(annotated, f"Postura: {status}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color_status, 2)
            cv2.putText(annotated, f"Pescoco: {neck_angle:.1f}deg", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 1)
            cv2.putText(annotated, f"Tronco:  {torso_angle:.1f}deg", (10, 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 1)

            self._draw_angle_line(annotated, shoulder_mid, nose, color_status)
            self._draw_angle_line(annotated, hip_mid, shoulder_mid, (0, 200, 255))

            state.update_posture(status, neck_angle, torso_angle, self._bad_count)
        else:
            cv2.putText(annotated, "Nenhuma pessoa detectada", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
            state.update_posture("Analisando...", 0.0, 0.0, 0)

        _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        state.update_frame(jpeg.tobytes())

    @staticmethod
    def _vertical_angle(base: tuple, tip: tuple) -> float:
        dx = tip[0] - base[0]
        dy = tip[1] - base[1]
        return math.degrees(math.atan2(abs(dx), abs(dy)))

    @staticmethod
    def _draw_angle_line(img, p1, p2, color):
        cv2.line(img,
                 (int(p1[0]), int(p1[1])),
                 (int(p2[0]), int(p2[1])),
                 color, 2)

    def close(self):
        self._landmarker.close()
