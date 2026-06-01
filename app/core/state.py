import threading
import time


class SharedState:
    def __init__(self):
        self._lock = threading.Lock()
        self.posture_status: str = "Analisando..."
        self.neck_angle: float = 0.0
        self.torso_angle: float = 0.0
        self.bad_frame_count: int = 0
        self.last_alert_time: float = 0.0
        self.annotated_frame: bytes | None = None
        self.running: bool = True

    def update_posture(self, status: str, neck: float, torso: float, bad_count: int) -> None:
        with self._lock:
            self.posture_status = status
            self.neck_angle = neck
            self.torso_angle = torso
            self.bad_frame_count = bad_count

    def update_frame(self, jpeg_bytes: bytes) -> None:
        with self._lock:
            self.annotated_frame = jpeg_bytes

    def get_frame(self) -> bytes | None:
        with self._lock:
            return self.annotated_frame

    def get_status_snapshot(self) -> dict:
        with self._lock:
            return {
                "status": self.posture_status,
                "neck_angle": self.neck_angle,
                "torso_angle": self.torso_angle,
                "bad_frame_count": self.bad_frame_count,
                "last_alert_time": self.last_alert_time,
            }

    def mark_alert_sent(self) -> None:
        with self._lock:
            self.last_alert_time = time.monotonic()
