import subprocess
import threading
import time

from .detector import PostureDetector
from .state import SharedState

COOLDOWN_SECONDS = 30
ALERT_MESSAGES = [
    "Por favor, corrija sua postura! Você está curvado.",
    "Atenção! Sua postura está incorreta. Sente-se ereto.",
    "Lembrete de postura: endireite as costas e levante a cabeça.",
    "Cuide da sua saúde! Corrija sua postura agora.",
]
_msg_index = 0


class PostureAlerter:
    def __init__(self, state: SharedState):
        self._state = state
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._alert_loop, daemon=True)
        self._thread.start()

    def _alert_loop(self) -> None:
        global _msg_index
        while self._state.running:
            snapshot = self._state.get_status_snapshot()
            if (
                snapshot["status"] == "Ruim"
                and snapshot["bad_frame_count"] >= PostureDetector.CONSECUTIVE_BAD_FRAMES
                and (time.monotonic() - snapshot["last_alert_time"]) >= COOLDOWN_SECONDS
            ):
                msg = ALERT_MESSAGES[_msg_index % len(ALERT_MESSAGES)]
                _msg_index += 1
                self._speak(msg)
                self._state.mark_alert_sent()
            time.sleep(1.0)

    @staticmethod
    def _speak(message: str) -> None:
        try:
            subprocess.Popen(
                ["espeak-ng", "-v", "pt-br", "-s", "135", "-p", "55", message],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            print(f"[ALERTA] {message}")
