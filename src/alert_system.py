"""
alert_system.py - Web-based audio alert logic (Client-Side).

Reads assets/alert.mp3, converts it to a Base64 data URI, and
provides a function to inject an <audio autoplay> tag into the
browser using st.components.v1.html.
"""

import base64
import os
import threading
import time

import streamlit.components.v1 as components

# ─── Load alert.mp3 as Base64 (once at import time) ───────────
ALERT_MP3_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'assets', 'alert.mp3'
)


def _load_audio_base64(file_path):
    """
    Read an MP3 file and return a Base64-encoded data URI string.

    Args:
        file_path: Path to the .mp3 file.

    Returns:
        A data URI string: 'data:audio/mp3;base64,...'
        or None if the file does not exist.
    """
    abs_path = os.path.abspath(file_path)
    if not os.path.exists(abs_path):
        print(f"[AlertSystem] Warning: Audio file not found at {abs_path}")
        return None

    with open(abs_path, 'rb') as f:
        audio_bytes = f.read()

    b64 = base64.b64encode(audio_bytes).decode('utf-8')
    return f"data:audio/mp3;base64,{b64}"


# Pre-load once so we don't re-read the file on every alert
ALERT_AUDIO_B64 = _load_audio_base64(ALERT_MP3_PATH)


class AlertSystem:
    """Thread-safe alert manager for browser-based audio playback."""

    def __init__(self, cooldown_seconds=5):
        """
        Initialize the alert system.

        Args:
            cooldown_seconds: Minimum seconds between consecutive alerts.
        """
        self._lock = threading.Lock()
        self._should_alert = False
        self._cooldown_seconds = cooldown_seconds
        self._last_alert_time = 0

    def trigger(self):
        """
        Set the alert flag (called from VideoProcessor thread).
        Respects cooldown to prevent rapid-fire alerts.
        """
        current_time = time.time()
        with self._lock:
            if (current_time - self._last_alert_time) >= self._cooldown_seconds:
                self._should_alert = True
                self._last_alert_time = current_time

    def consume(self):
        """
        Check and reset the alert flag (called from Streamlit main thread).

        Returns:
            True if an alert was triggered since last consume, False otherwise.
        """
        with self._lock:
            if self._should_alert:
                self._should_alert = False
                return True
            return False

    @staticmethod
    def play_alert_in_browser():
        """
        Inject an <audio autoplay> tag into the browser via
        st.components.v1.html to play assets/alert.mp3.

        The MP3 is embedded as a Base64 data URI so no file-serving
        or external URL is needed. The 'autoplay' attribute makes
        it trigger immediately when this function is called.
        """
        if not ALERT_AUDIO_B64:
            return

        components.html(
            f"""
            <audio autoplay>
                <source src="{ALERT_AUDIO_B64}" type="audio/mp3">
            </audio>
            """,
            height=0,
        )
