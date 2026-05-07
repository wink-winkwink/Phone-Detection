"""
app.py - Main Streamlit Web Application for Phone Usage Monitor.

Uses the VideoProcessor class pattern with Streamlit-WebRTC
for browser camera access and real-time YOLOv8 detection.
Audio alerts play on the client browser via Base64-embedded
JavaScript injection after user clicks 'Start Monitoring'.
"""

import time
import av
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

from src.detector import PhoneDetector
from src.tracker import PickupTracker
from src.alert_system import AlertSystem


# ─── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="📱 Phone Usage Monitor",
    page_icon="📱",
    layout="wide",
)

st.title("📱 AI Phone Usage Monitor")
st.markdown(
    "This app uses **YOLOv8** to detect when you pick up your phone. "
    "If you exceed **5 pickups per hour**, an alert will sound!"
)


# ─── Autoplay Policy: Require user click first ─────────────────
# Browsers block audio autoplay until the user interacts with the
# page. This button satisfies that requirement.
if 'audio_enabled' not in st.session_state:
    st.session_state.audio_enabled = False

if not st.session_state.audio_enabled:
    st.info("🔊 Click the button below to enable audio alerts, then start the camera.")
    if st.button("🔊 Start Monitoring (Enable Audio)", type="primary", use_container_width=True):
        st.session_state.audio_enabled = True
        # Inject a silent audio play to unlock the browser's audio context
        st.markdown(
            """
            <script>
                var ctx = new (window.AudioContext || window.webkitAudioContext)();
                ctx.resume().then(function() {
                    console.log('AudioContext unlocked by user click.');
                });
            </script>
            """,
            unsafe_allow_html=True,
        )
        st.rerun()
    st.stop()

st.success("🔊 Audio alerts enabled. Start the camera below.")


# ─── VideoProcessor Class ──────────────────────────────────────
class PhoneUsageVideoProcessor(VideoProcessorBase):
    """
    Custom VideoProcessor that handles per-frame detection,
    tracking, drawing, and alert triggering.
    """

    def __init__(self):
        self.detector = PhoneDetector(model_name='yolov8s.pt', confidence=0.25)
        self.tracker = PickupTracker(hourly_limit=5, cooldown_seconds=4)
        self.alert = AlertSystem(cooldown_seconds=5)

        # Shared state for the Streamlit UI to read
        self.pickup_count = 0
        self.is_holding = False
        self.limit_exceeded = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
        Called for each video frame received from the browser camera.
        """
        img = frame.to_ndarray(format="bgr24")

        # ── Detection ───────────────────────────────────────────
        detections = self.detector.detect(img)

        # ── Tracking ────────────────────────────────────────────
        status = self.tracker.update(detections)

        # Update shared state for UI
        self.pickup_count = status['pickup_count']
        self.is_holding = status['is_holding']
        self.limit_exceeded = status['limit_exceeded']

        # ── Draw Bounding Boxes ─────────────────────────────────
        for box in detections['persons']:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, "Person", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        for box in detections['phones']:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, "Phone", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # ── HUD Overlay ────────────────────────────────────────
        holding_text = "Holding" if status['is_holding'] else "Idle"
        count_text = f"Pickups: {status['pickup_count']}/5"

        cv2.putText(img, f"Status: {holding_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, count_text, (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # ── Alert Logic ────────────────────────────────────────
        if status['limit_exceeded']:
            cv2.putText(img, "LIMIT EXCEEDED!", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
            if detections['phones']:
                self.alert.trigger()

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ─── WebRTC Streamer ───────────────────────────────────────────
ctx = webrtc_streamer(
    key="phone-monitor",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=PhoneUsageVideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# ─── Sidebar Header ───────────────────────────────────────────
st.sidebar.header("📊 Live Stats")

# ─── Placeholders for live-updating UI ─────────────────────────
status_placeholder = st.sidebar.empty()

if ctx.state.playing and ctx.video_processor:
    # ── Live Polling Loop ──────────────────────────────────────
    processor = ctx.video_processor

    while ctx.state.playing:
        # Read shared state from the processor
        pickup_count = processor.pickup_count
        is_holding = processor.is_holding
        limit_exceeded = processor.limit_exceeded

        status_emoji = "🔴 Holding Phone" if is_holding else "🟢 Idle"
        limit_emoji = "⚠️ Limit Exceeded!" if limit_exceeded else "✅ Within Limit"

        # Update sidebar stats
        status_placeholder.markdown(
            f"""
            **Pickup Count:** `{pickup_count} / 5`

            **Status:** {status_emoji}

            **Limit:** {limit_emoji}
            """
        )

        # ── Browser Audio Alert (Base64 MP3 via st.components.v1.html)
        if processor.alert.consume():
            processor.alert.play_alert_in_browser()

        # Poll every 500ms
        time.sleep(0.5)

else:
    st.sidebar.info("📷 Click **START** above to begin monitoring.")

# ─── Reset Button ─────────────────────────────────────────────
if ctx.video_processor:
    if st.sidebar.button("🔄 Reset Counter"):
        ctx.video_processor.tracker.reset()
        st.rerun()
