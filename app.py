import time
from collections import Counter

import av
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode

from ultralytics import YOLO


st.set_page_config(
    page_title="YOLO Real-Time Object Tracker",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 YOLO Real-Time Object Detection & Tracking")
st.caption("Webcam → YOLO → ByteTrack → Live object IDs and counts")

@st.cache_resource
def load_model():
    # If yolo26n.pt is unavailable in your Ultralytics version,
    # replace it with a compatible pretrained model such as yolo11n.pt.
    return YOLO("yolo26n.pt")

model = load_model()

st.sidebar.header("Settings")
confidence = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.05,
    max_value=0.80,
    value=0.10,
    step=0.05,
)
iou = st.sidebar.slider(
    "IoU threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05,
)

st.sidebar.info(
    "Lower confidence can help with small objects such as phones, "
    "but may also create more false detections."
)

# Per-WebRTC-session state is kept inside the processor closure.
def make_callback():
    previous_time = time.time()
    unique_ids = set()

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        nonlocal previous_time, unique_ids

        image = frame.to_ndarray(format="bgr24")

        results = model.track(
            image,
            persist=True,
            tracker="bytetrack.yaml",
            conf=confidence,
            iou=iou,
            verbose=False,
        )

        result = results[0]
        annotated = result.plot()

        counts = Counter()
        tracked_ids = []

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                counts[class_name] += 1

                if box.id is not None:
                    track_id = int(box.id[0])
                    tracked_ids.append((class_name, track_id))
                    unique_ids.add((class_name, track_id))

        now = time.time()
        fps = 1.0 / max(now - previous_time, 1e-6)
        previous_time = now

        # Dashboard background
        dashboard_height = 75 + 28 * min(len(counts), 8)
        cv2.rectangle(
            annotated,
            (10, 10),
            (390, dashboard_height),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            annotated,
            f"FPS: {fps:.1f}",
            (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            annotated,
            f"Unique tracked IDs: {len(unique_ids)}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        y = 92
        for name, count in counts.most_common(8):
            cv2.putText(
                annotated,
                f"{name}: {count}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            y += 28

        return av.VideoFrame.from_ndarray(
            annotated,
            format="bgr24",
        )

    return video_frame_callback


# Create the callback once per Streamlit session.
if "video_callback" not in st.session_state:
    st.session_state.video_callback = make_callback()

rtc_configuration = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
    ]
}

st.write("### Live camera")
st.write("Click **START** and allow camera permission when your browser asks.")

webrtc_streamer(
    key="yolo-live-tracker",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=rtc_configuration,
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
    video_frame_callback=st.session_state.video_callback,
    async_processing=True,
    media_toggle_controls=True,
)

st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Detector", "YOLO")
with col2:
    st.metric("Tracker", "ByteTrack")
with col3:
    st.metric("Input", "Webcam")

st.warning(
    "For a cloud deployment, webcam video is processed on the server. "
    "Performance depends on the available CPU/GPU and network connection. "
    "If WebRTC cannot establish a connection on a particular network, "
    "a TURN server may be required."
)
