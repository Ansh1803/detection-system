import tempfile

import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import RTCConfiguration, webrtc_streamer
from ultralytics import YOLO

st.set_page_config(page_title="Helmet Detection", page_icon="", layout="wide")


@st.cache_resource
def load_model():
    return YOLO("models/best.pt")


model = load_model()
RTC_CFG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

if "mode" not in st.session_state:
    st.session_state.mode = "upload"

st.title(" Helmet Detection")
st.caption("YOLOv8 model that detects riders with and without helmets.")

conf = st.sidebar.slider("Confidence threshold", 0.1, 0.9, 0.4, 0.05)
st.sidebar.caption("Lower = more detections, higher = fewer but surer.")

b1, b2, _ = st.columns([1, 1, 4])
if b1.button("📷 Live camera", use_container_width=True):
    st.session_state.mode = "live"
if b2.button("⬆️ Upload", use_container_width=True):
    st.session_state.mode = "upload"


def count_classes(result):
    names = [model.names[int(c)] for c in result.boxes.cls]
    return {n: names.count(n) for n in sorted(set(names))}


def show_counts(counts):
    if not counts:
        st.info("No detections. Try lowering the confidence threshold.")
        return
    cols = st.columns(len(counts))
    for col, (name, n) in zip(cols, counts.items()):
        col.metric(name, n)


# ---------------- LIVE CAMERA ----------------
if st.session_state.mode == "live":
    with st.container(border=True):
        st.subheader("Live detection")
        st.write("Click **START**, then allow camera access in your browser.")

        def callback(frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            res = model.predict(img, conf=conf, verbose=False)[0]
            return av.VideoFrame.from_ndarray(res.plot(), format="bgr24")

        webrtc_streamer(
            key="live-helmet",
            video_frame_callback=callback,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration=RTC_CFG,
            async_processing=True,
        )

# ---------------- UPLOAD ----------------
else:
    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            st.subheader("Input")
            f = st.file_uploader(
                "Upload an image or a video",
                type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
            )
            if f and f.type.startswith("image"):
                st.image(f, caption="Original", use_container_width=True)
            elif f:
                st.video(f)

    with right:
        with st.container(border=True):
            st.subheader("Output")
            if not f:
                st.info("Upload a file on the left to see detections here.")
            elif f.type.startswith("image"):
                img = Image.open(f).convert("RGB")
                bgr = np.array(img)[:, :, ::-1]
                res = model.predict(bgr, conf=conf, verbose=False)[0]
                st.image(res.plot()[:, :, ::-1], caption="Detections", use_container_width=True)
                show_counts(count_classes(res))
            else:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tmp.write(f.getvalue())
                tmp.close()

                cap = cv2.VideoCapture(tmp.name)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                frame_box = st.empty()
                counts_box = st.empty()
                bar = st.progress(0)
                i = 0
                while cap.isOpened():
                    ok, frame = cap.read()
                    if not ok:
                        break
                    res = model.predict(frame, conf=conf, verbose=False)[0]
                    frame_box.image(res.plot()[:, :, ::-1], use_container_width=True)
                    with counts_box.container():
                        show_counts(count_classes(res))
                    i += 1
                    bar.progress(min(i / total, 1.0))
                cap.release()
                st.success("Video processed")