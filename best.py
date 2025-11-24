import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from gtts import gTTS
import os
import time
from typing import Optional
import requests
import collections
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import io

# Popup configuration
POPUP_DURATION = 2.0  # seconds to display the transient popup

def speak(text):
    """Converts text to speech and plays it using st.audio."""
    try:
        tts = gTTS(text=text, lang='en')
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        st.audio(audio_bytes, format='audio/mp3', autoplay=True)
    except Exception as e:
        st.error(f"Error in TTS: {e}")

def show_detection_popup(label: str, conf: float, severity: str = 'warning') -> None:
    """Record a detection in session state so the UI can show a transient popup.

    severity: 'warning'|'info' - determines whether the UI shows a warning or info popup.
    Also records a `pending_tts` text that can be consumed by a later text-to-speech step.
    """
    now = time.time()
    st.session_state['last_detection'] = {
        'label': label,
        'conf': conf,
        'time': now,
        'severity': severity,
    }

    # Cooldown to avoid speaking too often for the same object
    COOLDOWN_SECONDS = 5.0

    last_spoken_time = st.session_state.get('last_spoken_time', 0.0)
    last_spoken_label = st.session_state.get('last_spoken_label')

    if label != last_spoken_label or (now - last_spoken_time) > COOLDOWN_SECONDS:
        tts_text = f"{label} is prohibited, please collect"
        st.session_state['pending_tts'] = tts_text
        # The speak function will be called in the main loop to avoid issues with webrtc callbacks
        st.session_state['last_spoken_label'] = label
        st.session_state['last_spoken_time'] = now


# Ensure session state key exists
if 'last_detection' not in st.session_state:
    st.session_state['last_detection'] = None
if 'pending_tts' not in st.session_state:
    st.session_state['pending_tts'] = None
if 'last_spoken_label' not in st.session_state:
    st.session_state['last_spoken_label'] = None
if 'last_spoken_time' not in st.session_state:
    st.session_state['last_spoken_time'] = 0.0
if 'collected_items' not in st.session_state:
    st.session_state['collected_items'] = []
if 'session_ended' not in st.session_state:
    st.session_state['session_ended'] = False
if 'collecting_item' not in st.session_state:
    st.session_state['collecting_item'] = None

# Load your YOLO model
MODEL_PATH = "best.pt"

def download_model_from_url(url: str, dst: str) -> bool:
    """Download model from URL to dst. Returns True on success."""
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dst, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception as e:
        st.warning(f"Model download failed: {e}")
        return False


# Ensure model file exists or try to download from `st.secrets['MODEL_URL']` or
# environment variable `MODEL_URL`. Streamlit Cloud cannot store very large
# files in the repo; use Git LFS or host the model externally and set the URL.
if not os.path.exists(MODEL_PATH):
    model_url = None
    try:
        model_url = st.secrets.get('MODEL_URL') if hasattr(st, 'secrets') else None
    except Exception:
        model_url = None
    if not model_url:
        model_url = os.environ.get('MODEL_URL')

    if model_url:
        st.info("Model file not found locally — downloading model from provided URL...")
        ok = download_model_from_url(model_url, MODEL_PATH)
        if not ok:
            st.error("Failed to download model. Please upload `best.pt` manually or set a valid MODEL_URL.")
    else:
        st.warning("Model file `best.pt` not found. To deploy on Streamlit Cloud either: (1) add `best.pt` to the repo via Git LFS, or (2) provide a download URL in `st.secrets['MODEL_URL']` or env var `MODEL_URL`.")

model = YOLO(MODEL_PATH)

st.title("Exam Classroom Scanner")

if st.button("End Exam Scan"):
    st.session_state.session_ended = True

if st.session_state.session_ended:
    st.header("Collected Items Report")
    if not st.session_state.collected_items:
        st.write("No prohibited items were collected during the scan.")
    else:
        item_counts = collections.Counter(item['label'] for item in st.session_state.collected_items)
        st.table(item_counts)
else:
    st.write("This application uses a YOLO model to scan for prohibited items in an exam setting.")

    class VideoTransformer(VideoTransformerBase):
        def __init__(self):
            self.model = model

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")

            # Run YOLO prediction
            results = self.model.predict(img, conf=0.5)

            # Plot detections
            annotated_frame = results[0].plot()

            # If there are detections, extract the first label+confidence and show popup
            try:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    # Attempt robust extraction of class and confidence
                    try:
                        cls_list = boxes.cls.cpu().numpy().astype(int).tolist()
                        conf_list = boxes.conf.cpu().numpy().tolist()
                    except Exception:
                        try:
                            cls_list = boxes.cls.numpy().astype(int).tolist()
                            conf_list = boxes.conf.numpy().tolist()
                        except Exception:
                            cls_list = [int(x) for x in boxes.cls]
                            conf_list = [float(x) for x in boxes.conf]

                    cls0 = cls_list[0]
                    conf0 = conf_list[0]
                    label = self.model.names.get(cls0, str(cls0)) if hasattr(self.model, 'names') else str(cls0)
                    # This will update st.session_state.last_detection
                    show_detection_popup(label, conf0, severity='warning')

            except Exception:
                # Non-fatal: if extraction fails, silently continue
                pass

            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    webrtc_ctx = webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

    spoken_text_placeholder = st.empty()
    collect_button_placeholder = st.empty()
    collection_status_placeholder = st.empty()

    # Handle collection timer and status
    if st.session_state.collecting_item:
        collection_start_time = st.session_state.collecting_item['collection_start_time']
        elapsed_time = time.time() - collection_start_time
        if elapsed_time < 5:
            collection_status_placeholder.info(f"Collecting {st.session_state.collecting_item['label']}... {5 - int(elapsed_time)}s remaining")
        else:
            st.session_state.collected_items.append(st.session_state.collecting_item)
            collection_status_placeholder.success(f"Collected {st.session_state.collecting_item['label']}!")
            st.session_state.collecting_item = None
            st.session_state.last_detection = None # Clear last detection to hide collect button
            collection_status_placeholder.empty()
            st.experimental_rerun()

    # Render transient popup if a recent detection was recorded
    last = st.session_state.get('last_detection')
    if last is not None and (time.time() - last.get('time', 0.0)) < POPUP_DURATION:
        if last.get('severity', 'info') == 'warning':
            st.warning(f"Detected: {last['label']} ({last['conf']:.2f})", icon="⚠️")
        else:
            st.info(f"Detected: {last['label']} ({last['conf']:.2f})", icon="📣")

    # Update spoken text display and speak
    if st.session_state.get('pending_tts'):
        tts_text = st.session_state.get('pending_tts')
        spoken_text_placeholder.write(f"Speaking: {tts_text}")
        speak(tts_text)
        st.session_state['pending_tts'] = None


    # Display collect button if an item is detected
    if last is not None and not st.session_state.collecting_item:
        label = last['label']
        if collect_button_placeholder.button(f"Collect {label}"):
            st.session_state.collecting_item = {
                'label': label,
                'collection_start_time': time.time()
            }
            collect_button_placeholder.empty()
            st.experimental_rerun()
