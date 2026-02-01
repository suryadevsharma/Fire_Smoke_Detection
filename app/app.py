import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np
from datetime import datetime
import pandas as pd
import threading
import time

# ----------- WINSOUND (Windows Alarm Support) ------------
try:
    import winsound
    HAS_WINSOUND = True
except:
    HAS_WINSOUND = False

# ----------- GLOBAL ALARM CONTROL FLAGS -------------------
alarm_running = False
alarm_thread = None


def alarm_loop():
    """Continuously beeps while alarm_running == True."""
    global alarm_running
    while alarm_running:
        try:
            winsound.Beep(1200, 500)  # 1200 Hz, 500 ms
            time.sleep(0.2)           # small pause before next beep
        except Exception:
            break


def start_alarm():
    """Start the alarm thread if not already running."""
    global alarm_running, alarm_thread

    # Respect OS support + user toggle
    if not HAS_WINSOUND:
        return
    if not st.session_state.get("enable_alarm", True):
        return

    if not alarm_running:
        alarm_running = True
        alarm_thread = threading.Thread(target=alarm_loop, daemon=True)
        alarm_thread.start()


def stop_alarm():
    """Stop the alarm safely."""
    global alarm_running
    alarm_running = False


# ------------------ PAGE CONFIG & CSS ------------------
st.set_page_config(
    page_title="Fire & Smoke Detection Dashboard",
    layout="wide",
    page_icon="🔥",
)

st.markdown("""
<style>
.alert-banner {
    animation: blinker 1s linear infinite;
    background-color: #ff4b4b;
    padding: 10px;
    border-radius: 10px;
    color: white;
    text-align: center;
    font-weight: 700;
    font-size: 18px;
    margin-bottom: 10px;
}
@keyframes blinker {
  50% { opacity: 0; }
}
.stat-card {
    padding: 12px;
    border-radius: 12px;
    background-color: #262730;
    color: #ffffff;
    text-align: center;
}
.stat-number {
    font-size: 26px;
    font-weight: 700;
}
.stat-label {
    font-size: 13px;
    opacity: 0.8;
}
</style>
""", unsafe_allow_html=True)

# ------------------ SESSION STATE ------------------
if "logs" not in st.session_state:
    st.session_state["logs"] = []  # list of dicts

if "snapshots" not in st.session_state:
    # each item: {"time":..., "type":..., "confidence":..., "image": np.array}
    st.session_state["snapshots"] = []

if "fire_count" not in st.session_state:
    st.session_state["fire_count"] = 0
if "smoke_count" not in st.session_state:
    st.session_state["smoke_count"] = 0
if "last_conf" not in st.session_state:
    st.session_state["last_conf"] = 0.0

# Event flag: are we currently inside a continuous fire/smoke event?
if "event_active" not in st.session_state:
    st.session_state["event_active"] = False

# Alarm toggle state
if "enable_alarm" not in st.session_state:
    st.session_state["enable_alarm"] = True

st.title("🔥 Fire & Smoke Detection Dashboard")


# ------------------ HELPER: REGION OF INTEREST ------------------
def apply_roi(frame, roi_option: str):
    """
    Mask frame outside selected ROI, keeping same resolution.
    roi_option: 'Full frame', 'Top half', 'Bottom half', 'Center'
    """
    if roi_option == "Full frame":
        return frame

    h, w = frame.shape[:2]
    roi_frame = np.zeros_like(frame)

    if roi_option == "Top half":
        roi_frame[0:h // 2, :] = frame[0:h // 2, :]
    elif roi_option == "Bottom half":
        roi_frame[h // 2:h, :] = frame[h // 2:h, :]
    elif roi_option == "Center":
        h1 = h // 4
        h2 = h - h // 4
        w1 = w // 4
        w2 = w - w // 4
        roi_frame[h1:h2, w1:w2] = frame[h1:h2, w1:w2]
    else:
        # fallback
        roi_frame = frame

    return roi_frame


# ------------------ LOAD MODEL + SIDEBAR CONTROLS ------------------
with st.sidebar:
    st.title("⚙️ Controls")
    st.caption("Advanced Fire & Smoke Detection Dashboard")

    try:
        # Adjust path as per your folder structure
        model = YOLO("../models/best.pt")
        st.success("✅ Model loaded successfully")
    except Exception as e:
        st.error(f"❌ Error loading model:\n{e}")
        st.stop()

    conf_threshold = st.slider(
        "Detection Confidence Threshold",
        min_value=0.25,
        max_value=0.90,
        value=0.50,
        step=0.05,
    )

    show_only_fire_smoke = st.checkbox(
        "Show only Fire/Smoke alerts in banner",
        value=True
    )

    st.session_state["enable_alarm"] = st.checkbox(
        "Enable sound alarm (Windows only)",
        value=st.session_state["enable_alarm"],
        help="Uses winsound.Beep on Windows. On other OS, this has no effect."
    )

    st.markdown("### 🧭 Region of Interest")
    roi_option = st.selectbox(
        "Area to monitor",
        ["Full frame", "Top half", "Bottom half", "Center"],
        index=0
    )

    frame_skip = st.slider(
        "Frame skip (for performance)",
        min_value=0,
        max_value=10,
        value=0,
        help="Process every (N+1)th frame in video/webcam/IP cam."
    )

    st.markdown("### 🔎 Current Session")
    st.write(f"Fire detections: **{st.session_state['fire_count']}**")
    st.write(f"Smoke detections: **{st.session_state['smoke_count']}**")
    st.write(f"Max confidence: **{st.session_state['last_conf']:.2f}**")


tabs = st.tabs([
    "🖼 Image",
    "🎞 Video",
    "📷 Webcam",
    "🌐 IP Camera",
    "📊 Logs & Stats"
])


# ------------------ SNAPSHOT HELPER ------------------
def add_snapshot(frame_rgb, label, conf):
    """
    Store screenshot in memory along with type & confidence.
    Called only for NEW events (first frame where fire/smoke appears).
    """
    st.session_state["snapshots"].append({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": label,
        "confidence": round(conf, 3),
        "image": frame_rgb,
    })


# ------------------ COMMON HELPERS ------------------
def process_detections(results):
    """
    Extract fire/smoke info, update logs & stats, and control alarm.

    Event-based:
      - event_active False & fire found  -> new_event True
      - event_active True & fire found   -> new_event False
      - no fire                          -> event_active False

    Returns:
        alert (bool): fire/smoke present in this frame
        label_str (str): summary label
        severity_level (str): 🔥 / 🔥🔥 / 🔥🔥🔥 / Low / ""
        max_conf (float): max confidence
        new_event (bool): True only for FIRST frame of each event
    """
    names = model.names  # class names
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        # No detections -> stop alarm, end any active event
        stop_alarm()
        st.session_state["event_active"] = False
        return False, "", "", 0.0, False

    cls_list = boxes.cls.cpu().numpy()
    conf_list = boxes.conf.cpu().numpy()

    found_fire_or_smoke = False    # at least 1 fire/smoke
    max_conf = 0.0
    label_str = ""

    for cls_id, conf in zip(cls_list, conf_list):
        label = names[int(cls_id)].lower()
        conf_val = float(conf)

        if conf_val > max_conf:
            max_conf = conf_val

        if "fire" in label or "smoke" in label:
            found_fire_or_smoke = True
            label_str = label

            # Update counts
            if "fire" in label:
                st.session_state["fire_count"] += 1
            if "smoke" in label:
                st.session_state["smoke_count"] += 1

            # Log entry
            st.session_state["logs"].append({
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "type": label,
                "confidence": round(conf_val, 3)
            })

    # Update global max confidence
    st.session_state["last_conf"] = max(
        st.session_state["last_conf"], max_conf
    )

    # Alarm control
    if found_fire_or_smoke:
        start_alarm()
    else:
        stop_alarm()

    # ----- Event-based logic -----
    was_active = st.session_state["event_active"]
    if found_fire_or_smoke and not was_active:
        # New event starts here
        new_event = True
        st.session_state["event_active"] = True
    elif not found_fire_or_smoke:
        # Event ended
        new_event = False
        st.session_state["event_active"] = False
    else:
        # Continuing event
        new_event = False

    # Severity based on max_conf
    if max_conf >= 0.80:
        severity = "🔥🔥🔥 Critical"
    elif max_conf >= 0.60:
        severity = "🔥🔥 High"
    elif max_conf >= 0.40:
        severity = "🔥 Medium"
    elif max_conf > 0:
        severity = "Low"
    else:
        severity = ""

    return found_fire_or_smoke, label_str, severity, max_conf, new_event


def draw_frame_with_alert(frame, results):
    annotated = results[0].plot()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return annotated


# ------------------ TAB 1: IMAGE ------------------
with tabs[0]:
    st.subheader("🖼 Image Detection")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        uploaded_image = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_image is not None:
            # For each new image, treat as fresh event context
            st.session_state["event_active"] = False

            file_bytes = np.asarray(
                bytearray(uploaded_image.read()), dtype=np.uint8
            )
            img = cv2.imdecode(file_bytes, 1)

            # Apply ROI
            proc_img = apply_roi(img, roi_option)

            results = model.predict(proc_img, conf=conf_threshold, verbose=False)
            alert, label_str, severity, max_conf, new_event = process_detections(results)
            annotated = draw_frame_with_alert(proc_img, results)

            if alert:
                st.markdown(
                    f'<div class="alert-banner">🚨 ALERT: {label_str.upper()} DETECTED! | SEVERITY: {severity}</div>',
                    unsafe_allow_html=True
                )
                # For images, every detection is effectively its own event
                add_snapshot(annotated, label_str, max_conf)

            st.image(annotated, caption="Detection Result", use_column_width=True)

    with col_right:
        st.markdown("### ℹ Info")
        st.write(
            "- Upload an image containing fire or smoke.\n"
            "- The model will highlight regions with bounding boxes.\n"
            "- ROI lets you select which area is actually monitored.\n"
            "- An alert banner + continuous alarm will trigger when fire/smoke is detected.\n"
            "- A screenshot is stored in memory and shown in the Logs tab.\n"
            "- Data is cleared automatically when app is closed/refreshed."
        )


# ------------------ TAB 2: VIDEO ------------------
with tabs[1]:
    st.subheader("🎞 Video Detection")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        uploaded_video = st.file_uploader(
            "Upload a video",
            type=["mp4", "avi", "mov"]
        )

        if uploaded_video is not None:
            # New video -> reset event state
            st.session_state["event_active"] = False

            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())

            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()
            alert_placeholder = st.empty()

            frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1

                # Frame skip for performance
                if frame_skip > 0 and frame_idx % (frame_skip + 1) != 0:
                    # Just show original frame (no detection)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    stframe.image(frame_rgb, channels="RGB", use_column_width=True)
                    continue

                # Apply ROI
                proc_frame = apply_roi(frame, roi_option)

                results = model.predict(proc_frame, conf=conf_threshold, verbose=False)
                alert, label_str, severity, max_conf, new_event = process_detections(results)
                annotated = draw_frame_with_alert(proc_frame, results)

                if alert:
                    alert_placeholder.markdown(
                        f'<div class="alert-banner">🚨 ALERT: {label_str.upper()} DETECTED! | SEVERITY: {severity}</div>',
                        unsafe_allow_html=True
                    )
                    # Only first frame of each event -> snapshot
                    if new_event:
                        add_snapshot(annotated, label_str, max_conf)
                else:
                    if not show_only_fire_smoke:
                        alert_placeholder.empty()

                stframe.image(annotated, channels="RGB", use_column_width=True)

            cap.release()
            stop_alarm()

    with col_right:
        st.markdown("### ℹ Info")
        st.write(
            "- Upload CCTV footage or any recorded video.\n"
            "- The app will process frame-by-frame and show detections.\n"
            "- You can restrict detection to a region (ROI) and skip frames for speed.\n"
            "- Fire/Smoke alerts will blink above the video with severity + continuous alarm.\n"
            "- Only the **first frame** of each fire/smoke event is stored as a screenshot in memory."
        )


# ------------------ TAB 3: WEBCAM ------------------
with tabs[2]:
    st.subheader("📷 Webcam Live Detection")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        start_cam = st.button("▶ Start Webcam")

        if start_cam:
            # New webcam session -> reset event state
            st.session_state["event_active"] = False

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Could not access webcam. Check your camera permissions.")
            else:
                stframe = st.empty()
                alert_placeholder = st.empty()
                frame_idx = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_idx += 1

                    # Frame skip
                    if frame_skip > 0 and frame_idx % (frame_skip + 1) != 0:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        stframe.image(frame_rgb, channels="RGB", use_column_width=True)
                        continue

                    # Apply ROI
                    proc_frame = apply_roi(frame, roi_option)

                    results = model.predict(proc_frame, conf=conf_threshold, verbose=False)
                    alert, label_str, severity, max_conf, new_event = process_detections(results)
                    annotated = draw_frame_with_alert(proc_frame, results)

                    if alert:
                        alert_placeholder.markdown(
                            f'<div class="alert-banner">🚨 ALERT: {label_str.upper()} DETECTED! | SEVERITY: {severity}</div>',
                            unsafe_allow_html=True
                        )
                        # Only first frame of each event -> snapshot
                        if new_event:
                            add_snapshot(annotated, label_str, max_conf)
                    else:
                        if not show_only_fire_smoke:
                            alert_placeholder.empty()

                    stframe.image(annotated, channels="RGB", use_column_width=True)

                cap.release()
                stop_alarm()

    with col_right:
        st.markdown("### ℹ Info")
        st.write(
            "- Click **Start Webcam** to begin live detection.\n"
            "- Show fire/smoke source or a flame video to the camera.\n"
            "- ROI + frame skip help you focus and improve performance.\n"
            "- Alerts, continuous alarm, and severity label will appear when detected.\n"
            "- Only the **first frame** of each detection event is stored as a screenshot."
        )


# ------------------ TAB 4: IP CAMERA ------------------
with tabs[3]:
    st.subheader("🌐 IP Camera Stream Detection")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        ip_url = st.text_input(
            "Enter IP Camera URL (RTSP/HTTP)",
            value="",
            help=(
                "Example RTSP: rtsp://user:password@192.168.1.10:554/stream\n"
                "Example HTTP/MJPEG: http://192.168.1.10:8080/video"
            )
        )

        start_ip = st.button("▶ Connect & Start IP Stream")

        if start_ip:
            if not ip_url.strip():
                st.error("⚠ Please enter a valid IP camera URL.")
            else:
                # New IP cam session -> reset event state
                st.session_state["event_active"] = False

                cap = cv2.VideoCapture(ip_url)
                if not cap.isOpened():
                    st.error("❌ Could not open IP camera stream. Check URL / camera / network.")
                else:
                    stframe = st.empty()
                    alert_placeholder = st.empty()
                    frame_idx = 0

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            st.warning("Stream ended or cannot read frames from IP camera.")
                            break

                        frame_idx += 1

                        # Frame skip
                        if frame_skip > 0 and frame_idx % (frame_skip + 1) != 0:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            stframe.image(frame_rgb, channels="RGB", use_column_width=True)
                            continue

                        # Apply ROI
                        proc_frame = apply_roi(frame, roi_option)

                        results = model.predict(proc_frame, conf=conf_threshold, verbose=False)
                        alert, label_str, severity, max_conf, new_event = process_detections(results)
                        annotated = draw_frame_with_alert(proc_frame, results)

                        if alert:
                            alert_placeholder.markdown(
                                f'<div class="alert-banner">🚨 ALERT: {label_str.upper()} DETECTED! | SEVERITY: {severity}</div>',
                                unsafe_allow_html=True
                            )
                            if new_event:
                                add_snapshot(annotated, label_str, max_conf)
                        else:
                            if not show_only_fire_smoke:
                                alert_placeholder.empty()

                        stframe.image(annotated, channels="RGB", use_column_width=True)

                    cap.release()
                    stop_alarm()

    with col_right:
        st.markdown("### ℹ How to Use IP Camera")
        st.write(
            "- Make sure your IP camera is **reachable** from this machine (same network / port open).\n"
            "- Use the correct **stream URL** (RTSP or HTTP MJPEG):\n"
            "  - `rtsp://user:password@<ip>:<port>/...`\n"
            "  - `http://<ip>:<port>/video` (IP Webcam app, etc.)\n"
            "- ROI + frame skip also apply to this stream.\n"
            "- If it doesn’t connect, check IP, port, credentials, firewall, and that the stream works in VLC first."
        )


# ------------------ TAB 5: LOGS & STATS ------------------
with tabs[4]:
    st.subheader("📊 Detection Logs & Statistics")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-number">{st.session_state['fire_count']}</div>
                <div class="stat-label">Total Fire Detections</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-number">{st.session_state['smoke_count']}</div>
                <div class="stat-label">Total Smoke Detections</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-number">
                    {st.session_state['last_conf']:.2f}
                </div>
                <div class="stat-label">Max Confidence Seen</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### 📝 Detailed Detection Log")

    # Buttons for logs
    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        clear_btn = st.button("🧹 Clear Logs & Stats")
    with col_btn2:
        # Placeholder for download button (only shown if logs exist)
        download_placeholder = st.empty()

    if len(st.session_state["logs"]) == 0:
        st.info("No detections logged yet. Run image/video/webcam/IP cam first.")
    else:
        df_logs = pd.DataFrame(st.session_state["logs"])

        # ---- Filters ----
        st.markdown("#### 🔍 Filter Logs")
        fcol1, fcol2 = st.columns([1, 2])
        with fcol1:
            type_filter = st.selectbox(
                "Type",
                ["All", "fire", "smoke"],
                index=0
            )
        with fcol2:
            search_text = st.text_input(
                "Search (time / type / confidence contains)",
                ""
            )

        if type_filter != "All":
            df_logs = df_logs[df_logs["type"].str.contains(type_filter, case=False)]

        if search_text:
            lower = search_text.lower()
            df_logs = df_logs[df_logs.apply(
                lambda row: lower in (" ".join(map(str, row.values))).lower(),
                axis=1
            )]

        st.dataframe(df_logs, use_container_width=True)

        # ---- Simple chart ----
        if not df_logs.empty:
            st.markdown("#### 📈 Detection Count by Type")
            counts = df_logs["type"].value_counts().reset_index()
            counts.columns = ["type", "count"]
            counts = counts.set_index("type")
            st.bar_chart(counts)

        # CSV download (filtered)
        csv_data = df_logs.to_csv(index=False).encode("utf-8")
        download_placeholder.download_button(
            label="⬇ Download Logs as CSV (filtered)",
            data=csv_data,
            file_name="detection_logs.csv",
            mime="text/csv",
        )

    # Handle clear button AFTER displaying (so UI updates correctly next run)
    if 'clear_applied' not in st.session_state:
        st.session_state['clear_applied'] = False

    if clear_btn:
        st.session_state["logs"] = []
        st.session_state["snapshots"] = []
        st.session_state["fire_count"] = 0
        st.session_state["smoke_count"] = 0
        st.session_state["last_conf"] = 0.0
        st.session_state["event_active"] = False
        stop_alarm()
        st.success("✅ Logs & stats cleared for this session.")

    st.markdown("### 📸 Detection Screenshots (first frame of each event)")
    if len(st.session_state["snapshots"]) == 0:
        st.info("No screenshots yet. They will appear here when a fire/smoke event is detected.")
    else:
        for snap in st.session_state["snapshots"]:
            st.markdown(
                f"**Time:** {snap['time']} | **Type:** {snap['type']} | **Conf:** {snap['confidence']}"
            )
            st.image(snap["image"], width=320)
            st.markdown("---")
