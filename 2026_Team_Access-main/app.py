import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
from datetime import datetime
import pandas as pd

# -------------------------------------------------------------------
# 1. Page Configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title="ErgoGuard AI - Safety Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# Initialize Incident Log in Session State (persists across redraws)
if "incident_log" not in st.session_state:
    st.session_state.incident_log = []

# -------------------------------------------------------------------
# 2. Sidebar Controls (Min 30–75)
# -------------------------------------------------------------------
st.sidebar.title("🛡️ ErgoGuard Controls")
st.sidebar.markdown("---")

# Control 1: Video Source Selector
video_source_option = st.sidebar.radio(
    "Select Input Source:",
    ("Webcam Feed", "Pre-recorded Demo Video")
)

# Control 2: Risk Threshold Sliders
st.sidebar.subheader("Threshold Configuration")
back_threshold = st.sidebar.slider(
    "Ergonomic Back Angle Limit (°)", 
    min_value=90, 
    max_value=160, 
    value=130, 
    help="Angles below this value trigger a Poor Posture warning."
)

enable_face_blur = st.sidebar.checkbox("Enable Privacy Mode (Blur Face)", value=True)

# Cooldown to avoid spamming log entries (in seconds)
LOG_COOLDOWN = 2 
if "last_log_time" not in st.session_state:
    st.session_state.last_log_time = 0

# -------------------------------------------------------------------
# 3. Main Dashboard Layout (Min 0–30)
# -------------------------------------------------------------------
st.title("🛡️ ErgoGuard AI — Real-Time Safety & Ergonomic Monitor")
st.caption("Computer Vision Powered Workstation Risk Assessment System")

# Create 2 Main Columns
col_video, col_metrics = st.columns([2, 1])

with col_video:
    st.subheader("📹 Live Monitor")
    video_placeholder = st.empty()

with col_metrics:
    st.subheader("📊 Live Metrics")
    kpi_angle = st.metric(label="Back Angle", value="-- °")
    kpi_status = st.empty()
    kpi_score = st.metric(label="Safety Score", value="100%")

st.markdown("---")

# Incident Log Table Section (Min 100–120)
st.subheader("📋 Real-Time Incident Log")
table_placeholder = st.empty()

# -------------------------------------------------------------------
# 4. Helper Functions (Person 1 & 2 Core Logic Hooked Here)
# -------------------------------------------------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

def log_incident(incident_type, angle_val):
    current_time = time.time()
    # Log only if cooldown period has passed
    if current_time - st.session_state.last_log_time > LOG_COOLDOWN:
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.incident_log.append({
            "Timestamp": timestamp,
            "Violation Event": incident_type,
            "Angle Registered": f"{angle_val}°" if angle_val else "N/A",
            "Severity": "High" if "HAZARD" in incident_type else "Medium"
        })
        st.session_state.last_log_time = current_time

# -------------------------------------------------------------------
# 5. Video Loop Execution (Min 75–100)
# -------------------------------------------------------------------
if video_source_option == "Webcam Feed":
    cap = cv2.VideoCapture(0)
else:
    # Make sure to have a demo.mp4 in your directory or replace path
    cap = cv2.VideoCapture("demo.mp4") 

safety_violations_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.warning("Video stream ended or camera unreachable.")
        break

    # Convert frame BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    
    # Draw Hazard Boundary Box on frame (x1=350, y1=100, x2=550, y2=350)
    h, w, _ = image.shape
    cv2.rectangle(image, (int(w*0.6), int(h*0.2)), (int(w*0.9), int(h*0.7)), (255, 0, 0), 2)
    cv2.putText(image, "DANGER ZONE", (int(w*0.6) + 10, int(h*0.2) - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    status_str = "SAFE POSTURE"
    current_angle = 180

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Extract joints for posture angle
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        
        current_angle = int(calculate_angle(shoulder, hip, knee))

        # Check Privacy Toggle (Blur Head Area)
        if enable_face_blur:
            nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
            nx, ny = int(nose.x * w), int(nose.y * h)
            x1, y1 = max(0, nx - 60), max(0, ny - 60)
            x2, y2 = min(w, nx + 60), min(h, ny + 60)
            if x2 > x1 and y2 > y1:
                image[y1:y2, x1:x2] = cv2.GaussianBlur(image[y1:y2, x1:x2], (51, 51), 0)

        # 1. Posture Check
        if current_angle < back_threshold:
            status_str = "POOR ERGONOMICS DETECTED"
            log_incident("POOR POSTURE (BAD BEND)", current_angle)
            safety_violations_count += 1

        # 2. Hazard Zone Breach Check (Right Wrist inside bounding box)
        rw_x = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x
        rw_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
        if 0.6 < rw_x < 0.9 and 0.2 < rw_y < 0.7:
            status_str = "HAZARD ZONE BREACH!"
            log_incident("HAZARD ZONE VIOLATION", current_angle)
            safety_violations_count += 1

        # Draw skeleton overlay
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Update Dashboard Components
    video_placeholder.image(image, channels="RGB", use_container_width=True)
    kpi_angle.metric(label="Back Angle", value=f"{current_angle}°")

    # Dynamic status pill rendering
    if status_str == "SAFE POSTURE":
        kpi_status.success("🟢 STATUS: NORMAL")
    elif "ERGONOMICS" in status_str:
        kpi_status.warning(f"⚠️ {status_str}")
    else:
        kpi_status.error(f"🚨 {status_str}")

    # Safety Score calculation
    score = max(0, 100 - (len(st.session_state.incident_log) * 5))
    kpi_score.metric(label="Shift Safety Score", value=f"{score}%")

    # Render Incident Table
    if st.session_state.incident_log:
        df = pd.DataFrame(st.session_state.incident_log)
        table_placeholder.dataframe(df.iloc[::-1], use_container_width=True) # newest first
    else:
        table_placeholder.info("No violations logged yet.")

cap.release()