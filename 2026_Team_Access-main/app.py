import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time

# Custom Modules Import
from hazard_checker import HazardZone
from logger import init_log_file, log_incident
from report_generator import generate_summary

# Page Setup
st.set_page_config(page_title="ErgoGuard AI", layout="wide")
init_log_file()

st.sidebar.title("🛡️ ErgoGuard AI Controls")
hazard_threshold = st.sidebar.slider("Back Angle Risk Limit (Deg)", 90, 160, 130)

st.title("Worker Safety & Ergonomic Risk Monitoring System")

# Layout Columns
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Feed & Hazard Geofencing")
    frame_placeholder = st.empty()

with col2:
    st.subheader("Real-Time Analytics")
    kpi_angle = st.empty()
    kpi_status = st.empty()
    st.markdown("---")
    st.subheader("Aggregated Safety Report")
    report_box = st.empty()

# MediaPipe Setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
hazard = HazardZone()

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

# Cooldown timer to prevent logging multiple times per second
last_log_time = 0

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.error("Camera feed disconnected.")
        break

    h, w, _ = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Draw Danger Zone Rectangle (Red)
    cv2.rectangle(image, (hazard.x_min, hazard.y_min), (hazard.x_max, hazard.y_max), (0, 0, 255), 2)
    cv2.putText(image, "DANGER ZONE", (hazard.x_min + 10, hazard.y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    status = "SAFE"
    angle = 180
    current_time = time.time()

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Coordinates for Ergonomics
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        
        angle = int(calculate_angle(shoulder, hip, knee))

        # Check Wrist for Danger Zone Entry
        wrist_x = int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w)
        wrist_y = int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h)

        if hazard.is_inside(wrist_x, wrist_y):
            status = "HAZARD ZONE BREACH!"
            if current_time - last_log_time > 2: # Log every 2 seconds
                log_incident("Hazard Zone Breach", "HIGH", angle)
                last_log_time = current_time
        elif angle < hazard_threshold:
            status = "POOR ERGONOMICS DETECTED"
            if current_time - last_log_time > 2:
                log_incident("Poor Ergonomics", "MEDIUM", angle)
                last_log_time = current_time

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Render Streamlit UI
    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
    
    kpi_angle.metric(label="Back Angle", value=f"{angle}°")
    
    if status == "SAFE":
        kpi_status.success(f"Status: {status}")
    else:
        kpi_status.error(f"Status: {status}")

    # Display Live Analytics Summary
    summary = generate_summary()
    report_box.json(summary)

cap.release()