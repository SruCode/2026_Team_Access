import time
import winsound
import threading
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import mediapipe as mp
import matplotlib.pyplot as plt

# MediaPipe Solutions Import (Python 3.12 Compatible)
try:
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
except AttributeError:
    import mediapipe.python.solutions.pose as mp_pose
    import mediapipe.python.solutions.drawing_utils as mp_drawing

# Custom Modules Import
from hazard_checker import HazardZone
from logger import init_log_file, log_incident
from report_generator import generate_summary

# Page Setup
st.set_page_config(page_title="ErgoGuard AI - Multi-Worker 3D", layout="wide", page_icon="🛡️")
init_log_file()

st.sidebar.title("🛡️ ErgoGuard AI Controls")
hazard_threshold = st.sidebar.slider("Back Angle Risk Limit (Deg)", 90, 160, 130)
enable_audio = st.sidebar.toggle("Enable Loud Alarm", value=True)
enable_3d_graph = st.sidebar.toggle("Render 3D Skeleton Graph", value=True)
run_monitoring = st.sidebar.checkbox("Activate Camera Feed", value=True)

# Export Log Option
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Compliance Auditing")
try:
    df_logs = pd.read_csv("incidents.csv")
    st.sidebar.download_button(
        label="📥 Download Incident Report (CSV)",
        data=df_logs.to_csv(index=False),
        file_name="ergoguard_incidents.csv",
        mime="text/csv"
    )
except Exception:
    st.sidebar.info("No incident logs available yet.")

st.title("Worker Safety & Ergonomic Risk Monitoring System (3D Enabled)")

col1, col2 = st.columns([2.5, 1])

with col1:
    st.subheader("Live Feed & Hazard Geofencing")
    frame_placeholder = st.empty()
    if enable_3d_graph:
        graph_3d_placeholder = st.empty()

with col2:
    st.subheader("Real-Time Analytics")
    kpi_angle = st.empty()
    kpi_depth = st.empty()
    kpi_status = st.empty()
    st.markdown("---")
    st.subheader("Aggregated Safety Report")
    report_box = st.empty()

hazard = HazardZone()

def calculate_3d_angle(a, b, c):
    """Calculates 3D interior spatial angle at joint B using world coordinates (x, y, z)."""
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return int(np.degrees(angle))

def _play_beep():
    winsound.Beep(2500, 200)

def trigger_loud_buzzer():
    """Triggers high-decibel hardware beep non-blockingly via threading."""
    threading.Thread(target=_play_beep, daemon=True).start()

def plot_3d_pose(world_landmarks):
    """Renders 3D Matplotlib scatter plot of real-world joint coordinates."""
    fig = plt.figure(figsize=(4, 3), facecolor="#0e1117")
    ax = fig.add_subplot(111, projection='3d', facecolor="#0e1117")
    ax.tick_params(colors='white', labelsize=6)
    
    xs, ys, zs = [], [], []
    for lm in world_landmarks.landmark:
        xs.append(lm.x)
        ys.append(lm.z)  # Swap y & z for vertical elevation perspective
        zs.append(-lm.y)
        
    ax.scatter(xs, ys, zs, c='cyan', s=15)
    ax.set_title("3D Pose Landmarks (Meters)", color="white", fontsize=8)
    plt.tight_layout()
    return fig

# Initialize Pose Tracking
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

last_log_time = 0

if run_monitoring:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    frame_count = 0

    try:
        while cap.isOpened() and run_monitoring:
            ret, frame = cap.read()
            if not ret or frame is None:
                st.error("Camera feed disconnected or unavailable.")
                break

            frame_count += 1
            h, w, _ = frame.shape
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            # Draw Danger Zone Box
            cv2.rectangle(frame, (hazard.x_min, hazard.y_min), (hazard.x_max, hazard.y_max), (0, 0, 255), 2)
            cv2.putText(frame, "DANGER ZONE", (hazard.x_min + 10, hazard.y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            overall_status = "SAFE"
            current_time = time.time()
            angle_3d = 180
            z_depth_meters = 0.0

            if results.pose_landmarks and results.pose_world_landmarks:
                landmarks = results.pose_landmarks.landmark
                world_landmarks = results.pose_world_landmarks.landmark

                # 1. Calculate 3D Posture Angle
                s_3d = [world_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        world_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                        world_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
                h_3d = [world_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        world_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                        world_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]
                k_3d = [world_landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        world_landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
                        world_landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z]

                angle_3d = calculate_3d_angle(s_3d, h_3d, k_3d)

                # 2. Distance Z Depth
                z_depth_meters = round(abs(world_landmarks[mp_pose.PoseLandmark.NOSE.value].z), 2)

                # 3. 2D Wrist Points for Geofence Boundary Check
                r_wrist = [int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w),
                           int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h)]
                l_wrist = [int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * w),
                           int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * h)]

                breach_right = hazard.is_inside(r_wrist[0], r_wrist[1]) if hasattr(hazard, 'is_inside') else (hazard.x_min <= r_wrist[0] <= hazard.x_max and hazard.y_min <= r_wrist[1] <= hazard.y_max)
                breach_left = hazard.is_inside(l_wrist[0], l_wrist[1]) if hasattr(hazard, 'is_inside') else (hazard.x_min <= l_wrist[0] <= hazard.x_max and hazard.y_min <= l_wrist[1] <= hazard.y_max)

                # Draw Pose Skeleton
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                cv2.putText(frame, f"{angle_3d} Deg (3D)", (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w) + 10, 
                                                            int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Violations
                if breach_right or breach_left:
                    overall_status = "HAZARD ZONE BREACH!"
                    if current_time - last_log_time > 1.5:
                        log_incident("Hazard Zone Breach", "HIGH", angle_3d)
                        last_log_time = current_time
                        if enable_audio:
                            trigger_loud_buzzer()
                elif angle_3d < hazard_threshold:
                    overall_status = "POOR ERGONOMICS DETECTED"
                    if current_time - last_log_time > 1.5:
                        log_incident("Poor Ergonomics", "MEDIUM", angle_3d)
                        last_log_time = current_time
                        if enable_audio:
                            trigger_loud_buzzer()

                # Render 3D Skeleton Graph (Throttled every 5 frames)
                if enable_3d_graph and (frame_count % 5 == 0):
                    fig_3d = plot_3d_pose(results.pose_world_landmarks)
                    graph_3d_placeholder.pyplot(fig_3d)
                    plt.close(fig_3d)

            # Display Frame with updated width syntax
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", width="stretch")

            kpi_angle.metric(label="Live 3D Back Angle", value=f"{angle_3d}°")
            kpi_depth.metric(label="Depth from Cam", value=f"{z_depth_meters} m")

            if overall_status == "SAFE":
                kpi_status.success(f"Status: {overall_status}")
            elif overall_status == "HAZARD ZONE BREACH!":
                kpi_status.error(f"🚨 ALERT: {overall_status}")
            else:
                kpi_status.warning(f"⚠️ WARNING: {overall_status}")

            summary = generate_summary()
            report_box.json(summary)

            # Interruption check
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
else:
    frame_placeholder.info("Click 'Activate Camera Feed' in the sidebar to start monitoring.")