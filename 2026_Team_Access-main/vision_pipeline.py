import cv2
import numpy as np
import mediapipe as mp

class ErgoVisionPipeline:
    def __init__(self, source=0):
        """
        Initialize MediaPipe Pose and Video Capture.
        :param source: 0 for live webcam or 'video.mp4' for pre-recorded file.
        """
        self.cap = cv2.VideoCapture(source)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def calculate_angle(self, a, b, c):
        """Calculates internal angle theta at point B given 3 joints (A, B, C)."""
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360.0 - angle
        return angle

    def process_frame(self, frame):
        """
        Processes a single BGR frame, tracks body landmarks, 
        and extracts key metrics.
        """
        if frame is None:
            return None, {}

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        metrics = {
            "back_angle": 180,
            "right_wrist": (0, 0),
            "pose_detected": False
        }

        if results.pose_landmarks:
            metrics["pose_detected"] = True
            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape

            # Extract 2D coordinates for Hip-Shoulder-Knee posture tracking
            shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                        landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x, 
                   landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x, 
                    landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]

            # Wrist position for hazard detection
            wrist_x = int(landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w)
            wrist_y = int(landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h)

            metrics["back_angle"] = int(self.calculate_angle(shoulder, hip, knee))
            metrics["right_wrist"] = (wrist_x, wrist_y)

            # Draw skeleton overlays
            self.mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS
            )

        return frame, metrics

    def release(self):
        """Release camera hardware resources."""
        self.cap.release()


# Standalone Testing Block (For local verification)
if __name__ == "__main__":
    pipeline = ErgoVisionPipeline(source=0)
    print("Testing CV Pipeline. Press 'q' to exit...")

    while pipeline.cap.isOpened():
        ret, frame = pipeline.cap.read()
        if not ret:
            break

        processed_frame, metrics = pipeline.process_frame(frame)
        
        cv2.putText(
            processed_frame, 
            f"Angle: {metrics['back_angle']} deg", 
            (20, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, (0, 255, 0), 2
        )
        cv2.imshow("CV Lead Sandbox", processed_frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    pipeline.release()
    cv2.destroyAllWindows()