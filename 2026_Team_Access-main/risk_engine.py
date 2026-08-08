import datetime

class ErgoRiskEngine:
    def __init__(self, angle_warning=140, angle_danger=120):
        # Posture thresholds in degrees
        self.angle_warning = angle_warning
        self.angle_danger = angle_danger
        
        # Virtual Hazard Zone Coordinates: (x1, y1, x2, y2)
        # Defines a box in top-right portion of camera frame
        self.hazard_zone = (400, 50, 640, 350)
        
        # Array to store dynamic incident alerts
        self.incident_logs = []

    def evaluate_posture(self, back_angle):
        """Classifies back angle metric into risk states."""
        if back_angle < self.angle_danger:
            return "CRITICAL: HIGH ERGONOMIC STRAIN", "RED"
        elif back_angle < self.angle_warning:
            return "WARNING: MODERATE BENDING", "ORANGE"
        else:
            return "POSTURE SAFE", "GREEN"

    def check_hazard_zone(self, wrist_coords):
        """Evaluates whether wrist keypoint enters the danger zone."""
        wx, wy = wrist_coords
        hx1, hy1, hx2, hy2 = self.hazard_zone
        
        if hx1 <= wx <= hx2 and hy1 <= wy <= hy2:
            return True
        return False

    def process_metrics(self, metrics):
        """
        Consolidates vision pipeline metrics into real-time safety alerts.
        """
        back_angle = metrics.get("back_angle", 180)
        wrist_coords = metrics.get("right_wrist", (0, 0))
        
        posture_status, color = self.evaluate_posture(back_angle)
        in_hazard = self.check_hazard_zone(wrist_coords)
        
        overall_status = posture_status
        if in_hazard:
            overall_status = "CRITICAL: HAZARD ZONE BREACH!"
            color = "RED"
            
        # Log recent violations for report history
        if color in ["RED", "ORANGE"]:
            log_entry = {
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                "event": overall_status,
                "angle": back_angle
            }
            # Prevent rapid duplicate logging
            if len(self.incident_logs) == 0 or self.incident_logs[-1]["event"] != overall_status:
                self.incident_logs.append(log_entry)
                if len(self.incident_logs) > 10:
                    self.incident_logs.pop(0)

        return {
            "status": overall_status,
            "color": color,
            "in_hazard": in_hazard,
            "logs": self.incident_logs,
            "hazard_box": self.hazard_zone
        }