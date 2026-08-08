import cv2
from vision_pipeline import ErgoVisionPipeline
from risk_engine import ErgoRiskEngine

pipeline = ErgoVisionPipeline(source=0)
risk_engine = ErgoRiskEngine()

print("Testing CV + Risk Engine Integration... Press 'q' to stop.")

while pipeline.cap.isOpened():
    ret, frame = pipeline.cap.read()
    if not ret:
        break

    # 1. Run Person 1 Vision Pipeline
    processed_frame, metrics = pipeline.process_frame(frame)

    # 2. Run Person 2 Risk Engine
    risk_assessment = risk_engine.process_metrics(metrics)

    # 3. Draw Hazard Box Overlay
    hx1, hy1, hx2, hy2 = risk_assessment["hazard_box"]
    box_color = (0, 0, 255) if risk_assessment["in_hazard"] else (255, 0, 0)
    cv2.rectangle(processed_frame, (hx1, hy1), (hx2, hy2), box_color, 2)
    cv2.putText(processed_frame, "DANGER ZONE", (hx1 + 10, hy1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    # 4. Display Real-Time Status Text
    status_text = risk_assessment["status"]
    cv2.putText(processed_frame, status_text, (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Combined Test Sandbox", processed_frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

pipeline.release()
cv2.destroyAllWindows()