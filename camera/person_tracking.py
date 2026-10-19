import cv2
from ultralytics import YOLO
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "yolov8n-pose.pt"

# Load YOLOv8 Pose model
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Pose model not found: {MODEL_PATH}")

model = YOLO(str(MODEL_PATH))

# Open webcam
camera = cv2.VideoCapture(1)

if not camera.isOpened():
    raise RuntimeError(
        "Cannot open webcam. Close other camera applications and try again."
    )

print("SAARTHI Person Tracking Started")
print("Press Q to quit.")

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read a frame from the camera.")
            break

        # Detect poses and track people across frames
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.5,
            verbose=False
        )

        result = results[0]

        # Draw pose skeletons, boxes and tracking IDs
        annotated_frame = result.plot()

        # Show tracking status
        if result.boxes is not None and result.boxes.id is not None:
            person_ids = result.boxes.id.int().cpu().tolist()
            status = f"People tracked: {len(person_ids)}"
        else:
            status = "People tracked: 0"

        cv2.putText(
            annotated_frame,
            status,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("SAARTHI - Person Tracking", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    print("Person tracking stopped.")