import cv2
import math
import time
import csv
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# --------------------------------------------------
# SAARTHI - Movement Analysis Prototype
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "yolov8n-pose.pt"
ALERT_DIR = BASE_DIR / "alerts"
ALERT_FILE = ALERT_DIR / "movement_alert.csv"

# Settings for initial testing
CONFIDENCE = 0.5
MOVEMENT_THRESHOLD = 0.06
ALERT_COOLDOWN = 5

ALERT_DIR.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Pose model not found: {MODEL_PATH}")

model = YOLO(str(MODEL_PATH))

# Create event log if it does not exist
if not ALERT_FILE.exists():
    with open(ALERT_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Date", "Time", "Event", "Track_ID", "Movement"])

camera = cv2.VideoCapture(1)

if not camera.isOpened():
    raise RuntimeError(
        "Cannot open webcam. Close other camera applications and try again."
    )

# Store previous normalized shoulder positions by tracking ID
previous_positions = {}
last_alert_times = {}

print("SAARTHI Movement Analysis Started")
print("Move safely in front of the camera to test.")
print("Press Q to quit.")

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read camera frame.")
            break

        height, width = frame.shape[:2]

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=CONFIDENCE,
            verbose=False
        )

        result = results[0]
        display_frame = result.plot()
        current_time = time.time()

        if (
            result.keypoints is not None
            and result.boxes is not None
            and result.boxes.id is not None
        ):
            track_ids = result.boxes.id.int().cpu().tolist()
            keypoints = result.keypoints.xy.cpu().numpy()
            keypoint_conf = (
                result.keypoints.conf.cpu().numpy()
                if result.keypoints.conf is not None
                else None
            )

            for index, track_id in enumerate(track_ids):
                if index >= len(keypoints):
                    continue

                # COCO pose keypoints:
                # 5 = left shoulder, 6 = right shoulder
                left_shoulder = keypoints[index][5]
                right_shoulder = keypoints[index][6]

                # Skip unreliable or missing shoulder keypoints
                if keypoint_conf is not None:
                    if (
                        keypoint_conf[index][5] < 0.4
                        or keypoint_conf[index][6] < 0.4
                    ):
                        continue

                if (
                    left_shoulder[0] <= 0
                    or left_shoulder[1] <= 0
                    or right_shoulder[0] <= 0
                    or right_shoulder[1] <= 0
                ):
                    continue

                # Shoulder midpoint, normalized to frame dimensions
                center_x = (
                    left_shoulder[0] + right_shoulder[0]
                ) / (2 * width)

                center_y = (
                    left_shoulder[1] + right_shoulder[1]
                ) / (2 * height)

                current_position = (center_x, center_y)

                if track_id in previous_positions:
                    old_x, old_y = previous_positions[track_id]

                    movement = math.sqrt(
                        (center_x - old_x) ** 2
                        + (center_y - old_y) ** 2
                    )

                    label = f"ID {track_id} movement: {movement:.3f}"

                    if movement >= MOVEMENT_THRESHOLD:
                        label = f"ID {track_id}: SUDDEN MOVEMENT"

                        last_time = last_alert_times.get(track_id, 0)

                        if current_time - last_time >= ALERT_COOLDOWN:
                            now = datetime.now()

                            with open(
                                ALERT_FILE,
                                "a",
                                newline="",
                                encoding="utf-8"
                            ) as file:
                                writer = csv.writer(file)
                                writer.writerow([
                                    now.strftime("%Y-%m-%d"),
                                    now.strftime("%H:%M:%S"),
                                    "Sudden movement - review required",
                                    track_id,
                                    f"{movement:.3f}"
                                ])

                            last_alert_times[track_id] = current_time
                            print(
                                f"Movement event logged for ID {track_id}"
                            )

                    x = int(center_x * width)
                    y = int(center_y * height) - 20

                    cv2.putText(
                        display_frame,
                        label,
                        (max(10, x - 100), max(25, y)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 255) if movement >= MOVEMENT_THRESHOLD
                        else (0, 255, 0),
                        2
                    )

                previous_positions[track_id] = current_position

        cv2.putText(
            display_frame,
            "SAARTHI - Movement Analysis Prototype",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2
        )

        cv2.imshow("SAARTHI - Movement Analysis", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    print("Movement analysis stopped.")
    print(f"Event log: {ALERT_FILE}")