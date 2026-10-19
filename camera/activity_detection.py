import cv2
import os
import csv
import math
from datetime import datetime
from ultralytics import YOLO

# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "yolov8n-pose.pt"
)

ALERT_DIR = os.path.join(
    BASE_DIR,
    "alerts"
)

ACTIVITY_FILE = os.path.join(
    ALERT_DIR,
    "activity_alert.csv"
)


# ==========================================
# CREATE ALERT FOLDER
# ==========================================

os.makedirs(ALERT_DIR, exist_ok=True)


# ==========================================
# CREATE CSV
# ==========================================

if not os.path.exists(ACTIVITY_FILE):

    with open(ACTIVITY_FILE, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "Date",
            "Time",
            "Event"
        ])


# ==========================================
# LOAD YOLOv8-POSE
# ==========================================

print("Loading YOLOv8-Pose model...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully!")


# ==========================================
# CAMERA
# ==========================================

camera = cv2.VideoCapture(1)

if not camera.isOpened():

    print("ERROR: Camera could not be opened.")

    exit()


# ==========================================
# MOVEMENT SETTINGS
# ==========================================

previous_positions = {}

movement_threshold = 35

abnormal_detected = False


# ==========================================
# START
# ==========================================

print()
print("==========================================")
print("     SAARTHI ACTIVITY DETECTION")
print("==========================================")
print()
print("Press Q to quit")
print()


# ==========================================
# FUNCTION: DISTANCE
# ==========================================

def calculate_distance(point1, point2):

    x1, y1 = point1
    x2, y2 = point2

    return math.sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    success, frame = camera.read()

    if not success:

        print("Could not read camera frame.")

        break


    # ======================================
    # POSE DETECTION
    # ======================================

    results = model(
        frame,
        conf=0.5,
        verbose=False
    )


    current_positions = {}

    current_abnormal = False


    # ======================================
    # PROCESS PEOPLE
    # ======================================

    for person_id, result in enumerate(results):

        if result.keypoints is None:

            continue


        keypoints = result.keypoints.xy


        if keypoints is None:

            continue


        # ==================================
        # PROCESS EACH PERSON
        # ==================================

        for i in range(len(keypoints)):

            person = keypoints[i]

            # Need enough body points
            if len(person) < 17:

                continue


            # ----------------------------------
            # Use shoulder center as body point
            # ----------------------------------

            left_shoulder = person[5]
            right_shoulder = person[6]


            # Check if keypoints exist
            if (
                left_shoulder[0] == 0
                or right_shoulder[0] == 0
            ):

                continue


            center_x = (
                float(left_shoulder[0])
                + float(right_shoulder[0])
            ) / 2


            center_y = (
                float(left_shoulder[1])
                + float(right_shoulder[1])
            ) / 2


            current_position = (
                center_x,
                center_y
            )


            current_positions[i] = current_position


            # ==================================
            # MOVEMENT ANALYSIS
            # ==================================

            if i in previous_positions:

                movement = calculate_distance(
                    previous_positions[i],
                    current_position
                )


                # ==================================
                # SUDDEN MOVEMENT
                # ==================================

                if movement > movement_threshold:

                    current_abnormal = True


                    # Draw warning
                    cv2.putText(
                        frame,
                        "POTENTIAL ABNORMAL MOVEMENT",
                        (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )


            # ==================================
            # DRAW BODY POINT
            # ==================================

            cv2.circle(
                frame,
                (
                    int(center_x),
                    int(center_y)
                ),
                6,
                (0, 255, 255),
                -1
            )


    # ======================================
    # UPDATE POSITIONS
    # ======================================

    previous_positions = current_positions


    # ======================================
    # SAVE ALERT
    # ======================================

    if current_abnormal and not abnormal_detected:

        now = datetime.now()

        date = now.strftime("%Y-%m-%d")
        time = now.strftime("%H:%M:%S")


        with open(
            ACTIVITY_FILE,
            "a",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                date,
                time,
                "POTENTIAL ABNORMAL MOVEMENT"
            ])


        print()
        print("⚠ POTENTIAL ABNORMAL MOVEMENT DETECTED")
        print(f"Date: {date}")
        print(f"Time: {time}")
        print()


    abnormal_detected = current_abnormal


    # ======================================
    # SYSTEM TITLE
    # ======================================

    cv2.putText(
        frame,
        "SAARTHI - ACTIVITY MONITORING",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # ======================================
    # STATUS
    # ======================================

    if current_abnormal:

        cv2.putText(
            frame,
            "⚠ POTENTIAL ABNORMAL ACTIVITY",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    else:

        cv2.putText(
            frame,
            "STATUS: NORMAL",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


    # ======================================
    # DISPLAY
    # ======================================

    cv2.imshow(
        "SAARTHI - Activity Detection",
        frame
    )


    # ======================================
    # QUIT
    # ======================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ==========================================
# CLEANUP
# ==========================================

camera.release()

cv2.destroyAllWindows()


print()
print("Activity detection stopped.")
print()
print("Alert file:")
print(ACTIVITY_FILE)