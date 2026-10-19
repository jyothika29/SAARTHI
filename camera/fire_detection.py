import cv2
import os
import csv
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
    "fire_smoke_model.pt"
)

ALERT_DIR = os.path.join(
    BASE_DIR,
    "alerts"
)

ALERT_FILE = os.path.join(
    ALERT_DIR,
    "fire_alert.csv"
)


# ==========================================
# CREATE ALERT FOLDER
# ==========================================

os.makedirs(ALERT_DIR, exist_ok=True)


# ==========================================
# CREATE CSV
# ==========================================

if not os.path.exists(ALERT_FILE):

    with open(ALERT_FILE, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "Date",
            "Time",
            "Event"
        ])


# ==========================================
# LOAD FIRE MODEL
# ==========================================

print("Loading fire detection model...")

model = YOLO(MODEL_PATH)

print("Fire detection model loaded.")


# ==========================================
# CAMERA
# ==========================================

camera = cv2.VideoCapture(1)

if not camera.isOpened():

    print("ERROR: Camera could not be opened.")

    exit()


# ==========================================
# ALERT CONTROL
# ==========================================

fire_already_detected = False


# ==========================================
# START
# ==========================================

print()
print("==========================================")
print("       SAARTHI FIRE DETECTION")
print("==========================================")
print()
print("Press Q to quit")
print()


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    success, frame = camera.read()

    if not success:

        print("Could not read camera frame.")

        break


    # ======================================
    # AI FIRE DETECTION
    # ======================================

    results = model(
        frame,
        conf=0.40,
        verbose=False
    )


    current_fire = False


    # ======================================
    # PROCESS DETECTIONS
    # ======================================

    for result in results:

        if result.boxes is None:
            continue


        for box in result.boxes:

            # Coordinates
            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )


            # Confidence
            confidence = float(
                box.conf[0].cpu().numpy()
            )


            # Class ID
            class_id = int(
                box.cls[0].cpu().numpy()
            )


            # Class name
            class_name = model.names[class_id]

            class_name = class_name.lower()


            # ==================================
            # FIRE DETECTED
            # ==================================

            if "fire" in class_name:

                current_fire = True


                # Red box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    3
                )


                # Label
                cv2.putText(
                    frame,
                    f"FIRE {confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )


            # ==================================
            # SMOKE DETECTED
            # ==================================

            elif "smoke" in class_name:

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 165, 255),
                    3
                )


                cv2.putText(
                    frame,
                    f"SMOKE {confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 165, 255),
                    2
                )


    # ======================================
    # FIRE ALERT
    # ======================================

    if current_fire and not fire_already_detected:

        now = datetime.now()

        date = now.strftime("%Y-%m-%d")
        time = now.strftime("%H:%M:%S")


        with open(
            ALERT_FILE,
            "a",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                date,
                time,
                "FIRE DETECTED"
            ])


        print()
        print("🔥 FIRE DETECTED!")
        print(f"Date: {date}")
        print(f"Time: {time}")
        print()


    # Update state
    fire_already_detected = current_fire


    # ======================================
    # SYSTEM TITLE
    # ======================================

    cv2.putText(
        frame,
        "SAARTHI - FIRE DETECTION",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # ======================================
    # STATUS
    # ======================================

    if current_fire:

        cv2.putText(
            frame,
            "!!! FIRE ALERT !!!",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    else:

        cv2.putText(
            frame,
            "STATUS: SAFE",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


    # ======================================
    # DISPLAY
    # ======================================

    cv2.imshow(
        "SAARTHI - Fire Detection",
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
print("Fire detection stopped.")
print("Alert file:")
print(ALERT_FILE)