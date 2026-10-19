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

YOLO_MODEL_PATH = os.path.join(
    BASE_DIR, "models", "yolov8n-face.pt"
)

FACE_MODEL_PATH = os.path.join(
    BASE_DIR, "models", "face_model.yml"
)

LABEL_FILE_PATH = os.path.join(
    BASE_DIR, "models", "labels.txt"
)

ALERT_DIR = os.path.join(
    BASE_DIR, "alerts"
)

UNKNOWN_FILE = os.path.join(
    ALERT_DIR, "unknown_person.csv"
)


# ==========================================
# CREATE ALERT FOLDER
# ==========================================

os.makedirs(ALERT_DIR, exist_ok=True)


# ==========================================
# CREATE CSV IF NOT EXISTS
# ==========================================

if not os.path.exists(UNKNOWN_FILE):

    with open(UNKNOWN_FILE, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "Date",
            "Time",
            "Event"
        ])


# ==========================================
# LOAD YOLO MODEL
# ==========================================

print("Loading YOLO model...")

yolo_model = YOLO(YOLO_MODEL_PATH)


# ==========================================
# LOAD LBPH MODEL
# ==========================================

print("Loading face recognition model...")

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

face_recognizer.read(FACE_MODEL_PATH)


# ==========================================
# LOAD LABELS
# ==========================================

labels = {}

with open(LABEL_FILE_PATH, "r") as file:

    for line in file:

        line = line.strip()

        if line:

            parts = line.split(",")

            label_id = int(parts[0])
            name = parts[1]

            labels[label_id] = name


print("Registered people:", labels)


# ==========================================
# CAMERA
# ==========================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("ERROR: Camera could not be opened.")

    exit()


# ==========================================
# UNKNOWN PERSON CONTROL
# ==========================================

unknown_detected = False


# ==========================================
# START MESSAGE
# ==========================================

print()
print("==========================================")
print("     SAARTHI UNKNOWN PERSON DETECTION")
print("==========================================")
print()
print("Registered people:", list(labels.values()))
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
    # YOLO FACE DETECTION
    # ======================================

    results = yolo_model(
        frame,
        verbose=False
    )


    current_unknown = False


    # ======================================
    # PROCESS DETECTED FACES
    # ======================================

    for result in results:

        if result.boxes is None:
            continue


        for box in result.boxes:

            # Get coordinates
            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )


            # Keep coordinates inside frame
            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)


            if x2 <= x1 or y2 <= y1:
                continue


            # ==================================
            # CROP FACE
            # ==================================

            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue


            # ==================================
            # GRAYSCALE
            # ==================================

            gray_face = cv2.cvtColor(
                face,
                cv2.COLOR_BGR2GRAY
            )


            # ==================================
            # RESIZE
            # ==================================

            gray_face = cv2.resize(
                gray_face,
                (200, 200)
            )


            # ==================================
            # FACE RECOGNITION
            # ==================================

            label_id, confidence = (
                face_recognizer.predict(gray_face)
            )


            # ==================================
            # CHECK IDENTITY
            # ==================================

            if confidence < 60 and label_id in labels:

                name = labels[label_id]

            else:

                name = "Unknown"


            # ==================================
            # REGISTERED PERSON
            # ==================================

            if name != "Unknown":

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )


                cv2.putText(
                    frame,
                    f"{name} ({confidence:.1f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )


            # ==================================
            # UNKNOWN PERSON
            # ==================================

            else:

                current_unknown = True

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    3
                )


                cv2.putText(
                    frame,
                    "UNKNOWN PERSON",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )


    # ======================================
    # SAVE UNKNOWN PERSON EVENT
    # ======================================

    if current_unknown and not unknown_detected:

        now = datetime.now()

        date = now.strftime("%Y-%m-%d")
        time = now.strftime("%H:%M:%S")


        with open(
            UNKNOWN_FILE,
            "a",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                date,
                time,
                "UNKNOWN PERSON DETECTED"
            ])


        print(
            f"⚠ UNKNOWN PERSON DETECTED | "
            f"{date} | {time}"
        )


    # Update detection state
    unknown_detected = current_unknown


    # ======================================
    # DISPLAY SYSTEM NAME
    # ======================================

    cv2.putText(
        frame,
        "SAARTHI - UNKNOWN PERSON DETECTION",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ======================================
    # DISPLAY CAMERA
    # ======================================

    cv2.imshow(
        "SAARTHI - Unknown Person Detection",
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
print("==========================================")
print("Unknown person detection stopped.")
print("==========================================")
print()
print("Alert file:")
print(UNKNOWN_FILE)