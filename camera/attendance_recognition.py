import cv2
import os
import csv
from datetime import datetime
from ultralytics import YOLO

# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

YOLO_MODEL_PATH = os.path.join(
    BASE_DIR, "models", "yolov8n-face.pt"
)

FACE_MODEL_PATH = os.path.join(
    BASE_DIR, "models", "face_model.yml"
)

LABEL_FILE_PATH = os.path.join(
    BASE_DIR, "models", "labels.txt"
)

ATTENDANCE_DIR = os.path.join(
    BASE_DIR, "attendance"
)

ATTENDANCE_FILE = os.path.join(
    ATTENDANCE_DIR, "attendance.csv"
)


# ==========================================
# CREATE ATTENDANCE FOLDER
# ==========================================

os.makedirs(ATTENDANCE_DIR, exist_ok=True)


# ==========================================
# CREATE CSV IF IT DOES NOT EXIST
# ==========================================

if not os.path.exists(ATTENDANCE_FILE):

    with open(ATTENDANCE_FILE, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "Name",
            "Date",
            "Time",
            "Status"
        ])


# ==========================================
# LOAD YOLO MODEL
# ==========================================

print("Loading YOLO model...")

yolo_model = YOLO(YOLO_MODEL_PATH)


# ==========================================
# LOAD FACE RECOGNITION MODEL
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

camera = cv2.VideoCapture(1)

if not camera.isOpened():

    print("ERROR: Camera could not be opened.")

    exit()


# ==========================================
# INITIAL MODE
# ==========================================

mode = "ENTRY"


# ==========================================
# PREVENT DUPLICATE ATTENDANCE
# ==========================================

marked_entries = set()
marked_exits = set()


# ==========================================
# START
# ==========================================

print()
print("========================================")
print("       SAARTHI ATTENDANCE SYSTEM")
print("========================================")
print()
print("Press E = ENTRY")
print("Press X = EXIT")
print("Press Q = QUIT")
print()
print("Current Mode: ENTRY")
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

    results = yolo_model(frame, verbose=False)


    for result in results:

        if result.boxes is None:
            continue


        for box in result.boxes:

            # Get face coordinates
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

            label_id, confidence = face_recognizer.predict(
                gray_face
            )


            # ==================================
            # RECOGNITION DECISION
            # ==================================

            if confidence < 60 and label_id in labels:

                name = labels[label_id]

            else:

                name = "Unknown"


            # ==================================
            # KNOWN PERSON
            # ==================================

            if name != "Unknown":

                # Green rectangle
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )


                # Display name
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
                # ATTENDANCE MARKING
                # ==================================

                if mode == "ENTRY":

                    if name not in marked_entries:

                        now = datetime.now()

                        date = now.strftime("%Y-%m-%d")
                        time = now.strftime("%H:%M:%S")


                        with open(
                            ATTENDANCE_FILE,
                            "a",
                            newline=""
                        ) as file:

                            writer = csv.writer(file)

                            writer.writerow([
                                name,
                                date,
                                time,
                                "ENTRY"
                            ])


                        marked_entries.add(name)


                        print(
                            f"ENTRY recorded: "
                            f"{name} | "
                            f"{date} | "
                            f"{time}"
                        )


                elif mode == "EXIT":

                    if name not in marked_exits:

                        now = datetime.now()

                        date = now.strftime("%Y-%m-%d")
                        time = now.strftime("%H:%M:%S")


                        with open(
                            ATTENDANCE_FILE,
                            "a",
                            newline=""
                        ) as file:

                            writer = csv.writer(file)

                            writer.writerow([
                                name,
                                date,
                                time,
                                "EXIT"
                            ])


                        marked_exits.add(name)


                        print(
                            f"EXIT recorded: "
                            f"{name} | "
                            f"{date} | "
                            f"{time}"
                        )


            # ==================================
            # UNKNOWN PERSON
            # ==================================

            else:

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2
                )


                cv2.putText(
                    frame,
                    "Unknown",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )


    # ======================================
    # DISPLAY CURRENT MODE
    # ======================================

    cv2.putText(
        frame,
        f"MODE: {mode}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )


    # ======================================
    # DISPLAY CONTROLS
    # ======================================

    cv2.putText(
        frame,
        "E: ENTRY   X: EXIT   Q: QUIT",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # ======================================
    # SHOW CAMERA
    # ======================================

    cv2.imshow(
        "SAARTHI - Attendance System",
        frame
    )


    # ======================================
    # KEYBOARD CONTROLS
    # ======================================

    key = cv2.waitKey(1) & 0xFF


    # ENTRY MODE
    if key == ord("e"):

        mode = "ENTRY"

        print("Mode changed to ENTRY")


    # EXIT MODE
    elif key == ord("x"):

        mode = "EXIT"

        print("Mode changed to EXIT")


    # QUIT
    elif key == ord("q"):

        break


# ==========================================
# CLEANUP
# ==========================================

camera.release()

cv2.destroyAllWindows()


print()
print("========================================")
print("SAARTHI Attendance System Stopped")
print("========================================")
print()
print("Attendance file:")
print(ATTENDANCE_FILE)