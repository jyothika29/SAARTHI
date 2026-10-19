import cv2
import os
from ultralytics import YOLO


# ==========================================
# PROJECT PATH
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ==========================================
# MODEL PATHS
# ==========================================

YOLO_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "yolov8n-face.pt"
)

FACE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "face_model.yml"
)

LABEL_FILE_PATH = os.path.join(
    BASE_DIR,
    "models",
    "labels.txt"
)


# ==========================================
# LOAD YOLO FACE DETECTOR
# ==========================================

print("Loading YOLO face model...")

yolo_model = YOLO(YOLO_MODEL_PATH)


# ==========================================
# LOAD LBPH FACE RECOGNIZER
# ==========================================

print("Loading LBPH face recognition model...")

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


print("Registered people:")

for label_id, name in labels.items():
    print(label_id, "->", name)


# ==========================================
# OPEN CAMERA
# ==========================================

camera = cv2.VideoCapture(1)


if not camera.isOpened():

    print("ERROR: Could not open camera.")

    exit()


print("\nLive face recognition started.")
print("Press Q to quit.")


# ==========================================
# LIVE RECOGNITION
# ==========================================

while True:

    ret, frame = camera.read()

    if not ret:

        print("ERROR: Could not read camera frame.")

        break


    # --------------------------------------
    # YOLO FACE DETECTION
    # --------------------------------------

    results = yolo_model(frame, verbose=False)


    for result in results:

        if result.boxes is None:
            continue


        for box in result.boxes:

            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)


            # Keep coordinates inside frame
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)


            # Extract face
            face = frame[y1:y2, x1:x2]


            if face.size == 0:
                continue


            # --------------------------------------
            # PREPROCESS FACE FOR LBPH
            # --------------------------------------

            gray_face = cv2.cvtColor(
                face,
                cv2.COLOR_BGR2GRAY
            )


            gray_face = cv2.resize(
                gray_face,
                (200, 200)
            )


            # --------------------------------------
            # LBPH PREDICTION
            # --------------------------------------

            label_id, confidence = face_recognizer.predict(
                gray_face
            )


            # LBPH confidence is a distance:
            # lower = better match

            if confidence < 60 and label_id in labels:

                name = labels[label_id]

            else:

                name = "Unknown"


            # --------------------------------------
            # DRAW RESULT
            # --------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            cv2.putText(
                frame,
                name,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )


            # Display confidence/distance
            cv2.putText(
                frame,
                f"Distance: {confidence:.1f}",
                (x1, y2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )


    # --------------------------------------
    # DISPLAY CAMERA
    # --------------------------------------

    cv2.imshow(
        "SAARTHI - Live Face Recognition",
        frame
    )


    # --------------------------------------
    # EXIT
    # --------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ==========================================
# CLEANUP
# ==========================================

camera.release()

cv2.destroyAllWindows()

print("Live recognition stopped.")