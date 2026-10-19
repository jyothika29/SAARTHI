from ultralytics import YOLO
import cv2
import os

# ---------------------------------------------
# SAARTHI - Face Image Collection
# ---------------------------------------------

PERSON_NAME = "Jyothika"

# SAARTHI project directory
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# Dataset inside SAARTHI
SAVE_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    PERSON_NAME
)

# YOLO face model
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "yolov8n-face.pt"
)


os.makedirs(SAVE_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print("ERROR: Face model not found.")
    print(MODEL_PATH)
    exit()

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

existing_files = [
    f for f in os.listdir(SAVE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

count = len(existing_files)

print("----------------------------------------")
print("SAARTHI - Face Image Collection")
print("----------------------------------------")
print("Person:", PERSON_NAME)
print("Save folder:", SAVE_DIR)
print()
print("Press S -> Save face image")
print("Press Q -> Quit")
print("----------------------------------------")

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera frame.")
        break

    results = model(frame, verbose=False)

    face_detected = False
    face = None

    for result in results:

        for box in result.boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            confidence = float(box.conf[0])

            if confidence >= 0.5:

                face_detected = True

                h, w = frame.shape[:2]

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Face {confidence:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                face = frame[y1:y2, x1:x2]

                break

        if face_detected:
            break

    cv2.putText(
        frame,
        "Press S to save | Q to quit",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Images: {count}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "SAARTHI - Face Collection",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):

        if face_detected and face is not None and face.size > 0:

            count += 1

            filename = os.path.join(
                SAVE_DIR,
                f"{count}.jpg"
            )

            cv2.imwrite(
                filename,
                face
            )

            print(
                f"Saved image {count}: {filename}"
            )

        else:
            print("No face detected.")

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("----------------------------------------")
print("Face collection completed.")
print("Person:", PERSON_NAME)
print("Total images:", count)
print("Saved in:", SAVE_DIR)
print("----------------------------------------")