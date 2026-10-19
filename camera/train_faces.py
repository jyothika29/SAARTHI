import cv2
import os
import numpy as np

# ---------------------------------------------
# SAARTHI - Face Recognition Training
# ---------------------------------------------

# SAARTHI project directory
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# Dataset inside SAARTHI
DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset"
)

# Models folder inside SAARTHI
MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


MODEL_FILE = os.path.join(
    MODEL_DIR,
    "face_model.yml"
)

LABEL_FILE = os.path.join(
    MODEL_DIR,
    "labels.txt"
)

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------
# Create LBPH recognizer
# ---------------------------------------------

recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_names = {}

current_label = 0

# ---------------------------------------------
# Read dataset
# ---------------------------------------------

for person_name in sorted(os.listdir(DATASET_DIR)):

    person_path = os.path.join(
        DATASET_DIR,
        person_name
    )

    if not os.path.isdir(person_path):
        continue

    print(
        f"Reading images for: {person_name}"
    )

    label_names[current_label] = person_name

    for filename in os.listdir(person_path):

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        image_path = os.path.join(
            person_path,
            filename
        )

        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            print(
                "Could not read:",
                image_path
            )
            continue

        image = cv2.resize(
            image,
            (200, 200)
        )

        faces.append(image)
        labels.append(current_label)

    current_label += 1

# ---------------------------------------------
# Check dataset
# ---------------------------------------------

if len(faces) == 0:

    print("ERROR: No training images found.")
    print("Dataset location:")
    print(DATASET_DIR)
    exit()

print("----------------------------------------")
print("Training face recognition model...")
print("Total images:", len(faces))
print("Total people:", len(label_names))
print("----------------------------------------")

labels = np.array(
    labels,
    dtype=np.int32
)

# ---------------------------------------------
# Train model
# ---------------------------------------------

recognizer.train(
    faces,
    labels
)

# Save model
recognizer.write(
    MODEL_FILE
)

# Save names
with open(
    LABEL_FILE,
    "w",
    encoding="utf-8"
) as file:

    for label, name in label_names.items():

        file.write(
            f"{label},{name}\n"
        )

# ---------------------------------------------
# Completed
# ---------------------------------------------

print("----------------------------------------")
print("TRAINING COMPLETED")
print("----------------------------------------")

print("Model saved:")
print(MODEL_FILE)

print()

print("Labels saved:")
print(LABEL_FILE)

print()

print("Registered people:")

for label, name in label_names.items():

    print(
        label,
        "->",
        name
    )

print("----------------------------------------")