import cv2
import os
from ultralytics import YOLO

# ==========================================
# PROJECT PATH
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "yolov8n-pose.pt"
)

# ==========================================
# LOAD POSE MODEL
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

print("Camera started.")
print("Press Q to quit.")

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

    # ======================================
    # DRAW POSE
    # ======================================

    annotated_frame = results[0].plot()

    # ======================================
    # DISPLAY
    # ======================================

    cv2.imshow(
        "SAARTHI - Pose Detection",
        annotated_frame
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

print("Pose detection stopped.")