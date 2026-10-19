import cv2
import csv
import threading
import time
import math

from datetime import datetime
from pathlib import Path
from ultralytics import YOLO


# ==================================================
# PROJECT PATHS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "models"
ALERT_DIR = PROJECT_DIR / "alerts"

POSE_MODEL_PATH = MODELS_DIR / "yolov8n-pose.pt"
FACE_MODEL_PATH = MODELS_DIR / "yolov8n-face.pt"
LBPH_MODEL_PATH = MODELS_DIR / "face_model.yml"
LABEL_FILE_PATH = MODELS_DIR / "labels.txt"

UNKNOWN_FILE = ALERT_DIR / "unknown_person.csv"
ACTIVITY_FILE = ALERT_DIR / "activity_alert.csv"

ALERT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# CSV FILE INITIALIZATION
# ==================================================

def create_csv_if_missing(path):
    if not path.exists():
        with open(
            path,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            writer.writerow(["Date", "Time", "Event"])


create_csv_if_missing(UNKNOWN_FILE)
create_csv_if_missing(ACTIVITY_FILE)


# ==================================================
# MOVEMENT SETTINGS
# ==================================================

# Basic shoulder movement threshold.
# This is NOT a validated violence detector.
MOVEMENT_THRESHOLD = 0.12

# Minimum time between alerts.
ALERT_COOLDOWN = 5.0


# ==================================================
# LOAD REGISTERED FACE LABELS
# ==================================================

def load_labels():
    labels = {}

    if not LABEL_FILE_PATH.exists():
        raise FileNotFoundError(
            f"Face labels not found: {LABEL_FILE_PATH}"
        )

    with open(
        LABEL_FILE_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:
            line = line.strip()

            if not line:
                continue

            parts = line.split(",", 1)

            if len(parts) == 2:
                label_id = int(parts[0].strip())
                name = parts[1].strip()
                labels[label_id] = name

    return labels


# ==================================================
# CAMERA STREAM CLASS
# ==================================================

class WebcamStream:

    def __init__(self):
        self.camera = None

        self.pose_model = None
        self.face_model = None
        self.face_recognizer = None

        self.labels = {}

        self.running = False
        self.thread = None

        self.lock = threading.Lock()
        self.latest_frame = None

        self.people_count = 0
        self.unknown_count = 0
        self.known_people = []

        self.error = None
        self.unknown_detected = False

        self.previous_positions = {}
        self.last_activity_alert = {}

        self.activity_detected = False

    # --------------------------------------------------
    # START CAMERA
    # --------------------------------------------------

    def start(self):

        if self.running:
            return True, "Camera is already running."

        try:
            self.error = None

            print("SAARTHI: Loading YOLOv8 Pose model...")

            self.pose_model = YOLO(
                str(POSE_MODEL_PATH)
            )

            print("SAARTHI: Loading YOLO face model...")

            self.face_model = YOLO(
                str(FACE_MODEL_PATH)
            )

            print("SAARTHI: Loading registered face labels...")

            self.labels = load_labels()

            print("SAARTHI: Loading LBPH recognizer...")

            if not LBPH_MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"LBPH model not found: {LBPH_MODEL_PATH}"
                )

            if not hasattr(cv2, "face"):
                raise RuntimeError(
                    "OpenCV face module is missing. "
                    "Install opencv-contrib-python in the "
                    "active Python environment."
                )

            self.face_recognizer = (
                cv2.face.LBPHFaceRecognizer_create()
            )

            self.face_recognizer.read(
                str(LBPH_MODEL_PATH)
            )

            print(
                "SAARTHI registered people:",
                self.labels
            )

            # Open only one webcam.
            print("SAARTHI: Opening webcam...")

            self.camera = cv2.VideoCapture(
                1,
                cv2.CAP_DSHOW
            )

            if not self.camera.isOpened():
                self.camera.release()
                self.camera = None

                self.error = (
                    "Could not open webcam. "
                    "Close other camera applications and try again."
                )

                return False, self.error

            self.unknown_detected = False
            self.previous_positions.clear()
            self.last_activity_alert.clear()
            self.activity_detected = False

            with self.lock:
                self.latest_frame = None
                self.people_count = 0
                self.unknown_count = 0
                self.known_people = []

            self.running = True

            self.thread = threading.Thread(
                target=self._process_frames,
                daemon=True
            )

            self.thread.start()

            return True, "SAARTHI camera monitoring started."

        except Exception as exc:
            self.error = str(exc)
            self.running = False

            self._release_camera()

            print("SAARTHI startup error:", exc)

            return False, self.error

    # --------------------------------------------------
    # RECORD UNKNOWN PERSON EVENT
    # --------------------------------------------------

    def _record_unknown_event(self):

        now = datetime.now()

        try:
            with open(
                UNKNOWN_FILE,
                "a",
                newline="",
                encoding="utf-8"
            ) as file:

                writer = csv.writer(file)

                writer.writerow([
                    now.strftime("%Y-%m-%d"),
                    now.strftime("%H:%M:%S"),
                    "UNKNOWN FACE DETECTED"
                ])

            print(
                "SAARTHI WARNING: Unknown face detected at",
                now.strftime("%Y-%m-%d %H:%M:%S")
            )

        except OSError as exc:
            print(
                "Could not save unknown-person event:",
                exc
            )

    # --------------------------------------------------
    # RECORD POTENTIAL ABNORMAL MOVEMENT
    # --------------------------------------------------

    def _record_activity_event(self):

        now = datetime.now()

        try:
            with open(
                ACTIVITY_FILE,
                "a",
                newline="",
                encoding="utf-8"
            ) as file:

                writer = csv.writer(file)

                writer.writerow([
                    now.strftime("%Y-%m-%d"),
                    now.strftime("%H:%M:%S"),
                    "POTENTIAL ABNORMAL MOVEMENT"
                ])

            print(
                "SAARTHI WARNING: Potential abnormal movement at",
                now.strftime("%Y-%m-%d %H:%M:%S")
            )

        except OSError as exc:
            print(
                "Could not save activity alert:",
                exc
            )

    # --------------------------------------------------
    # ANALYZE BODY MOVEMENT
    # --------------------------------------------------

    def _analyze_movement(self, pose_result, frame):

        current_positions = {}
        activity_detected = False

        height, width = frame.shape[:2]

        if (
            pose_result.keypoints is None
            or pose_result.boxes is None
        ):
            self.previous_positions = {}
            self.activity_detected = False
            return False

        keypoints = pose_result.keypoints.xy.cpu().numpy()

        confidences = None

        if pose_result.keypoints.conf is not None:
            confidences = (
                pose_result.keypoints.conf.cpu().numpy()
            )

        current_time = time.time()

        for index, person in enumerate(keypoints):

            if len(person) < 17:
                continue

            left_shoulder = person[5]
            right_shoulder = person[6]

            # Ignore missing shoulder points.
            if (
                left_shoulder[0] <= 0
                or left_shoulder[1] <= 0
                or right_shoulder[0] <= 0
                or right_shoulder[1] <= 0
            ):
                continue

            # Ignore uncertain shoulder detections.
            if confidences is not None:

                if (
                    confidences[index][5] < 0.4
                    or confidences[index][6] < 0.4
                ):
                    continue

            center_x = (
                left_shoulder[0] + right_shoulder[0]
            ) / (2.0 * width)

            center_y = (
                left_shoulder[1] + right_shoulder[1]
            ) / (2.0 * height)

            current_position = (center_x, center_y)

            current_positions[index] = current_position

            if index in self.previous_positions:

                old_x, old_y = self.previous_positions[index]

                movement = math.sqrt(
                    (center_x - old_x) ** 2
                    + (center_y - old_y) ** 2
                )

                if movement >= MOVEMENT_THRESHOLD:

                    activity_detected = True

                    cv2.putText(
                        frame,
                        "POTENTIAL ABNORMAL MOVEMENT",
                        (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 0, 255),
                        2
                    )

                    last_alert = (
                        self.last_activity_alert.get(
                            index,
                            0
                        )
                    )

                    if (
                        current_time - last_alert
                        >= ALERT_COOLDOWN
                    ):
                        self._record_activity_event()

                        self.last_activity_alert[index] = (
                            current_time
                        )

        self.previous_positions = current_positions
        self.activity_detected = activity_detected

        return activity_detected

    # --------------------------------------------------
    # PROCESS CAMERA FRAMES
    # --------------------------------------------------

    def _process_frames(self):

        try:
            while self.running:

                camera = self.camera

                if camera is None:
                    break

                # Read a webcam frame.
                success, frame = camera.read()

                if not success:
                    self.error = "Could not read webcam frame."

                    print(
                        "SAARTHI: Webcam frame read failed."
                    )

                    time.sleep(0.1)
                    continue

                # Detect people and body poses.
                pose_results = self.pose_model.predict(
                    source=frame,
                    classes=[0],
                    conf=0.5,
                    verbose=False
                )

                pose_result = pose_results[0]

                people_count = (
                    len(pose_result.boxes)
                    if pose_result.boxes is not None
                    else 0
                )

                annotated = pose_result.plot()

                # Analyze movement.
                activity_detected = self._analyze_movement(
                    pose_result,
                    annotated
                )

                if activity_detected:
                    print(
                        "SAARTHI: Movement threshold exceeded."
                    )

                # Detect faces.
                face_results = self.face_model.predict(
                    source=frame,
                    conf=0.5,
                    verbose=False
                )

                current_unknown = False
                known_people = []
                unknown_count = 0

                for result in face_results:

                    if result.boxes is None:
                        continue

                    for box in result.boxes:

                        x1, y1, x2, y2 = (
                            box.xyxy[0]
                            .cpu()
                            .numpy()
                            .astype(int)
                        )

                        x1 = max(0, x1)
                        y1 = max(0, y1)

                        x2 = min(
                            frame.shape[1],
                            x2
                        )

                        y2 = min(
                            frame.shape[0],
                            y2
                        )

                        if x2 <= x1 or y2 <= y1:
                            continue

                        face = frame[y1:y2, x1:x2]

                        if face.size == 0:
                            continue

                        gray_face = cv2.cvtColor(
                            face,
                            cv2.COLOR_BGR2GRAY
                        )

                        gray_face = cv2.resize(
                            gray_face,
                            (200, 200)
                        )

                        label_id, confidence = (
                            self.face_recognizer.predict(
                                gray_face
                            )
                        )

                        # Recognize registered faces.
                        if (
                            confidence < 80
                            and label_id in self.labels
                        ):

                            name = self.labels[label_id]
                            known_people.append(name)

                            color = (0, 255, 0)

                            text = (
                                f"{name} ({confidence:.1f})"
                            )

                        else:
                            name = "UNKNOWN"
                            current_unknown = True
                            unknown_count += 1

                            color = (0, 0, 255)

                            text = (
                                "UNKNOWN - VERIFY IDENTITY"
                            )

                        cv2.rectangle(
                            annotated,
                            (x1, y1),
                            (x2, y2),
                            color,
                            2
                        )

                        cv2.putText(
                            annotated,
                            text,
                            (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2
                        )

                # Record a new unknown-person event.
                if (
                    current_unknown
                    and not self.unknown_detected
                ):
                    self._record_unknown_event()

                self.unknown_detected = current_unknown

                # Update the latest frame and camera status.
                with self.lock:

                    self.latest_frame = annotated
                    self.people_count = people_count
                    self.unknown_count = unknown_count

                    self.known_people = sorted(
                        set(known_people)
                    )

                    self.activity_detected = (
                        activity_detected
                    )

                self.error = None

        except Exception as exc:

            self.error = str(exc)

            print(
                "SAARTHI camera processing error:",
                exc
            )

        finally:

            self.running = False
            self._release_camera()

    # --------------------------------------------------
    # RETURN JPEG FRAME
    # --------------------------------------------------

    def get_frame(self):

        with self.lock:

            if self.latest_frame is None:
                return None

            success, buffer = cv2.imencode(
                ".jpg",
                self.latest_frame
            )

            if not success:
                return None

            return buffer.tobytes()

    # --------------------------------------------------
    # CAMERA STATUS
    # --------------------------------------------------

    def get_status(self):

        with self.lock:

            return {
                "running": self.running,
                "people_count": self.people_count,
                "unknown_count": self.unknown_count,
                "known_people": self.known_people.copy(),
                "activity_detected": self.activity_detected,
                "error": self.error
            }

    # --------------------------------------------------
    # STOP CAMERA
    # --------------------------------------------------

    def stop(self):

        self.running = False

        if (
            self.thread
            and self.thread.is_alive()
            and self.thread != threading.current_thread()
        ):
            self.thread.join(timeout=3)

        self._release_camera()

        return True, "Camera monitoring stopped."

    # --------------------------------------------------
    # RELEASE CAMERA
    # --------------------------------------------------

    def _release_camera(self):

        camera = self.camera
        self.camera = None

        if camera is not None:
            camera.release()


# ==================================================
# SHARED CAMERA INSTANCE
# ==================================================

webcam_stream = WebcamStream()