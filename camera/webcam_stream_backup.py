import cv2
import threading
import time
from pathlib import Path
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "yolov8n-pose.pt"


class WebcamStream:

    def __init__(self):
        self.camera = None
        self.model = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.latest_frame = None
        self.people_count = 0
        self.error = None

    def start(self):
        if self.running:
            return True, "Camera is already running."

        if not MODEL_PATH.exists():
            self.error = f"Model not found: {MODEL_PATH}"
            return False, self.error

        try:
            self.error = None

            print("SAARTHI: Loading YOLOv8 Pose model...")
            self.model = YOLO(str(MODEL_PATH))

            print("SAARTHI: Opening webcam...")
            self.camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)

            if not self.camera.isOpened():
                self.camera.release()
                self.camera = None

                self.error = (
                    "Could not open webcam. "
                    "Close other camera applications and try again."
                )
                return False, self.error

            self.running = True

            self.thread = threading.Thread(
                target=self._process_frames,
                daemon=True
            )
            self.thread.start()

            return True, "Camera monitoring started."

        except Exception as exc:
            self.error = str(exc)
            self.running = False
            self._release_camera()

            print("SAARTHI camera startup error:", exc)
            return False, self.error

    def _process_frames(self):
        try:
            while self.running:
                camera = self.camera

                if camera is None:
                    break

                success, frame = camera.read()

                if not success:
                    self.error = "Could not read webcam frame."
                    time.sleep(0.1)
                    continue

                results = self.model.predict(
                    source=frame,
                    classes=[0],
                    conf=0.5,
                    verbose=False
                )

                count = len(results[0].boxes)
                annotated = results[0].plot()

                with self.lock:
                    self.latest_frame = annotated
                    self.people_count = count

        except Exception as exc:
            self.error = str(exc)
            print("SAARTHI camera processing error:", exc)

        finally:
            self.running = False
            self._release_camera()

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

    def get_status(self):
        return {
            "running": self.running,
            "people_count": self.people_count,
            "error": self.error
        }

    def stop(self):
        self.running = False

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3)

        self._release_camera()

        return True, "Camera monitoring stopped."

    def _release_camera(self):
        camera = self.camera
        self.camera = None

        if camera is not None:
            camera.release()


webcam_stream = WebcamStream()