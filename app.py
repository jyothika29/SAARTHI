from flask import Flask, render_template, Response, jsonify, request
from camera.webcam_stream import webcam_stream
from pathlib import Path
import csv
from werkzeug.utils import secure_filename

import time
import uuid
import threading
import torch
import serial
import atexit

from models.model import HybridVideoClassifier
from src.video_utils import load_video_frames
from configs.config import NUM_FRAMES


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
MODEL_PATH = BASE_DIR / "models" / "hybrid_model3.pth"

UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}
MAX_UPLOAD_MB = 200

app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

DEVICE = torch.device("cpu")
# ---------------- ESP32 BUZZER ----------------

ESP32_PORT = "COM7"
ESP32_BAUD = 115200

buzzer_serial = None
buzzer_lock = threading.Lock()


def connect_buzzer():
    global buzzer_serial

    try:
        buzzer_serial = serial.Serial(
            ESP32_PORT,
            ESP32_BAUD,
            timeout=1,
            write_timeout=1
        )

        # ESP32 may restart when the serial port opens.
        time.sleep(2)

        print("SAARTHI: ESP32 buzzer connected on", ESP32_PORT)

    except serial.SerialException as error:
        buzzer_serial = None
        print("SAARTHI: Buzzer unavailable:", error)


def trigger_buzzer():
    """Request a brief beep. The ESP32 stops it after one second."""
    with buzzer_lock:
        if buzzer_serial is None or not buzzer_serial.is_open:
            print("SAARTHI: Buzzer is not connected.")
            return False

        try:
            buzzer_serial.write(b"ALERT\n")
            buzzer_serial.flush()
            return True

        except serial.SerialException as error:
            print("SAARTHI: Buzzer command failed:", error)
            return False


def close_buzzer():
    global buzzer_serial

    with buzzer_lock:
        if buzzer_serial is not None:
            try:
                buzzer_serial.close()
            except serial.SerialException:
                pass

            buzzer_serial = None


atexit.register(close_buzzer)


# ---------------- LOAD AI MODEL ----------------

model = None
model_error = None
model_lock = threading.Lock()


def load_ai_model():
    global model, model_error

    if not MODEL_PATH.is_file():
        model_error = f"Model weights not found: {MODEL_PATH}"
        app.logger.error(model_error)
        return

    try:
        loaded_model = HybridVideoClassifier(
            freeze_backbone=True,
            fine_tune_layer4=True
        )

        try:
            weights = torch.load(
                MODEL_PATH,
                map_location=DEVICE,
                weights_only=True
            )
        except TypeError:
            weights = torch.load(
                MODEL_PATH,
                map_location=DEVICE
            )

        if isinstance(weights, dict) and "state_dict" in weights:
            weights = weights["state_dict"]

        loaded_model.load_state_dict(weights, strict=True)
        loaded_model.to(DEVICE)
        loaded_model.eval()

        model = loaded_model
        model_error = None

        app.logger.info("SAARTHI video model loaded successfully.")

    except Exception as error:
        model = None
        model_error = str(error)
        app.logger.exception("Could not load the video model.")


load_ai_model()


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )

# ---------------- BUZZER TEST API ----------------

@app.route("/api/buzzer/test", methods=["POST"])
def test_buzzer():
    success = trigger_buzzer()

    return jsonify({
        "success": success,
        "message": (
            "Buzzer command sent."
            if success
            else "Buzzer is not connected. Check the Flask terminal."
        )
    }), 200 if success else 503
# ---------------- DASHBOARD ----------------

@app.route("/")
def dashboard():
    return render_template("index.html")
# ---------------- ATTENDANCE API ----------------

ATTENDANCE_FILE = BASE_DIR / "attendance" / "attendance.csv"


@app.route("/api/attendance/today")
def attendance_today():
    from datetime import datetime

    today = datetime.now().strftime("%Y-%m-%d")
    records = []
    present_people = set()

    try:
        if ATTENDANCE_FILE.exists():
            with open(
                ATTENDANCE_FILE,
                "r",
                newline="",
                encoding="utf-8-sig"
            ) as file:

                reader = csv.reader(file)

                for row in reader:
                    if len(row) < 4:
                        continue

                    name, date, event_time, status = [
                        value.strip() for value in row[:4]
                    ]

                    # Skip a header row if one exists.
                    if name.lower() == "name":
                        continue

                    # Only return records from today.
                    if date != today:
                        continue

                    record = {
                        "name": name,
                        "date": date,
                        "time": event_time,
                        "status": status.upper()
                    }

                    records.append(record)

                    if status.upper() == "ENTRY" and name:
                        present_people.add(name)

        records.reverse()

        return jsonify({
            "success": True,
            "date": today,
            "count": len(present_people),
            "records": records
        })

    except (OSError, csv.Error) as error:
        app.logger.exception("Could not read attendance file.")

        return jsonify({
            "success": False,
            "message": "Could not read attendance records."
        }), 500


# ---------------- RECENT ACTIVITY API ----------------

UNKNOWN_PERSON_FILE = BASE_DIR / "alerts" / "unknown_person.csv"
ACTIVITY_ALERT_FILE = BASE_DIR / "alerts" / "activity_alert.csv"


@app.route("/api/alerts/recent")
def recent_alerts():
    events = []

    alert_files = [
        UNKNOWN_PERSON_FILE,
        ACTIVITY_ALERT_FILE
    ]

    try:
        for alert_file in alert_files:

            if not alert_file.is_file():
                continue

            with open(
                alert_file,
                "r",
                newline="",
                encoding="utf-8-sig"
            ) as file:

                reader = csv.DictReader(file)

                # Normalize CSV column names.
                if reader.fieldnames:
                    reader.fieldnames = [
                        name.strip().lstrip("\ufeff")
                        if name else ""
                        for name in reader.fieldnames
                    ]

                for row in reader:

                    # Normalize each row's column names too.
                    cleaned_row = {
                        str(key).strip().lstrip("\ufeff"): (
                            value.strip() if isinstance(value, str)
                            else value
                        )
                        for key, value in row.items()
                        if key is not None
                    }

                    date = cleaned_row.get("Date", "")
                    event_time = cleaned_row.get("Time", "")

                    # Support common alternative event column names.
                    event_name = (
                        cleaned_row.get("Event")
                        or cleaned_row.get("Alert")
                        or cleaned_row.get("Activity")
                        or cleaned_row.get("Event Type")
                        or cleaned_row.get("event")
                        or "Safety event detected"
                    )

                    # Avoid displaying empty or malformed rows.
                    if not date and not event_time and not event_name:
                        continue

                    events.append({
                        "date": date,
                        "time": event_time,
                        "event": event_name
                    })

        # Newest date and time first.
        events.sort(
            key=lambda event: (
                event["date"],
                event["time"]
            ),
            reverse=True
        )

        return jsonify({
            "success": True,
            "events": events[:10]
        })

    except (OSError, csv.Error):
        app.logger.exception(
            "Could not read the alert log files."
        )

        return jsonify({
            "success": False,
            "message": "Could not read the alert logs.",
            "events": []
        }), 500

# ---------------- CAMERA START ----------------

@app.route("/api/camera/start", methods=["POST"])
def start_camera():
    try:
        success, message = webcam_stream.start()

        return jsonify({
            "success": success,
            "message": message
        }), 200 if success else 500

    except Exception as exc:
        app.logger.exception("Camera start failed")

        return jsonify({
            "success": False,
            "message": str(exc)
        }), 500


# ---------------- CAMERA STOP ----------------

@app.route("/api/camera/stop", methods=["POST"])
def stop_camera():
    try:
        success, message = webcam_stream.stop()

        return jsonify({
            "success": success,
            "message": message
        }), 200 if success else 500

    except Exception as exc:
        app.logger.exception("Camera stop failed")

        return jsonify({
            "success": False,
            "message": str(exc)
        }), 500


# ---------------- CAMERA STATUS ----------------

@app.route("/api/camera/status", methods=["GET"])
def camera_status():
    return jsonify(webcam_stream.get_status())
# ---------------- LIVE CAMERA VIDEO FEED ----------------

@app.route("/video_feed")
def video_feed():

    def generate():
        while True:
            if not webcam_stream.running:
                break

            frame = webcam_stream.get_frame()

            if frame is None:
                time.sleep(0.1)
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Cache-Control: no-cache\r\n\r\n"
                + frame
                + b"\r\n"
            )

            time.sleep(0.03)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


# ---------------- VIDEO UPLOAD AND AI ANALYSIS ----------------

@app.route("/api/video/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({
            "success": False,
            "message": "Please select a video file."
        }), 400

    video = request.files["video"]

    if not video.filename:
        return jsonify({
            "success": False,
            "message": "No video was selected."
        }), 400

    if not allowed_file(video.filename):
        return jsonify({
            "success": False,
            "message": "Unsupported format. Use MP4, AVI, MOV or MKV."
        }), 400

    if model is None:
        return jsonify({
            "success": False,
            "message": (
                "AI model could not be loaded. "
                "Check the terminal for details."
            ),
            "model_error": model_error
        }), 503

    safe_name = secure_filename(video.filename)
    unique_name = f"{uuid.uuid4().hex}_{safe_name}"
    save_path = UPLOAD_DIR / unique_name

    try:
        video.save(save_path)

        frames = load_video_frames(
            str(save_path),
            num_frames=NUM_FRAMES
        )

        frames = frames.unsqueeze(0).to(DEVICE)

        with model_lock, torch.inference_mode():
            logits, uncertainty = model(frames)

        raw_logit = float(logits.reshape(-1)[0].item())

        score = float(
            torch.sigmoid(logits.reshape(-1)[0]).item()
        )

        uncertainty_score = float(
            uncertainty.reshape(-1)[0].item()
        )

        return jsonify({
            "success": True,
            "message": (
                "Video analysis completed. Results are experimental "
                "and require human review."
            ),
            "filename": unique_name,
            "frames_analyzed": NUM_FRAMES,
            "experimental_score": round(score, 4),
            "raw_model_output": round(raw_logit, 4),
            "uncertainty": round(uncertainty_score, 4),
            "result": (
                "Experimental model output — human review required."
            ),
            "warning": (
                "This model output does not establish that violence "
                "or abuse occurred. Training labels and accuracy "
                "must be verified."
            )
        })

    except Exception:
        app.logger.exception("Video analysis failed.")

        return jsonify({
            "success": False,
            "message": (
                "Video analysis failed. Check the Flask terminal "
                "for the error details."
            )
        }), 500

    finally:
        try:
            if save_path.exists():
                save_path.unlink()
        except OSError:
            app.logger.warning(
                "Could not remove temporary uploaded video."
            )


# ---------------- ERROR HANDLING ----------------

@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "success": False,
        "message": (
            f"Video is too large. Maximum size is {MAX_UPLOAD_MB} MB."
        )
    }), 413


# ---------------- RUN APPLICATION ----------------

if __name__ == "__main__":
    connect_buzzer()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True
    )