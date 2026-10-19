from ultralytics import YOLO
import cv2

# Load face detection model
model = YOLO("../models/yolov8n-face.pt")

# Open laptop camera
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Camera could not be opened.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read frame.")
        break

    # Detect faces
    results = model(frame, verbose=False)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            confidence = float(box.conf[0])

            if confidence >= 0.5:
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
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

    cv2.imshow("SAARTHI - Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()