import cv2
from ultralytics import YOLO

def main():
    model = YOLO("yolov8n-pose.pt")
    camera = cv2.VideoCapture(1)

    if not camera.isOpened():
        print("ERROR: Could not open camera.")
        return

    print("SAARTHI: Person detection started.")
    print("Press Q to exit.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Cannot read camera frame.")
                break

            results = model.predict(
                source=frame,
                classes=[0],
                conf=0.5,
                verbose=False
            )

            cv2.imshow(
                "SAARTHI - Person Detection",
                results[0].plot()
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()