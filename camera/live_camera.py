import cv2

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Camera could not be opened.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read frame.")
        break

    cv2.imshow("SAARTHI - Live Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()