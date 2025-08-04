from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture(r'car_counting_project\2103099-uhd_3840_2160_30fps.mp4')
if not cap.isOpened():
    print("Error: Cannot open video file")
    exit()

model = YOLO('yolov8n.pt')
cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)  # Allows window resizing

while True:
    success, img = cap.read()
    if not success:
        print("Video ended or error reading frame.")
        break

    img = cv2.resize(img, (1280, 720))  # Resize to fit your screen or GPU power

    results = model(img, stream=True)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = round(float(box.conf[0]), 2)
            cls = int(box.cls[0])
            class_name = model.names[cls]

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cvzone.putTextRect(img, f'{class_name} {conf}', (x1, y1 - 10),
                               scale=1, thickness=1, colorR=(0, 255, 0), offset=5)

    cv2.imshow("YOLO Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
