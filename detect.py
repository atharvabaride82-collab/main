import os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np
from ultralytics import YOLO

SOURCE = 0
MODEL_CANDIDATES = [
    "runs/detect/final/weights/best.pt",
    "runs/detect/test_run/weights/best.pt",
]
CONF = 0.25
IMGSZ = 640

model_path = next((p for p in MODEL_CANDIDATES if os.path.exists(p)), None)
if model_path is None:
    raise SystemExit("No model found. Checked: " + ", ".join(MODEL_CANDIDATES))
print("Using model:", model_path)

model = YOLO(model_path)
print("Classes:", model.names)
garbage_ids = [i for i, n in model.names.items() if n.lower() == "garbage"] or None
model.predict(np.zeros((480, 640, 3), np.uint8), imgsz=IMGSZ, verbose=False)

cap = cv2.VideoCapture(SOURCE)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not cap.isOpened():
    raise SystemExit("Could not open the webcam.")

print("Press Q in the video window to quit.")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Could not read a frame from the webcam.")
        break

    frame = cv2.resize(frame, (640, 480))
    res = model.predict(frame, imgsz=IMGSZ, conf=CONF,
                        classes=garbage_ids, verbose=False)[0]

    count = 0
    for b, c in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist()):
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"Garbage {c:.2f}", (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        count += 1

    if count:
        text, color = f"Garbage detected: {count}", (0, 0, 255)
    else:
        text, color = "No garbage detected", (0, 160, 0)
    cv2.rectangle(frame, (0, 0), (640, 36), (255, 255, 255), -1)
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Smart Garbage Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
