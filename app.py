import os
import time
import threading
import webbrowser

import cv2
import numpy as np
from flask import Flask, Response
from ultralytics import YOLO

# ---------------- settings (change only if needed) ----------------
SOURCE = 0            # 0 = laptop webcam. For a phone: "http://PHONE_IP:8080/video"
MODEL_CANDIDATES = [
    "runs/detect/final/weights/best.pt",
    "runs/detect/test_run/weights/best.pt",
]
CONF = 0.35           # confidence threshold
IMGSZ = 640           # use 416 if it is slow
PORT = 5000
# -------------------------------------------------------------------

model_path = next((p for p in MODEL_CANDIDATES if os.path.exists(p)), None)
if model_path is None:
    raise SystemExit("No model found. Checked: " + ", ".join(MODEL_CANDIDATES))
print("Using model:", model_path)

model = YOLO(model_path)
print("Classes:", model.names)
garbage_ids = [i for i, n in model.names.items() if n.lower() == "garbage"] or None
model.predict(np.zeros((480, 640, 3), np.uint8), imgsz=IMGSZ, verbose=False)  # warm-up

latest = {"jpeg": None}


def worker():
    """Reads the camera, runs YOLO, and stores the newest annotated frame."""
    cap = cv2.VideoCapture(SOURCE)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(1)
            cap.release()
            cap = cv2.VideoCapture(SOURCE)
            continue

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

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            latest["jpeg"] = buf.tobytes()


threading.Thread(target=worker, daemon=True).start()

app = Flask(__name__)

PAGE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Smart Garbage Detection</title>
  <style>
    body { font-family: Arial, sans-serif; background: #f4f6f8; margin: 0; text-align: center; }
    h1 { margin: 24px 0 8px; color: #1f3864; }
    p { color: #555; margin: 0 0 16px; }
    img { width: 90%; max-width: 900px; border: 3px solid #1f3864; border-radius: 8px; }
  </style>
</head>
<body>
  <h1>Smart Garbage Detection</h1>
  <p>Live detection using YOLO11 and OpenCV</p>
  <img src="/video" alt="Live video">
</body>
</html>"""


@app.route("/")
def index():
    return PAGE


def stream():
    while True:
        jpeg = latest["jpeg"]
        if jpeg is not None:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
        time.sleep(0.03)


@app.route("/video")
def video():
    return Response(stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    url = f"http://127.0.0.1:{PORT}"
    print("Open", url, "in your browser (Ctrl+C to stop)")
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=PORT, threaded=True)
