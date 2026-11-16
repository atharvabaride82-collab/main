import os
import time
import threading
import webbrowser
import smtplib
from datetime import datetime
from email.message import EmailMessage

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request
from ultralytics import YOLO

# Keep these settings aligned with your existing project.
SOURCE = 0
MODEL_CANDIDATES = [
    "runs/detect/final/weights/best.pt",
    "runs/detect/test_run/weights/best.pt",
]
CONF = 0.25
IMGSZ = 640
PORT = 5000

# Set these as environment variables in the terminal; never hard-code passwords.
GMAIL_SENDER = os.getenv("GMAIL_SENDER", "").strip()
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "").replace(" ", "").strip()
GMAIL_RECEIVER = os.getenv("GMAIL_RECEIVER", "").strip()
EMAIL_COOLDOWN_SECONDS = 60  # Helps prevent repeated emails.

model_path = next((p for p in MODEL_CANDIDATES if os.path.exists(p)), None)
if model_path is None:
    raise SystemExit(
        "No model found. Checked: " + ", ".join(MODEL_CANDIDATES)
        + "\nRun this file from your smart_garbage_detection project directory."
    )

print("Using model:", model_path)
model = YOLO(model_path)
print("Model classes:", model.names)

# Filters to a class named exactly 'garbage'; if not present, all model classes are used.
garbage_ids = [i for i, name in model.names.items()
               if str(name).lower() == "garbage"] or None

app = Flask(__name__)
lock = threading.Lock()
state = {
    "jpeg": None,
    "count": 0,
    "camera_active": False,
    "running": True,
    "message": "Starting camera...",
    "email_status": "Gmail is not configured." if not (
        GMAIL_SENDER and GMAIL_APP_PASSWORD and GMAIL_RECEIVER
    ) else "Gmail configured; waiting for detection.",
}
last_email_time = 0.0
object_was_visible = False
last_no_detection_time = time.monotonic()


def send_email(image_bytes, count, detected_at):
    """Send the annotated detection screenshot as a Gmail attachment."""
    if not (GMAIL_SENDER and GMAIL_APP_PASSWORD and GMAIL_RECEIVER):
        status = "Not sent: configure GMAIL_SENDER, GMAIL_APP_PASSWORD, GMAIL_RECEIVER."
        print(status)
    else:
        msg = EmailMessage()
        msg["Subject"] = f"Garbage detected - {detected_at}"
        msg["From"] = GMAIL_SENDER
        msg["To"] = GMAIL_RECEIVER
        msg.set_content(
            f"Smart Garbage Detection detected {count} object(s) at {detected_at}.\n"
            "The attached screenshot includes the red detection box(es)."
        )
        filename = "garbage_detection_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
        msg.add_attachment(image_bytes, maintype="image", subtype="jpeg", filename=filename)
        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as smtp:
                smtp.login(GMAIL_SENDER, GMAIL_APP_PASSWORD)
                smtp.send_message(msg)
            status = f"Screenshot emailed to {GMAIL_RECEIVER} at {datetime.now():%H:%M:%S}."
        except Exception as exc:
            status = f"Email failed: {exc}"
        print(status)

    with lock:
        state["email_status"] = status


def camera_worker():
    global last_email_time, object_was_visible, last_no_detection_time

    cap = cv2.VideoCapture(SOURCE)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    try:
        while True:
            with lock:
                running = state["running"]

            ok, frame = cap.read()
            if not ok:
                with lock:
                    state["camera_active"] = False
                    state["message"] = "Camera unavailable. Check permissions or close other camera apps."
                time.sleep(1)
                cap.release()
                cap = cv2.VideoCapture(SOURCE)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                continue

            frame = cv2.resize(frame, (640, 480))
            count = 0

            if running:
                result = model.predict(
                    frame, imgsz=IMGSZ, conf=CONF,
                    classes=garbage_ids, verbose=False
                )[0]

                for box, confidence in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
                    x1, y1, x2, y2 = map(int, box)
                    # Red bounding box and red label background.
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    label = f"Garbage {confidence:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    ty = max(y1, th + 8)
                    cv2.rectangle(frame, (x1, ty-th-8), (x1+tw+6, ty), (0, 0, 255), -1)
                    cv2.putText(frame, label, (x1+3, ty-4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    count += 1

                now = time.monotonic()
                if count > 0:
                    with lock:
                        state["message"] = f"Garbage detected: {count}. Red box shown."
                    # Email once when a new detection event begins, not on every frame.
                    if not object_was_visible and now - last_email_time >= EMAIL_COOLDOWN_SECONDS:
                        ok_jpg, buffer = cv2.imencode(
                            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90]
                        )
                        if ok_jpg:
                            threading.Thread(
                                target=send_email,
                                args=(buffer.tobytes(), count, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                                daemon=True,
                            ).start()
                            last_email_time = now
                    object_was_visible = True
                else:
                    if object_was_visible:
                        last_no_detection_time = now
                    if now - last_no_detection_time >= 3:
                        object_was_visible = False
                    with lock:
                        state["message"] = "No garbage detected."
                with lock:
                    state["count"] = count
            else:
                with lock:
                    state["message"] = "Detection paused."
                    state["count"] = 0

            status_text = f"Garbage detected: {count}" if count else (
                "DETECTION PAUSED" if not running else "No garbage detected"
            )
            cv2.rectangle(frame, (0, 0), (640, 38), (255, 255, 255), -1)
            cv2.putText(frame, status_text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255) if count else (0, 130, 0), 2)

            ok_jpg, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            with lock:
                state["camera_active"] = True
                if ok_jpg:
                    state["jpeg"] = buffer.tobytes()
            time.sleep(0.01)
    finally:
        cap.release()


# Simple page: live camera, detection count, camera/email status, Start and Pause.
PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Smart Garbage Detection</title>
<style>
body{margin:0;padding:24px;font-family:Arial,sans-serif;background:#f3f6fb;color:#1d2939;text-align:center}
main{max-width:900px;margin:auto}h1{margin-bottom:8px}
#status,.card{background:white;border-radius:8px;padding:14px;margin:12px auto}
#video{width:100%;max-width:800px;height:auto;background:#111;border-radius:8px}
.card{text-align:left;line-height:1.9}
button{border:0;border-radius:6px;padding:10px 16px;margin:6px;cursor:pointer;background:#2457a7;color:white;font-weight:bold}
button:disabled{opacity:.5}.small{color:#667085;font-size:13px}
</style>
</head>
<body><main>
<h1>Smart Garbage Detection</h1>
<p class="small">Live camera • Red box means garbage detected • Screenshot sent by Gmail</p>
<div id="status">Connecting to camera…</div>
<img id="video" src="/video" alt="Live webcam with red garbage detection boxes">
<div class="card">
<strong>Objects in current frame:</strong> <span id="count">0</span><br>
<strong>Camera:</strong> <span id="camera">Checking…</span><br>
<strong>Email status:</strong> <span id="email">Waiting…</span>
</div>
<button id="start" onclick="setRunning(true)">Start Detection</button>
<button id="pause" onclick="setRunning(false)">Pause Detection</button>
</main>
<script>
async function refreshStatus(){
  try{
    const r=await fetch('/api/status'); const d=await r.json();
    document.getElementById('status').textContent=d.message;
    document.getElementById('count').textContent=d.count;
    document.getElementById('camera').textContent=d.camera_active?'Active':'Offline';
    document.getElementById('email').textContent=d.email_status;
    document.getElementById('start').disabled=d.running;
    document.getElementById('pause').disabled=!d.running;
  }catch(e){document.getElementById('status').textContent='Cannot reach Flask server. Check terminal.';}
}
async function setRunning(running){
  try{
    await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({running})});
    refreshStatus();
  }catch(e){document.getElementById('status').textContent='Could not change detection state.';}
}
refreshStatus(); setInterval(refreshStatus,1000);
</script>
</body></html>"""


@app.route("/")
def index():
    return PAGE


def stream():
    while True:
        with lock:
            jpeg = state["jpeg"]
        if jpeg is not None:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(0.03)


@app.route("/video")
def video():
    return Response(stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    with lock:
        return jsonify({
            "count": state["count"],
            "camera_active": state["camera_active"],
            "running": state["running"],
            "message": state["message"],
            "email_status": state["email_status"],
        })


@app.route("/api/control", methods=["POST"])
def api_control():
    data = request.get_json(silent=True) or {}
    if not isinstance(data.get("running"), bool):
        return jsonify({"ok": False, "error": "Expected running to be true or false."}), 400
    with lock:
        state["running"] = data["running"]
    return jsonify({"ok": True, "running": data["running"]})


if __name__ == "__main__":
    url = f"http://127.0.0.1:{PORT}"
    print("Open", url, "in your browser (Ctrl+C to stop)")
    if not (GMAIL_SENDER and GMAIL_APP_PASSWORD and GMAIL_RECEIVER):
        print("Gmail alert not configured. Set GMAIL_SENDER, GMAIL_APP_PASSWORD, and GMAIL_RECEIVER.")
    threading.Thread(target=camera_worker, daemon=True).start()
    # Preserve automatic opening in the default browser (Firefox if configured as default).
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=PORT, threaded=True, use_reloader=False)
