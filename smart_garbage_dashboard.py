import os
import time
import threading
import webbrowser
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_file
from ultralytics import YOLO

# ---------------- settings (keep your existing project setup) ----------------
SOURCE = 0  # 0 = laptop webcam. For a phone: "http://PHONE_IP:8080/video"
MODEL_CANDIDATES = [
    "runs/detect/final/weights/best.pt",
    "runs/detect/test_run/weights/best.pt",
]
CONF = 0.25
IMGSZ = 640
PORT = 5000

SHOW_ZONE = False
ZONE = (160, 120, 480, 400)
# ------------------------------------------------------------------------------

model_path = next((p for p in MODEL_CANDIDATES if os.path.exists(p)), None)
if model_path is None:
    raise SystemExit(
        "No model found. Checked: " + ", ".join(MODEL_CANDIDATES)
        + "\nRun this file from your project directory so the runs/ folder can be found."
    )

print("Using model:", model_path)
model_label = "final" if "/final/" in model_path else "test_run"
model = YOLO(model_path)
print("Classes:", model.names)

garbage_ids = [i for i, n in model.names.items() if n.lower() == "garbage"] or None
model.predict(np.zeros((480, 640, 3), np.uint8), imgsz=IMGSZ, verbose=False)

app = Flask(__name__)

# Shared state is protected because the camera worker and Flask routes run separately.
state_lock = threading.Lock()
latest = {
    "jpeg": None,
    "frame": None,
    "camera_active": False,
    "running": True,
    "current_count": 0,
    "highest_confidence": 0.0,
    "total_detections": 0,
    "history": [],
    "confidence": CONF,
    "last_update": None,
    "error": None,
}


def add_history(count, confidences):
    """Store a small recent-detection list for the dashboard."""
    if not count:
        return
    now = datetime.now().strftime("%H:%M:%S")
    items = []
    for confidence in confidences:
        items.append({
            "time": now,
            "object": "Garbage",
            "confidence": round(float(confidence) * 100, 1),
            "status": "Detected",
        })
    with state_lock:
        latest["history"] = (items + latest["history"])[:12]


def worker():
    """Continuously read the camera and run YOLO while detection is enabled."""
    cap = cv2.VideoCapture(SOURCE)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while True:
        with state_lock:
            is_running = latest["running"]
            confidence_threshold = latest["confidence"]

        ok, frame = cap.read()
        if not ok:
            with state_lock:
                latest["camera_active"] = False
                latest["error"] = "Camera unavailable. Check permissions or whether another app is using it."
            time.sleep(1)
            cap.release()
            cap = cv2.VideoCapture(SOURCE)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue

        frame = cv2.resize(frame, (640, 480))
        with state_lock:
            latest["camera_active"] = True
            latest["error"] = None

        count = 0
        confidences = []
        in_zone = False

        if is_running:
            result = model.predict(
                frame,
                imgsz=IMGSZ,
                conf=confidence_threshold,
                classes=garbage_ids,
                verbose=False,
            )[0]

            for box, confidence in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                label = f"Garbage {confidence:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                ty = max(y1, th + 8)
                cv2.rectangle(frame, (x1, ty - th - 8), (x1 + tw + 6, ty), (0, 0, 255), -1)
                cv2.putText(
                    frame, label, (x1 + 3, ty - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )
                count += 1
                confidences.append(float(confidence))
                cx, cy = (x1 + x2) // 2, y2
                if SHOW_ZONE and ZONE[0] <= cx <= ZONE[2] and ZONE[1] <= cy <= ZONE[3]:
                    in_zone = True

            if SHOW_ZONE:
                zone_color = (0, 0, 255) if in_zone else (0, 200, 0)
                cv2.rectangle(frame, (ZONE[0], ZONE[1]), (ZONE[2], ZONE[3]), zone_color, 3)
                if in_zone:
                    cv2.putText(
                        frame, "ALERT: Garbage in restricted area!", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                    )

            status_text = f"Garbage detected: {count}" if count else "No garbage detected"
            status_color = (0, 0, 255) if count else (0, 150, 0)
            with state_lock:
                latest["current_count"] = count
                latest["highest_confidence"] = max(confidences, default=0.0)
                latest["total_detections"] += count
                latest["last_update"] = datetime.now().strftime("%H:%M:%S")
            add_history(count, confidences)
        else:
            status_text = "DETECTION PAUSED"
            status_color = (0, 120, 200)

        cv2.rectangle(frame, (0, 0), (640, 38), (255, 255, 255), -1)
        cv2.putText(frame, status_text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(
            frame, f"YOLO | {model_label}", (500, 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1
        )

        with state_lock:
            latest["frame"] = frame.copy()
            ok_jpeg, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok_jpeg:
                latest["jpeg"] = buffer.tobytes()

        time.sleep(0.01)


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Smart Garbage Detection</title>
<style>
:root { --navy:#142b4a; --blue:#2457a7; --green:#15966a; --red:#d94444; --muted:#718096; --bg:#f3f6fb; }
* { box-sizing:border-box; }
body { margin:0; font-family:Inter,Arial,sans-serif; background:var(--bg); color:#1c2b3a; }
header { background:linear-gradient(120deg,#102744,#2457a7); color:white; padding:22px 5%; display:flex; justify-content:space-between; align-items:center; gap:18px; flex-wrap:wrap; }
.brand { display:flex; align-items:center; gap:13px; }
.logo { width:46px; height:46px; border-radius:13px; background:#ffffff20; display:grid; place-items:center; font-size:25px; }
h1 { margin:0; font-size:clamp(20px,3vw,28px); }
.subtitle { margin:6px 0 0; color:#dbe7f8; font-size:13px; }
.status-pill { border:1px solid #ffffff55; border-radius:999px; padding:10px 14px; font-weight:700; font-size:12px; letter-spacing:.5px; }
.status-dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:#53e3a5; margin-right:7px; }
main { max-width:1400px; margin:25px auto; padding:0 22px 35px; }
.section-title { margin:0 0 14px; font-size:17px; }
.layout { display:grid; grid-template-columns:minmax(0,1.8fr) minmax(270px,1fr); gap:20px; align-items:start; }
.card { background:white; border:1px solid #e2e8f0; border-radius:16px; box-shadow:0 5px 20px #1e3a5f0b; overflow:hidden; }
.card-head { display:flex; justify-content:space-between; align-items:center; gap:10px; padding:17px 19px; border-bottom:1px solid #edf1f6; }
.card-head h2 { font-size:15px; margin:0; }
.live-tag { color:var(--red); font-size:11px; font-weight:800; letter-spacing:.7px; }
.video-wrap { background:#101820; min-height:230px; display:grid; place-items:center; }
#video { display:block; width:100%; height:auto; max-height:600px; object-fit:contain; }
.video-note { padding:12px 18px; color:var(--muted); font-size:12px; }
.stats { display:grid; grid-template-columns:1fr 1fr; gap:12px; padding:16px; }
.stat { background:#f7f9fc; border:1px solid #e8edf5; padding:16px; border-radius:12px; min-width:0; }
.stat-label { color:var(--muted); font-size:12px; margin-bottom:9px; }
.stat-value { font-size:25px; font-weight:800; color:var(--navy); overflow-wrap:anywhere; }
.stat-sub { color:var(--muted); font-size:11px; margin-top:6px; }
.controls { display:flex; flex-wrap:wrap; gap:10px; padding:16px; border-top:1px solid #edf1f6; }
button { border:0; border-radius:9px; padding:11px 15px; font-weight:700; cursor:pointer; transition:transform .15s,opacity .15s; }
button:hover { transform:translateY(-1px); }
button:disabled { opacity:.55; cursor:not-allowed; transform:none; }
.primary { background:var(--blue); color:white; }
.success { background:#e2f7ed; color:#08794f; }
.neutral { background:#edf2f7; color:#334155; }
.danger { background:#fff0f0; color:#b42323; }
.settings { margin-top:20px; padding:20px; }
.setting-row { display:flex; align-items:center; justify-content:space-between; gap:18px; flex-wrap:wrap; }
input[type=range] { width:min(440px,65vw); accent-color:var(--blue); }
.muted { color:var(--muted); font-size:12px; }
.history { margin-top:20px; }
.table-wrap { overflow-x:auto; }
table { border-collapse:collapse; width:100%; min-width:400px; }
th,td { text-align:left; padding:13px 17px; border-bottom:1px solid #edf1f6; font-size:13px; }
th { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.5px; }
.badge { display:inline-block; border-radius:999px; padding:5px 9px; background:#e2f7ed; color:#08794f; font-size:11px; font-weight:700; }
#message { margin:0 0 15px; color:var(--muted); font-size:13px; min-height:18px; }
footer { text-align:center; color:var(--muted); padding:18px; font-size:12px; }
@media(max-width:850px) { .layout { grid-template-columns:1fr; } main { padding:0 13px 25px; } header { padding:18px; } }
</style>
</head>
<body>
<header>
  <div class="brand">
    <div class="logo">♻</div>
    <div><h1>Smart Garbage Detection</h1><p class="subtitle">AI-powered waste monitoring dashboard · YOLO + OpenCV</p></div>
  </div>
  <div class="status-pill"><span class="status-dot" id="statusDot"></span><span id="systemStatus">CONNECTING</span></div>
</header>

<main>
  <p id="message" role="status">Connecting to the detection system…</p>
  <div class="layout">
    <section class="card">
      <div class="card-head"><h2>Live Camera Feed</h2><span class="live-tag">● LIVE</span></div>
      <div class="video-wrap"><img id="video" src="/video" alt="Live webcam feed with garbage detection boxes"></div>
      <div class="video-note">Detection boxes and confidence labels are drawn by the running YOLO model.</div>
      <div class="controls">
        <button class="primary" id="startBtn" onclick="setRunning(true)">▶ Start Detection</button>
        <button class="neutral" id="pauseBtn" onclick="setRunning(false)">Ⅱ Pause Detection</button>
        <button class="success" onclick="captureFrame()">▣ Capture Screenshot</button>
        <button class="danger" onclick="resetStats()">↻ Reset Statistics</button>
      </div>
    </section>

    <aside class="card">
      <div class="card-head"><h2>Live Statistics</h2><span class="muted">Updates automatically</span></div>
      <div class="stats">
        <div class="stat"><div class="stat-label">Objects in current frame</div><div class="stat-value" id="currentCount">—</div><div class="stat-sub">Garbage objects detected now</div></div>
        <div class="stat"><div class="stat-label">Highest confidence</div><div class="stat-value" id="confidenceValue">—</div><div class="stat-sub">Best detection in current frame</div></div>
        <div class="stat"><div class="stat-label">Camera</div><div class="stat-value" id="cameraStatus">Checking</div><div class="stat-sub">Laptop webcam connection</div></div>
        <div class="stat"><div class="stat-label">AI model</div><div class="stat-value" id="modelName">YOLO</div><div class="stat-sub">Loaded trained weights</div></div>
      </div>
      <div class="card-head"><h2>Session Summary</h2></div>
      <div class="stats">
        <div class="stat"><div class="stat-label">Detection observations</div><div class="stat-value" id="totalCount">0</div><div class="stat-sub">Counted across processed frames</div></div>
        <div class="stat"><div class="stat-label">Confidence threshold</div><div class="stat-value" id="thresholdCard">25%</div><div class="stat-sub">Minimum accepted confidence</div></div>
      </div>
    </aside>
  </div>

  <section class="card settings">
    <div class="setting-row">
      <div><h2 class="section-title">Detection Settings</h2><div class="muted">Adjust the minimum confidence used by YOLO.</div></div>
      <div><strong id="thresholdLabel">25%</strong></div>
    </div>
    <div class="setting-row" style="margin-top:14px">
      <input id="threshold" type="range" min="10" max="90" step="5" value="25" aria-label="Confidence threshold">
      <span class="muted">Lower = more detections, potentially more false positives.</span>
    </div>
  </section>

  <section class="card history">
    <div class="card-head"><h2>Recent Detections</h2><span class="muted">Latest 12 observations</span></div>
    <div class="table-wrap">
      <table><thead><tr><th>Time</th><th>Object</th><th>Confidence</th><th>Status</th></tr></thead>
      <tbody id="historyBody"><tr><td colspan="4" class="muted">Waiting for detections…</td></tr></tbody></table>
    </div>
  </section>
</main>
<footer>Smart Garbage Detection · Local application · Camera video is processed on this computer</footer>

<script>
const message = document.getElementById('message');
let thresholdTimer = null;

async function api(url, options={}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok || data.ok === false) throw new Error(data.error || 'Request failed');
  return data;
}
async function setRunning(running) {
  try {
    const data = await api('/api/control', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({running})
    });
    message.textContent = running ? 'Detection resumed.' : 'Detection paused. The camera feed remains open.';
    refreshStatus();
  } catch (e) { message.textContent = 'Could not change detection state: ' + e.message; }
}
async function resetStats() {
  try {
    await api('/api/reset', {method:'POST'});
    message.textContent = 'Session statistics and recent detection history have been reset.';
    refreshStatus();
  } catch (e) { message.textContent = 'Could not reset statistics: ' + e.message; }
}
async function captureFrame() {
  try {
    const response = await fetch('/api/capture', {method:'POST'});
    const data = await response.json();
    if (!response.ok || data.ok === false) throw new Error(data.error || 'Capture failed');
    message.textContent = 'Screenshot saved: ' + data.filename;
  } catch (e) { message.textContent = 'Could not capture screenshot: ' + e.message; }
}
document.getElementById('threshold').addEventListener('input', event => {
  const value = Number(event.target.value);
  document.getElementById('thresholdLabel').textContent = value + '%';
  document.getElementById('thresholdCard').textContent = value + '%';
  clearTimeout(thresholdTimer);
  thresholdTimer = setTimeout(async () => {
    try {
      await api('/api/confidence', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body:JSON.stringify({confidence:value/100})
      });
      message.textContent = 'Confidence threshold updated to ' + value + '%.';
    } catch (e) { message.textContent = 'Could not update threshold: ' + e.message; }
  }, 180);
});
async function refreshStatus() {
  try {
    const data = await api('/api/status');
    document.getElementById('systemStatus').textContent = data.running ? 'SYSTEM RUNNING' : 'DETECTION PAUSED';
    document.getElementById('statusDot').style.background = data.running ? '#53e3a5' : '#ffca65';
    document.getElementById('currentCount').textContent = data.current_count;
    document.getElementById('confidenceValue').textContent = data.highest_confidence + '%';
    document.getElementById('cameraStatus').textContent = data.camera_active ? 'Active' : 'Offline';
    document.getElementById('cameraStatus').style.color = data.camera_active ? '#15966a' : '#d94444';
    document.getElementById('modelName').textContent = data.model_label;
    document.getElementById('totalCount').textContent = data.total_detections;
    document.getElementById('thresholdLabel').textContent = Math.round(data.confidence*100) + '%';
    document.getElementById('thresholdCard').textContent = Math.round(data.confidence*100) + '%';
    document.getElementById('threshold').value = Math.round(data.confidence*100);
    document.getElementById('startBtn').disabled = data.running;
    document.getElementById('pauseBtn').disabled = !data.running;
    if (!data.camera_active && data.error) message.textContent = data.error;
    const body = document.getElementById('historyBody');
    if (!data.history.length) {
      body.innerHTML = '<tr><td colspan="4" class="muted">No detections recorded yet. Show an item to the camera.</td></tr>';
    } else {
      body.innerHTML = data.history.map(item =>
        `<tr><td>${item.time}</td><td>${item.object}</td><td>${item.confidence}%</td><td><span class="badge">${item.status}</span></td></tr>`
      ).join('');
    }
  } catch (e) {
    document.getElementById('systemStatus').textContent = 'CONNECTION ISSUE';
    message.textContent = 'Cannot reach the Flask server. Check the terminal for errors.';
  }
}
refreshStatus();
setInterval(refreshStatus, 1000);
</script>
</body>
</html>"""


@app.route("/")
def index():
    return PAGE


def stream():
    while True:
        with state_lock:
            jpeg = latest["jpeg"]
        if jpeg is not None:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(0.03)


@app.route("/video")
def video():
    return Response(stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    with state_lock:
        return jsonify({
            "ok": True,
            "running": latest["running"],
            "camera_active": latest["camera_active"],
            "current_count": latest["current_count"],
            "highest_confidence": round(latest["highest_confidence"] * 100, 1),
            "total_detections": latest["total_detections"],
            "confidence": latest["confidence"],
            "history": latest["history"],
            "last_update": latest["last_update"],
            "error": latest["error"],
            "model_label": model_label,
        })


@app.route("/api/control", methods=["POST"])
def api_control():
    data = request.get_json(silent=True) or {}
    if not isinstance(data.get("running"), bool):
        return jsonify({"ok": False, "error": "Expected running to be true or false."}), 400
    with state_lock:
        latest["running"] = data["running"]
    return jsonify({"ok": True, "running": data["running"]})


@app.route("/api/confidence", methods=["POST"])
def api_confidence():
    data = request.get_json(silent=True) or {}
    try:
        value = float(data.get("confidence"))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Confidence must be a number between 0.10 and 0.90."}), 400
    if not 0.10 <= value <= 0.90:
        return jsonify({"ok": False, "error": "Confidence must be between 0.10 and 0.90."}), 400
    with state_lock:
        latest["confidence"] = value
    return jsonify({"ok": True, "confidence": value})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    with state_lock:
        latest["total_detections"] = 0
        latest["history"] = []
    return jsonify({"ok": True})


@app.route("/api/capture", methods=["POST"])
def api_capture():
    with state_lock:
        frame = None if latest["frame"] is None else latest["frame"].copy()
    if frame is None:
        return jsonify({"ok": False, "error": "No camera frame is available yet."}), 503

    capture_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")
    os.makedirs(capture_dir, exist_ok=True)
    filename = "garbage_detection_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
    filepath = os.path.join(capture_dir, filename)
    if not cv2.imwrite(filepath, frame):
        return jsonify({"ok": False, "error": "Could not save the screenshot."}), 500
    return jsonify({"ok": True, "filename": os.path.join("captures", filename)})


if __name__ == "__main__":
    # Keep the existing browser-launch behavior so your default Firefox setup is preserved.
    url = f"http://127.0.0.1:{PORT}"
    print("Open", url, "in your browser (Ctrl+C to stop)")
    threading.Thread(target=worker, daemon=True).start()
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=PORT, threaded=True, use_reloader=False)
