# ♻️ Smart Garbage Detection System

A real-time AI-powered garbage detection system developed using **Python, YOLO, OpenCV, and Flask**. The system uses a camera to identify garbage, highlights detected objects with red bounding boxes, displays results on a web dashboard, and sends automated Gmail notifications with detection screenshots.

---

## 📌 Project Overview

The Smart Garbage Detection System aims to automate garbage identification through computer vision and deep learning.

The application captures live video from a camera, processes frames using a trained YOLO object detection model, and identifies garbage in real time.

When garbage is detected:
- A red bounding box highlights the detected object.
- The detection count is updated.
- The live detection result is displayed on a Flask web dashboard.
- A screenshot of the detection is sent to a configured Gmail account.

The system provides a simple interface for monitoring garbage detection and controlling the detection process.

---

## ✨ Key Features

- **Real-Time Detection:** Detect garbage using a live camera feed.
- **YOLO Integration:** Uses a trained YOLO model for object detection.
- **Bounding Box Visualization:** Highlights detected garbage using red bounding boxes.
- **Live Web Dashboard:** Displays the camera feed and detection information.
- **Detection Counter:** Shows the number of detected objects in the current frame.
- **Start/Pause Controls:** Allows users to control the detection process.
- **Gmail Notifications:** Sends email alerts when a new garbage detection event occurs.
- **Screenshot Capture:** Attaches the annotated detection image to the notification email.
- **Automatic Browser Launch:** Opens the dashboard automatically when the application starts.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core application development |
| YOLO (Ultralytics) | Garbage object detection |
| OpenCV | Camera access and image processing |
| Flask | Backend web server |
| HTML | Dashboard structure |
| CSS | Dashboard styling |
| JavaScript | Live status updates and dashboard controls |
| SMTP | Email notification delivery |
| Threading | Background camera processing and email sending |

---

## 🏗️ System Architecture

```text
             CAMERA / WEBCAM
                    |
                    v
             OPENCV CAPTURE
                    |
                    v
              YOLO MODEL
                    |
                    v
           GARBAGE DETECTION
                    |
                    v
          BOUNDING BOX DRAWING
                    |
             +------+------+
             |             |
             v             v
        FLASK SERVER   SCREENSHOT
             |             |
             v             v
       WEB DASHBOARD   GMAIL ALERT
             |
             v
       LIVE MONITORING
```

---

## 📂 Project Structure

```text
smart_garbage_detection/
│
├── smart_garbage_detection_simple_email.py
│
├── runs/
│   └── detect/
│       ├── final/
│       │   └── weights/
│       │       └── best.pt
│       │
│       └── test_run/
│           └── weights/
│               └── best.pt
│
├── venv/
│
└── README.md
```

**Note:** The YOLO model path depends on the model available in your project. The `venv` directory is your local Python virtual environment and should not be uploaded to GitHub.

---

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate into the project:

```bash
cd smart_garbage_detection
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install ultralytics opencv-python Flask
```

The application also uses Python standard libraries for threading, email delivery, and browser integration.

### 4. Configure the YOLO Model

Place your trained YOLO model (`best.pt`) in the expected model directory.

Make sure the model path in the Python script matches your actual model location.

---

## 📧 Gmail Notification Configuration

The system uses Gmail SMTP to send detection screenshots.

### Step 1: Generate an App Password

1. Open your Google Account security settings.
2. Enable 2-Step Verification.
3. Generate a Gmail App Password.
4. Copy the generated password.

### Step 2: Configure Environment Variables

Run the following commands in your terminal:

```bash
export GMAIL_SENDER="your_email@gmail.com"

export GMAIL_RECEIVER="receiver_email@gmail.com"

export GMAIL_APP_PASSWORD="your_gmail_app_password"
```

**Security Warning:** Never upload your Gmail credentials or App Password to GitHub.

Use environment variables instead of hardcoding credentials in your Python source code.

---

## ▶️ Running the Application

After completing the installation and configuration:

```bash
python smart_garbage_detection_simple_email.py
```

The Flask application starts locally and automatically opens the dashboard in your default browser.

Open manually if required:

```text
http://127.0.0.1:5000
```

---

## 🖥️ Dashboard Functionality

The web dashboard provides:

| Component | Function |
|---|---|
| Live Camera Feed | Displays real-time camera footage |
| Detection Status | Shows whether garbage is detected |
| Object Counter | Displays the current detection count |
| Camera Status | Indicates camera availability |
| Email Status | Displays email notification status |
| Start Detection | Starts the detection process |
| Pause Detection | Pauses the detection process |

---

## 📩 Email Alert Workflow

When the system identifies a new garbage detection event:

1. YOLO detects garbage in the camera frame.
2. OpenCV draws a red bounding box around the detected object.
3. The system captures the annotated frame.
4. The screenshot is attached to an email.
5. Gmail SMTP sends the notification to the configured receiver.

The application includes a cooldown mechanism to reduce repeated email notifications.

---

## 🔒 Security Considerations

- Keep Gmail App Passwords private.
- Do not commit `.env` files or credentials.
- Add virtual environments to `.gitignore`.
- Keep trained model files separate if they are too large for normal GitHub uploads.
- Use environment variables for sensitive configuration.

Suggested `.gitignore`:

```gitignore
venv/
__pycache__/
*.pyc
.env
.DS_Store
```

---

## 🚀 Future Improvements

Potential improvements include:

- Multiple camera support.
- Garbage classification into different categories.
- Detection history and analytics.
- Database integration.
- Detection location tracking.
- Cloud-based monitoring.
- Real-time notification through mobile applications.
- Smart waste-bin monitoring.
- Integration with IoT devices and robotics systems.

---

## 🎯 Project Objective

The main objective of this project is to demonstrate how artificial intelligence and computer vision can be used to automate garbage detection and improve waste-monitoring systems.

By combining YOLO, OpenCV, Flask, and email automation, the system provides a foundation for developing intelligent waste-management applications.

---

## 👨‍💻 Developer

**Project:** Smart Garbage Detection Systemminutes

**Domain:** Artificial Intelligence | Computer Vision | Deep Learning | Web Development

**Programming Language:** Python

<img width="1920" height="1080" alt="Screenshot from 2026-11-17 00-22-34" src="https://github.com/user-attachments/assets/7fccefe5-b135-4269-9de2-77ae10839e3a" />
<img width="1920" height="1080" alt="Screenshot from 2026-11-17 00-23-06" src="https://github.com/user-attachments/assets/0265139a-b368-4dfc-9994-7c70e24b2668" />
<img width="1920" height="1080" alt="Screenshot from 2026-11-17 00-23-31" src="https://github.com/user-attachments/assets/9752e326-4f40-4f31-9be0-379daf0a42e1" />


