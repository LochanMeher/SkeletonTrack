# SkeletonTrack — Real-Time Multi-Person Pose Detection & Tracking

**SkeletonTrack** is a computer vision-based system designed to detect, track, and visualize multiple people in real time using an Android smartphone camera as the video source.

The system combines deep learning-based human pose estimation, centroid-based multi-object tracking, Kalman filtering, and a Flask REST API to provide persistent person identification and live skeleton visualization through an interactive web dashboard.

## Key Features

- **Real-Time Pose Estimation:** Detects human body keypoints from a live camera stream using MoveNet inference.
- **Multi-Person Tracking:** Tracks multiple individuals and assigns persistent IDs using centroid-based tracking.
- **Skeleton Smoothing:** Applies Kalman filtering to reduce keypoint jitter and improve tracking stability.
- **Android Camera Integration:** Uses an Android smartphone as a wireless IP camera through the IP Webcam application.
- **Live Visualization:** Displays detected skeletons, body keypoints, confidence scores, and person IDs.
- **Web Dashboard:** Provides a browser-based interface to monitor tracked individuals and system performance.
- **REST API:** Exposes real-time pose information, individual tracking data, and alert information.
- **Pose-Based Alerts:** Detects configured pose-related events, including hands-up, sitting, lying down, and falling.
- **Performance Monitoring:** Displays FPS, inference latency, API response time, and tracking statistics.

## System Architecture

The application follows a modular computer vision pipeline:

1. **Video Acquisition:** Captures live video from an Android smartphone using an IP Webcam stream.
2. **Frame Processing:** Resizes incoming frames to 640 × 480 for efficient processing.
3. **Pose Estimation:** Uses MoveNet inference to identify human body keypoints.
4. **Multi-Object Tracking:** Associates detected individuals with tracked objects using centroid-based tracking.
5. **Keypoint Smoothing:** Applies Kalman filtering to stabilize detected skeleton coordinates.
6. **Alert Processing:** Evaluates tracked pose information using a configurable pose alert engine.
7. **Visualization:** Renders skeletons and tracking information using OpenCV.
8. **API Integration:** Publishes tracking and alert information through Flask REST endpoints.
9. **Dashboard:** Retrieves API data and displays real-time tracking information in a web browser.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Computer Vision | OpenCV |
| Deep Learning | TensorFlow, MoveNet |
| Numerical Computing | NumPy, SciPy |
| Tracking | Centroid-Based Multi-Object Tracking |
| Filtering | Kalman Filter |
| Backend | Flask, Flask-CORS |
| Frontend | HTML5, CSS3, JavaScript |
| Visualization | OpenCV, HTML Canvas |
| Camera Integration | Android IP Webcam |
| Version Control | Git, GitHub |

## Project Structure

```text
SkeletonTrack/
│
├── main.py
│
├── dashboard.html
│
├── requirements.txt
│
├── run.ps1
│
├── utils/
│   ├── streaming.py
│   ├── tracker.py
│   ├── visualizer.py
│   ├── movenet_inference.py
│   └── pose_alerts.py
│
├── .gitignore
│
└── README.md
```

### Core Modules

| File | Responsibility |
|---|---|
| `main.py` | Main application, inference pipeline, tracking integration, and Flask API |
| `streaming.py` | Android camera stream acquisition and reconnection |
| `tracker.py` | Centroid tracking and Kalman-based keypoint smoothing |
| `visualizer.py` | Skeleton and pose visualization |
| `movenet_inference.py` | Human pose estimation |
| `pose_alerts.py` | Pose-based alert generation and configuration |
| `dashboard.html` | Interactive browser-based monitoring dashboard |

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/LochanMeher/SkeletonTrack.git

cd SkeletonTrack
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows:**

```powershell
.\venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip

pip install -r requirements.txt
```

### Dependencies

The project includes the following Python packages:

- NumPy
- OpenCV
- TensorFlow
- Flask
- Flask-CORS
- SciPy
- Scikit-learn
- MediaPipe

## Camera Configuration

SkeletonTrack uses an Android smartphone as a wireless camera.

1. Install the **IP Webcam** application on your Android smartphone.
2. Connect your smartphone and computer to the same Wi-Fi network.
3. Start the IP Webcam server.
4. Find the IP address and video streaming port displayed by the application.
5. Open `main.py`.
6. Update the camera URL:

```python
ANDROID_CAMERA_URL = "http://YOUR_PHONE_IP:8080/video"
```

Replace `YOUR_PHONE_IP` with your smartphone's actual local IP address.

Make sure the video stream is accessible from your computer before starting the application.

## Running the Application

### Option 1: Direct Execution

Activate your virtual environment and execute:

```bash
python main.py
```

### Option 2: PowerShell Script

On Windows, you can use the included startup script:

```powershell
.\run.ps1
```

The script activates the virtual environment and starts the application.

### Access the Web Dashboard

After starting the application, open `dashboard.html` in your web browser.

The Flask API runs on:

```text
http://localhost:5000
```

The dashboard connects to the API and displays the available tracking information.

**Note:** Keep the Python application running while accessing the dashboard.

## REST API Documentation

SkeletonTrack provides REST endpoints for retrieving pose information and managing alerts.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Check API status |
| GET | `/api/poses` | Retrieve all tracked persons and pose data |
| GET | `/api/person/{person_id}` | Retrieve a specific person's pose |
| GET | `/api/alerts` | Retrieve active pose alerts |
| GET | `/api/alerts/{person_id}` | Retrieve alerts for an individual |
| GET | `/api/alerts/config` | Retrieve alert configuration |
| POST | `/api/alerts/config` | Update alert configuration |
| GET | `/api/alerts/history` | Retrieve recent alert history |

### Example API Requests

Check application status:

```bash
curl http://localhost:5000/api/health
```

Retrieve all tracked persons:

```bash
curl http://localhost:5000/api/poses
```

Retrieve active alerts:

```bash
curl http://localhost:5000/api/alerts
```

Retrieve a specific person's tracking information:

```bash
curl http://localhost:5000/api/person/1
```

## Performance & Configuration

The main application includes configurable parameters:

| Parameter | Default Value | Purpose |
|---|---|---|
| Target FPS | 30 | Target processing frame rate |
| Frame Resolution | 640 × 480 | Processing resolution |
| Confidence Threshold | 0.5 | Detection confidence setting |
| Maximum Person TTL | 10 frames | Tracking persistence threshold |
| Maximum Kalman Tracks | 5 | Configured person slots |
| Keypoints | 33 | Configured skeleton keypoints |

Actual FPS and inference latency depend on the hardware, camera stream, and processing conditions.

## Applications & Use Cases

SkeletonTrack can be extended for applications involving:

- Human activity recognition
- Multi-person movement analysis
- Smart surveillance systems
- Human-computer interaction
- Fitness and exercise monitoring
- Gesture recognition
- Pose-based event detection
- Real-time computer vision applications

## Future Improvements

- Improve tracking accuracy in crowded environments.
- Integrate advanced multi-object tracking algorithms such as DeepSORT or ByteTrack.
- Optimize inference performance for edge devices.
- Add historical movement analytics.
- Introduce configurable alert notifications.
- Implement recording and playback of tracking sessions.
- Add support for multiple camera streams.
- Improve dashboard analytics and visualization.

## Contributing

Contributions, suggestions, and improvements are welcome.

1. Fork this repository.
2. Create a feature branch.
3. Implement your changes.
4. Commit your changes.
5. Submit a pull request.

## Author

**Lochan Meher**

Computer Science & Engineering | Artificial Intelligence & Machine Learning

- GitHub: [@LochanMeher](https://github.com/LochanMeher)
- Repository: [SkeletonTrack](https://github.com/LochanMeher/SkeletonTrack)

---

**SkeletonTrack — Detect. Track. Visualize.**

*Building intelligent real-time human pose tracking systems using computer vision and deep learning.*
