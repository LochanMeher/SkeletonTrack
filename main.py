"""
Real-time Pose Detection from Android Phone Camera
- Streams from IP Webcam app on Android
- MediaPipe BlazePose inference on laptop
- Centroid-based MOT tracking
- Kalman filtering for skeleton smoothing
- REST API for pose queries
- Real-time visualization
"""

import cv2
import numpy as np
import time
from collections import defaultdict
from threading import Thread
from flask import Flask, jsonify, request
from flask_cors import CORS
import json
from utils.streaming import AndroidCameraCapture
from utils.tracker import CentroidTracker, KalmanSmoother
from utils.visualizer import PoseVisualizer
from utils.movenet_inference import MoveNetInference

# Import custom modules
from utils.streaming import AndroidCameraCapture
from utils.tracker import CentroidTracker, KalmanSmoother
from utils.visualizer import PoseVisualizer

# ==================== CONFIG ====================
ANDROID_CAMERA_URL = "http://192.168.1.6:8080/video"  # CHANGE TO YOUR PHONE IP
TARGET_FPS = 30
CONFIDENCE_THRESHOLD = 0.5
MAX_PERSON_TTL = 10  # Frames until person is considered "lost"

# ==================== GLOBALS ====================
frame_buffer = None
pose_results = {}
persons_state = {}
lock = __import__('threading').Lock()

# ==================== MOVENET SETUP (MULTI-PERSON) ====================
from utils.movenet_inference import MoveNetInference

pose_detector = MoveNetInference()

# ==================== ALERT ENGINE SETUP ====================
from utils.pose_alerts import PoseAlertEngine

alert_engine = PoseAlertEngine(confidence_threshold=0.3, temporal_window=3)
alert_history = []  # Store all alerts for logging

# ==================== TRACKING SETUP ====================
tracker = CentroidTracker(maxDisappeared=MAX_PERSON_TTL)
kalman_smoother = KalmanSmoother(num_people=5, num_keypoints=33)

# ==================== FLASK API SETUP ====================
app = Flask(__name__)
CORS(app)

@app.route('/api/poses', methods=['GET'])
def get_poses():
    """REST API: Get latest poses for all tracked persons"""
    with lock:
        return jsonify({
            'timestamp': time.time(),
            'persons': pose_results
        })

@app.route('/api/person/<int:person_id>', methods=['GET'])
def get_person(person_id):
    """REST API: Get pose for specific person ID"""
    with lock:
        if str(person_id) in pose_results:
            return jsonify({
                'person_id': person_id,
                'pose': pose_results[str(person_id)]
            })
        return jsonify({'error': 'Person not found'}), 404

@app.route('/api/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({'status': 'running'})


@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Get all active alerts"""
    with lock:
        alerts = alert_engine.get_active_alerts()
        return jsonify({
            'timestamp': time.time(),
            'active_alerts': len(alerts),
            'alerts': [
                {
                    'person_id': alert.person_id,
                    'pose_type': alert.pose_type,
                    'alert_type': alert.alert_type,
                    'confidence': alert.confidence,
                    'timestamp': alert.timestamp,
                    'message': alert.message
                }
                for alert in alerts
            ]
        })

@app.route('/api/alerts/<int:person_id>', methods=['GET'])
def get_person_alerts(person_id):
    """Get alerts for specific person"""
    with lock:
        alerts = alert_engine.get_person_alerts(person_id)
        return jsonify({
            'person_id': person_id,
            'alerts': [
                {
                    'pose_type': alert.pose_type,
                    'alert_type': alert.alert_type,
                    'confidence': alert.confidence,
                    'timestamp': alert.timestamp,
                    'message': alert.message
                }
                for alert in alerts
            ]
        })

@app.route('/api/alerts/config', methods=['GET', 'POST'])
def alert_config():
    """Get/set alert configuration"""
    if request.method == 'GET':
        return jsonify(alert_engine.get_config())
    
    elif request.method == 'POST':
        data = request.get_json()
        if 'confidence_threshold' in data:
            alert_engine.confidence_threshold = data['confidence_threshold']
        if 'temporal_window' in data:
            alert_engine.temporal_window = data['temporal_window']
        if 'enabled_poses' in data:
            for pose, enabled in data['enabled_poses'].items():
                alert_engine.set_pose_enabled(pose, enabled)
        return jsonify({'status': 'updated'})

@app.route('/api/alerts/history', methods=['GET'])
def get_alert_history():
    """Get alert history (last 100)"""
    with lock:
        return jsonify({
            'total': len(alert_history),
            'recent': [
                {
                    'person_id': alert.person_id,
                    'pose_type': alert.pose_type,
                    'alert_type': alert.alert_type,
                    'timestamp': alert.timestamp,
                    'message': alert.message
                }
                for alert in alert_history[-100:]
            ]
        })
# ==================== MAIN INFERENCE LOOP ====================
def inference_loop():
    """Main pose detection and tracking loop"""
    global frame_buffer, pose_results, persons_state
    
    # Open Android camera stream
    cap = AndroidCameraCapture(ANDROID_CAMERA_URL)
    if not cap.is_open():
        print("❌ Failed to connect to Android camera. Check IP address.")
        return
    
    print("✓ Connected to Android camera")
    
    # Visualization
    visualizer = PoseVisualizer()
    
    frame_count = 0
    fps_clock = time.time()
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame. Reconnecting...")
            cap.reconnect()
            continue
        
        frame_count += 1
        
        # Frame resizing for performance (4GB RAM constraint)
        frame_resized = cv2.resize(frame, (640, 480))
        
        # ========== POSE INFERENCE (MOVENET MULTI-PERSON) ==========
        start_inference = time.time()
        detections = pose_detector.detect(frame_resized)
        inference_time = (time.time() - start_inference) * 1000  # ms
        
        # ========== MULTI-PERSON TRACKING ==========
        objects = tracker.update([d['centroid'] for d in detections])
        
        # ========== POSE-BASED ALERTS ==========
        new_alerts = alert_engine.update(pose_results)
        
        if new_alerts:
            print(f"DEBUG: New alerts created: {len(new_alerts)}")
            for alert in new_alerts:
                alert_history.append(alert)
                print(f"🚨 {alert.message} (Confidence: {alert.confidence:.2f})")
        
        # Debug: Show pose history
        if pose_results:
            for person_id, poses in alert_engine.pose_history.items():
                print(f"DEBUG: Person {person_id} pose_history = {poses}")
        
        # Debug: Show active alerts in engine
        active = alert_engine.get_active_alerts()
        if active:
            print(f"DEBUG: Active alerts in engine: {len(active)}")
            
        # ========== ASSOCIATE DETECTIONS TO TRACKED IDS ==========
        with lock:
            pose_results = {}
            
            for person_id, centroid in objects.items():
                person_id_str = str(person_id)
                
                # Find matching detection
                matched_detection = None
                min_dist = float('inf')
                
                for det in detections:
                    dist = np.linalg.norm(np.array(det['centroid']) - np.array(centroid))
                    if dist < min_dist:
                        min_dist = dist
                        matched_detection = det
                
                if matched_detection and min_dist < 100:  # Distance threshold
                    # Apply Kalman smoothing
                    smooth_keypoints = kalman_smoother.smooth(person_id, matched_detection['keypoints'])
                    
                    pose_results[person_id_str] = {
                        'person_id': person_id,
                        'bbox': list(matched_detection['bbox']),
                        'centroid': list(centroid),
                        'keypoints': smooth_keypoints,
                        'confidence': float(matched_detection['confidence']),
                        'timestamp': time.time()
                    }
        
        # ========== VISUALIZATION ==========
        frame_viz = visualizer.draw_poses(frame_resized, pose_results)
        
         # FPS counter
        if frame_count % 30 == 0:
            elapsed = time.time() - fps_clock
            fps = 30 / elapsed
            fps_clock = time.time()
            print(f"FPS: {fps:.1f} | Inference: {inference_time:.1f}ms | People tracked: {len(pose_results)}")
        
        # Display
        cv2.putText(frame_viz, f"FPS: {fps:.1f} | Latency: {inference_time:.0f}ms", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Pose Detection', frame_viz)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    pose_detector.close()

# ==================== FLASK API THREAD ====================
def run_api():
    """Run Flask API in background thread"""
    print("✓ Starting REST API on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

# ==================== MAIN ====================
if __name__ == '__main__':
    print("=" * 60)
    print("POSE DETECTION: Android Camera + MediaPipe BlazePose")
    print("=" * 60)
    print(f"\n📱 Android Camera URL: {ANDROID_CAMERA_URL}")
    print("   (Change this to your phone's IP in main.py)")
    print(f"🎯 Target: {TARGET_FPS} FPS, {CONFIDENCE_THRESHOLD*100:.0f}% confidence threshold")
    print(f"👥 Multi-person tracking: 0-5 people, persistent IDs")
    print(f"\n📊 REST API: http://localhost:5000/api/poses")
    print("🔍 Press 'q' to exit\n")
    
    # Start API in background thread
    api_thread = Thread(target=run_api, daemon=True)
    api_thread.start()
    
    # Start main inference loop
    time.sleep(2)  # Wait for API to start
    inference_loop()