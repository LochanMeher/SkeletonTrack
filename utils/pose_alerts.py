"""
Pose-Based Alert Engine
Detects specific body poses and triggers alerts
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict
import time
import json

class AlertType(Enum):
    """Alert severity levels"""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

class PoseType(Enum):
    """Supported pose types"""
    HANDS_UP = "hands_up"
    SITTING = "sitting"
    LYING_DOWN = "lying_down"
    FALLING = "falling"
    PHONE_USE = "phone_use"

@dataclass
class Alert:
    """Single alert instance"""
    person_id: int
    pose_type: str
    alert_type: str
    confidence: float
    timestamp: float
    message: str

class PoseAlertEngine:
    """Detects poses and generates alerts"""
    
    def __init__(self, confidence_threshold=0.3, temporal_window=3):
        self.confidence_threshold = confidence_threshold
        self.temporal_window = temporal_window
        self.pose_history = {}
        self.active_alerts = {}
        self.last_alert_time = {}
        self.enabled_poses = {
            PoseType.HANDS_UP.value: True,
            PoseType.SITTING.value: True,
            PoseType.LYING_DOWN.value: True,
            PoseType.FALLING.value: True,
            PoseType.PHONE_USE.value: False,
        }
    
    def update(self, pose_results):
        """Update alerts based on current pose detections"""
        new_alerts = []
        current_time = time.time()
        
        for person_id in list(self.pose_history.keys()):
            if person_id not in pose_results:
                del self.pose_history[person_id]
                if person_id in self.active_alerts:
                    del self.active_alerts[person_id]
        
        for person_id_str, person_data in pose_results.items():
            person_id = int(person_id_str)
            keypoints = person_data['keypoints']
            
            if person_id not in self.pose_history:
                self.pose_history[person_id] = {}
                self.active_alerts[person_id] = []
            
            poses_detected = {}
            
            if self.enabled_poses[PoseType.HANDS_UP.value]:
                hands_up, conf = self._detect_hands_up(keypoints)
                if hands_up:
                    poses_detected[PoseType.HANDS_UP.value] = conf
            
            if self.enabled_poses[PoseType.SITTING.value]:
                sitting, conf = self._detect_sitting(keypoints)
                if sitting:
                    poses_detected[PoseType.SITTING.value] = conf
            
            if self.enabled_poses[PoseType.LYING_DOWN.value]:
                lying, conf = self._detect_lying_down(keypoints)
                if lying:
                    poses_detected[PoseType.LYING_DOWN.value] = conf
            
            if self.enabled_poses[PoseType.FALLING.value]:
                falling, conf = self._detect_falling(keypoints, person_id)
                if falling:
                    poses_detected[PoseType.FALLING.value] = conf
            
            for pose_name in list(self.pose_history[person_id].keys()):
                if pose_name in poses_detected:
                    self.pose_history[person_id][pose_name] += 1
                else:
                    self.pose_history[person_id][pose_name] = 0
            
            for pose_name in poses_detected:
                if pose_name not in self.pose_history[person_id]:
                    self.pose_history[person_id][pose_name] = 1
                else:
                    self.pose_history[person_id][pose_name] += 1
            
            for pose_name, frame_count in list(self.pose_history[person_id].items()):
                if frame_count >= self.temporal_window:
                    confidence = poses_detected.get(pose_name, 0)
                    alert_key = (person_id, pose_name)
                    if alert_key not in self.last_alert_time or (current_time - self.last_alert_time[alert_key]) > 1.0:
                        alert = Alert(
                            person_id=person_id,
                            pose_type=pose_name,
                            alert_type=self._get_alert_type(pose_name),
                            confidence=confidence,
                            timestamp=current_time,
                            message=self._get_alert_message(pose_name, person_id)
                        )
                        new_alerts.append(alert)
                        self.last_alert_time[alert_key] = current_time
                        self.active_alerts[person_id] = [a for a in self.active_alerts[person_id] if a.pose_type != pose_name]
                        self.active_alerts[person_id].append(alert)
        
        return new_alerts
    
    def _detect_hands_up(self, keypoints):
        """Detect: Both hands above shoulders"""
        try:
            if len(keypoints) < 17:
                return False, 0
            
            l_wrist = keypoints[15]
            r_wrist = keypoints[16]
            l_shoulder = keypoints[11]
            r_shoulder = keypoints[12]
            
            print(f"DEBUG hands_up: l_wrist_conf={l_wrist['confidence']:.2f}")
            print(f"DEBUG hands_up: r_wrist_conf={r_wrist['confidence']:.2f}")
            
            if any(kp['confidence'] < self.confidence_threshold for kp in [l_wrist, r_wrist, l_shoulder, r_shoulder]):
                print(f"DEBUG: Confidence too low, threshold={self.confidence_threshold}")
                return False, 0
            
            hands_up = (l_wrist['y'] < l_shoulder['y'] and r_wrist['y'] < r_shoulder['y'])
            print(f"DEBUG: hands_up={hands_up}")
            conf = (l_wrist['confidence'] + r_wrist['confidence']) / 2
            return hands_up, conf
        except Exception as e:
            print(f"DEBUG hands_up error: {e}")
            return False, 0
    
    def _detect_sitting(self, keypoints):
        """Detect: Sitting"""
        try:
            if len(keypoints) < 27:
                return False, 0
            
            l_hip = keypoints[23]
            r_hip = keypoints[24]
            l_knee = keypoints[25]
            r_knee = keypoints[26]
            
            if any(kp['confidence'] < self.confidence_threshold for kp in [l_hip, r_hip, l_knee, r_knee]):
                return False, 0
            
            hip_y = (l_hip['y'] + r_hip['y']) / 2
            knee_y = (l_knee['y'] + r_knee['y']) / 2
            sitting = knee_y > hip_y
            conf = (l_hip['confidence'] + r_hip['confidence']) / 2
            return sitting, conf
        except Exception as e:
            return False, 0
    
    def _detect_lying_down(self, keypoints):
        """Detect: Lying down"""
        try:
            if len(keypoints) < 17:
                return False, 0
            
            valid_kps = [kp for kp in keypoints if kp['confidence'] > self.confidence_threshold]
            if len(valid_kps) < 5:
                return False, 0
            
            x_coords = [kp['x'] for kp in valid_kps]
            y_coords = [kp['y'] for kp in valid_kps]
            
            x_range = max(x_coords) - min(x_coords)
            y_range = max(y_coords) - min(y_coords)
            
            lying = x_range > y_range * 1.5
            avg_conf = sum(kp['confidence'] for kp in valid_kps) / len(valid_kps)
            return lying, avg_conf
        except Exception as e:
            return False, 0
    
    def _detect_falling(self, keypoints, person_id):
        """Detect: Falling risk"""
        try:
            lying, conf = self._detect_lying_down(keypoints)
            return lying, conf
        except Exception as e:
            return False, 0
    
    def _get_alert_type(self, pose_name):
        """Get alert severity level"""
        if pose_name in [PoseType.FALLING.value]:
            return AlertType.CRITICAL.value
        elif pose_name in [PoseType.LYING_DOWN.value]:
            return AlertType.WARNING.value
        else:
            return AlertType.INFO.value
    
    def _get_alert_message(self, pose_name, person_id):
        """Generate human-readable alert message"""
        messages = {
            PoseType.HANDS_UP.value: f"Person {person_id}: Hands raised",
            PoseType.SITTING.value: f"Person {person_id}: Sitting detected",
            PoseType.LYING_DOWN.value: f"Person {person_id}: Lying down",
            PoseType.FALLING.value: f"Person {person_id}: FALLING DETECTED",
            PoseType.PHONE_USE.value: f"Person {person_id}: Using phone",
        }
        return messages.get(pose_name, f"Person {person_id}: {pose_name}")
    
    def get_active_alerts(self):
        """Get all active alerts"""
        all_alerts = []
        for alerts in self.active_alerts.values():
            all_alerts.extend(alerts)
        return all_alerts
    
    def get_person_alerts(self, person_id):
        """Get alerts for specific person"""
        return self.active_alerts.get(person_id, [])
    
    def set_pose_enabled(self, pose_name, enabled):
        """Enable/disable specific pose detection"""
        if pose_name in self.enabled_poses:
            self.enabled_poses[pose_name] = enabled
    
    def get_config(self):
        """Get current configuration"""
        return {
            'confidence_threshold': self.confidence_threshold,
            'temporal_window': self.temporal_window,
            'enabled_poses': self.enabled_poses
        }