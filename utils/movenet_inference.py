"""
MediaPipe Pose for multi-person detection
Detects multiple people by running inference on person crops
"""

import cv2
import mediapipe as mp
import numpy as np

class MoveNetInference:
    """MediaPipe Pose detector (multi-person compatible)"""
    
    KEYPOINT_NAMES = [
        'nose', 'l_eye_inner', 'l_eye', 'l_eye_outer',
        'r_eye_inner', 'r_eye', 'r_eye_outer',
        'l_ear', 'r_ear', 'mouth_left', 'mouth_right',
        'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow',
        'l_wrist', 'r_wrist', 'l_pinky', 'r_pinky',
        'l_index', 'r_index', 'l_thumb', 'r_thumb',
        'l_hip', 'r_hip', 'l_knee', 'r_knee',
        'l_ankle', 'r_ankle', 'l_heel', 'r_heel',
        'l_foot_index', 'r_foot_index'
    ]
    
    def __init__(self):
        """Initialize MediaPipe Pose"""
        mp_pose = mp.solutions.pose
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("✓ MediaPipe Pose loaded successfully")
    
    def detect(self, frame):
        """
        Detect persons using grid-based approach for multi-person
        
        Args:
            frame: Input image (BGR)
        
        Returns:
            List of detected persons with keypoints
        """
        h, w = frame.shape[:2]
        persons = []
        
        # Try detecting full frame first
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Extract person from full frame
            person = self._extract_person(results, frame)
            if person:
                persons.append(person)
        
        # Also try detecting on left and right halves for multiple people
        # Left half
        left_half = frame[:, :w//2]
        results_left = self.pose.process(cv2.cvtColor(left_half, cv2.COLOR_BGR2RGB))
        if results_left.pose_landmarks:
            person = self._extract_person(results_left, left_half, x_offset=0)
            if person and not self._is_duplicate(person, persons):
                persons.append(person)
        
        # Right half
        right_half = frame[:, w//2:]
        results_right = self.pose.process(cv2.cvtColor(right_half, cv2.COLOR_BGR2RGB))
        if results_right.pose_landmarks:
            person = self._extract_person(results_right, right_half, x_offset=w//2)
            if person and not self._is_duplicate(person, persons):
                persons.append(person)
        
        return persons
    
    def _extract_person(self, results, frame, x_offset=0):
        """Extract person data from MediaPipe results"""
        if not results.pose_landmarks:
            return None
        
        h, w = frame.shape[:2]
        
        # Get keypoints
        keypoints = []
        x_coords, y_coords = [], []
        
        for lm in results.pose_landmarks.landmark:
            x = int(lm.x * w) + x_offset
            y = int(lm.y * h)
            conf = lm.visibility
            
            keypoints.append({
                'x': x,
                'y': y,
                'z': lm.z * w,
                'confidence': conf
            })
            
            if conf > 0.5:
                x_coords.append(x)
                y_coords.append(y)
        
        if not x_coords or not y_coords:
            return None
        
        # Bounding box
        x_min = max(0, min(x_coords))
        x_max = min(frame.shape[1] + x_offset, max(x_coords))
        y_min = max(0, min(y_coords))
        y_max = min(h, max(y_coords))
        
        cx = (x_min + x_max) // 2
        cy = (y_min + y_max) // 2
        
        avg_conf = np.mean([kp['confidence'] for kp in keypoints if kp['confidence'] > 0.3])
        
        return {
            'centroid': (cx, cy),
            'bbox': (x_min, y_min, x_max, y_max),
            'keypoints': keypoints,
            'confidence': float(avg_conf)
        }
    
    def _is_duplicate(self, person, persons, threshold=50):
        """Check if person is duplicate of existing detection"""
        for existing in persons:
            dist = np.linalg.norm(
                np.array(person['centroid']) - np.array(existing['centroid'])
            )
            if dist < threshold:
                return True
        return False