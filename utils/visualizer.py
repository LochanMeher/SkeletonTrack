"""
Pose visualization with MediaPipe skeleton
"""

import cv2
import numpy as np

class PoseVisualizer:
    """Draw pose skeletons on frames"""
    
    # MoveNet 17 keypoints (COCO format)
    KEYPOINT_NAMES = [
        'nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear',
        'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow',
        'l_wrist', 'r_wrist', 'l_hip', 'r_hip',
        'l_knee', 'r_knee', 'l_ankle', 'r_ankle'
    ]
    
    # Skeleton connections (COCO format - for 17 keypoints)
    SKELETON = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (0, 5), (0, 6),  # Shoulders
        (5, 7), (7, 9),  # Left arm
        (6, 8), (8, 10),  # Right arm
        (5, 6), (5, 11), (6, 12),  # Torso
        (11, 12),  # Hip line
        (11, 13), (13, 15),  # Left leg
        (12, 14), (14, 16)  # Right leg
    ]
    
    # Colors for different persons
    COLORS = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (128, 255, 0),  # Spring green
    ]
    
    def __init__(self):
        pass
    
    def draw_poses(self, frame, pose_results):
        """
        Draw all poses on frame
        
        Args:
            frame: Input image
            pose_results: {person_id: {keypoints: [...], bbox: (...), ...}}
        
        Returns:
            Annotated frame
        """
        frame_out = frame.copy()
        
        for person_id_str, person_data in pose_results.items():
            person_id = int(person_id_str)
            color = self.COLORS[person_id % len(self.COLORS)]
            
            keypoints = person_data['keypoints']
            bbox = person_data['bbox']
            
            # Draw bounding box
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(frame_out, (x_min, y_min), (x_max, y_max), color, 2)
            
            # Draw person ID
            cv2.putText(frame_out, f"ID: {person_id}", 
                       (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, color, 2)
            
            # Draw skeleton
            for start_idx, end_idx in self.SKELETON:
                if start_idx < len(keypoints) and end_idx < len(keypoints):
                    start_kp = keypoints[start_idx]
                    end_kp = keypoints[end_idx]
                    
                    if start_kp['confidence'] > 0.5 and end_kp['confidence'] > 0.5:
                        start_pos = (int(start_kp['x']), int(start_kp['y']))
                        end_pos = (int(end_kp['x']), int(end_kp['y']))
                        cv2.line(frame_out, start_pos, end_pos, color, 2)
            
            # Draw keypoints (circles)
            for kp_idx, kp in enumerate(keypoints):
                if kp['confidence'] > 0.5:
                    pos = (int(kp['x']), int(kp['y']))
                    cv2.circle(frame_out, pos, 4, color, -1)
                    cv2.circle(frame_out, pos, 4, (255, 255, 255), 1)
        
        return frame_out