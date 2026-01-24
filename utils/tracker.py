"""
Multi-Object Tracking (MOT) with Kalman Smoothing
- Centroid-based association
- Persistent person IDs
- Kalman filter for skeleton smoothing
"""

import numpy as np
from scipy.spatial.distance import euclidean
from collections import defaultdict
import uuid

class CentroidTracker:
    """
    Centroid-based multi-object tracker
    - Assigns unique IDs to persons
    - Tracks centroids across frames
    - Handles occlusion/disappearance
    """
    
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = {}  # {id: centroid}
        self.disappeared = {}  # {id: frames_disappeared}
        self.maxDisappeared = maxDisappeared
    
    def register(self, centroid):
        """Register new person"""
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1
    
    def deregister(self, objectID):
        """Remove person after too long missing"""
        del self.objects[objectID]
        del self.disappeared[objectID]
    
    def update(self, rects):
        """
        Update tracker with new detections
        
        Args:
            rects: List of (cx, cy) centroids
        
        Returns:
            objects: {person_id: (cx, cy)}
        """
        
        if len(rects) == 0:
            # No detections: increment disappearance counter
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects
        
        # Match detections to existing objects
        if len(self.objects) == 0:
            # First frame: register all
            for rect in rects:
                self.register(rect)
        else:
            # Associate centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            
            # Compute distances
            D = np.zeros((len(objectCentroids), len(rects)))
            for i, objCentroid in enumerate(objectCentroids):
                for j, rect in enumerate(rects):
                    D[i, j] = euclidean(objCentroid, rect)
            
            # Hungarian algorithm (greedy for simplicity)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)
            
            usedRows = set()
            usedCols = set()
            
            for row in rows:
                if row in usedRows:
                    continue
                if cols[row] in usedCols:
                    continue
                if D[row, cols[row]] > 50:  # Distance threshold
                    continue
                
                objectID = objectIDs[row]
                self.objects[objectID] = rects[cols[row]]
                self.disappeared[objectID] = 0
                
                usedRows.add(row)
                usedCols.add(cols[row])
            
            # Register new objects
            unusedRows = set(range(D.shape[0])) - usedRows
            unusedCols = set(range(D.shape[1])) - usedCols
            
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(rects[col])
        
        return self.objects


class KalmanSmoother:
    """
    Kalman filter for skeleton keypoint smoothing
    Reduces jitter in pose landmarks across frames
    """
    
    def __init__(self, num_people=5, num_keypoints=33, 
                 process_variance=0.01, measurement_variance=4.0):
        self.num_people = num_people
        self.num_keypoints = num_keypoints
        self.process_var = process_variance
        self.measurement_var = measurement_variance
        self.states = {}  # {person_id: {kp_idx: KalmanState}}
    
    def smooth(self, person_id, keypoints):
        """
        Smooth keypoints for a person
        
        Args:
            person_id: Unique person identifier
            keypoints: List of dicts with 'x', 'y', 'z', 'confidence'
        
        Returns:
            Smoothed keypoints (same structure)
        """
        
        person_id = str(person_id)
        
        if person_id not in self.states:
            self.states[person_id] = {}
        
        smoothed = []
        
        for kp_idx, kp in enumerate(keypoints):
            if kp_idx not in self.states[person_id]:
                # Initialize new keypoint filter
                self.states[person_id][kp_idx] = {
                    'x': {'value': kp['x'], 'error': self.measurement_var},
                    'y': {'value': kp['y'], 'error': self.measurement_var},
                    'z': {'value': kp['z'], 'error': self.measurement_var}
                }
                smoothed_kp = kp.copy()
            else:
                state = self.states[person_id][kp_idx]
                smoothed_kp = kp.copy()
                
                # Apply Kalman filter to each coordinate
                for coord in ['x', 'y', 'z']:
                    measurement = kp[coord]
                    s = state[coord]
                    
                    # Predict
                    pred_value = s['value']
                    pred_error = s['error'] + self.process_var
                    
                    # Update
                    kalman_gain = pred_error / (pred_error + self.measurement_var)
                    s['value'] = pred_value + kalman_gain * (measurement - pred_value)
                    s['error'] = (1 - kalman_gain) * pred_error
                    
                    smoothed_kp[coord] = s['value']
            
            smoothed.append(smoothed_kp)
        
        return smoothed