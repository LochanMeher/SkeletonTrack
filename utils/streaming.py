"""
Android camera streaming via IP Webcam app
"""

import cv2
import time
from threading import Thread

class AndroidCameraCapture:
    """Wrapper for Android IP Webcam streaming"""
    
    def __init__(self, url, buffer_size=1):
        self.url = url
        self.cap = None
        self.frame = None
        self.grabbed = False
        self.buffer_size = buffer_size
        self.is_running = False
        self._connect()
    
    def _connect(self):
        """Connect to Android camera"""
        try:
            self.cap = cv2.VideoCapture(self.url)
            # Set buffer size to reduce latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
            self.is_running = True
            
            # Start reading thread
            self.read_thread = Thread(target=self._read_frames, daemon=True)
            self.read_thread.start()
            
            time.sleep(1)  # Wait for first frame
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            self.is_running = False
            return False
    
    def _read_frames(self):
        """Background thread to continuously read frames"""
        while self.is_running:
            self.grabbed, self.frame = self.cap.read()
            if not self.grabbed:
                print("⚠️ Lost connection, attempting to reconnect...")
                self.reconnect()
    
    def read(self):
        """Get next frame"""
        return self.grabbed, self.frame
    
    def reconnect(self):
        """Reconnect on disconnect"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        time.sleep(2)
        self._connect()
    
    def is_open(self):
        """Check if connection is open"""
        return self.is_running and self.cap is not None
    
    def release(self):
        """Release resources"""
        self.is_running = False
        if self.cap:
            self.cap.release()