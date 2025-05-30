import cv2
import mediapipe as mp
import torch
from ultralytics import YOLO
import numpy as np
from scipy.spatial.transform import Rotation as R
from collections import deque
import datetime
import os
import time
from typing import Dict, List, Tuple, Optional
import logging

# Configuration
FACE_MODEL_PATH = "face_landmarker.task"
YOLO_MODEL_NAME = 'yolov12n.pt'
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
CONFIDENCE_THRESHOLD = 0.4
ALERT_WINDOW_SECONDS = 3
FPS_ESTIMATE = 10
ALERT_FRAMES_THRESHOLD_RATIO = 0.6

# Classes to monitor
MONITORED_CLASSES = {
    'cell phone': 'Phone Detected', 
    'person': 'Multiple People', 
    'book': 'Book Detected'
}

# Eye landmark indices for MediaPipe
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473
LEFT_EYE_CORNERS = [33, 133]
RIGHT_EYE_CORNERS = [362, 263]

# Global detector instance
_detector_instance = None

class CheatingDetector:
    """MediaPipe and YOLO based cheating detection system"""
    
    def __init__(self):
        self.setup_models()
        self.setup_alert_system()
        self.setup_logging()
        
        # FPS tracking
        self.last_time = time.time()
        self.frame_count = 0
        self.fps = 0.0
        
    def setup_logging(self):
        """Setup detailed logging for detection events"""
        # Create logger for detection events
        self.logger = logging.getLogger('cheating_detector')
        self.logger.setLevel(logging.INFO)
        
        # Create console handler if not exists
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # Create detailed formatter
            formatter = logging.Formatter(
                '[%(asctime)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
    def update_fps(self):
        """Calculate and update FPS"""
        current_time = time.time()
        self.frame_count += 1
        
        # Calculate FPS every second
        if current_time - self.last_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_time)
            self.frame_count = 0
            self.last_time = current_time
        
    def setup_models(self):
        """Initialize MediaPipe and YOLO models"""
        # MediaPipe Face detector setup
        try:
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=FACE_MODEL_PATH),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,  # Changed to IMAGE for FastAPI
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_facial_transformation_matrixes=True
            )
            self.face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
            print("✅ MediaPipe Face landmarker loaded successfully")
        except Exception as e:
            print(f"❌ MediaPipe Face landmarker error: {e}")
            self.face_landmarker = None
        
        # YOLO setup
        try:
            self.yolo_model = YOLO(YOLO_MODEL_NAME)
            self.yolo_model.to(DEVICE)
            self.valid_classes = {k: v for k, v in MONITORED_CLASSES.items() 
                                if k in self.yolo_model.names.values()}
            print(f"✅ YOLO loaded successfully, monitoring: {list(self.valid_classes.keys())}")
        except Exception as e:
            print(f"❌ YOLO error: {e}")
            self.yolo_model = None
    
    def setup_alert_system(self):
        """Initialize alert tracking system"""
        max_frames = int(FPS_ESTIMATE * ALERT_WINDOW_SECONDS)
        self.recent_detections = {alert: deque(maxlen=max_frames) 
                                for alert in self.valid_classes.values()}
        
    def get_head_pose(self, transformation_matrix) -> str:
        """Calculate head pose from MediaPipe transformation matrix"""
        try:
            rotation_matrix = np.array(transformation_matrix).reshape(4, 4)[:3, :3]
            euler_angles = R.from_matrix(rotation_matrix).as_euler('yxz', degrees=True)
            yaw, pitch = euler_angles[0], euler_angles[1]
            
            if abs(yaw) <= 12 and abs(pitch) <= 10:
                return "Looking at Screen"
            elif pitch < -20: 
                return "Looking Up"
            elif pitch > 18: 
                return "Looking Down"
            elif yaw > 30: 
                return "Looking Left"
            elif yaw < -30: 
                return "Looking Right"
            else:
                return "Head Turned"
        except Exception as e:
            print(f"❌ Head pose calculation error: {e}")
            return "Unknown"
    
    def get_eye_gaze(self, landmarks, frame_width: int, frame_height: int) -> str:
        """Calculate gaze direction using MediaPipe iris landmarks"""
        try:
            # Get iris centers
            left_iris = landmarks[LEFT_IRIS_CENTER]
            right_iris = landmarks[RIGHT_IRIS_CENTER]
            
            # Calculate eye centers
            left_corner_inner = landmarks[LEFT_EYE_CORNERS[0]]
            left_corner_outer = landmarks[LEFT_EYE_CORNERS[1]]
            right_corner_inner = landmarks[RIGHT_EYE_CORNERS[0]]
            right_corner_outer = landmarks[RIGHT_EYE_CORNERS[1]]
            
            left_center_x = (left_corner_inner.x + left_corner_outer.x) / 2
            right_center_x = (right_corner_inner.x + right_corner_outer.x) / 2
            
            # Calculate eye widths
            left_width = abs(left_corner_outer.x - left_corner_inner.x)
            right_width = abs(right_corner_outer.x - right_corner_inner.x)
            
            # Calculate iris position relative to eye center
            left_ratio = (left_iris.x - left_center_x) / (left_width / 2) if left_width > 0 else 0
            right_ratio = (right_iris.x - right_center_x) / (right_width / 2) if right_width > 0 else 0
            
            # Average both eyes for final gaze direction
            combined_ratio = (left_ratio + right_ratio) / 2
            
            # Determine gaze direction
            if abs(combined_ratio) <= 0.2:
                return "Looking at Screen"
            elif combined_ratio > 0.4:
                return "Looking Right"
            elif combined_ratio < -0.4:
                return "Looking Left"
            else:
                return "Eyes Moving"
                
        except Exception as e:
            print(f"❌ Gaze calculation error: {e}")
            return "Unknown"
    
    def process_yolo_detections(self, yolo_results) -> Tuple[List[str], List[str]]:
        """Process YOLO detections and generate alerts"""
        current_detections = {alert: False for alert in self.valid_classes.values()}
        person_count = 0
        detected_objects = []
        
        if yolo_results.boxes is not None:
            for box in yolo_results.boxes:
                class_name = self.yolo_model.names[int(box.cls.item())]
                confidence = float(box.conf.item())
                
                if confidence >= CONFIDENCE_THRESHOLD:
                    detected_objects.append(f"{class_name} ({confidence:.2f})")
                    
                    if class_name == 'person':
                        person_count += 1
                    elif class_name in self.valid_classes:
                        current_detections[self.valid_classes[class_name]] = True
        
        # Check for multiple people
        if 'Multiple People' in current_detections:
            current_detections['Multiple People'] = person_count > 1
        
        # Generate alerts based on detection history
        alerts = []
        for alert_name, deque_obj in self.recent_detections.items():
            deque_obj.append(current_detections[alert_name])
            
            if len(deque_obj) == deque_obj.maxlen:
                alert_count = sum(deque_obj)
                threshold = deque_obj.maxlen * ALERT_FRAMES_THRESHOLD_RATIO
                if alert_count >= threshold:
                    alerts.append(f"ALERT: {alert_name}")
        
        return detected_objects, alerts
    
    def log_detection_results(self, results: Dict):
        """Log detection results in the desired format"""
        # Update FPS
        self.update_fps()
        
        # Format detected objects
        objects_str = ", ".join(results['detected_objects']) if results['detected_objects'] else 'None'
        
        # Format alerts
        alerts_str = ", ".join(results['alerts']) if results['alerts'] else 'None'
        
        # Create log message
        log_message = (
            f"FPS: {self.fps:.1f} | "
            f"Head: {results['head_pose']} | "
            f"Eyes: {results['eye_gaze']} | "
            f"Objects: {objects_str} | "
            f"Alerts: {alerts_str}"
        )
        
        # Log the message
        self.logger.info(log_message)
    
    def detect_frame(self, frame: np.ndarray) -> Dict:
        """
        Main detection function for a single frame
        Returns detection results as dictionary
        """
        frame_height, frame_width = frame.shape[:2]
        
        # Initialize results
        results = {
            'head_pose': 'No Face',
            'eye_gaze': 'No Face',
            'detected_objects': [],
            'alerts': [],
            'face_detected': False,
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'fps': self.fps
        }
        
        # MediaPipe Face Detection and Gaze Analysis
        if self.face_landmarker:
            try:
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                # Detect face landmarks
                face_result = self.face_landmarker.detect(mp_image)
                
                if face_result and face_result.face_landmarks:
                    results['face_detected'] = True
                    landmarks = face_result.face_landmarks[0]
                    
                    # Calculate head pose
                    if face_result.facial_transformation_matrixes:
                        results['head_pose'] = self.get_head_pose(
                            face_result.facial_transformation_matrixes[0])
                    
                    # Calculate gaze direction
                    results['eye_gaze'] = self.get_eye_gaze(
                        landmarks, frame_width, frame_height)
                        
            except Exception as e:
                print(f"❌ MediaPipe detection error: {e}")
                results['head_pose'] = 'Error'
                results['eye_gaze'] = 'Error'
        
        # YOLO Object Detection
        if self.yolo_model:
            try:
                yolo_results = self.yolo_model.predict(
                    source=frame,
                    device=DEVICE,
                    verbose=False,
                    conf=CONFIDENCE_THRESHOLD,
                    imgsz=640
                )[0]
                
                detected_objects, alerts = self.process_yolo_detections(yolo_results)
                results['detected_objects'] = detected_objects
                results['alerts'] = alerts
                
            except Exception as e:
                print(f"❌ YOLO detection error: {e}")
        
        # Log the results
        self.log_detection_results(results)
        
        return results

def get_detector() -> CheatingDetector:
    """Get or create detector instance (singleton pattern)"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = CheatingDetector()
    return _detector_instance

def detect_cheating(frame: np.ndarray) -> Dict:
    """
    Main function to detect cheating behaviors in a frame
    
    Args:
        frame: OpenCV image frame (BGR format)
        
    Returns:
        Dictionary containing detection results:
        {
            'head_pose': str,
            'eye_gaze': str,
            'detected_objects': List[str],
            'alerts': List[str],
            'face_detected': bool,
            'timestamp': str,
            'fps': float
        }
    """
    detector = get_detector()
    return detector.detect_frame(frame)

# Additional utility functions for FastAPI integration

def initialize_detector() -> bool:
    """Initialize the detector and return success status"""
    try:
        get_detector()
        return True
    except Exception as e:
        logging.error(f"Failed to initialize detector: {e}")
        return False

def get_detector_status() -> Dict:
    """Get current detector status"""
    detector = get_detector()
    return {
        'mediapipe_loaded': detector.face_landmarker is not None,
        'yolo_loaded': detector.yolo_model is not None,
        'device': DEVICE,
        'monitored_classes': list(detector.valid_classes.keys()),
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'current_fps': detector.fps
    }

def detect_cheating_batch(frames: List[np.ndarray]) -> List[Dict]:
    """Process multiple frames for batch detection"""
    detector = get_detector()
    results = []
    
    for frame in frames:
        result = detector.detect_frame(frame)
        results.append(result)
    
    return results

# For backward compatibility and testing
if __name__ == "__main__":
    # Test the detector
    print("Testing Cheating Detector...")
    
    # Initialize detector
    if initialize_detector():
        print("✅ Detector initialized successfully")
        
        # Print status
        status = get_detector_status()
        print(f"📊 Detector Status:")
        for key, value in status.items():
            print(f"  {key}: {value}")
        
        # Test with webcam if available
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                print("🎥 Testing with webcam...")
                ret, frame = cap.read()
                if ret:
                    result = detect_cheating(frame)
                    print(f"📋 Sample Detection Result:")
                    for key, value in result.items():
                        print(f"  {key}: {value}")
                cap.release()
            else:
                print("⚠️  No webcam available for testing")
        except Exception as e:
            print(f"❌ Webcam test failed: {e}")
    else:
        print("❌ Failed to initialize detector")