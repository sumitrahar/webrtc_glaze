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
# Removed sys import as StreamHandler(sys.stdout) will be removed

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
        self.setup_logging() # Setup logging first
        self.setup_models()
        self.setup_alert_system()

        # FPS tracking
        self.last_time = time.time()
        self.frame_count = 0
        self.fps = 0.0

    def setup_logging(self):
        """Setup detailed logging for detection events to a file"""
        self.logger = logging.getLogger('cheating_detector')
        self.logger.setLevel(logging.INFO)

        # Clear existing handlers to avoid duplicates if re-initialized
        self.logger.handlers.clear()

        # Create file handler
        log_file_path = "detector_events.log"
        try:
            file_handler = logging.FileHandler(log_file_path, mode='a') # 'a' for append
            file_handler.setLevel(logging.INFO)

            # Create detailed formatter
            formatter = logging.Formatter(
                '[%(asctime)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
            # Initial log to confirm file setup (optional, will go to file)
            self.logger.info(f"Logging initialized. Detector events will be saved to {os.path.abspath(log_file_path)}")

        except Exception as e:
            # Fallback to console print if file logging setup fails
            print(f"CRITICAL: Failed to set up file logging for detector: {e}. Logs will not be saved to file.")


        # Prevent propagation to avoid duplicate logs if root logger is configured
        self.logger.propagate = False

    def update_fps(self):
        """Calculate and update FPS"""
        current_time = time.time()
        self.frame_count += 1

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
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_facial_transformation_matrixes=True
            )
            self.face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
            print("✅ MediaPipe Face landmarker loaded successfully") # Startup console message
        except Exception as e:
            print(f"❌ MediaPipe Face landmarker error: {e}") # Startup console message
            self.face_landmarker = None

        # YOLO setup
        try:
            self.yolo_model = YOLO(YOLO_MODEL_NAME)
            self.yolo_model.to(DEVICE)
            self.valid_classes = {k: v for k, v in MONITORED_CLASSES.items()
                                if k in self.yolo_model.names.values()}
            print(f"✅ YOLO loaded successfully, monitoring: {list(self.valid_classes.keys())}") # Startup console message
        except Exception as e:
            print(f"❌ YOLO error: {e}") # Startup console message
            self.yolo_model = None

    def setup_alert_system(self):
        """Initialize alert tracking system"""
        max_frames = int(FPS_ESTIMATE * ALERT_WINDOW_SECONDS)
        # Ensure valid_classes is initialized before this is called if setup_models can fail
        if hasattr(self, 'valid_classes') and self.valid_classes:
            self.recent_detections = {alert: deque(maxlen=max_frames)
                                    for alert in self.valid_classes.values()}
        else:
            self.recent_detections = {} # Handle case where yolo might not load
            print("⚠️ Alert system initialized with no valid classes due to YOLO model setup issue.")


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
            self.logger.error(f"Head pose calculation error: {e}")
            return "Unknown"

    def get_eye_gaze(self, landmarks, frame_width: int, frame_height: int) -> str:
        """Calculate gaze direction using MediaPipe iris landmarks"""
        try:
            left_iris = landmarks[LEFT_IRIS_CENTER]
            right_iris = landmarks[RIGHT_IRIS_CENTER]

            left_corner_inner = landmarks[LEFT_EYE_CORNERS[0]]
            left_corner_outer = landmarks[LEFT_EYE_CORNERS[1]]
            right_corner_inner = landmarks[RIGHT_EYE_CORNERS[0]]
            right_corner_outer = landmarks[RIGHT_EYE_CORNERS[1]]

            left_center_x = (left_corner_inner.x + left_corner_outer.x) / 2
            right_center_x = (right_corner_inner.x + right_corner_outer.x) / 2

            left_width = abs(left_corner_outer.x - left_corner_inner.x)
            right_width = abs(right_corner_outer.x - right_corner_inner.x)

            left_ratio = (left_iris.x - left_center_x) / (left_width / 2) if left_width > 0 else 0
            right_ratio = (right_iris.x - right_center_x) / (right_width / 2) if right_width > 0 else 0

            combined_ratio = (left_ratio + right_ratio) / 2

            if abs(combined_ratio) <= 0.2:
                return "Looking at Screen"
            elif combined_ratio > 0.4:
                return "Looking Right"
            elif combined_ratio < -0.4:
                return "Looking Left"
            else:
                return "Eyes Moving"

        except Exception as e:
            self.logger.error(f"Gaze calculation error: {e}")
            return "Unknown"

    def process_yolo_detections(self, yolo_results) -> Tuple[List[str], List[str]]:
        """Process YOLO detections and generate alerts"""
        # Ensure recent_detections is initialized
        if not hasattr(self, 'recent_detections') or not self.recent_detections:
             # This can happen if YOLO failed to load and valid_classes was empty
            if hasattr(self, 'valid_classes') and self.valid_classes: # If yolo loaded but valid_classes is empty for some reason
                 self.setup_alert_system() 
            else: # If yolo truly failed
                 return [], []


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
        
        if 'Multiple People' in current_detections: # Ensure 'Multiple People' is a key
            current_detections['Multiple People'] = person_count > 1
        elif 'Multiple People' in self.valid_classes.values(): # If it should exist but wasn't in current_detections
            # This case implies an issue if 'Multiple People' is a target alert but not in current_detections keys
            # For safety, initialize it if it's a valid alert type
            current_detections['Multiple People'] = person_count > 1


        alerts = []
        for alert_name, deque_obj in self.recent_detections.items():
            # Ensure alert_name from recent_detections is valid in current_detections
            # (handles cases where MONITORED_CLASSES might change or have issues)
            is_detected_currently = current_detections.get(alert_name, False)
            deque_obj.append(is_detected_currently)

            if len(deque_obj) == deque_obj.maxlen:
                alert_count = sum(deque_obj)
                threshold = deque_obj.maxlen * ALERT_FRAMES_THRESHOLD_RATIO
                if alert_count >= threshold:
                    alerts.append(f"ALERT: {alert_name}")

        return detected_objects, alerts

    def log_detection_results(self, results: Dict):
        """Log detection results to the configured logger (file)"""
        self.update_fps()

        objects_str = ", ".join(results['detected_objects']) if results['detected_objects'] else 'None'
        alerts_str = ", ".join(results['alerts']) if results['alerts'] else 'None'

        log_message = (
            f"FPS: {self.fps:.1f} | "
            f"Head: {results['head_pose']} | "
            f"Eyes: {results['eye_gaze']} | "
            f"Objects: {objects_str} | "
            f"Alerts: {alerts_str}"
        )

        self.logger.info(log_message)
        # REMOVED: print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {log_message}")

    def detect_frame(self, frame: np.ndarray) -> Dict:
        """Main detection function for a single frame"""
        frame_height, frame_width = frame.shape[:2]
        results = {
            'head_pose': 'No Face',
            'eye_gaze': 'No Face',
            'detected_objects': [],
            'alerts': [],
            'face_detected': False,
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'fps': self.fps
        }

        if self.face_landmarker:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                face_result = self.face_landmarker.detect(mp_image)

                if face_result and face_result.face_landmarks:
                    results['face_detected'] = True
                    landmarks = face_result.face_landmarks[0]
                    if face_result.facial_transformation_matrixes:
                        results['head_pose'] = self.get_head_pose(
                            face_result.facial_transformation_matrixes[0])
                    results['eye_gaze'] = self.get_eye_gaze(
                        landmarks, frame_width, frame_height)
            except Exception as e:
                self.logger.error(f"MediaPipe detection error: {e}")
                results['head_pose'] = 'Error'
                results['eye_gaze'] = 'Error'

        if self.yolo_model:
            try:
                yolo_results_list = self.yolo_model.predict(
                    source=frame,
                    device=DEVICE,
                    verbose=False,
                    conf=CONFIDENCE_THRESHOLD,
                    imgsz=640
                )
                if yolo_results_list: # yolo.predict now returns a list
                    yolo_results = yolo_results_list[0]
                    detected_objects, alerts = self.process_yolo_detections(yolo_results)
                    results['detected_objects'] = detected_objects
                    results['alerts'] = alerts
                else:
                    self.logger.warning("YOLO prediction returned empty list.")

            except Exception as e:
                self.logger.error(f"YOLO detection error: {e}")
        
        self.log_detection_results(results)
        return results

def get_detector() -> CheatingDetector:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = CheatingDetector()
    return _detector_instance

def detect_cheating(frame: np.ndarray) -> Dict:
    detector = get_detector()
    return detector.detect_frame(frame)

def initialize_detector() -> bool:
    try:
        get_detector() # This will initialize if _detector_instance is None
        # Check if models actually loaded (optional, for more robust status)
        detector = get_detector()
        if detector.face_landmarker and detector.yolo_model:
             detector.logger.info("Detector initialized successfully (from initialize_detector call).")
             return True
        else:
            detector.logger.error("Detector initialization called, but one or more models failed to load.")
            return False
    except Exception as e:
        # If logger isn't set up yet (e.g., error in CheatingDetector.__init__ before setup_logging)
        # this log might not go to file. Print as fallback.
        print(f"CRITICAL: Failed to initialize detector: {e}")
        if _detector_instance and hasattr(_detector_instance, 'logger'):
            _detector_instance.logger.error(f"Failed to initialize detector: {e}")
        return False


def get_detector_status() -> Dict:
    detector = get_detector()
    return {
        'mediapipe_loaded': detector.face_landmarker is not None,
        'yolo_loaded': detector.yolo_model is not None,
        'device': DEVICE,
        'monitored_classes': list(detector.valid_classes.keys()) if hasattr(detector, 'valid_classes') and detector.valid_classes else [],
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'current_fps': detector.fps
    }

def detect_cheating_batch(frames: List[np.ndarray]) -> List[Dict]:
    detector = get_detector()
    results_list = []
    for frame in frames:
        result = detector.detect_frame(frame)
        results_list.append(result)
    return results_list

if __name__ == "__main__":
    print("Testing Cheating Detector (direct script run)...")
    if initialize_detector():
        print("✅ Detector initialized successfully (from __main__)")
        status = get_detector_status()
        print(f"📊 Detector Status:")
        for key, value in status.items():
            print(f"  {key}: {value}")
        
        # Sample logging to file if detector is used
        _detector_instance.logger.info("Detector test run from __main__ block started.")
        
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                print("🎥 Testing with webcam (from __main__)...")
                ret, frame = cap.read()
                if ret:
                    result = detect_cheating(frame) # This will log to file
                    print(f"📋 Sample Detection Result (logged to file, snippet below):")
                    print(f"  Head: {result['head_pose']}, Eyes: {result['eye_gaze']}")
                cap.release()
            else:
                print("⚠️  No webcam available for testing (from __main__)")
        except Exception as e:
            print(f"❌ Webcam test failed (from __main__): {e}")
        
        _detector_instance.logger.info("Detector test run from __main__ block finished.")
    else:
        print("❌ Failed to initialize detector (from __main__)")