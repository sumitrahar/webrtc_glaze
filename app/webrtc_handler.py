from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import base64
import json
import logging
import sys
from typing import Dict, List
import asyncio
from datetime import datetime

# Import the detection function
from app.detectors import detect_cheating, get_detector_status, initialize_detector

router = APIRouter()

# Enhanced logging setup
def setup_logging():
    """Setup enhanced logging for the WebRTC handler"""
    # Create logger
    logger = logging.getLogger('webrtc_handler')
    logger.setLevel(logging.INFO)
    
    # Create console handler if not exists
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '[%(asctime)s] WebRTC: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

# Store active WebSocket connections
active_connections: List[WebSocket] = []

class ConnectionManager:
    """Manage WebSocket connections for real-time monitoring"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_count = 0
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_count += 1
        logger.info(f"📱 New WebSocket connection established. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"📱 WebSocket connection closed. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"❌ Error sending message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"❌ Error broadcasting message: {e}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()

def decode_base64_image(base64_string: str) -> np.ndarray:
    """Convert base64 string to OpenCV image"""
    try:
        # Remove data URL prefix if present
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        # Decode base64 to bytes
        image_bytes = base64.b64decode(base64_string)
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        return image
    except Exception as e:
        logger.error(f"❌ Error decoding base64 image: {e}")
        raise

def encode_image_to_base64(image: np.ndarray) -> str:
    """Convert OpenCV image to base64 string"""
    try:
        # Encode image to JPEG
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        
        # Convert to base64
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return f"data:image/jpeg;base64,{image_base64}"
    except Exception as e:
        logger.error(f"❌ Error encoding image to base64: {e}")
        raise

@router.get("/status")
async def get_status():
    """Get detector status and health check"""
    try:
        status = get_detector_status()
        status.update({
            'status': 'healthy',
            'active_connections': len(manager.active_connections),
            'timestamp': datetime.now().isoformat()
        })
        logger.info(f"📊 Status check - Connections: {len(manager.active_connections)}, MediaPipe: {status.get('mediapipe_loaded')}, YOLO: {status.get('yolo_loaded')}")
        return JSONResponse(content=status)
    except Exception as e:
        logger.error(f"❌ Status check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={'status': 'error', 'message': str(e)}
        )

@router.post("/detect")
async def detect_frame(data: dict):
    """Process a single frame for cheating detection"""
    try:
        if 'image' not in data:
            raise HTTPException(status_code=400, detail="No image data provided")
        
        # Decode the image
        image = decode_base64_image(data['image'])
        logger.info(f"🖼️  Processing single frame - Size: {image.shape}")
        
        # Run detection
        result = detect_cheating(image)
        
        # Add processing info
        result.update({
            'status': 'success',
            'processed_at': datetime.now().isoformat(),
            'image_shape': image.shape
        })
        
        # Log detection summary
        objects_str = ", ".join(result['detected_objects']) if result['detected_objects'] else 'None'
        alerts_str = ", ".join(result['alerts']) if result['alerts'] else 'None'
        logger.info(f"🔍 Single Frame Detection - Head: {result['head_pose']}, Eyes: {result['eye_gaze']}, Objects: {objects_str}, Alerts: {alerts_str}")
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"❌ Detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time cheating detection"""
    await manager.connect(websocket)
    frame_counter = 0
    
    try:
        while True:
            # Receive data from client
            data = await websocket.receive_text()
            
            try:
                # Parse JSON data
                message = json.loads(data)
                
                if message.get('type') == 'frame':
                    frame_counter += 1
                    
                    # Process frame for cheating detection
                    if 'image' in message:
                        # Decode image
                        image = decode_base64_image(message['image'])
                        
                        # Log frame processing (less frequent to avoid spam)
                        if frame_counter % 30 == 0:  # Log every 30th frame
                            logger.info(f"🎥 Processing WebSocket frame #{frame_counter} - Size: {image.shape}")
                        
                        # Run detection
                        result = detect_cheating(image)
                        
                        # Send result back to client
                        response = {
                            'type': 'detection_result',
                            'data': result,
                            'timestamp': datetime.now().isoformat(),
                            'frame_number': frame_counter
                        }
                        
                        await manager.send_personal_message(response, websocket)
                        
                        # If alerts detected, broadcast to all connections and log
                        if result.get('alerts'):
                            alert_broadcast = {
                                'type': 'alert_broadcast',
                                'alerts': result['alerts'],
                                'timestamp': datetime.now().isoformat(),
                                'frame_number': frame_counter
                            }
                            await manager.broadcast(alert_broadcast)
                            
                            # Log critical alerts
                            alerts_str = ", ".join(result['alerts'])
                            logger.warning(f"🚨 CRITICAL ALERT DETECTED - Frame #{frame_counter}: {alerts_str}")
                
                elif message.get('type') == 'ping':
                    # Respond to ping
                    pong_response = {
                        'type': 'pong',
                        'timestamp': datetime.now().isoformat()
                    }
                    await manager.send_personal_message(pong_response, websocket)
                    logger.debug("🏓 Ping-pong received")
                
                elif message.get('type') == 'get_status':
                    # Send current detector status
                    status = get_detector_status()
                    status_response = {
                        'type': 'status',
                        'data': status,
                        'timestamp': datetime.now().isoformat()
                    }
                    await manager.send_personal_message(status_response, websocket)
                    logger.info("📊 Status requested via WebSocket")
                
            except json.JSONDecodeError:
                error_response = {
                    'type': 'error',
                    'message': 'Invalid JSON format',
                    'timestamp': datetime.now().isoformat()
                }
                await manager.send_personal_message(error_response, websocket)
                logger.error("❌ Invalid JSON received via WebSocket")
                
            except Exception as e:
                logger.error(f"❌ Error processing WebSocket message: {e}")
                error_response = {
                    'type': 'error',
                    'message': f'Processing error: {str(e)}',
                    'timestamp': datetime.now().isoformat()
                }
                await manager.send_personal_message(error_response, websocket)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info(f"📱 WebSocket connection closed by client after {frame_counter} frames")
    
    except Exception as e:
        logger.error(f"❌ WebSocket error after {frame_counter} frames: {e}")
        manager.disconnect(websocket)

@router.get("/connections")
async def get_active_connections():
    """Get number of active WebSocket connections"""
    connections_info = {
        'active_connections': len(manager.active_connections),
        'timestamp': datetime.now().isoformat()
    }
    logger.info(f"📱 Connection status requested: {len(manager.active_connections)} active")
    return connections_info

@router.post("/initialize")
async def initialize():
    """Initialize or reinitialize the detector"""
    try:
        logger.info("🔄 Initializing detector...")
        success = initialize_detector()
        if success:
            status = get_detector_status()
            logger.info("✅ Detector initialization successful")
            return JSONResponse(content={
                'status': 'initialized',
                'detector_status': status,
                'timestamp': datetime.now().isoformat()
            })
        else:
            logger.error("❌ Detector initialization failed")
            return JSONResponse(
                status_code=500,
                content={
                    'status': 'failed',
                    'message': 'Failed to initialize detector',
                    'timestamp': datetime.now().isoformat()
                }
            )
    except Exception as e:
        logger.error(f"❌ Initialization error: {e}")
        return JSONResponse(
            status_code=500,
            content={
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
        )

# Additional utility endpoints

@router.post("/test")
async def test_detector():
    """Test the detector with a sample image"""
    try:
        logger.info("🧪 Running detector test...")
        
        # Create a simple test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(test_image, "Test Image", (200, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # Run detection
        result = detect_cheating(test_image)
        
        logger.info(f"✅ Detector test completed - Head: {result['head_pose']}, Eyes: {result['eye_gaze']}")
        
        return JSONResponse(content={
            'status': 'test_completed',
            'result': result,
            'message': 'Detector test completed successfully',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Detector test failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                'status': 'test_failed',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
        )

# Initialize detector on module load
logger.info("🚀 Starting WebRTC handler initialization...")
try:
    initialize_detector()
    logger.info("✅ WebRTC handler initialized successfully with cheating detector")
except Exception as e:
    logger.error(f"❌ Failed to initialize WebRTC handler: {e}")