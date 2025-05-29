from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import base64
import json
import logging
from typing import Dict, List
import asyncio
from datetime import datetime

# Import the detection function
from app.detectors import detect_cheating, get_detector_status, initialize_detector

router = APIRouter()
logger = logging.getLogger(__name__)

# Store active WebSocket connections
active_connections: List[WebSocket] = []

class ConnectionManager:
    """Manage WebSocket connections for real-time monitoring"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")
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
        logger.error(f"Error decoding base64 image: {e}")
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
        logger.error(f"Error encoding image to base64: {e}")
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
        return JSONResponse(content=status)
    except Exception as e:
        logger.error(f"Status check failed: {e}")
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
        
        # Run detection
        result = detect_cheating(image)
        
        # Add processing info
        result.update({
            'status': 'success',
            'processed_at': datetime.now().isoformat(),
            'image_shape': image.shape
        })
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time cheating detection"""
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive data from client
            data = await websocket.receive_text()
            
            try:
                # Parse JSON data
                message = json.loads(data)
                
                if message.get('type') == 'frame':
                    # Process frame for cheating detection
                    if 'image' in message:
                        # Decode image
                        image = decode_base64_image(message['image'])
                        
                        # Run detection
                        result = detect_cheating(image)
                        
                        # Send result back to client
                        response = {
                            'type': 'detection_result',
                            'data': result,
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        await manager.send_personal_message(response, websocket)
                        
                        # If alerts detected, broadcast to all connections
                        if result.get('alerts'):
                            alert_broadcast = {
                                'type': 'alert_broadcast',
                                'alerts': result['alerts'],
                                'timestamp': datetime.now().isoformat()
                            }
                            await manager.broadcast(alert_broadcast)
                
                elif message.get('type') == 'ping':
                    # Respond to ping
                    pong_response = {
                        'type': 'pong',
                        'timestamp': datetime.now().isoformat()
                    }
                    await manager.send_personal_message(pong_response, websocket)
                
                elif message.get('type') == 'get_status':
                    # Send current detector status
                    status = get_detector_status()
                    status_response = {
                        'type': 'status',
                        'data': status,
                        'timestamp': datetime.now().isoformat()
                    }
                    await manager.send_personal_message(status_response, websocket)
                
            except json.JSONDecodeError:
                error_response = {
                    'type': 'error',
                    'message': 'Invalid JSON format',
                    'timestamp': datetime.now().isoformat()
                }
                await manager.send_personal_message(error_response, websocket)
                
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
                error_response = {
                    'type': 'error',
                    'message': f'Processing error: {str(e)}',
                    'timestamp': datetime.now().isoformat()
                }
                await manager.send_personal_message(error_response, websocket)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket connection closed by client")
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@router.get("/connections")
async def get_active_connections():
    """Get number of active WebSocket connections"""
    return {
        'active_connections': len(manager.active_connections),
        'timestamp': datetime.now().isoformat()
    }

@router.post("/initialize")
async def initialize():
    """Initialize or reinitialize the detector"""
    try:
        success = initialize_detector()
        if success:
            status = get_detector_status()
            return JSONResponse(content={
                'status': 'initialized',
                'detector_status': status,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return JSONResponse(
                status_code=500,
                content={
                    'status': 'failed',
                    'message': 'Failed to initialize detector',
                    'timestamp': datetime.now().isoformat()
                }
            )
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
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
        # Create a simple test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(test_image, "Test Image", (200, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # Run detection
        result = detect_cheating(test_image)
        
        return JSONResponse(content={
            'status': 'test_completed',
            'result': result,
            'message': 'Detector test completed successfully',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Detector test failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                'status': 'test_failed',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
        )

# Initialize detector on module load
logger.info("Initializing cheating detector...")
try:
    initialize_detector()
    logger.info("✅ Cheating detector initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize cheating detector: {e}")