from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import base64
import json
import logging
# Removed sys import
import os # For log file path
from typing import Dict, List
import asyncio
from datetime import datetime

# Import the detection function
from app.detectors import detect_cheating, get_detector_status, initialize_detector

router = APIRouter()

# Enhanced logging setup
def setup_webrtc_logging(): # Renamed to avoid conflict if imported elsewhere
    """Setup enhanced logging for the WebRTC handler to a file"""
    logger_instance = logging.getLogger('webrtc_handler')
    logger_instance.setLevel(logging.INFO)
    logger_instance.handlers.clear()

    log_file_path = "webrtc_events.log"
    try:
        file_handler = logging.FileHandler(log_file_path, mode='a') # 'a' for append
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '[%(asctime)s] WebRTC: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        logger_instance.addHandler(file_handler)
        logger_instance.info(f"Logging initialized. WebRTC events will be saved to {os.path.abspath(log_file_path)}")
    except Exception as e:
        print(f"CRITICAL: Failed to set up file logging for WebRTC handler: {e}. Logs will not be saved to file.")


    logger_instance.propagate = False
    return logger_instance

logger = setup_webrtc_logging()

active_connections: List[WebSocket] = [] # This seems unused, ConnectionManager has its own

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        # self.connection_count = 0 # Redundant, len(self.active_connections) is the count

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        # self.connection_count += 1 # Removed
        logger.info(f"📱 New WebSocket connection established. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        # Log only if it was actually removed or if it's a new disconnect call
        logger.info(f"📱 WebSocket connection closed/removed. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            # Check if websocket is still active before sending (basic check)
            if websocket.client_state.CONNECTED or websocket.application_state.CONNECTED : # FastAPI 0.95+
                await websocket.send_text(json.dumps(message))
            else:
                logger.warning(f"Attempted to send to a disconnected WebSocket. Message: {message.get('type')}")
        except RuntimeError as e: # Catches "Cannot call 'send' once a close message has been sent."
            logger.error(f"❌ Runtime error sending message (likely already closed): {e}. Message type: {message.get('type')}")
            self.disconnect(websocket) # Ensure it's removed
        except Exception as e:
            logger.error(f"❌ Unexpected error sending message: {e}. Message type: {message.get('type')}")
            self.disconnect(websocket)

    async def broadcast(self, message: dict):
        disconnected_sockets = []
        for connection in self.active_connections:
            try:
                if connection.client_state.CONNECTED or connection.application_state.CONNECTED:
                    await connection.send_text(json.dumps(message))
                else:
                    disconnected_sockets.append(connection)
            except RuntimeError as e:
                logger.error(f"❌ Runtime error broadcasting message (likely already closed for one socket): {e}")
                disconnected_sockets.append(connection)
            except Exception as e:
                logger.error(f"❌ Unexpected error broadcasting message: {e}")
                disconnected_sockets.append(connection)
        
        for conn in disconnected_sockets:
            self.disconnect(conn)

manager = ConnectionManager()

def decode_base64_image(base64_string: str) -> np.ndarray:
    try:
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        image_bytes = base64.b64decode(base64_string)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image, imdecode returned None")
        return image
    except Exception as e:
        logger.error(f"❌ Error decoding base64 image: {e}")
        raise ValueError(f"Base64 decoding failed: {e}")


def encode_image_to_base64(image: np.ndarray) -> str:
    try:
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{image_base64}"
    except Exception as e:
        logger.error(f"❌ Error encoding image to base64: {e}")
        raise

@router.get("/status")
async def get_status_endpoint(): # Renamed to avoid conflict with status variable
    try:
        detector_status_data = get_detector_status()
        detector_status_data.update({
            'status': 'healthy',
            'active_connections': len(manager.active_connections),
            'timestamp': datetime.now().isoformat()
        })
        logger.info(f"📊 Status check - Connections: {len(manager.active_connections)}, MediaPipe: {detector_status_data.get('mediapipe_loaded')}, YOLO: {detector_status_data.get('yolo_loaded')}")
        return JSONResponse(content=detector_status_data)
    except Exception as e:
        logger.error(f"❌ Status check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={'status': 'error', 'message': str(e)}
        )

@router.post("/detect")
async def detect_frame_endpoint(data: dict): # Renamed
    try:
        if 'image' not in data:
            raise HTTPException(status_code=400, detail="No image data provided")
        
        image = decode_base64_image(data['image'])
        logger.info(f"🖼️  Processing single frame (HTTP POST) - Size: {image.shape}")
        
        result = detect_cheating(image) # This will log details to detector_events.log
        
        result.update({
            'status': 'success',
            'processed_at': datetime.now().isoformat(),
            'image_shape': image.shape
        })
        return JSONResponse(content=result)
    except ValueError as ve: # Specific error for decoding
        logger.error(f"❌ Detection failed (HTTP POST) due to image decoding: {ve}")
        raise HTTPException(status_code=400, detail=f"Image processing error: {str(ve)}")
    except Exception as e:
        logger.error(f"❌ Detection failed (HTTP POST): {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    logger.info("🔌 WebSocket connection attempt received")
    await manager.connect(websocket)
    frame_counter = 0
    client_disconnected_gracefully = False

    try:
        welcome_message = {
            'type': 'connection_established',
            'message': 'Connected to cheating detection service',
            'timestamp': datetime.now().isoformat()
        }
        await manager.send_personal_message(welcome_message, websocket)
        
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message.get('type') == 'frame':
                    frame_counter += 1
                    if 'image' in message:
                        try:
                            image = decode_base64_image(message['image'])
                        except ValueError as img_err:
                            logger.error(f"WebSocket: Image decoding error for frame #{frame_counter}: {img_err}")
                            error_response = {
                                'type': 'error',
                                'message': f'Image decoding error: {str(img_err)}',
                                'timestamp': datetime.now().isoformat()
                            }
                            await manager.send_personal_message(error_response, websocket)
                            continue # Skip processing this frame

                        result = detect_cheating(image) # Logs to detector_events.log
                        
                        response = {
                            'type': 'detection_result',
                            'data': result,
                            'timestamp': datetime.now().isoformat(),
                            'frame_number': frame_counter
                        }
                        await manager.send_personal_message(response, websocket)
                        
                        if result.get('alerts'):
                            alert_broadcast = {
                                'type': 'alert_broadcast',
                                'alerts': result['alerts'],
                                'timestamp': datetime.now().isoformat(),
                                'frame_number': frame_counter
                            }
                            # await manager.broadcast(alert_broadcast) # Decide if alerts should be broadcast
                            logger.warning(f"🚨 CRITICAL ALERT (WS Frame #{frame_counter}): {', '.join(result['alerts'])}")
                
                elif message.get('type') == 'ping':
                    pong_response = {'type': 'pong', 'timestamp': datetime.now().isoformat()}
                    await manager.send_personal_message(pong_response, websocket)
                
                elif message.get('type') == 'get_status':
                    status_data = get_detector_status()
                    status_response = {'type': 'status', 'data': status_data, 'timestamp': datetime.now().isoformat()}
                    await manager.send_personal_message(status_response, websocket)
                    logger.info("📊 Status requested via WebSocket")
                
            except json.JSONDecodeError:
                logger.error("❌ Invalid JSON received via WebSocket")
                error_response = {'type': 'error', 'message': 'Invalid JSON format', 'timestamp': datetime.now().isoformat()}
                await manager.send_personal_message(error_response, websocket)
                
            except WebSocketDisconnect: # This should be caught by the outer try-except
                client_disconnected_gracefully = True
                logger.info(f"📱 WebSocket client disconnected gracefully during receive. Frames: {frame_counter}")
                break # Exit while loop

            except RuntimeError as rterr: # e.g. sending on a closed socket
                 if "Cannot call 'send' once a close message has been sent" in str(rterr) or \
                    "Cannot call 'receive' once a disconnect message has been received" in str(rterr):
                    logger.warning(f"RuntimeError indicating WebSocket already closing/closed: {rterr}. Frames: {frame_counter}")
                    break # Exit while loop
                 else:
                    logger.error(f"❌ Unhandled RuntimeError in WebSocket loop: {rterr}. Frames: {frame_counter}")
                    # Potentially send error message if socket seems alive
                    error_response = {'type': 'error', 'message': f'Server processing error: {str(rterr)}', 'timestamp': datetime.now().isoformat()}
                    await manager.send_personal_message(error_response, websocket) # This might also fail


            except Exception as e: # Catch other errors during message processing
                logger.error(f"❌ Error processing WebSocket message (Frame #{frame_counter}): {e}")
                error_response = {
                    'type': 'error',
                    'message': f'Server processing error: {str(e)}',
                    'timestamp': datetime.now().isoformat()
                }
                try:
                    await manager.send_personal_message(error_response, websocket)
                except Exception as send_exc:
                    logger.error(f"❌ Failed to send error message to socket after processing error: {send_exc}")
                # Depending on error, might want to break or continue
                # If it's a framing/data error from client, continue. If server state error, might break.

    except WebSocketDisconnect:
        client_disconnected_gracefully = True # Redundant if already set, but safe
        logger.info(f"📱 WebSocket connection closed by client (outer handler). Frames processed: {frame_counter}")
    
    except Exception as e: # Errors during initial connect or unhandled in loop
        logger.error(f"❌ Unhandled WebSocket error (Frames: {frame_counter}): {e}")
        # Try to send a final error message if the exception isn't a disconnect itself
        if not isinstance(e, WebSocketDisconnect):
            try:
                error_response = {'type': 'error', 'message': f'Unhandled WebSocket error: {str(e)}', 'timestamp': datetime.now().isoformat()}
                await manager.send_personal_message(error_response, websocket)
            except Exception as final_send_err:
                logger.error(f"❌ Failed to send final error message on unhandled WebSocket error: {final_send_err}")
    finally:
        manager.disconnect(websocket)
        if not client_disconnected_gracefully:
             logger.info(f"📱 WebSocket connection ended (possibly uncleanly or due to server error). Frames processed: {frame_counter}")


@router.get("/connections")
async def get_active_connections():
    connections_info = {
        'active_connections': len(manager.active_connections),
        'timestamp': datetime.now().isoformat()
    }
    logger.info(f"📱 Connection status requested (HTTP): {len(manager.active_connections)} active")
    return connections_info

@router.post("/initialize")
async def initialize_detector_endpoint(): # Renamed
    try:
        logger.info("🔄 Initializing detector (HTTP POST)...")
        success = initialize_detector()
        if success:
            status_data = get_detector_status()
            logger.info("✅ Detector initialization successful (HTTP POST)")
            return JSONResponse(content={
                'status': 'initialized',
                'detector_status': status_data,
                'timestamp': datetime.now().isoformat()
            })
        else:
            logger.error("❌ Detector initialization failed (HTTP POST)")
            return JSONResponse(
                status_code=500,
                content={'status': 'failed', 'message': 'Failed to initialize detector', 'timestamp': datetime.now().isoformat()}
            )
    except Exception as e:
        logger.error(f"❌ Initialization error (HTTP POST): {e}")
        return JSONResponse(
            status_code=500,
            content={'status': 'error', 'message': str(e), 'timestamp': datetime.now().isoformat()}
        )

@router.post("/test")
async def test_detector_endpoint(): # Renamed
    try:
        logger.info("🧪 Running detector test (HTTP POST)...")
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(test_image, "Test Image", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        result = detect_cheating(test_image) # Logs to detector_events.log
        logger.info(f"✅ Detector test completed (HTTP POST). Result summary: Head={result['head_pose']}")
        
        return JSONResponse(content={
            'status': 'test_completed',
            'result': result, # Full result can be large, consider summarizing
            'message': 'Detector test completed successfully. Details in detector_events.log.',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"❌ Detector test failed (HTTP POST): {e}")
        return JSONResponse(
            status_code=500,
            content={'status': 'test_failed', 'message': str(e), 'timestamp': datetime.now().isoformat()}
        )

@router.get("/ws-test") # This is fine for a simple test endpoint
async def websocket_test_endpoint():
    logger.info("🧪 WebSocket test endpoint /ws-test accessed (HTTP GET)")
    return {
        'message': 'WebSocket route /ws is registered. This is an HTTP test endpoint.',
        'active_connections_on_manager': len(manager.active_connections),
        'timestamp': datetime.now().isoformat()
    }

# Initialize detector on module load (detector logs to its file)
logger.info("🚀 WebRTC handler module loading, attempting to initialize detector...")
try:
    if initialize_detector(): # This now returns bool and logs internally
        logger.info("✅ WebRTC handler: Cheating detector initialized successfully during module load.")
    else:
        logger.error("❌ WebRTC handler: Cheating detector failed to initialize during module load. Check detector_events.log.")
except Exception as e:
    logger.critical(f"❌ CRITICAL error during detector initialization in WebRTC handler module load: {e}")