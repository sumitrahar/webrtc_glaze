// script.js - Fixed version for cheating detection

// DOM elements
const videoElem = document.getElementById("webcam-video");
const toggleBtn = document.getElementById("toggle-webcam");
const statusDiv = document.getElementById("status");

// WebSocket variable
let ws = null;
// MediaStream from webcam
let stream = null;
// Flag to track if webcam is on
let webcamOn = false;
// Frame sending interval
let frameInterval = null;

// Add status logging function
function updateStatus(message, type = 'info') {
    console.log(message);
    if (statusDiv) {
        const timestamp = new Date().toLocaleTimeString();
        const statusClass = type === 'error' ? 'error' : type === 'success' ? 'success' : 'info';
        statusDiv.innerHTML = `<span class="${statusClass}">[${timestamp}] ${message}</span>`;
    }
}

// Turn on webcam: request media, set video src, open websocket and start sending frames
async function startWebcam() {
    try {
        updateStatus("Requesting webcam access...");
        
        // Request webcam video stream
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480 }, 
            audio: false 
        });
        
        videoElem.srcObject = stream;
        await videoElem.play();
        
        updateStatus("Webcam started, connecting to detection service...");

        // FIXED: Correct WebSocket URL - exactly matching the backend endpoint
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProtocol}//${window.location.host}/ws`;
        
        console.log("Connecting to WebSocket:", wsUrl);
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            updateStatus("✅ Connected to cheating detection service", 'success');
            console.log("WebSocket connected successfully");
            // Start sending frames periodically
            startSendingFrames();
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                handleWebSocketMessage(data);
            } catch (e) {
                console.error("Error parsing WebSocket message:", e);
            }
        };

        ws.onclose = (event) => {
            updateStatus(`❌ Disconnected from detection service (Code: ${event.code})`, 'error');
            console.log("WebSocket disconnected with code:", event.code, "reason:", event.reason);
            stopSendingFrames();
        };

        ws.onerror = (err) => {
            updateStatus("❌ Detection service connection error", 'error');
            console.error("WebSocket error:", err);
        };

        webcamOn = true;
        toggleBtn.textContent = "Stop Detection";
        toggleBtn.style.backgroundColor = "#dc3545"; // Red color when stopping

    } catch (err) {
        updateStatus(`❌ Could not access webcam: ${err.message}`, 'error');
        console.error("Webcam error:", err);
    }
}

// Stop webcam: stop video stream tracks, close websocket, update UI
function stopWebcam() {
    updateStatus("Stopping webcam and detection...");
    
    // Stop sending frames
    stopSendingFrames();
    
    // Stop video stream
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }

    // Close WebSocket
    if (ws) {
        ws.close();
        ws = null;
    }

    // Reset video element
    videoElem.pause();
    videoElem.srcObject = null;

    webcamOn = false;
    toggleBtn.textContent = "Start Detection";
    toggleBtn.style.backgroundColor = "#007bff"; // Blue color when starting
    
    updateStatus("Detection stopped");
}

// Start sending frames at regular intervals
function startSendingFrames() {
    if (frameInterval) {
        clearInterval(frameInterval);
    }
    
    // Send frames every 200ms (5 FPS) to avoid overwhelming the server
    frameInterval = setInterval(() => {
        sendFrame();
    }, 200);
}

// Stop sending frames
function stopSendingFrames() {
    if (frameInterval) {
        clearInterval(frameInterval);
        frameInterval = null;
    }
}

// Capture current video frame, convert to base64, send over websocket
function sendFrame() {
    if (!webcamOn || !ws || ws.readyState !== WebSocket.OPEN || !videoElem.videoWidth) {
        return;
    }

    try {
        // Create a canvas to capture current video frame
        const canvas = document.createElement("canvas");
        canvas.width = videoElem.videoWidth;
        canvas.height = videoElem.videoHeight;
        const ctx = canvas.getContext("2d");

        // Draw current frame to canvas
        ctx.drawImage(videoElem, 0, 0, canvas.width, canvas.height);

        // Convert to base64 JPEG
        const base64Image = canvas.toDataURL("image/jpeg", 0.8); // 80% quality

        // Send proper JSON message format expected by the WebSocket handler
        const message = {
            type: "frame",
            image: base64Image,
            timestamp: new Date().toISOString()
        };

        ws.send(JSON.stringify(message));

    } catch (error) {
        console.error("Error sending frame:", error);
        updateStatus("❌ Error sending frame to detection service", 'error');
    }
}

// Handle incoming WebSocket messages
function handleWebSocketMessage(data) {
    switch (data.type) {
        case 'detection_result':
            handleDetectionResult(data.data);
            break;
            
        case 'alert_broadcast':
            handleAlert(data.alerts);
            break;
            
        case 'error':
            updateStatus(`❌ Detection error: ${data.message}`, 'error');
            break;
            
        case 'pong':
            console.log("Received pong from server");
            break;
            
        case 'status':
            console.log("Detector status:", data.data);
            break;
            
        default:
            console.log("Unknown message type:", data.type);
    }
}

// Handle detection results
function handleDetectionResult(result) {
    // Log detection results to console (this will show the formatted logs)
    console.log("Detection Result:", result);
    
    // Update status with key information
    const alerts = result.alerts && result.alerts.length > 0 ? result.alerts.join(', ') : 'None';
    const statusMessage = `👁️ ${result.head_pose} | 👀 ${result.eye_gaze} | 🚨 Alerts: ${alerts}`;
    
    // Color code based on alerts
    const statusType = result.alerts && result.alerts.length > 0 ? 'error' : 'success';
    updateStatus(statusMessage, statusType);
}

// Handle critical alerts
function handleAlert(alerts) {
    const alertMessage = `🚨 CRITICAL ALERT: ${alerts.join(', ')}`;
    updateStatus(alertMessage, 'error');
    
    // You could add additional alert handling here:
    // - Play a sound
    // - Show a popup
    // - Send notification to supervisor
    console.warn("CRITICAL ALERT DETECTED:", alerts);
}

// Toggle button click handler
toggleBtn.addEventListener("click", () => {
    if (webcamOn) {
        stopWebcam();
    } else {
        startWebcam();
    }
});

// Send periodic ping to keep connection alive
function startPingInterval() {
    setInterval(() => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: "ping" }));
        }
    }, 30000); // Ping every 30 seconds
}

// Initialize ping interval when page loads
window.addEventListener('load', () => {
    updateStatus("Cheating detection system ready");
    startPingInterval();
});

// Handle page unload - cleanup connections
window.addEventListener('beforeunload', () => {
    if (webcamOn) {
        stopWebcam();
    }
});

// Add keyboard shortcut (Space bar) to toggle webcam
document.addEventListener('keydown', (event) => {
    if (event.code === 'Space' && event.target.tagName !== 'INPUT') {
        event.preventDefault();
        toggleBtn.click();
    }
});

// Test WebSocket connection function
async function testWebSocketConnection() {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}/ws`;
    
    console.log("Testing WebSocket connection to:", wsUrl);
    
    try {
        const testWs = new WebSocket(wsUrl);
        
        testWs.onopen = () => {
            console.log("✅ WebSocket test connection successful");
            testWs.close();
        };
        
        testWs.onerror = (error) => {
            console.error("❌ WebSocket test connection failed:", error);
        };
        
        testWs.onclose = (event) => {
            console.log("WebSocket test connection closed:", event.code, event.reason);
        };
    } catch (error) {
        console.error("❌ WebSocket test failed:", error);
    }
}

// Add a button to test connection (for debugging)
console.log("WebSocket test function available: testWebSocketConnection()");