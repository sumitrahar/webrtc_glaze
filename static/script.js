// script.js

// DOM elements
const videoElem = document.getElementById("webcam-video");
const toggleBtn = document.getElementById("toggle-webcam");

// WebSocket variable
let ws = null;
// MediaStream from webcam
let stream = null;
// Flag to track if webcam is on
let webcamOn = false;

// Turn on webcam: request media, set video src, open websocket and start sending frames
async function startWebcam() {
    try {
        // Request webcam video stream
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        videoElem.srcObject = stream;
        await videoElem.play();

        // Open WebSocket connection to backend
        ws = new WebSocket("ws://localhost:8000/ws/video");

        ws.onopen = () => {
            console.log("WebSocket connected");
            // Start sending frames periodically
            sendFrames();
        };

        ws.onclose = () => {
            console.log("WebSocket disconnected");
        };

        ws.onerror = (err) => {
            console.error("WebSocket error:", err);
        };

        webcamOn = true;
        toggleBtn.textContent = "Turn Off Webcam";

    } catch (err) {
        alert("Could not access webcam: " + err.message);
    }
}

// Stop webcam: stop video stream tracks, close websocket, update UI
function stopWebcam() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }

    if (ws) {
        ws.close();
        ws = null;
    }

    videoElem.pause();
    videoElem.srcObject = null;

    webcamOn = false;
    toggleBtn.textContent = "Turn On Webcam";
}

// Capture current video frame, convert to JPEG base64, send over websocket
function sendFrames() {
    if (!webcamOn || ws.readyState !== WebSocket.OPEN) return;

    // Create a canvas to draw current video frame
    const canvas = document.createElement("canvas");
    canvas.width = videoElem.videoWidth;
    canvas.height = videoElem.videoHeight;
    const ctx = canvas.getContext("2d");

    // Draw current frame
    ctx.drawImage(videoElem, 0, 0, canvas.width, canvas.height);

    // Get base64 jpeg string
    const base64Image = canvas.toDataURL("image/jpeg");

    // Send frame over websocket
    ws.send(base64Image);

    // Repeat every 100ms (10 FPS)
    setTimeout(sendFrames, 100);
}

// Toggle button click handler
toggleBtn.addEventListener("click", () => {
    if (webcamOn) {
        stopWebcam();
    } else {
        startWebcam();
    }
});