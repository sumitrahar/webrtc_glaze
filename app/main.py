# # app/main.py

# from fastapi import FastAPI, Request
# from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles
# from starlette.middleware.cors import CORSMiddleware
# from app.webrtc_handler import router as webrtc_router

# # Initialize FastAPI app
# app = FastAPI()

# # Enable CORS for all domains (adjust if needed for security)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Replace with specific origins in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Mount the static directory to serve frontend files (HTML, CSS, JS)
# app.mount("/static", StaticFiles(directory="static"), name="static")

# # Include the WebRTC router for frame processing endpoint
# app.include_router(webrtc_router, prefix="/webrtc")

# # Serve the frontend HTML at the root path
# @app.get("/", response_class=HTMLResponse)
# async def get_index():
#     with open("templates/index.html", "r") as f:
#         html_content = f.read()
#     return HTMLResponse(content=html_content)

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from app.webrtc_handler import router as webrtc_router

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for all domains (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory for JS, CSS
app.mount("/static", StaticFiles(directory="static"), name="static")

# 👇 Include WebSocket router WITHOUT prefix
app.include_router(webrtc_router)

# Serve the frontend HTML page at the root URL
@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("templates/index.html", "r") as f:
        return HTMLResponse(content=f.read())