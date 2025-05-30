from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from app.webrtc_handler import router as webrtc_router
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Cheating Detection System", version="1.0.0")

# Enable CORS for all domains (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory for JS, CSS
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include WebSocket router WITHOUT prefix (critical for WebSocket routing)
app.include_router(webrtc_router)

logger.info("✅ FastAPI app configured with WebSocket routes")

# Serve the frontend HTML page at the root URL
@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the main HTML page"""
    try:
        with open("templates/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        logger.error("❌ templates/index.html not found")
        return HTMLResponse(content="<h1>Error: Template file not found</h1>", status_code=500)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {
        "status": "healthy",
        "message": "Cheating Detection System is running",
        "routes_registered": True
    }

# Debug endpoint to list all routes
@app.get("/debug/routes")
async def list_routes():
    """Debug endpoint to list all registered routes"""
    routes = []
    for route in app.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            routes.append({
                "path": route.path,
                "methods": list(route.methods) if route.methods else ["WebSocket"]
            })
    return {"routes": routes}

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting Cheating Detection System...")
    logger.info("📱 WebSocket endpoint available at: /ws")
    logger.info("🌐 Frontend available at: http://localhost:8000")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 Shutting down Cheating Detection System...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)