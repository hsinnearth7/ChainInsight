"""ChainInsight Live — FastAPI application entry point."""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.api.routes import trigger_pipeline_from_path
from app.config import BASE_DIR, CORS_ORIGINS, RAW_DIR
from app.db.models import init_db
from app.ws.routes import ws_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Track watchdog observer for shutdown
_watchdog_observer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown events."""
    global _watchdog_observer
    logger.info("ChainInsight Live starting up...")
    init_db()
    logger.info("Database initialized")

    # Start watchdog file monitor
    try:
        from app.watcher import start_watcher

        loop = asyncio.get_running_loop()

        def on_csv_detected(file_path: str):
            """Watchdog callback — bridge sync thread to async pipeline trigger."""
            logger.info("Watchdog: new CSV detected — %s", file_path)
            asyncio.run_coroutine_threadsafe(
                trigger_pipeline_from_path(file_path), loop
            )

        _watchdog_observer = start_watcher(
            watch_dir=str(RAW_DIR),
            callback=on_csv_detected,
            debounce_seconds=2.0,
        )
        logger.info("Watchdog file monitor started — watching %s", RAW_DIR)
    except ImportError:
        logger.warning("watchdog not installed — file monitoring disabled")
    except Exception:
        logger.exception("Failed to start watchdog")

    yield

    # Shutdown
    if _watchdog_observer is not None:
        _watchdog_observer.stop()
        _watchdog_observer.join(timeout=5)
        logger.info("Watchdog stopped")
    logger.info("ChainInsight Live shutting down...")


app = FastAPI(
    title="ChainInsight Live",
    description="Real-time supply chain inventory analytics platform",
    version="3.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["X-API-Key", "Content-Type"],
)

@app.get("/api/health")
def health_check():
    """Health check endpoint (no auth required)."""
    return {"status": "ok", "name": "ChainInsight Live", "version": "3.0.0"}

# Register WebSocket routes first (before API and static mounts)
app.include_router(ws_router)

# Register API routes
app.include_router(api_router)


# Mount React SPA (production build) as catch-all — must be last
_frontend_dist = BASE_DIR / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="spa")
    logger.info("React SPA mounted from %s", _frontend_dist)
else:
    @app.get("/")
    def root():
        return {
            "name": "ChainInsight Live",
            "version": "3.0.0",
            "docs": "/docs",
            "status": "running",
            "note": "React frontend not built yet. Run 'cd frontend && npm run build'",
        }
