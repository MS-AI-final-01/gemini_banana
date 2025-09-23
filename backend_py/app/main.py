from contextlib import asynccontextmanager
from fastapi import FastAPI
import os
import logging
from .settings import settings
from .routes.health import router as health_router
from .routes.api import router as api_router
from .routes.generate import router as generate_router
from .routes.recommend import router as recommend_router
from .routes.recommend_external import router as recommend_external_router
from .routes.recommend_positions import router as recommend_positions_router
from .routes.proxy import router as proxy_router
from .routes.tips import router as tips_router
from .routes.tryon_video import router as tryon_video_router
from .routes.evaluate import router as evaluate_router
from .routes.search import router as search_router
from fastapi.middleware.cors import CORSMiddleware
from .middleware.logging import LoggingMiddleware

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Application startup completed")
    yield
    # Shutdown
    logger.info("Application shutdown")

app = FastAPI(title="AI Virtual Try-On API (Python)", version="1.0.0", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL, "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register custom middleware
app.add_middleware(LoggingMiddleware)

# Routers
app.include_router(health_router)
app.include_router(api_router)
app.include_router(generate_router)
app.include_router(recommend_router)
app.include_router(recommend_external_router)
app.include_router(recommend_positions_router)
app.include_router(proxy_router)
app.include_router(tips_router)
app.include_router(evaluate_router)
app.include_router(tryon_video_router)
app.include_router(search_router)

@app.get("/")
def root():
    return {"message": "AI Virtual Try-On (Python)"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.NODE_ENV != "production",
    )
