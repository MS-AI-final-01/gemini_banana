from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
import os
import logging
import time

# 기존 import 유지
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

# Redis 캐싱 관련 import (안전하게 처리)
REDIS_AVAILABLE = False
try:
    from .services.cache_manager import CacheService
    REDIS_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("Redis 캐싱 모듈 로드 성공")
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Redis 캐싱 모듈 로드 실패: {e}")
    REDIS_AVAILABLE = False

from fastapi.middleware.cors import CORSMiddleware
from .middleware.logging import LoggingMiddleware

# Configure logging (기존 설정 유지)
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'DEBUG').upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 라이프사이클 관리 (Redis 캐싱 추가)"""
    # Startup
    logger.info("Application startup started")

    try:
        # Redis 캐시 매니저 초기화 (안전하게 처리)
        if REDIS_AVAILABLE:
            cache_enabled = os.getenv("CACHE_ENABLED", "false").lower() == "true"
            if cache_enabled:
                cache_manager = CacheService.get_instance()
                if cache_manager:
                    logger.info("Redis 캐시 시스템 활성화")

                    # 인기 상품 캐시 워밍업 (선택적)
                    try:
                        popular_products = list(range(1, 101))  # 첫 100개 상품
                        stats = cache_manager.warm_up_cache(popular_products, batch_size=50)
                        logger.info(f"캐시 워밍업 완료: {stats}")
                    except Exception as e:
                        logger.warning(f"캐시 워밍업 실패 (무시 가능): {e}")
                else:
                    logger.warning("Redis 캐시 매니저 초기화 실패")
            else:
                logger.info("Redis 캐시가 비활성화되어 있습니다 (CACHE_ENABLED=false)")
        else:
            logger.info("Redis 캐싱 모듈이 로드되지 않았습니다")

        logger.info("Application startup completed")

        yield

    finally:
        # Shutdown
        logger.info("Application shutdown started")

        # Redis 연결 정리 (안전하게 처리)
        if REDIS_AVAILABLE:
            try:
                CacheService.close_connections()
                logger.info("캐시 연결 정리 완료")
            except Exception as e:
                logger.warning(f"캐시 연결 정리 중 오류 (무시 가능): {e}")

        logger.info("Application shutdown completed")

# FastAPI 앱 생성
app = FastAPI(
    title="AI Virtual Try-On API (Python)" + (" with Redis Cache" if REDIS_AVAILABLE else ""),
    version="2.0.0" if REDIS_AVAILABLE else "1.0.0",
    lifespan=lifespan
)

# 응답 시간 미들웨어 추가 (성능 모니터링용)
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(f"{process_time * 1000:.1f}ms")
    return response

# CORS (기존 설정 유지)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL, "http://localhost:5173", "http://localhost:5174", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register custom middleware (기존 유지)
app.add_middleware(LoggingMiddleware)

# 기존 라우터들 포함 (모두 유지)
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

# Redis 캐싱 라우터 추가 (안전하게 처리)
if REDIS_AVAILABLE:
    try:
        from .routes.similarity import router as similarity_router
        from .routes.admin import router as admin_router

        app.include_router(similarity_router, prefix="/api/v1")
        app.include_router(admin_router, prefix="/api/v1")
        logger.info("Redis 캐싱 라우터 등록 완료")
    except ImportError as e:
        logger.warning(f"Redis 캐싱 라우터 등록 실패: {e}")

@app.get("/")
def root():
    """루트 엔드포인트"""
    cache_status = "disabled"

    if REDIS_AVAILABLE:
        cache_enabled = os.getenv("CACHE_ENABLED", "false").lower() == "true"
        if cache_enabled:
            try:
                cache_manager = CacheService.get_instance()
                cache_status = "active" if cache_manager else "failed"
            except:
                cache_status = "error"

    endpoints = {
        "health": "/health",
        "docs": "/docs"
    }

    if REDIS_AVAILABLE and cache_status == "active":
        endpoints.update({
            "similarity_search": "/api/v1/similarity/search",
            "cache_admin": "/api/v1/admin/cache/stats"
        })

    return {
        "message": "AI Virtual Try-On (Python)" + (" with Redis Cache" if REDIS_AVAILABLE else ""),
        "version": "2.0.0" if REDIS_AVAILABLE else "1.0.0",
        "cache_status": cache_status,
        "redis_available": REDIS_AVAILABLE,
        "endpoints": endpoints
    }

@app.get("/cache/status")
def cache_status():
    """캐시 상태 확인 엔드포인트"""
    if not REDIS_AVAILABLE:
        return {"status": "unavailable", "message": "Redis 모듈이 로드되지 않았습니다"}

    cache_enabled = os.getenv("CACHE_ENABLED", "false").lower() == "true"
    if not cache_enabled:
        return {"status": "disabled", "message": "캐시가 비활성화되어 있습니다 (CACHE_ENABLED=false)"}

    try:
        cache_manager = CacheService.get_instance()
        if cache_manager:
            return {
                "status": "active",
                "message": "캐시가 정상 작동 중입니다",
                "redis_connected": True
            }
        else:
            return {
                "status": "failed",
                "message": "캐시 초기화 실패",
                "redis_connected": False
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"캐시 상태 확인 실패: {str(e)}",
            "redis_connected": False
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.NODE_ENV != "production",
    )
