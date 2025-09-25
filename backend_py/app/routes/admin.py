from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Optional
import logging
import os

# 환경 변수에서 직접 읽기
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "false").lower() == "true"

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])

# 캐시 관련 모듈 안전하게 import
try:
    from ..services.cache_manager import CacheService
    CACHE_SERVICE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"캐시 서비스 모듈 로드 실패: {e}")
    CACHE_SERVICE_AVAILABLE = False

@router.get("/cache/status")
async def get_cache_status():
    """캐시 시스템 상태 확인"""
    if not CACHE_ENABLED:
        return {
            "status": "disabled",
            "message": "캐시가 환경 변수에서 비활성화되어 있습니다",
            "cache_enabled": False,
            "service_available": CACHE_SERVICE_AVAILABLE
        }

    if not CACHE_SERVICE_AVAILABLE:
        return {
            "status": "unavailable",
            "message": "캐시 서비스 모듈을 로드할 수 없습니다",
            "cache_enabled": CACHE_ENABLED,
            "service_available": False
        }

    try:
        cache_manager = CacheService.get_instance()
        if cache_manager:
            stats = cache_manager.get_cache_stats()
            return {
                "status": "active",
                "cache_stats": stats,
                "cache_enabled": True,
                "service_available": True
            }
        else:
            return {
                "status": "failed",
                "message": "캐시 매니저 초기화 실패",
                "cache_enabled": CACHE_ENABLED,
                "service_available": True
            }
    except Exception as e:
        logger.error(f"캐시 상태 조회 실패: {e}")
        return {
            "status": "error",
            "message": f"캐시 상태 조회 실패: {str(e)}",
            "cache_enabled": CACHE_ENABLED,
            "service_available": CACHE_SERVICE_AVAILABLE
        }

if CACHE_SERVICE_AVAILABLE:
    @router.post("/cache/warm-up")
    async def warm_up_cache(
        product_ids: List[int],
        batch_size: int = Query(100, ge=10, le=500, description="배치 크기")
    ):
        """
        캐시 워밍업 - 인기 상품들을 미리 캐시에 로딩
        """
        if not CACHE_ENABLED:
            raise HTTPException(status_code=400, detail="캐시가 비활성화되어 있습니다")

        cache_manager = CacheService.get_instance()
        if not cache_manager:
            raise HTTPException(status_code=400, detail="캐시 매니저를 사용할 수 없습니다")

        if len(product_ids) > 1000:
            raise HTTPException(status_code=400, detail="한 번에 최대 1000개 상품만 워밍업 가능합니다")

        try:
            logger.info(f"캐시 워밍업 시작: {len(product_ids)}개 상품")
            stats = cache_manager.warm_up_cache(product_ids, batch_size=batch_size)

            return {
                "status": "success",
                "message": f"캐시 워밍업 완료",
                "total_products": len(product_ids),
                "batch_size": batch_size,
                "stats": stats
            }
        except Exception as e:
            logger.error(f"캐시 워밍업 실패: {e}")
            raise HTTPException(status_code=500, detail=f"캐시 워밍업 실패: {str(e)}")

else:
    @router.post("/cache/warm-up")
    async def warm_up_cache_unavailable():
        raise HTTPException(
            status_code=503,
            detail="캐시 서비스를 사용할 수 없습니다. Redis 모듈을 설정하세요."
        )
