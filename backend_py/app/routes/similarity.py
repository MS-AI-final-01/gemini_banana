import asyncio
from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Optional
import logging
import time
import os

# 환경 변수에서 직접 읽기 (config 모듈 없이)
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "false").lower() == "true"

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/similarity", tags=["similarity"])

# 캐시 관련 모듈 안전하게 import
try:
    from ..services.vector_search import VectorSearchService
    VECTOR_SEARCH_AVAILABLE = True
except ImportError as e:
    logger.warning(f"벡터 검색 모듈 로드 실패: {e}")
    VECTOR_SEARCH_AVAILABLE = False

@router.get("/status")
async def similarity_status():
    """유사도 검색 시스템 상태 확인"""
    return {
        "status": "active",
        "cache_enabled": CACHE_ENABLED,
        "vector_search_available": VECTOR_SEARCH_AVAILABLE,
        "message": "유사도 검색 API가 준비되었습니다"
    }

if VECTOR_SEARCH_AVAILABLE:
    @router.post("/search")
    async def vector_similarity_search(
        query_vector: List[float],
        limit: int = Query(10, ge=1, le=100, description="검색 결과 개수"),
        use_cache: bool = Query(True, description="캐시 사용 여부")
    ) -> Dict:
        """
        벡터 유사도 검색
        - Redis 캐시 우선 검색
        - 캐시 미스 시 PostgreSQL 검색
        - 결과를 Redis에 캐싱
        """
        start_time = time.time()

        # 입력 검증
        if not query_vector:
            raise HTTPException(status_code=400, detail="query_vector가 필요합니다")

        if len(query_vector) != 1024:
            raise HTTPException(
                status_code=400,
                detail=f"벡터 차원이 올바르지 않습니다. 예상: 1024, 실제: {len(query_vector)}"
            )

        try:
            # 벡터 검색 서비스 사용
            vector_service = VectorSearchService()
            search_result = await vector_service.search_similar_products(
                query_vector=query_vector,
                limit=limit,
                use_cache=use_cache and CACHE_ENABLED
            )

            # 응답 포맷팅
            total_time = (time.time() - start_time) * 1000

            return {
                "status": "success",
                "query_info": {
                    "vector_dimension": len(query_vector),
                    "result_limit": limit,
                    "cache_enabled": use_cache and CACHE_ENABLED,
                    "total_response_time_ms": f"{total_time:.1f}"
                },
                "search_result": search_result
            }

        except Exception as e:
            logger.error(f"벡터 검색 실패: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"벡터 검색 중 오류가 발생했습니다: {str(e)}"
            )

    @router.post("/search/legacy")
    async def vector_similarity_search_legacy(
        query_vector: List[float],
        limit: int = Query(10, ge=1, le=100)
    ) -> Dict:
        """
        기존 벡터 검색 (캐시 미사용)
        PostgreSQL 직접 검색
        """
        return await vector_similarity_search(
            query_vector=query_vector,
            limit=limit,
            use_cache=False
        )

else:
    @router.post("/search")
    async def vector_similarity_search_unavailable():
        raise HTTPException(
            status_code=503,
            detail="벡터 검색 서비스를 사용할 수 없습니다. Redis 캐싱 모듈을 설정하세요."
        )
