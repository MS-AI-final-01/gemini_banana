from typing import List, Dict, Optional
import logging
import time
import os

# 환경 변수에서 직접 읽기 (config 모듈 없이)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
POSTGRES_URL = os.getenv("POSTGRES_URL", "")

logger = logging.getLogger(__name__)

# 캐시 매니저 안전하게 import
try:
    from .cache_manager import CacheService
    CACHE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"캐시 매니저 import 실패: {e}")
    CACHE_AVAILABLE = False

# PostgreSQL 관련 모듈 안전하게 import
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    logger.warning("psycopg2 모듈이 없습니다. PostgreSQL 직접 검색 불가")
    POSTGRES_AVAILABLE = False

class VectorSearchService:
    """
    벡터 검색 서비스
    캐시 우선 검색 후 PostgreSQL 폴백
    """

    def __init__(self):
        self.cache_manager = CacheService.get_instance() if CACHE_AVAILABLE else None
        self.postgres_conn = None

    def _connect_postgres(self):
        """PostgreSQL 직접 연결 (캐시 우회용)"""
        if not POSTGRES_AVAILABLE:
            raise Exception("psycopg2 모듈이 설치되지 않았습니다")

        if not POSTGRES_URL or POSTGRES_URL == "postgresql://username:password@hostname:5432/database?client_encoding=utf8":
            raise Exception("실제 PostgreSQL 연결 문자열이 설정되지 않았습니다")

        if not self.postgres_conn:
            try:
                self.postgres_conn = psycopg2.connect(
                    POSTGRES_URL,
                    client_encoding='utf8',
                    sslmode='require',
                    cursor_factory=RealDictCursor
                )
                logger.debug("PostgreSQL 직접 연결 생성")
            except Exception as e:
                logger.error(f"PostgreSQL 연결 실패: {e}")
                raise

    async def search_similar_products(
        self,
        query_vector: List[float],
        limit: int = 10,
        use_cache: bool = True
    ) -> Dict:
        """
        벡터 유사도 검색 (캐시 우선)
        """
        start_time = time.time()
        source = "unknown"

        try:
            # 캐시 검색 시도
            if use_cache and self.cache_manager:
                try:
                    results = self.cache_manager.smart_vector_search(
                        query_vector=query_vector,
                        limit=limit,
                        use_cache=True,
                        cache_results=True
                    )
                    source = "cache" if results else "cache_miss"

                    if results:
                        response_time = (time.time() - start_time) * 1000
                        logger.info(f"캐시 검색 성공: {len(results)}개 결과 ({response_time:.1f}ms)")

                        return {
                            "results": results,
                            "response_time_ms": f"{response_time:.1f}",
                            "result_count": len(results),
                            "source": source
                        }
                except Exception as e:
                    logger.warning(f"캐시 검색 실패, PostgreSQL 폴백: {e}")
                    source = "cache_error"

            # PostgreSQL 직접 검색 (캐시 실패 시)
            results = await self._direct_postgres_search(query_vector, limit)
            source = "postgres" if source == "unknown" else f"{source}_fallback"

            response_time = (time.time() - start_time) * 1000
            logger.info(f"PostgreSQL 검색 완료: {len(results)}개 결과 ({response_time:.1f}ms)")

            return {
                "results": results,
                "response_time_ms": f"{response_time:.1f}",
                "result_count": len(results),
                "source": source
            }

        except Exception as e:
            logger.error(f"벡터 검색 전체 실패: {e}")
            response_time = (time.time() - start_time) * 1000
            return {
                "results": [],
                "response_time_ms": f"{response_time:.1f}",
                "result_count": 0,
                "source": "error",
                "error": str(e)
            }

    async def _direct_postgres_search(self, query_vector: List[float], limit: int) -> List[Dict]:
        """PostgreSQL 직접 검색"""
        if not POSTGRES_AVAILABLE:
            logger.error("PostgreSQL 검색 불가: psycopg2 모듈 없음")
            return []

        try:
            self._connect_postgres()

            with self.postgres_conn.cursor() as cur:
                # pgvector 코사인 유사도 검색
                search_sql = """
                    SELECT 
                        p.pos,
                        p."Product_N" as product_name,
                        p."Product_B" as brand,
                        p."Product_P" as price,
                        p."Category" as category,
                        p."Product_U" as product_url,
                        p."Product_img_U" as image_url,
                        e.dense_vector <-> %s::vector as distance
                    FROM products3 p
                    JOIN embeddings3 e ON p.pos = e.pos
                    ORDER BY e.dense_vector <-> %s::vector
                    LIMIT %s
                """

                cur.execute(search_sql, (query_vector, query_vector, limit))
                raw_results = cur.fetchall()

                # 결과 포맷팅
                results = []
                for row in raw_results:
                    result = dict(row)
                    result['similarity'] = 1 - float(result['distance'])  # 거리를 유사도로 변환
                    del result['distance']
                    results.append(result)

                return results

        except Exception as e:
            logger.error(f"PostgreSQL 검색 실패: {e}")
            return []

    async def get_product_detail(self, pos: int, use_cache: bool = True) -> Optional[Dict]:
        """상품 상세 정보 조회"""
        if use_cache and self.cache_manager:
            try:
                product = self.cache_manager.get_product_with_cache(pos)
                if product:
                    logger.debug(f"상품 캐시 히트: pos={pos}")
                    return product
            except Exception as e:
                logger.warning(f"상품 캐시 조회 실패: {e}")

        # PostgreSQL 직접 조회
        if not POSTGRES_AVAILABLE:
            return None

        try:
            self._connect_postgres()

            with self.postgres_conn.cursor() as cur:
                cur.execute('SELECT * FROM products3 WHERE pos = %s', (pos,))
                product_data = cur.fetchone()

                if product_data:
                    product_dict = dict(product_data)
                    logger.debug(f"PostgreSQL에서 상품 조회: pos={pos}")
                    return product_dict
                else:
                    return None

        except Exception as e:
            logger.error(f"상품 조회 실패 (pos: {pos}): {e}")
            return None

    def __del__(self):
        """소멸자: 연결 정리"""
        if self.postgres_conn:
            try:
                self.postgres_conn.close()
            except:
                pass
