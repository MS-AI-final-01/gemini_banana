from typing import Optional
import logging
import os

logger = logging.getLogger(__name__)

# 임시로 환경 변수에서 직접 읽어오기 (redis_config 없이)
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "false").lower() == "true"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
POSTGRES_URL = os.getenv("POSTGRES_URL", "")
CACHE_DEFAULT_TTL = int(os.getenv("CACHE_DEFAULT_TTL", "1800"))

class CacheService:
    """
    캐시 서비스 싱글톤 매니저 (임시 버전)
    Redis 관련 모듈을 선택적으로 로드
    """
    _instance = None

    @classmethod
    def get_instance(cls):
        """캐시 매니저 인스턴스 반환"""
        if not CACHE_ENABLED:
            logger.info("캐시가 비활성화되어 있습니다")
            return None

        # Redis 모듈이 없으면 None 반환
        try:
            import redis
        except ImportError:
            logger.warning("redis 패키지가 설치되지 않았습니다. 캐시 기능 비활성화")
            return None

        # redis_vector_cache 모듈이 없으면 None 반환
        try:
            from ..utils.redis_vector_cache import RedisVectorCacheManager
        except ImportError:
            logger.warning("redis_vector_cache 모듈이 없습니다. 캐시 기능 비활성화")
            return None

        if cls._instance is None:
            try:
                cls._instance = RedisVectorCacheManager(
                    redis_url=REDIS_URL,
                    postgres_url=POSTGRES_URL,
                    default_ttl=CACHE_DEFAULT_TTL
                )
                cls._instance.connect_postgres()
                logger.info("Redis 캐시 매니저 초기화 완료")
            except Exception as e:
                logger.error(f"캐시 매니저 초기화 실패: {e}")
                cls._instance = None

        return cls._instance

    @classmethod
    def close_connections(cls):
        """연결 종료"""
        if cls._instance:
            try:
                cls._instance.disconnect_postgres()
                logger.info("캐시 매니저 연결 종료 완료")
            except Exception as e:
                logger.error(f"연결 종료 중 오류: {e}")
            finally:
                cls._instance = None

    @classmethod
    def is_enabled(cls) -> bool:
        """캐시 활성화 상태 확인"""
        return CACHE_ENABLED and cls.get_instance() is not None
