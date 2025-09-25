import redis
import psycopg2
from psycopg2.extras import RealDictCursor
import json
import numpy as np
from datetime import datetime, date
from decimal import Decimal
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class RedisVectorCacheManager:
    """Redis를 사용한 벡터 캐싱 매니저 (수정된 버전)"""

    def __init__(self, redis_url: str, postgres_url: str, default_ttl: int = 1800):
        self.redis_url = redis_url
        self.postgres_url = postgres_url
        self.default_ttl = default_ttl
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.postgres_conn = None

    def connect_postgres(self):
        """PostgreSQL 연결"""
        try:
            self.postgres_conn = psycopg2.connect(
                self.postgres_url,
                client_encoding='utf8',
                sslmode='require',
                cursor_factory=RealDictCursor
            )
            logger.debug("PostgreSQL 연결 성공")
        except Exception as e:
            logger.error(f"PostgreSQL 연결 실패: {e}")
            raise

    def disconnect_postgres(self):
        """PostgreSQL 연결 해제"""
        if self.postgres_conn:
            self.postgres_conn.close()
            self.postgres_conn = None

    def serialize_for_redis(self, data):
        """Redis 저장을 위한 JSON 직렬화 (datetime 처리)"""
        def convert_types(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, date):
                return obj.isoformat()
            elif isinstance(obj, Decimal):
                return float(obj)
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            else:
                return str(obj)

        if isinstance(data, dict):
            return {k: convert_types(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [convert_types(item) for item in data]
        else:
            return convert_types(data)

    def safe_json_dumps(self, data):
        """안전한 JSON 직렬화"""
        try:
            return json.dumps(self.serialize_for_redis(data), ensure_ascii=False)
        except Exception as e:
            logger.warning(f"JSON 직렬화 오류: {e}")
            # 최후의 수단: 문제 있는 필드 제거
            if isinstance(data, dict):
                clean_data = {}
                for k, v in data.items():
                    try:
                        json.dumps(v)
                        clean_data[k] = v
                    except:
                        clean_data[k] = str(v)  # 문자열로 변환
                return json.dumps(clean_data, ensure_ascii=False)
            else:
                return json.dumps(str(data), ensure_ascii=False)

    def parse_vector_string(self, vector_str):
        """문자열 형태의 벡터를 리스트로 변환"""
        try:
            # JSON 파싱 시도
            if isinstance(vector_str, str) and vector_str.startswith('['):
                return json.loads(vector_str)
        except:
            pass

        try:
            # ast.literal_eval 시도
            import ast
            return ast.literal_eval(vector_str)
        except:
            pass

        # 이미 리스트라면 그대로 반환
        if isinstance(vector_str, list):
            return vector_str

        raise ValueError(f"벡터 파싱 실패: {str(vector_str)[:100]}...")

    def cache_vector(self, pos: int, vector: List[float], ttl: Optional[int] = None) -> bool:
        """벡터를 Redis에 캐싱 (수정된 버전)"""
        try:
            # 벡터가 문자열이면 파싱
            if isinstance(vector, str):
                vector = self.parse_vector_string(vector)

            # 벡터 유효성 검사
            if not isinstance(vector, list) or len(vector) != 1024:
                logger.error(f"잘못된 벡터 형식 (pos: {pos}): {type(vector)}, 길이: {len(vector) if isinstance(vector, list) else 'N/A'}")
                return False

            key = f"vector:{pos}"
            ttl = ttl or self.default_ttl

            # JSON으로 직렬화하여 저장
            vector_json = json.dumps(vector)
            success = self.redis_client.setex(key, ttl, vector_json)

            if success:
                logger.debug(f"벡터 캐싱 성공 (pos: {pos})")
                return True
            else:
                logger.error(f"벡터 캐싱 실패 (pos: {pos})")
                return False

        except Exception as e:
            logger.error(f"벡터 캐싱 실패 (pos: {pos}): {e}")
            return False

    def cache_product(self, pos: int, product_data: Dict, ttl: Optional[int] = None) -> bool:
        """상품 정보를 Redis에 캐싱 (수정된 버전)"""
        try:
            key = f"product:{pos}"
            ttl = ttl or self.default_ttl

            # 안전한 JSON 직렬화 사용
            product_json = self.safe_json_dumps(product_data)
            success = self.redis_client.setex(key, ttl, product_json)

            if success:
                logger.debug(f"상품 캐싱 성공 (pos: {pos})")
                return True
            else:
                logger.error(f"상품 캐싱 실패 (pos: {pos})")
                return False

        except Exception as e:
            logger.error(f"상품 정보 캐싱 실패 (pos: {pos}): {e}")
            return False

    def get_cached_vector(self, pos: int) -> Optional[List[float]]:
        """캐시된 벡터 조회"""
        try:
            key = f"vector:{pos}"
            cached_data = self.redis_client.get(key)

            if cached_data:
                vector = json.loads(cached_data)
                if isinstance(vector, list) and len(vector) == 1024:
                    return vector
                else:
                    logger.warning(f"캐시된 벡터 형식 오류 (pos: {pos})")

        except Exception as e:
            logger.error(f"벡터 캐시 조회 실패 (pos: {pos}): {e}")

        return None

    def get_cached_product(self, pos: int) -> Optional[Dict]:
        """캐시된 상품 정보 조회"""
        try:
            key = f"product:{pos}"
            cached_data = self.redis_client.get(key)

            if cached_data:
                return json.loads(cached_data)

        except Exception as e:
            logger.error(f"상품 캐시 조회 실패 (pos: {pos}): {e}")

        return None

    def warm_up_cache(self, product_ids: List[int], batch_size: int = 50) -> Dict:
        """캐시 워밍업 (수정된 버전)"""
        if not self.postgres_conn:
            self.connect_postgres()

        total = len(product_ids)
        vector_success = 0
        product_success = 0
        errors = 0

        try:
            with self.postgres_conn.cursor() as cur:
                for i in range(0, total, batch_size):
                    batch = product_ids[i:i + batch_size]

                    try:
                        # 벡터 데이터 조회
                        vector_sql = "SELECT pos, dense_vector FROM embeddings3 WHERE pos = ANY(%s)"
                        cur.execute(vector_sql, (batch,))
                        vectors = cur.fetchall()

                        for row in vectors:
                            pos = row['pos']
                            vector_data = row['dense_vector']

                            try:
                                # 벡터 데이터 파싱 및 캐싱
                                if isinstance(vector_data, str):
                                    vector = self.parse_vector_string(vector_data)
                                elif isinstance(vector_data, list):
                                    vector = vector_data
                                else:
                                    logger.warning(f"알 수 없는 벡터 형식 (pos: {pos}): {type(vector_data)}")
                                    errors += 1
                                    continue

                                if self.cache_vector(pos, vector):
                                    vector_success += 1
                                else:
                                    errors += 1

                            except Exception as e:
                                logger.error(f"벡터 처리 실패 (pos: {pos}): {e}")
                                errors += 1

                        # 상품 데이터 조회
                        product_sql = "SELECT * FROM products3 WHERE pos = ANY(%s)"
                        cur.execute(product_sql, (batch,))
                        products = cur.fetchall()

                        for row in products:
                            pos = row['pos']
                            product_data = dict(row)

                            if self.cache_product(pos, product_data):
                                product_success += 1
                            else:
                                errors += 1

                    except Exception as e:
                        logger.error(f"배치 처리 실패 ({i}-{i+batch_size}): {e}")
                        errors += batch_size

                    # 진행률 로깅
                    processed = min(i + batch_size, total)
                    logger.info(f"캐시 워밍업 진행: {processed}/{total}")

        except Exception as e:
            logger.error(f"캐시 워밍업 실패: {e}")

        stats = {
            'vectors': vector_success,
            'products': product_success,
            'errors': errors
        }

        logger.info(f"캐시 워밍업 완료: 벡터 {vector_success}개, 상품 {product_success}개")
        return stats

    def get_cache_stats(self) -> Dict:
        """캐시 통계 조회"""
        try:
            info = self.redis_client.info()
            return {
                'redis_version': info.get('redis_version'),
                'used_memory_human': info.get('used_memory_human'),
                'connected_clients': info.get('connected_clients'),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'total_keys': len(self.redis_client.keys('*'))
            }
        except Exception as e:
            logger.error(f"캐시 통계 조회 실패: {e}")
            return {}
