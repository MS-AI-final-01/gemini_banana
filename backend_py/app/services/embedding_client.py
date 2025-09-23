"""
임베딩 서버 HTTP 클라이언트
VM에서 실행되는 임베딩 서버와 통신
"""
import os
import logging
from typing import List, Optional, Dict, Any
import httpx


class EmbeddingClient:
    """
    임베딩 서버와 통신하는 HTTP 클라이언트
    """
    
    def __init__(self, base_url: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.base_url = base_url or os.getenv("EMBEDDING_SERVER_URL", "http://localhost:8001")
        self.timeout = 60.0
        
    def available(self) -> bool:
        """
        임베딩 서버 사용 가능 여부 확인
        """
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.base_url}/health")
                return response.status_code == 200
        except Exception as e:
            self.logger.warning(f"Embedding server health check failed: {e}")
            return False
    
    def get_embedding(self, text: str) -> List[float]:
        """
        텍스트를 임베딩 벡터로 변환
        
        Args:
            text: 변환할 텍스트
            
        Returns:
            List[float]: 임베딩 벡터
            
        Raises:
            RuntimeError: 임베딩 서버 오류 시
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/embed",
                    json={"text": text.strip()}
                )
                response.raise_for_status()
                
                data = response.json()
                embedding = data.get("embedding")
                
                if not embedding or not isinstance(embedding, list):
                    raise RuntimeError("Invalid embedding response format")
                
                return embedding
                
        except httpx.HTTPStatusError as e:
            self.logger.error(f"Embedding server HTTP error: {e.response.status_code} - {e.response.text}")
            raise RuntimeError(f"Embedding server HTTP error: {e.response.status_code}")
            
        except httpx.TimeoutException:
            self.logger.error("Embedding server timeout")
            raise RuntimeError("Embedding server timeout")
            
        except Exception as e:
            self.logger.error(f"Embedding server error: {e}")
            raise RuntimeError(f"Embedding server error: {e}")
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        여러 텍스트를 배치로 임베딩 벡터로 변환
        
        Args:
            texts: 변환할 텍스트 리스트
            
        Returns:
            List[List[float]]: 임베딩 벡터 리스트
        """
        if not texts:
            return []
        
        # 현재는 순차 처리 (배치 API가 없을 경우)
        embeddings = []
        for text in texts:
            try:
                embedding = self.get_embedding(text)
                embeddings.append(embedding)
            except Exception as e:
                self.logger.warning(f"Failed to get embedding for text '{text[:50]}...': {e}")
                # 실패한 경우 빈 벡터로 대체 (길이는 다른 임베딩과 맞춰야 함)
                if embeddings:
                    embeddings.append([0.0] * len(embeddings[0]))
                else:
                    # 첫 번째가 실패한 경우 기본 길이 사용
                    embeddings.append([0.0] * 1024)  # bge-m3 기본 길이
        
        return embeddings
    
    def get_colbert_embedding(self, text: str) -> Dict[str, Any]:
        """
        텍스트를 Dense + ColBERT 벡터로 변환
        
        Args:
            text: 변환할 텍스트
            
        Returns:
            Dict[str, Any]: {
                "dense_embedding": List[float],
                "colbert_embeddings": List[List[float]],
                "model_name": str,
                "text_length": int,
                "num_tokens": int
            }
            
        Raises:
            RuntimeError: 임베딩 서버 오류 시
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/embed/colbert",
                    json={"text": text.strip()}
                )
                response.raise_for_status()
                
                data = response.json()
                
                # 응답 검증
                required_fields = ["dense_embedding", "colbert_embeddings", "model_name", "text_length", "num_tokens"]
                for field in required_fields:
                    if field not in data:
                        raise RuntimeError(f"Missing field in ColBERT response: {field}")
                
                return data
                
        except httpx.HTTPStatusError as e:
            self.logger.error(f"ColBERT embedding server HTTP error: {e.response.status_code} - {e.response.text}")
            raise RuntimeError(f"ColBERT embedding server HTTP error: {e.response.status_code}")
            
        except httpx.TimeoutException:
            self.logger.error("ColBERT embedding server timeout")
            raise RuntimeError("ColBERT embedding server timeout")
            
        except Exception as e:
            self.logger.error(f"ColBERT embedding server error: {e}")
            raise RuntimeError(f"ColBERT embedding server error: {e}")
    
    def get_server_info(self) -> dict:
        """
        임베딩 서버 정보 조회
        
        Returns:
            dict: 서버 정보
        """
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.base_url}/health")
                response.raise_for_status()
                return response.json()
        except Exception as e:
            self.logger.error(f"Failed to get server info: {e}")
            return {"error": str(e)}


# 전역 인스턴스
embedding_client = EmbeddingClient()
