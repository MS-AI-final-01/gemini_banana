from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import re

# Robust gender detectors (English word-boundary safe; Korean keyword safe)
RE_UNISEX = re.compile(r"(?:\buni(?:sex)?\b|男女|공용|유니섹스|남녀|남여|공용/유니섹스|all\s*genders?)", re.I)
RE_KIDS = re.compile(r"(?:\bkid(?:s)?\b|\bchild(?:ren)?\b|\byouth\b|\bjunior\b|boys?\s*&\s*girls?|아동|키즈)", re.I)
RE_FEMALE = re.compile(r"(?:\bwomen\b|\bwoman\b|\bfemale\b|\bladies\b|\blady\b|\bgirls?\b|여성|여자|우먼)", re.I)
RE_MALE = re.compile(r"(?:\bmen\b|\bman\b|\bmale\b|\bboys?\b|\bmens\b|\bman's\b|\bmans\b|남성|남자|맨)", re.I)

try:
    import faiss
    from sqlalchemy import create_engine, text  # type: ignore
    from sqlalchemy.engine import Engine  # type: ignore
except Exception:  # Optional dependency
    faiss = None  # type: ignore
    create_engine = None  # type: ignore
    text = None  # type: ignore
    class Engine:  # type: ignore
        pass


@dataclass
class DbConfig:
    host: str = os.getenv("DB_HOST", "")
    port: int = int(os.getenv("DB_PORT", "5432"))
    name: str = os.getenv("DB_NAME", "")
    user: str = os.getenv("DB_USER", "")
    password: str = os.getenv("DB_PASSWORD", "")
    sslmode: str = os.getenv("DB_SSLMODE", "require")

    @property
    def url(self) -> str:
        # Allow hard-disable via env flag without editing DB_* vars
        flag = os.getenv("DB_RECO_ENABLED", "").strip().lower()
        # Default remains enabled when flag is unset; only explicit false/0/off disables
        if flag in {"0", "false", "off", "no"}:
            return ""
        if not (self.host and self.user):
            return ""
        return (
            f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}?sslmode={self.sslmode}"
        )


def _normalize_slot(raw: Optional[str]) -> str:
    c = (str(raw or "").strip().lower())
    if not c:
        return "unknown"
    
    # DB 카테고리 매핑만 사용
    if c in ["man_outer", "woman_outer"]:
        return "outer"
    elif c in ["man_top", "woman_top"]:
        return "top"
    elif c in ["man_bottom", "woman_bottom"]:
        return "pants"
    elif c in ["man_shoes", "woman_shoes"]:
        return "shoes"
    elif c == "woman_dress_skirt":
        return "pants"  # 드레스/스커트는 하의로 분류
    
    # 알 수 없는 카테고리는 그대로 반환
    return c


def _normalize_gender(raw: Optional[str]) -> str:
    g = str(raw or "").strip()
    if not g:
        return "unknown"
    # 언더스코어/대시/슬래시 등을 공백으로 치환해 단어 경계 인식 강화
    g = re.sub(r"[_\-\/]+", " ", g)

    # 공용/키즈 우선 판정
    if RE_UNISEX.search(g):
        return "unisex"
    if RE_KIDS.search(g):
        return "kids"

    # 단어 경계 사용으로 'women' 안의 'men' 오탐 방지
    female = bool(RE_FEMALE.search(g))
    male = bool(RE_MALE.search(g))
    if female and male:
        return "unisex"
    if female:
        return "female"
    if male:
        return "male"
    return g.lower()


def _parse_price(price_raw: Optional[str]) -> int:
    """가격 문자열을 정수로 안전하게 파싱"""
    if not price_raw:
        return 0
    
    # 문자열로 변환
    price_str = str(price_raw).strip()
    
    # 숫자가 아닌 문자 제거 (쉼표, 원, 공백 등)
    price_clean = re.sub(r"[^0-9]", "", price_str)
    
    # 숫자가 없으면 0 반환
    if not price_clean:
        return 0
    
    try:
        return int(price_clean)
    except ValueError:
        return 0


class IndexOnlyRecommender:
    """
    DB 기반 하이브리드 추천 시스템 (BGE-M3 + ColBERT MaxSim)
    - Dense: DB에서 로드한 BGE-M3 dense 벡터 → FAISS (IP)
    - Rerank: DB에서 로드한 ColBERT 벡터로 MaxSim 계산
    - 최종 점수 = w_dense * Dense + w_maxsim * MaxSim (후보 집합 내 min-max 정규화 후 결합)
    - DB에서 벡터화된 데이터를 로드하여 메모리에 캐싱 (FlagEmbedding 모델 불필요)
    """

    def __init__(
        self,
        cfg: Optional[DbConfig] = None,
        w_dense: float = 0.5,
        w_maxsim: float = 0.5
    ):
        self.logger = logging.getLogger(__name__)
        self.cfg = cfg or DbConfig()
        self.w_dense = w_dense
        self.w_maxsim = w_maxsim
        
        # DB 연결
        self.engine: Optional[Engine] = None
        
        # 데이터 저장소
        self.dense_index = None
        self.metadata = None
        self.dense_vectors = None
        self.colbert_data = {}
        self.colbert_loaded = False
        self.colbert_dim = None
        self.colbert_cache = {}
        self.cache_limit = 1000
        
        # DB 연결 및 초기화
        if self.cfg.url and create_engine is not None and text is not None:
            try:
                self.engine = create_engine(
                    self.cfg.url,
                    pool_pre_ping=True,
                    connect_args={
                        "keepalives": 1,
                        "keepalives_idle": 30,
                        "keepalives_interval": 10,
                        "keepalives_count": 5,
                        "connect_timeout": int(os.getenv("DB_CONNECT_TIMEOUT", "5")),
                    },
                )
                with self.engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                self.logger.info("[IndexOnlyRecommender] DB 연결 성공")
                self._load_all()
            except Exception as exc:
                self.logger.exception("[IndexOnlyRecommender] 초기화 실패: %s", exc)
                self.engine = None

    def _load_all(self) -> None:
        """DB에서 모든 데이터 로드"""
        try:
            self._load_products_from_db()
            self._load_embeddings_from_db()
            self._build_faiss_index()
            self._build_colbert_index()
            self.logger.info("[IndexOnlyRecommender] DB에서 모든 데이터 로드 완료")
        except Exception as e:
            self.logger.error(f"[IndexOnlyRecommender] DB 데이터 로드 실패: {e}")
            raise

    def _load_products_from_db(self) -> None:
        """DB에서 상품 메타데이터 로드"""
        assert self.engine is not None and text is not None
        
        with self.engine.begin() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT pos,
                           "Product_U",
                           "Product_img_U",
                           "Product_N",
                           "Product_Desc",
                           "Product_P",
                           "Category",
                           "Product_B",
                           "Product_G",
                           "Image_P"
                    FROM public.products2
                    ORDER BY pos ASC
                    """
                )
            ).mappings().all()
        
        # pandas DataFrame으로 변환
        data = []
        for r in rows:
            data.append({
                "pos": int(r.get("pos")),
                "Product_U": r.get("Product_U"),
                "Product_img_U": r.get("Product_img_U"),
                "Product_N": r.get("Product_N"),
                "Product_Desc": r.get("Product_Desc"),
                "Product_P": _parse_price(r.get("Product_P")),
                "Category": r.get("Category"),
                "Product_B": r.get("Product_B"),
                "Product_G": r.get("Product_G"),
                "Image_P": r.get("Image_P")
            })
        
        self.metadata = pd.DataFrame(data)
        self.logger.info(f"[IndexOnlyRecommender] 상품 메타데이터 로드: {len(self.metadata):,}개 상품")

    def _load_embeddings_from_db(self) -> None:
        """DB에서 embeddings2의 모든 데이터 로드"""
        assert self.engine is not None and text is not None
        
        with self.engine.begin() as conn:
            data = conn.execute(
                text(
                    """
                    SELECT pos, dense, colbert_vecs, colbert_offsets
                    FROM public.embeddings2
                    WHERE dense IS NOT NULL
                    ORDER BY pos ASC
                    """
                )
            ).all()
        
        if not data:
            self.logger.error("[IndexOnlyRecommender] No embeddings2 data found")
            return
        
        # Dense 벡터들 추출
        dense_vectors = []
        colbert_vecs_list = []
        colbert_offsets_list = []
        
        for row in data:
            pos, dense_bytes, colbert_vecs_bytes, colbert_offsets_bytes = row
            
            # Dense 벡터
            if dense_bytes:
                dense_vec = np.frombuffer(dense_bytes, dtype=np.float32)
                dense_vectors.append(dense_vec)
            else:
                dense_vectors.append(None)
            
            # ColBERT 벡터들
            if colbert_vecs_bytes:
                colbert_vecs_1d = np.frombuffer(colbert_vecs_bytes, dtype=np.float32)
                # 1차원을 2차원으로 reshape (토큰 수 x 차원)
                # DB에 저장된 형태에 따라 적절히 reshape
                if len(colbert_vecs_1d) > 0:
                    # 일반적으로 ColBERT는 80토큰 x 1024차원 형태
                    # 80 * 1024 = 81920이므로 이를 기준으로 reshape 시도
                    if len(colbert_vecs_1d) % 1024 == 0:
                        num_tokens = len(colbert_vecs_1d) // 1024
                        colbert_vecs = colbert_vecs_1d.reshape(num_tokens, 1024)
                    else:
                        # 1024로 나누어떨어지지 않으면 1차원 그대로 사용
                        colbert_vecs = colbert_vecs_1d.reshape(-1, 1)
                    colbert_vecs_list.append(colbert_vecs)
                else:
                    colbert_vecs_list.append(None)
            else:
                colbert_vecs_list.append(None)
            
            # ColBERT 오프셋
            if colbert_offsets_bytes:
                offsets = np.frombuffer(colbert_offsets_bytes, dtype=np.int64)
                colbert_offsets_list.append(offsets)
            else:
                colbert_offsets_list.append(None)
        
        # Dense 벡터들을 numpy 배열로 변환
        valid_dense = [v for v in dense_vectors if v is not None]
        if valid_dense:
            self.dense_vectors = np.vstack(valid_dense)
            self.logger.info(f"[IndexOnlyRecommender] Dense 벡터 로드: {self.dense_vectors.shape}")
        else:
            self.dense_vectors = None
            self.logger.warning("[IndexOnlyRecommender] No valid dense vectors found")
        
        # ColBERT 데이터를 상품별로 저장 (개별 접근을 위해)
        self.colbert_data = {}
        for i, (vecs, offsets) in enumerate(zip(colbert_vecs_list, colbert_offsets_list)):
            if vecs is not None and offsets is not None and len(offsets) >= 2:
                # 각 상품의 ColBERT 벡터들을 저장
                self.colbert_data[i] = {
                    'vecs': vecs,
                    'start_offset': int(offsets[0]),
                    'end_offset': int(offsets[1])
                }
        
        self.logger.info(f"[IndexOnlyRecommender] ColBERT 데이터 로드: {len(self.colbert_data)}개 상품")
        
        # 디버깅: 첫 번째 벡터의 shape 확인
        if self.colbert_data:
            first_key = next(iter(self.colbert_data.keys()))
            first_vecs = self.colbert_data[first_key]['vecs']
            self.logger.info(f"[DEBUG] 첫 번째 ColBERT 벡터 shape: {first_vecs.shape}, dtype: {first_vecs.dtype}")
            
            # ColBERT 벡터 통계
            total_tokens = sum(len(data['vecs']) for data in self.colbert_data.values())
            self.logger.info(f"[DEBUG] ColBERT 총 토큰 수: {total_tokens}, 상품 수: {len(self.colbert_data)}")


    def _build_faiss_index(self) -> None:
        """Dense 벡터들로 FAISS 인덱스 구축"""
        if self.dense_vectors is None:
            self.logger.error("[IndexOnlyRecommender] Dense vectors not loaded")
            return
        
        # FAISS 인덱스 생성 (Inner Product)
        dimension = self.dense_vectors.shape[1]
        self.dense_index = faiss.IndexFlatIP(dimension)
        
        # 벡터들을 인덱스에 추가
        self.dense_index.add(self.dense_vectors.astype(np.float32))
        
        self.logger.info(f"[IndexOnlyRecommender] FAISS 인덱스 구축: {self.dense_index.ntotal:,}개 벡터, d={dimension}")

    def _build_colbert_index(self) -> None:
        """ColBERT 인덱스 구축 (DB 기반)"""
        if not hasattr(self, 'colbert_data') or not self.colbert_data:
            self.logger.warning("[IndexOnlyRecommender] ColBERT 데이터가 없어서 인덱스를 구축할 수 없습니다")
            return
        
        # 첫 번째 유효한 벡터에서 차원 확인
        first_vecs = next(iter(self.colbert_data.values()))['vecs']
        
        # 벡터 차원 안전하게 확인
        if first_vecs.ndim < 2:
            self.logger.warning("[IndexOnlyRecommender] ColBERT 벡터가 2차원이 아닙니다. shape: %s", first_vecs.shape)
            self.colbert_loaded = False
            return
        
        self.colbert_dim = first_vecs.shape[1]
        
        self.colbert_loaded = True
        self.colbert_cache = {}
        self.cache_limit = 1000
        
        self.logger.info(f"[IndexOnlyRecommender] ColBERT 인덱스: {len(self.colbert_data)}개 상품 (d={self.colbert_dim})")

    def available(self) -> bool:
        """추천 시스템 사용 가능 여부"""
        return (
            self.engine is not None and
            self.dense_index is not None and 
            self.metadata is not None
        )

    def _minmax(self, x: np.ndarray) -> np.ndarray:
        """Min-max 정규화"""
        x = np.asarray(x, dtype="float32")
        if x.size == 0:
            return x
        x_min, x_max = float(np.min(x)), float(np.max(x))
        if x_max - x_min < 1e-12:
            return np.zeros_like(x, dtype="float32")
        return (x - x_min) / (x_max - x_min)


    def _get_doc_tokens(self, product_id: int) -> Optional[np.ndarray]:
        """상품의 ColBERT 토큰 벡터들 가져오기 (DB 기반)"""
        if product_id in self.colbert_cache:
            return self.colbert_cache[product_id]
        
        if product_id in self.colbert_data:
            vecs = self.colbert_data[product_id]['vecs']
            
            # 캐시 관리
            if len(self.colbert_cache) >= self.cache_limit:
                self.colbert_cache.pop(next(iter(self.colbert_cache)))
            
            self.colbert_cache[product_id] = vecs
            return vecs
        
        return None

    def compute_maxsim(self, q_tok: np.ndarray, d_tok: np.ndarray, per_token_mode: str = "raw") -> float:
        """ColBERT MaxSim 계산"""
        if q_tok.size == 0 or d_tok.size == 0:
            return 0.0
        
        if q_tok.ndim != 2 or d_tok.ndim != 2 or q_tok.shape[1] != d_tok.shape[1]:
            raise RuntimeError("MaxSim 입력 차원 오류")
        
        # 정규화
        q = (q_tok.T / (np.linalg.norm(q_tok, axis=1) + 1e-8)).T
        d = (d_tok.T / (np.linalg.norm(d_tok, axis=1) + 1e-8)).T
        
        # 유사도 계산
        sims = np.dot(q, d.T)  # (Q, L)
        per_tok = np.max(sims, axis=1)  # (Q,)
        
        if per_token_mode == "relu":
            per_tok = np.maximum(per_tok, 0.0)
        elif per_token_mode == "shift01":
            per_tok = (per_tok + 1.0) * 0.5
        elif per_token_mode != "raw":
            raise ValueError("per_token_mode ∈ {raw,relu,shift01}")
        
        return float(per_tok.sum())
    
    @staticmethod
    def _minmax_norm(x: np.ndarray) -> np.ndarray:
        """Min-Max 정규화"""
        x = np.asarray(x, dtype="float32")
        if x.size == 0:
            return x
        x_min, x_max = float(np.min(x)), float(np.max(x))
        if x_max - x_min < 1e-12:
            return np.zeros_like(x, dtype="float32")
        return (x - x_min) / (x_max - x_min)
    
    def combine_scores(self, dense_scores, maxsim_scores, w_dense=0.5, w_maxsim=0.5, use_norm=True):
        """Dense와 ColBERT 점수를 결합"""
        d = np.asarray(dense_scores, dtype="float32")
        if maxsim_scores is None:
            d_used = self._minmax_norm(d) if use_norm else d
            m_used = np.zeros_like(d_used, dtype="float32")
            combined = w_dense * d_used
            return combined, d_used, m_used
        
        m = np.asarray(maxsim_scores, dtype="float32")
        finite_m = np.isfinite(m)
        
        if use_norm:
            d_used = self._minmax_norm(d)
            m_used = np.zeros_like(m, dtype="float32")
            if finite_m.any():
                m_used[finite_m] = self._minmax_norm(m[finite_m])
        else:
            d_used = d
            m_used = np.zeros_like(m, dtype="float32")
            if finite_m.any():
                m_used[finite_m] = m[finite_m]
        
        combined = w_dense * d_used + w_maxsim * m_used
        return combined, d_used, m_used

    def recommend_indices(
        self,
        self_idx: int,
        top_n: int = 5,
        cand_k: int = 50,
        same_category: bool = True,
        w_dense: Optional[float] = None,
        w_maxsim: Optional[float] = None
    ) -> List[int]:
        """인덱스 기반 추천"""
        if not self.available():
            raise RuntimeError("IndexOnlyRecommender unavailable")
        
        w_dense = w_dense or self.w_dense
        w_maxsim = w_maxsim or self.w_maxsim
        
        # 자기 자신 제외
        exclude_ids = {self_idx}
        
        # 카테고리 필터링
        target_category = None
        if same_category and 0 <= self_idx < len(self.metadata):
            target_category = str(self.metadata.iloc[self_idx].get("Category", ""))
        
        # self_idx로 DB에서 직접 벡터 가져오기
        if 0 <= self_idx < len(self.dense_vectors):
            query_vec = self.dense_vectors[self_idx].reshape(1, -1)
        else:
            raise ValueError(f"Invalid self_idx: {self_idx}")
        
        # Dense 검색 (쿼리 벡터로 직접 검색)
        dense_scores, cand = self.dense_index.search(query_vec, cand_k)
        dense_scores = dense_scores[0]
        cand_ids = cand[0].astype(int)
        
        # 자기 자신 제외
        keep_mask = np.array([pid not in exclude_ids for pid in cand_ids], dtype=bool)
        cand_ids = cand_ids[keep_mask]
        dense_scores = dense_scores[keep_mask]
        
        if cand_ids.size == 0:
            return []
        
        # MaxSim 재랭킹 (쿼리 상품의 ColBERT 벡터 사용)
        maxsim_scores = None
        if self.colbert_loaded and self_idx in self.colbert_data:
            q_tok = self.colbert_data[self_idx]['vecs']
            
            maxsim_list = []
            for pid in cand_ids:
                doc_tok = self._get_doc_tokens(pid)
                if doc_tok is None or doc_tok.size == 0:
                    maxsim_list.append(float("-inf"))
                else:
                    maxsim_list.append(self.compute_maxsim(q_tok, doc_tok))
            
            maxsim_scores = np.array(maxsim_list, dtype="float32")
        
        # 점수 결합
        if maxsim_scores is not None:
            dense_norm = self._minmax(dense_scores)
            maxsim_norm = self._minmax(maxsim_scores)
            combined = w_dense * dense_norm + w_maxsim * maxsim_norm
        else:
            combined = dense_scores
        
        # 정렬 및 상위 선택
        order = np.argsort(combined)[::-1]
        final_ids = cand_ids[order][:top_n]
        final_scores = combined[order][:top_n]
        
        return list(zip(final_ids.tolist(), final_scores.tolist()))

    def recommend(
        self,
        positions: List[int],
        *,
        top_k: int = 5,
        w_dense: Optional[float] = None,
        w_maxsim: Optional[float] = None,
        same_category: bool = True
    ) -> List[Dict]:
        """위치 기반 추천 (기존 API 호환)"""
        if not self.available():
            raise RuntimeError("IndexOnlyRecommender unavailable")
        
        if not positions:
            return []
        
        # 각 위치에 대해 개별 검색 수행
        all_recommendations = []
        for pos in positions:
            if 0 <= pos < len(self.metadata):
                rec_results = self.recommend_indices(
                    self_idx=pos,
                    top_n=top_k,
                    cand_k=50,
                    same_category=same_category,
                    w_dense=w_dense,
                    w_maxsim=w_maxsim
                )
                
                for rank, (idx, score) in enumerate(rec_results, 1):
                    if 0 <= idx < len(self.metadata):
                        row = self.metadata.iloc[idx]
                        product = {
                            "id": str(idx),
                            "pos": int(idx),
                            "title": str(row.get("Product_N", "")),
                            "price": int(row.get("Product_P", 0)),
                            "tags": [str(row.get("Product_B", "")), str(row.get("Product_G", ""))],
                            "category": _normalize_slot(row.get("Category", "")),
                            "gender": _normalize_gender(row.get("Product_G", "")),
                            "imageUrl": str(row.get("Product_img_U", "")),
                            "productUrl": str(row.get("Product_U", "")),
                            "score": float(score)  # 실제 추천 점수
                        }
                        all_recommendations.append(product)
        
        # 중복 제거 및 상위 k개 선택
        seen = set()
        unique_recommendations = []
        for rec in all_recommendations:
            if rec["pos"] not in seen:
                seen.add(rec["pos"])
                unique_recommendations.append(rec)
                if len(unique_recommendations) >= top_k:
                    break
        
        return unique_recommendations

    def recommend_by_embedding(
        self,
        query_embedding: List[float],
        *,
        query_colbert: Optional[List[List[float]]] = None,
        category: Optional[str] = None,
        top_k: int = 5,
        w_dense: Optional[float] = None,
        w_maxsim: Optional[float] = None
    ) -> List[Dict]:
        """임베딩 기반 추천 (Dense + ColBERT 하이브리드)"""
        if not self.available():
            raise RuntimeError("IndexOnlyRecommender unavailable")
        
        w_dense = w_dense or self.w_dense
        w_maxsim = w_maxsim or self.w_maxsim
        
        # 쿼리 벡터를 FAISS에 직접 검색
        query_vec = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        
        # 더 많은 후보를 가져와서 ColBERT 재랭킹 수행
        dense_scores, cand = self.dense_index.search(query_vec, top_k * 10)  # 더 많은 후보
        dense_scores = dense_scores[0]
        cand_ids = cand[0].astype(int)
        
        # ColBERT 재랭킹 (제공된 경우)
        if query_colbert is not None and self.colbert_loaded:
            query_colbert_vec = np.array(query_colbert, dtype=np.float32)
            maxsim_scores = []
            
            for cand_id in cand_ids:
                if cand_id in self.colbert_data:
                    doc_tokens = self.colbert_data[cand_id]['vecs']
                    if doc_tokens.size > 0:
                        maxsim_score = self.compute_maxsim(query_colbert_vec, doc_tokens)
                        maxsim_scores.append(maxsim_score)
                    else:
                        maxsim_scores.append(float('-inf'))
                else:
                    maxsim_scores.append(float('-inf'))
            
            maxsim_scores = np.array(maxsim_scores, dtype=np.float32)
            
            # Dense + ColBERT 점수 결합
            combined, dense_used, maxsim_used = self.combine_scores(
                dense_scores, maxsim_scores, w_dense, w_maxsim, use_norm=True
            )
        else:
            # ColBERT가 없으면 Dense 점수만 사용
            combined = dense_scores
        
        # 정렬
        order = np.argsort(combined)[::-1]
        final_ids = cand_ids[order][:top_k]
        
        # 결과 생성
        results = []
        for rank, idx in enumerate(final_ids, 1):
            if 0 <= idx < len(self.metadata):
                row = self.metadata.iloc[idx]
                
                # 카테고리 필터링
                if category:
                    product_category = _normalize_slot(row.get("Category", ""))
                    if product_category != category:
                        continue
                
                product = {
                    "id": str(idx),
                    "pos": int(idx),
                    "title": str(row.get("Product_N", "")),
                    "price": int(row.get("Product_P", 0)),
                    "tags": [str(row.get("Product_B", "")), str(row.get("Product_G", ""))],
                    "category": _normalize_slot(row.get("Category", "")),
                    "gender": _normalize_gender(row.get("Product_G", "")),
                    "imageUrl": str(row.get("Product_img_U", "")),
                    "productUrl": str(row.get("Product_U", "")),
                    "score": float(combined[rank - 1])
                }
                results.append(product)
        
        return results


class DbPosRecommender:
    """
    DB-backed recommender. Loads products and embeddings into memory (NumPy) and
    performs cosine similarity + price weighting, similar to PosRecommender.
    Tables expected:
      public.products2(pos, "Product_U", "Product_Desc", "Product_P", "Category")
      public.embeddings2(pos, dense BYTEA) - uses dense column only for legacy compatibility
    """

    def __init__(self, cfg: Optional[DbConfig] = None) -> None:
        self.cfg = cfg or DbConfig()
        self.logger = logging.getLogger(__name__)
        self.engine: Optional[Engine] = None
        self.products: List[Dict] = []
        self.emb: Optional[np.ndarray] = None
        self.emb_norm: Optional[np.ndarray] = None
        self.prices: Optional[np.ndarray] = None

        if self.cfg.url and create_engine is not None and text is not None:
            try:
                self.engine = create_engine(
                    self.cfg.url,
                    pool_pre_ping=True,
                    connect_args={
                        # Keep TCP healthy
                        "keepalives": 1,
                        "keepalives_idle": 30,
                        "keepalives_interval": 10,
                        "keepalives_count": 5,
                        # Fast fail when DB is unreachable (seconds)
                        "connect_timeout": int(os.getenv("DB_CONNECT_TIMEOUT", "5")),
                    },
                )
                with self.engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                self.logger.info("[DbPosRecommender] Connected to DB host=%s db=%s", self.cfg.host, self.cfg.name)
                self._load_all()
                if self.available():
                    self.logger.info(
                        "[DbPosRecommender] Loaded %d products and embeddings", len(self.products)
                    )
                else:
                    self.logger.warning("[DbPosRecommender] Loaded data but recommender marked unavailable")
            except Exception as exc:
                self.logger.exception("[DbPosRecommender] Initialization failed: %s", exc)
                # Leave unavailable; route will fall back
                self.engine = None

    def _load_all(self) -> None:
        assert self.engine is not None and text is not None
        # Load products from products2 (updated to use new table)
        with self.engine.begin() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT pos,
                           "Product_U",
                           "Product_img_U",
                           "Product_N",
                           "Product_Desc",
                           "Product_P",
                           "Category",
                           "Product_B",
                           "Product_G",
                           "Image_P"
                    FROM public.products2
                    ORDER BY pos ASC
                    """
                )
            ).mappings().all()
        self.products = []
        for r in rows:
            title = r.get("Product_N") or r.get("Product_Desc") or ""
            brand = r.get("Product_B")
            gender_raw = r.get("Product_G")
            tags: List[str] = []
            if brand:
                tags.append(str(brand))
            if gender_raw:
                tags.append(str(gender_raw))
            image_url = r.get("Product_img_U") or r.get("Image_P") or None
            product_url = r.get("Product_U")
            category_raw = r.get("Category")
            norm_cat = _normalize_slot(category_raw)
            gender_norm = _normalize_gender(gender_raw)
            self.products.append(
                {
                    "id": str(r.get("pos")),
                    "pos": int(r.get("pos")),
                    "title": str(title),
                    "price": _parse_price(r.get("Product_P")),
                    "tags": tags,
                    "category": norm_cat,
                    "gender": gender_norm,
                    "imageUrl": image_url,
                    "productUrl": product_url,
                }
            )

        # Load embeddings from embeddings2 (dense column only for legacy compatibility)
        with self.engine.begin() as conn:
            data = conn.execute(
                text(
                    """
                    SELECT pos, dense
                    FROM public.embeddings2
                    WHERE dense IS NOT NULL
                    ORDER BY pos ASC
                    """
                )
            ).all()
        
        if not data:
            self.logger.error("[DbPosRecommender] No embeddings2 data found")
            self.products = []
            self.emb = None
            return
        
        # Convert BYTEA to numpy array
        dense_vectors = []
        for row in data:
            if row[1]:  # dense column is not None
                dense_vec = np.frombuffer(row[1], dtype=np.float32)
                dense_vectors.append(dense_vec)
        
        if not dense_vectors:
            self.logger.error("[DbPosRecommender] No valid dense vectors found")
            self.products = []
            self.emb = None
            return
        
        mat = np.vstack(dense_vectors)

        # Sanity check
        if len(self.products) != mat.shape[0]:
            # mismatch: mark unavailable
            self.logger.error(
                "[DbPosRecommender] Product/embedding count mismatch: products=%d, embeddings_rows=%d",
                len(self.products),
                mat.shape[0],
            )
            self.products = []
            self.emb = None
            self.emb_norm = None
            self.prices = None
            return

        self.emb = mat.astype(np.float32, copy=False)
        norms = np.linalg.norm(self.emb, axis=1)
        norms[norms == 0] = 1e-8
        self.emb_norm = self.emb / norms[:, None]
        self.prices = np.array([p["price"] for p in self.products], dtype=np.float32)

    def available(self) -> bool:
        return self.emb_norm is not None and len(self.products) > 0

    def _calculate_similarity_scores(
        self, 
        query_vec: np.ndarray, 
        *, 
        alpha: float = 0.38, 
        w1: float = 0.97, 
        w2: float = 0.03
    ) -> np.ndarray:
        """
        코사인 유사도 + 가격 가중치 계산 공통 함수
        
        Args:
            query_vec: 정규화된 쿼리 벡터
            alpha: 가격 가중치 파라미터
            w1: 유사도 가중치
            w2: 가격 가중치
            
        Returns:
            np.ndarray: 최종 점수 배열
        """
        emb_norm = self.emb_norm  # type: ignore[assignment]
        prices = self.prices  # type: ignore[assignment]
        
        # 코사인 유사도 계산
        sim = emb_norm @ query_vec
        
        # 가격 가중치 계산
        avg_price = float(prices.mean())
        clog = np.log1p(prices)
        qlog = np.log1p(avg_price)
        price_score = np.exp(-alpha * np.abs(clog - qlog))
        
        # 최종 점수 계산
        total = w1 * sim + w2 * price_score
        
        return total

    def recommend(
        self,
        positions: List[int],
        *,
        top_k: int = 5,
        alpha: float = 0.38,
        w1: float = 0.97,
        w2: float = 0.03,
    ) -> List[Dict]:
        if not self.available():
            raise RuntimeError("DbPosRecommender unavailable")

        n = len(self.products)
        if any(p < 0 or p >= n for p in positions):
            raise ValueError("positions out of range")

        k = max(1, min(int(top_k), n))
        emb_norm = self.emb_norm  # type: ignore[assignment]

        # 쿼리 벡터 생성 및 정규화
        q = emb_norm[positions].mean(axis=0)
        qn = np.linalg.norm(q)
        if qn == 0:
            qn = 1e-8
        q = q / qn

        # 공통 함수로 점수 계산
        total = self._calculate_similarity_scores(q, alpha=alpha, w1=w1, w2=w2)
        total[np.array(positions, dtype=int)] = -np.inf

        if k >= n:
            top_idx = np.argsort(-total)
        else:
            part = np.argpartition(-total, kth=k - 1)[:k]
            top_idx = part[np.argsort(-total[part])]

        out: List[Dict] = []
        for i in top_idx.tolist():
            p = dict(self.products[i])
            p["score"] = float(total[i])
            out.append(p)
        return out

    def recommend_by_embedding(
        self,
        query_embedding: List[float],
        *,
        category: Optional[str] = None,
        top_k: int = 5,
        alpha: float = 0.38,
        w1: float = 0.97,
        w2: float = 0.03,
    ) -> List[Dict]:
        """
        외부 이미지에서 생성된 임베딩 벡터로 추천
        
        Args:
            query_embedding: 쿼리 임베딩 벡터
            category: 카테고리 필터 (top, pants, shoes, outer)
            top_k: 반환할 추천 개수
            alpha: 가격 가중치 파라미터
            w1: 유사도 가중치
            w2: 가격 가중치
            
        Returns:
            List[Dict]: 추천 아이템 리스트
        """
        if not self.available():
            raise RuntimeError("DbPosRecommender unavailable")

        n = len(self.products)
        k = max(1, min(int(top_k), n))

        # 쿼리 벡터 정규화
        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            query_norm = 1e-8
        query_vec = query_vec / query_norm

        # 공통 함수로 점수 계산
        total = self._calculate_similarity_scores(query_vec, alpha=alpha, w1=w1, w2=w2)

        # 카테고리 필터링
        if category:
            category_indices = []
            for i, product in enumerate(self.products):
                product_category = _normalize_slot(product.get("category", ""))
                if product_category == category:
                    category_indices.append(i)
            
            if not category_indices:
                return []
            
            # 카테고리 필터링된 인덱스로 점수 재계산
            filtered_total = np.full(n, -np.inf)
            for idx in category_indices:
                filtered_total[idx] = total[idx]
            total = filtered_total

        # 상위 k개 선택
        if k >= n:
            top_idx = np.argsort(-total)
        else:
            part = np.argpartition(-total, kth=k - 1)[:k]
            top_idx = part[np.argsort(-total[part])]

        out: List[Dict] = []
        for i in top_idx.tolist():
            if total[i] == -np.inf:  # 필터링된 항목 건너뛰기
                continue
            p = dict(self.products[i])
            p["score"] = float(total[i])
            out.append(p)
        
        return out


_flag = os.getenv("DB_RECO_ENABLED", "").strip().lower()
# Enabled by default unless explicitly disabled via env
_db_enabled = False if _flag in {"0", "false", "off", "no"} else True
db_pos_recommender = DbPosRecommender() if _db_enabled else DbPosRecommender(DbConfig(host="", user=""))

# IndexOnlyRecommender 인스턴스 생성 (DB 기반)
if _db_enabled:
    try:
        index_only_recommender = IndexOnlyRecommender()
    except Exception as e:
        logging.getLogger(__name__).warning(f"IndexOnlyRecommender 초기화 실패: {e}")
        index_only_recommender = None
else:
    index_only_recommender = None
