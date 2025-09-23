from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import re

# Robust gender detectors (English word-boundary safe; Korean keyword safe)
RE_UNISEX = re.compile(r"(?:\buni(?:sex)?\b|男女|공용|유니섹스|남녀|남여|공용/유니섹스|all\s*genders?)", re.I)
RE_KIDS = re.compile(r"(?:\bkid(?:s)?\b|\bchild(?:ren)?\b|\byouth\b|\bjunior\b|boys?\s*&\s*girls?|아동|키즈)", re.I)
RE_FEMALE = re.compile(r"(?:\bwomen\b|\bwoman\b|\bfemale\b|\bladies\b|\blady\b|\bgirls?\b|여성|여자|우먼)", re.I)
RE_MALE = re.compile(r"(?:\bmen\b|\bman\b|\bmale\b|\bboys?\b|\bmens\b|\bman's\b|\bmans\b|남성|남자|맨)", re.I)

try:
    from sqlalchemy import create_engine, text  # type: ignore
    from sqlalchemy.engine import Engine  # type: ignore
except Exception:  # Optional dependency
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


class DbPosRecommender:
    """
    DB-backed recommender. Loads products and embeddings into memory (NumPy) and
    performs cosine similarity + price weighting, similar to PosRecommender.
    Tables expected:
      public.products(pos, "Product_U", "Product_Desc", "Product_P", "Category")
      public.embeddings(pos, col_0.. or value JSON/array)
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
        # Load products
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
                    FROM public.products
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
                    "price": int(r.get("Product_P") or 0),
                    "tags": tags,
                    "category": norm_cat,
                    "gender": gender_norm,
                    "imageUrl": image_url,
                    "productUrl": product_url,
                }
            )

        # Load embeddings (supports col_0.. or value)
        with self.engine.begin() as conn:
            cols = [c[0] for c in conn.execute(
                text(
                    """
                    SELECT column_name FROM information_schema.columns
                    WHERE table_schema='public' AND table_name='embeddings'
                    ORDER BY ordinal_position
                    """
                )
            ).all()]

        vector_cols = [c for c in cols if c.startswith("col_")]
        if vector_cols:
            with self.engine.begin() as conn:
                col_list = ", ".join(['pos'] + vector_cols)
                data = conn.execute(text(f"SELECT {col_list} FROM public.embeddings ORDER BY pos ASC")).all()
            mat = np.array([list(row)[1:] for row in data], dtype=np.float32)
        else:
            with self.engine.begin() as conn:
                data = conn.execute(text('SELECT pos, "value" FROM public.embeddings ORDER BY pos ASC')).all()
            # Assume DB driver returns Python list/JSON for value
            mat = np.array([np.array(row[1], dtype=np.float32) for row in data], dtype=np.float32)

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
