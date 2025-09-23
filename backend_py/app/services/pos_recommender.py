from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .catalog import get_catalog_service
from .db_recommender import IndexOnlyRecommender


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_EMBED_PATH = ROOT_DIR / "data" / "embeddings.npy"


class PosRecommender:
    """
    Lightweight, file-based recommender using precomputed item embeddings (NxD float32).
    Avoids heavy deps (pandas/scikit/psycopg) – only numpy.
    """

    def __init__(self) -> None:
        # config
        self.embed_path = Path(os.getenv("POS_REC_EMBEDDINGS_PATH", str(DEFAULT_EMBED_PATH)))
        # state
        self._emb: Optional[np.ndarray] = None
        self._emb_norm: Optional[np.ndarray] = None
        self._prices: Optional[np.ndarray] = None
        self._count: int = 0
        self._dim: int = 0

        self._load_if_available()

    def _load_if_available(self) -> None:
        try:
            if not self.embed_path.exists():
                return
            emb = np.load(self.embed_path)
            if not isinstance(emb, np.ndarray):
                return
            if emb.dtype != np.float32:
                emb = emb.astype(np.float32, copy=False)
            self._emb = emb
            self._count, self._dim = emb.shape[0], int(emb.shape[1])
            # precompute L2 norms
            norms = np.linalg.norm(emb, axis=1)
            norms[norms == 0] = 1e-8
            self._emb_norm = emb / norms[:, None]

            # align with catalog prices (assumes same ordering by pos)
            catalog = get_catalog_service().get_all()
            if len(catalog) != self._count:
                # length mismatch – mark unavailable
                self._emb = None
                self._emb_norm = None
                self._prices = None
                self._count = 0
                return
            prices = np.array([int(p.get("price", 0)) for p in catalog], dtype=np.float32)
            self._prices = prices
        except Exception:
            self._emb = None
            self._emb_norm = None
            self._prices = None
            self._count = 0

    def available(self) -> bool:
        return self._emb_norm is not None and self._prices is not None and self._count > 0

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
            raise RuntimeError("PosRecommender not available (missing embeddings or catalog mismatch)")

        # validate
        n = self._count
        if any(p < 0 or p >= n for p in positions):
            raise ValueError("positions out of range")
        k = max(1, min(int(top_k), n))

        emb_norm = self._emb_norm  # type: ignore[assignment]
        prices = self._prices  # type: ignore[assignment]

        # query embedding (mean of selected)
        q = emb_norm[positions].mean(axis=0)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            q_norm = 1e-8
        q = q / q_norm

        # cosine similarity via dot with normalized vectors
        sim = emb_norm @ q  # shape (N,)

        # price score
        qprice = float(prices[positions].mean())
        clog = np.log1p(prices)
        qlog = np.log1p(qprice)
        price_score = np.exp(-alpha * np.abs(clog - qlog))

        # weighted total
        total = w1 * sim + w2 * price_score
        total[np.array(positions, dtype=int)] = -np.inf  # exclude query items

        # top-k
        if k >= n:
            top_idx = np.argsort(-total)
        else:
            part = np.argpartition(-total, kth=k - 1)[:k]
            top_idx = part[np.argsort(-total[part])]

        # map to internal RecommendationItem-like dicts
        catalog = get_catalog_service().get_all()
        out: List[Dict] = []
        for idx in top_idx.tolist():
            p = catalog[idx]
            out.append({
                "id": str(p.get("id")),
                "title": p.get("title") or "",
                "price": int(p.get("price", 0)),
                "tags": p.get("tags") or [],
                "category": p.get("category") or "top",
                "imageUrl": p.get("imageUrl"),
                "productUrl": p.get("productUrl"),
                "score": float(total[idx]),
            })
        return out


@lru_cache(maxsize=1)
def get_pos_recommender() -> IndexOnlyRecommender:
    """
    IndexOnlyRecommender를 반환하는 함수 (기존 PosRecommender 대신)
    """
    try:
        return IndexOnlyRecommender()
    except Exception as e:
        # IndexOnlyRecommender 실패 시 기존 PosRecommender로 폴백
        return PosRecommender()

