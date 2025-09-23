from __future__ import annotations

from functools import lru_cache
from typing import List, Optional, Dict, Tuple

import json
import os

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..services.catalog import get_catalog_service
from ..services.azure_openai_service import azure_openai_service

router = APIRouter(prefix="/api/search", tags=["Search"])


@lru_cache(maxsize=1)
def _load_text_index() -> Tuple[Optional[np.ndarray], List[str]]:
    """Load precomputed product text embeddings and ids.

    Returns (embeddings_matrix or None, ids list). If files are missing, embeddings is None.
    """
    try:
        from pathlib import Path

        root = Path(__file__).resolve().parents[3]
        emb_path = root / "data" / "text_embeddings.npy"
        ids_path = root / "data" / "text_ids.json"
        if not emb_path.exists() or not ids_path.exists():
            return None, []
        embs = np.load(str(emb_path))  # shape: (N, D)
        # L2 normalize for cosine similarity
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
        embs = embs / norms
        ids: List[str] = json.loads(ids_path.read_text(encoding="utf-8"))
        if len(ids) != embs.shape[0]:
            # Mismatch; ignore embeddings
            return None, []
        return embs.astype(np.float32), ids
    except Exception:
        return None, []


def _embed_query(text: str) -> Optional[np.ndarray]:
    """Create embedding for the query using OpenAI API if available.
    Falls back to None if not configured.
    """
    try:
        import openai  # type: ignore

        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        if not api_key:
            return None
        client = openai.OpenAI(api_key=api_key)  # type: ignore
        resp = client.embeddings.create(model=model, input=text)
        vec = np.array(resp.data[0].embedding, dtype=np.float32)
        # L2 normalize
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        return vec
    except Exception:
        return None


def _filter_products(products: List[Dict], category: Optional[str], min_price: Optional[int], max_price: Optional[int]) -> List[Dict]:
    out = products
    if category:
        out = [p for p in out if str(p.get("category", "")).lower() == category.lower()]
    if min_price is not None:
        out = [p for p in out if int(p.get("price", 0)) >= int(min_price)]
    if max_price is not None:
        out = [p for p in out if int(p.get("price", 0)) <= int(max_price)]
    return out


@router.get("/semantic")
def semantic_search(
    q: str = Query("", description="Search query text"),
    limit: int = Query(24, ge=1, le=100),
    category: Optional[str] = None,
    minPrice: Optional[int] = None,
    maxPrice: Optional[int] = None,
) -> List[Dict]:
    """Semantic search over catalog.

    Tries vector similarity if precomputed embeddings + API key are available.
    Otherwise falls back to substring scoring via CatalogService.search.
    """
    svc = get_catalog_service()
    all_products = svc.get_all()
    # apply coarse filters first to shrink candidates
    filtered = _filter_products(all_products, category, minPrice, maxPrice)

    # Attempt vector search
    embs, ids = _load_text_index()
    q_vec = _embed_query(q) if q else None
    if embs is not None and q_vec is not None and len(ids) == embs.shape[0] and len(ids) > 0:
        # Map id -> product
        id_to_product = {str(p.get("id")): p for p in all_products}
        # Get candidate indices under current filters (if any), else all
        if filtered is not all_products:
            filtered_ids = set(str(p.get("id")) for p in filtered)
            mask = np.array([1 if pid in filtered_ids else 0 for pid in ids], dtype=bool)
            if mask.any():
                embs_view = embs[mask]
                ids_view = [pid for pid, m in zip(ids, mask) if m]
            else:
                embs_view = embs
                ids_view = ids
        else:
            embs_view = embs
            ids_view = ids

        # cosine similarity = q dot v (after L2 normalize)
        sims = embs_view @ q_vec.reshape(-1, 1)  # (N,1)
        top_k = int(min(limit * 4, sims.shape[0]))
        idx = np.argpartition(-sims.squeeze(1), top_k - 1)[:top_k]
        top_pairs = sorted(((float(sims[i]), ids_view[i]) for i in idx), reverse=True)
        results: List[Dict] = []
        for _score, pid in top_pairs:
            p = id_to_product.get(str(pid))
            if not p:
                continue
            copy = dict(p)
            copy["score"] = _score
            results.append(copy)
            if len(results) >= limit:
                break
        return results

    # Fallback: substring matching
    if not q.strip():
        # empty query -> return recent subset under filters
        return filtered[:limit]
    words = [w for w in q.strip().split() if w]
    res = svc.search(words, max_results=limit * 2, products=filtered)
    return res[:limit]


# ---------------------- Natural language parse ---------------------- #

class ParseRequest(BaseModel):
    text: str


class ParseResponse(BaseModel):
    category: Optional[str] = None
    tokens: List[str] = []
    colors: List[str] = []
    gender: Optional[str] = None
    priceRange: Optional[Dict[str, int]] = None
    source: str = "fallback"


_COLOR_MAP = {
    "black": ["블랙", "검정", "검은", "흑", "black"],
    "white": ["화이트", "하양", "흰", "white"],
    "gray": ["그레이", "회색", "gray", "grey"],
    "beige": ["베이지", "beige"],
    "brown": ["브라운", "갈색", "brown"],
    "navy": ["네이비", "navy"],
    "blue": ["블루", "파랑", "청", "blue"],
    "green": ["그린", "초록", "녹", "green"],
    "red": ["레드", "빨강", "red"],
    "pink": ["핑크", "분홍", "pink"],
    "purple": ["퍼플", "보라", "purple"],
    "yellow": ["옐로우", "노랑", "yellow"],
    "orange": ["오렌지", "주황", "orange"],
}


_CATEGORY_SYNONYMS = {
    "top": [
        "상의", "탑", "티", "티셔츠", "반팔", "긴팔", "맨투맨", "후드",
        "니트", "스웨터", "셔츠", "블라우스", "폴로", "피케", "피케티",
    ],
    "pants": [
        "하의", "바지", "슬랙스", "팬츠", "데님", "청바지", "조거", "트레이닝",
        "치노", "와이드", "테이퍼드", "스커트", "치마", "원피스",
    ],
    "shoes": [
        "신발", "스니커즈", "운동화", "로퍼", "부츠", "샌들", "힐", "구두",
    ],
    "outer": [
        "아우터", "자켓", "재킷", "자켓", "코트", "패딩", "점퍼", "가디건",
        "야상", "블루종", "바람막이",
    ],
    "accessories": [
        "모자", "캡", "비니", "가방", "백팩", "토트", "숄더", "벨트", "시계",
        "양말", "넥타이", "머플러", "액세서리", "주얼리",
    ],
}


def _fallback_parse(text: str) -> ParseResponse:
    t = (text or "").strip().lower()
    resp = ParseResponse()

    # Detect category by synonyms
    for cat, syns in _CATEGORY_SYNONYMS.items():
        for s in syns:
            if s.lower() in t:
                resp.category = cat
                break
        if resp.category:
            break

    # Extract colors
    colors: List[str] = []
    for norm, syns in _COLOR_MAP.items():
        if any(s.lower() in t for s in syns):
            colors.append(norm)
    resp.colors = colors

    # Gender hints
    if any(k in t for k in ["남성", "남자", "man", "male", "남자용", "신사"]):
        resp.gender = "male"
    elif any(k in t for k in ["여성", "여자", "woman", "female", "여자용", "숙녀"]):
        resp.gender = "female"
    elif any(k in t for k in ["공용", "유니섹스", "unisex", "남녀공용"]):
        resp.gender = "unisex"
    elif any(k in t for k in ["키즈", "아동", "kids", "child", "children"]):
        resp.gender = "kids"

    # Price range detection (very light heuristics for KRW)
    import re

    # Normalize common units like 만원/천원/k
    def parse_price(piece: str) -> Optional[int]:
        piece = piece.strip()
        try:
            if piece.endswith("만원") or piece.endswith("만 원") or piece.endswith("만"):
                num = float(re.sub(r"[^0-9.]", "", piece))
                return int(num * 10000)
            if piece.endswith("천원") or piece.endswith("천 원"):
                num = float(re.sub(r"[^0-9.]", "", piece))
                return int(num * 1000)
            if piece.lower().endswith("k"):
                num = float(re.sub(r"[^0-9.]", "", piece))
                return int(num * 1000)
            # plain number may be won already
            digits = re.sub(r"[^0-9]", "", piece)
            if digits:
                return int(digits)
        except Exception:
            return None
        return None

    pr: Dict[str, int] = {}
    # Patterns like '5만원 이하', '10만 원 이하', '3~5만원'
    m = re.search(r"(\d+[\.,]?\d*)\s*(만|만원|만 원|천|천원|k)?\s*(이하|이내|under|<=)", t)
    if m:
        price = parse_price(m.group(0))
        if price:
            pr["max"] = price
    else:
        m2 = re.search(r"(\d+[\.,]?\d*)\s*~\s*(\d+[\.,]?\d*)\s*(만|만원|만 원|천|천원|k)?", t)
        if m2:
            p1 = parse_price(m2.group(1) + (m2.group(3) or ""))
            p2 = parse_price(m2.group(2) + (m2.group(3) or ""))
            if p1 and p2:
                pr["min"], pr["max"] = min(p1, p2), max(p1, p2)
    if pr:
        resp.priceRange = pr

    # Build tokens by stripping stopwords and keeping alphanumerics/Korean
    words = [w for w in re.split(r"[^0-9A-Za-z가-힣\+/#-]+", t) if w]
    # Remove generic words
    stop = {"좀", "매우", "정도", "같은", "원", "가격", "의", "and", "or", "under"}
    tokens = [w for w in words if w not in stop and len(w) > 1]
    # Avoid duplicating color words
    color_words = {s.lower() for syns in _COLOR_MAP.values() for s in syns}
    tokens = [w for w in tokens if w not in color_words]
    resp.tokens = tokens[:8]
    return resp


@router.post("/parse")
def parse_text(req: ParseRequest) -> ParseResponse:
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    # Try Azure OpenAI if available
    if azure_openai_service.available():
        try:
            data = azure_openai_service.parse_search_text(text)
            # Normalize Azure result to our schema
            out = ParseResponse(
                category=(data.get("category") or None),
                tokens=[str(t) for t in (data.get("tokens") or [])][:8],
                colors=[str(c).lower() for c in (data.get("colors") or [])],
                gender=(data.get("gender") or None),
                priceRange=data.get("priceRange") or None,
                source="ai",
            )
            # Light post-normalization of category
            if out.category:
                c = out.category.lower()
                # Map a few common variants
                if c in ["bottom", "bottoms", "skirt", "dress"]:
                    out.category = "pants"
                elif c in ["top", "tops", "tshirt", "shirt", "knit", "hoodie"]:
                    out.category = "top"
                elif c in ["shoe", "sneakers", "boots", "heels"]:
                    out.category = "shoes"
                elif c in ["outerwear", "jacket", "coat", "padding", "cardigan"]:
                    out.category = "outer"
                elif c in ["acc", "accessory", "bag", "hat", "belt", "watch", "socks"]:
                    out.category = "accessories"
            return out
        except Exception:
            pass

    # Fallback rules
    return _fallback_parse(text)
