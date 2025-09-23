from __future__ import annotations

from typing import List, Optional, Dict
import re
from urllib.parse import urlparse, parse_qs

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field

from ..models import RecommendationItem
from ..services.external_recommender import external_recommender
from ..services.pos_recommender import get_pos_recommender
from ..services.db_recommender import db_pos_recommender
from ..services.catalog import get_catalog_service
from ..services.llm_ranker import llm_ranker


router = APIRouter(prefix="/api/recommend", tags=["Recommendations"])


class SelectedItem(BaseModel):
    pos: int
    category: Optional[str] = None
    title: Optional[str] = None
    tags: Optional[List[str]] = None
    price: Optional[int] = None
    brand: Optional[str] = None
    gender: Optional[str] = None
    productUrl: Optional[str] = None
    imageUrl: Optional[str] = None
    description: Optional[str] = None


class PositionsRequest(BaseModel):
    positions: List[int] = Field(..., description="Selected product positions (0-based)")
    top_k: int = Field(3, ge=1, le=200, description="Pool size before final cut")
    final_k: int = Field(3, ge=1, le=50, description="Final number of items to return")
    alpha: float = Field(0.38, ge=0.0, le=10.0)
    w1: float = Field(0.97, ge=0.0, le=1.0)
    w2: float = Field(0.03, ge=0.0, le=1.0)
    categories: Optional[List[str]] = Field(default=None, description="Restrict to these categories (e.g., ['top'])")
    use_llm_rerank: Optional[bool] = Field(default=None, description="Enable Azure LLM rerank when available")
    items: Optional[List[SelectedItem]] = Field(default=None, description="Optional: full metadata of selected products")


def _product_id_from_url_raw(u: str) -> Optional[str]:
    try:
        p = urlparse(str(u))
        # 1) query params common keys
        q = parse_qs(p.query or "")
        for k in ("id", "pid", "productId", "product_id", "goodsNo", "goods_no"):
            if k in q and q[k]:
                return str(q[k][0])
        # 2) last meaningful path segment
        segs = [s for s in (p.path or "").split("/") if s]
        if segs:
            last = segs[-1]
            # strip extension
            if "." in last:
                last = last.split(".")[0]
            # accept tokens containing digits (e.g., 12345, AB-123)
            if re.search(r"\d", last):
                return last
        return None
    except Exception:
        return None


def _key(it: Dict) -> str:
    _id = it.get("id")
    if _id is not None and str(_id).strip() != "":
        return f"id:{str(_id).strip()}"
    pu = it.get("productUrl") or it.get("product_url")
    if pu:
        pid = _product_id_from_url_raw(str(pu))
        if pid:
            # include host to avoid cross-site collisions
            try:
                host = urlparse(str(pu)).netloc.lower()
            except Exception:
                host = ""
            return f"pid:{host}:{pid}"
        return f"url:{str(pu).strip().lower()}"
    iu = it.get("imageUrl") or it.get("image_url")
    if iu:
        return f"img:{str(iu).strip().lower()}"
    title = str(it.get("title") or "").strip().lower()
    price = str(it.get("price") or "0").strip()
    return f"tp:{title}|{price}"

COLOR_WORDS = {
    # EN
    "black","white","gray","grey","navy","blue","light","sky","red","pink","purple","green","olive","khaki","yellow","beige","brown","cream","ivory","orange","silver","gold",
    # KR
    "블랙","화이트","그레이","네이비","파랑","라이트","하늘","빨강","레드","핑크","보라","초록","그린","올리브","카키","노랑","베이지","브라운","갈색","크림","아이보리","오렌지","실버","골드",
}

_NON_ALNUM = re.compile(r"[^a-z0-9가-힣]+")

def _title_core(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", t)
    tokens = [tok for tok in _NON_ALNUM.split(t) if tok]
    tokens = [tok for tok in tokens if not tok.isdigit() and tok not in COLOR_WORDS]
    core = " ".join(tokens[:4])
    return core or t.strip()

def _brand_of(it: Dict) -> str:
    b = it.get("brandName")
    if b:
        return str(b).strip().lower()
    tags = it.get("tags") or []
    return str(tags[0]).strip().lower() if tags else ""

def _url_root(it: Dict) -> str:
    u = it.get("productUrl") or it.get("product_url")
    if not u:
        return ""
    try:
        p = urlparse(str(u))
        path = "/".join([seg for seg in p.path.split("/") if seg][:2])
        return f"{p.netloc.lower()}/{path.lower()}"
    except Exception:
        return ""

def _signature(it: Dict) -> str:
    return f"{_brand_of(it)}|{_title_core(str(it.get('title') or ''))}|{_url_root(it)}"

def _diversify_pick(candidates: List[Dict], k: int) -> List[Dict]:
    out: List[Dict] = []
    seen_sig: set[str] = set()
    for it in candidates:
        sig = _signature(it)
        if sig in seen_sig:
            continue
        out.append(it)
        seen_sig.add(sig)
        if len(out) >= k:
            return out
    seen_keys = {_key(x) for x in out}
    for it in candidates:
        if len(out) >= k:
            break
        kkey = _key(it)
        if kkey in seen_keys:
            continue
        out.append(it)
        seen_keys.add(kkey)
    return out

def _normalize_category(value: str | None) -> str:
    v = (value or "").strip().lower()
    if not v:
        return "unknown"
    if "outer" in v or "jacket" in v or "coat" in v:
        return "outer"
    if "top" in v or "shirt" in v or "tee" in v or "상의" in v:
        return "top"
    if (
        "pant" in v or "bottom" in v or "하의" in v or "denim" in v or "skirt" in v
    ):
        return "pants"
    if "shoe" in v or "sneaker" in v or "신발" in v:
        return "shoes"
    if "access" in v:
        return "accessories"
    return v


def _infer_target_categories_from_positions(positions: List[int]) -> List[str]:
    svc = get_catalog_service()
    catalog = svc.get_all()
    cats: List[str] = []
    for p in positions:
        if 0 <= p < len(catalog):
            cats.append(_normalize_category(str(catalog[p].get("category"))))
    if not cats:
        return []
    # majority category as first, keep unique order
    from collections import Counter

    cnt = Counter(cats)
    majority = max(cnt.items(), key=lambda kv: kv[1])[0]
    ordered_unique: List[str] = []
    for c in [majority] + cats:
        if c not in ordered_unique:
            ordered_unique.append(c)
    return ordered_unique


@router.post("/by-positions", response_model=List[RecommendationItem])
def recommend_by_positions(req: PositionsRequest, response: Response) -> List[RecommendationItem]:
    # determine target categories (priority: explicit -> from items -> from positions)
    if req.categories:
        target_cats = [_normalize_category(c) for c in req.categories]
    elif req.items:
        target_cats = []
        for it in req.items:
            if it.category:
                c = _normalize_category(it.category)
                if c not in target_cats:
                    target_cats.append(c)
    else:
        target_cats = _infer_target_categories_from_positions(req.positions)
    target_cats = [c for c in target_cats if c in {"top", "pants", "shoes", "outer", "accessories"}] or []

    # Prefer DB recommender if available, then file-based, then external
    pool: List[Dict] | None = None
    if db_pos_recommender.available():
        try:
            items = db_pos_recommender.recommend(
                positions=req.positions,
                top_k=req.top_k,
                alpha=req.alpha,
                w1=req.w1,
                w2=req.w2,
            )
            pool = items
        except Exception as e:
            # fall through to file/external
            pass

    # Prefer internal (file-based) recommender when available
    if pool is None:
        pos_rec = get_pos_recommender()
        if pos_rec.available():
            try:
                items = pos_rec.recommend(
                    positions=req.positions,
                    top_k=req.top_k,
                    alpha=req.alpha,
                    w1=req.w1,
                    w2=req.w2,
                )
                pool = items
            except Exception as e:
                # fall back to external if configured
                if external_recommender.available():
                    try:
                        items = external_recommender.recommend_by_positions(
                            positions=req.positions,
                            top_k=req.top_k,
                            alpha=req.alpha,
                            w1=req.w1,
                            w2=req.w2,
                        )
                        pool = items
                    except Exception as e2:
                        raise HTTPException(status_code=500, detail=str(e2))
                else:
                    raise HTTPException(status_code=500, detail=str(e))

    # If internal not available, try external
    if pool is None and external_recommender.available():
        try:
            items = external_recommender.recommend_by_positions(
                positions=req.positions,
                top_k=req.top_k,
                alpha=req.alpha,
                w1=req.w1,
                w2=req.w2,
            )
            pool = items
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    if pool is None:
        raise HTTPException(status_code=503, detail="No recommender available (internal/external)")

    # Build per-category top-N so that every dressed category surfaces results
    selected: List[Dict] = []
    # use_llm_rerank가 명시적으로 False면 LLM 사용 안함, None이면 기본값 사용
    if req.use_llm_rerank is not None:
        use_llm = req.use_llm_rerank
    else:
        use_llm = llm_ranker.available()

    # Helper: build simple analysis context
    def build_analysis_for(cat: str) -> Dict:
        if req.items:
            t: List[str] = []
            for it in req.items:
                if it.category and _normalize_category(it.category) != cat:
                    continue
                if it.brand:
                    t.append(str(it.brand))
                if it.gender:
                    t.append(str(it.gender))
                if it.tags:
                    t.extend([str(x) for x in it.tags])
                if it.title:
                    t.extend(str(it.title).split()[:8])
                if it.description:
                    t.extend(str(it.description).split()[:8])
            return {"categories": [cat], "tags": t[:30]}
        # fallback to catalog tags from positions
        svc = get_catalog_service()
        catalog = svc.get_all()
        t: List[str] = []
        for p in req.positions:
            if 0 <= p < len(catalog) and _normalize_category(str(catalog[p].get("category"))) == cat:
                t.extend(catalog[p].get("tags") or [])
        return {"categories": [cat], "tags": t[:20]}

    if target_cats:
        # produce top final_k for each target category
        cat_pool_map: Dict[str, List[Dict]] = {}
        cat_pick_map: Dict[str, List[Dict]] = {}
        for cat in target_cats:
            cat_pool = [it for it in pool if _normalize_category(str(it.get("category"))) == cat]
            # If not enough results for this category, try boosting pool once
            if len(cat_pool) < req.final_k:
                try:
                    booster_k = max(req.top_k * 5, 200)
                    # try DB first
                    boosted: List[Dict] | None = None
                    if db_pos_recommender.available():
                        boosted = db_pos_recommender.recommend(
                            positions=req.positions,
                            top_k=booster_k,
                            alpha=req.alpha,
                            w1=req.w1,
                            w2=req.w2,
                        )
                    elif get_pos_recommender().available():
                        boosted = get_pos_recommender().recommend(
                            positions=req.positions,
                            top_k=booster_k,
                            alpha=req.alpha,
                            w1=req.w1,
                            w2=req.w2,
                        )
                    if boosted:
                        more = [it for it in boosted if _normalize_category(str(it.get("category"))) == cat]
                        # merge unique
                        seen = {_key(x) for x in cat_pool}
                        for it in more:
                            k = _key(it)
                            if k not in seen:
                                cat_pool.append(it)
                                seen.add(k)
                            if len(cat_pool) >= req.final_k * 3:
                                break
                except Exception:
                    pass
            if not cat_pool:
                continue
            # sort by score desc
            try:
                cat_pool.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
            except Exception:
                pass
            # Fallback fill from catalog if still lacking
            if len(cat_pool) < req.final_k:
                try:
                    svc = get_catalog_service()
                    catalog = [p for p in svc.get_all() if _normalize_category(str(p.get("category"))) == cat]
                    # push items not already present by dedup key
                    seen = {_key(x) for x in cat_pool}
                    for p in catalog:
                        temp = {
                            "id": str(p.get("id")),
                            "pos": p.get("pos"),
                            "title": p.get("title") or "",
                            "price": int(p.get("price") or 0),
                            "tags": p.get("tags") or [],
                            "category": p.get("category") or cat,
                            "imageUrl": p.get("imageUrl"),
                            "productUrl": p.get("productUrl"),
                            "score": 0.0,
                        }
                        k = _key(temp)
                        if k in seen:
                            continue
                        cat_pool.append(temp)
                        if len(cat_pool) >= req.final_k:
                            break
                except Exception:
                    pass
            cat_pick_map[cat] = _diversify_pick(cat_pool, req.final_k)
            cat_pool_map[cat] = cat_pool

        # Global de-dup across categories while keeping per-category quotas
        seen_ids: set[str] = set()
        final_by_cat: Dict[str, List[Dict]] = {c: [] for c in target_cats}
        # First pass: keep order within each category picks
        for cat in target_cats:
            picks = cat_pick_map.get(cat, [])
            for it in picks:
                _id = _key(it)
                if _id in seen_ids:
                    continue
                final_by_cat[cat].append(it)
                seen_ids.add(_id)
                if len(final_by_cat[cat]) >= req.final_k:
                    break
        # Second pass: top-up from each category's pool to meet final_k per category
        for cat in target_cats:
            if len(final_by_cat[cat]) >= req.final_k:
                continue
            pool_cat = cat_pool_map.get(cat, [])
            for it in pool_cat:
                if len(final_by_cat[cat]) >= req.final_k:
                    break
                _id = _key(it)
                if _id in seen_ids:
                    continue
                final_by_cat[cat].append(it)
                seen_ids.add(_id)

        # Debug headers with counts
        try:
            response.headers["X-Rec-Categories"] = ",".join(target_cats)
            response.headers["X-Rec-Counts-Initial"] = ",".join(str(len(cat_pool_map.get(c, []))) for c in target_cats)
            response.headers["X-Rec-Counts-Final"] = ",".join(str(len(final_by_cat.get(c, []))) for c in target_cats)
        except Exception:
            pass

        # Flatten respecting category order
        final_items = []
        for cat in target_cats:
            final_items.extend(final_by_cat[cat])
    else:
        # No category target -> fall back to global top-N
        try:
            pool.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        except Exception:
            pass
        final_items = pool[: req.final_k]
    # Ensure 'pos' is populated when missing (derive from id when numeric)
    norm_items = []
    for it in final_items:
        d = dict(it)
        if d.get("pos") is None and d.get("id") is not None:
            try:
                d["pos"] = int(d.get("id"))
            except Exception:
                pass
        norm_items.append(RecommendationItem(**d))
    return norm_items

