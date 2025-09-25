from __future__ import annotations

from datetime import datetime
import json
from typing import Any, Dict, List, Tuple

from fastapi import APIRouter, Response

from ..models import (
    CategoryRecommendations,
    ClothingItems,
    RecommendationItem,
    RecommendationOptions,
    RecommendationRequest,
    RecommendationResponse,
)
from ..services.azure_openai_service import azure_openai_service
from ..services.catalog import get_catalog_service
from ..services.db_recommender import _normalize_gender as _db_norm_gender
from ..services.db_recommender import db_pos_recommender, index_only_recommender
from ..services.embedding_client import embedding_client
from ..services.llm_ranker import llm_ranker

router = APIRouter(prefix="/api/recommend", tags=["Recommendations"])


def _candidate_budget(opts: RecommendationOptions) -> int:
    base = opts.maxPerCategory if opts.maxPerCategory is not None else 3
    return base * 4


def _convert_analysis_to_text(analysis: dict) -> str:
    """임베딩 서버에 넘길 텍스트로 분석 결과를 정규화"""
    text_parts: List[str] = []

    overall = analysis.get("overall_style")
    if isinstance(overall, list) and overall:
        text_parts.append(f"Overall style: {', '.join(str(v) for v in overall)}")

    slot_details = analysis.get("slotDetails")
    if not isinstance(slot_details, dict):
        slot_details = {}

    for category in ["top", "pants", "shoes", "outer"]:
        tokens = analysis.get(category)
        if isinstance(tokens, list) and tokens:
            token_text = ", ".join(str(t) for t in tokens if str(t).strip())
            if token_text:
                text_parts.append(f"{category} keywords: {token_text}")

        detail = slot_details.get(category)
        if isinstance(detail, dict) and detail:
            meta_bits: List[str] = []
            garment = detail.get("garment")
            color = detail.get("color")
            style_tags = detail.get("styleTags") if isinstance(detail.get("styleTags"), list) else []

            for value in [garment, color]:
                if isinstance(value, str) and value.strip():
                    meta_bits.append(value.strip())
            meta_bits.extend(str(tag).strip() for tag in style_tags if str(tag).strip())

            if meta_bits:
                text_parts.append(f"{category} detail: {', '.join(meta_bits)}")

            description = detail.get("description")
            if isinstance(description, str) and description.strip():
                text_parts.append(f"{category} description: {description.strip()}")

    colors = analysis.get("colors")
    if isinstance(colors, list) and colors:
        color_text = ", ".join(str(c) for c in colors if str(c).strip())
        if color_text:
            text_parts.append(f"Colors: {color_text}")

    tags = analysis.get("tags")
    if isinstance(tags, list) and tags:
        tag_text = ", ".join(str(t) for t in tags if str(t).strip())
        if tag_text:
            text_parts.append(f"Tags: {tag_text}")

    captions = analysis.get("captions")
    if isinstance(captions, list) and captions:
        caption_text = "; ".join(str(c) for c in captions if str(c).strip())
        if caption_text:
            text_parts.append(f"Description: {caption_text}")

    return ". ".join(text_parts) if text_parts else "casual style clothing"


def _parse_clothing_analysis(raw: str) -> Tuple[List[str], Dict[str, Any]]:
    """Parse JSON clothing analysis output into keywords and structured detail."""
    if not raw:
        return [], {}

    cleaned = raw.strip()
    if not cleaned:
        return [], {}

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            color = str(data.get("color") or "").strip() or None
            garment = str(data.get("garment") or "").strip() or None
            style_tags_raw = data.get("styleTags")
            if isinstance(style_tags_raw, list):
                style_tags = [str(tag).strip() for tag in style_tags_raw if str(tag).strip()]
            elif isinstance(style_tags_raw, str):
                style_tags = [tok.strip() for tok in style_tags_raw.split(",") if tok.strip()]
            else:
                style_tags = []
            description = str(data.get("description") or "").strip()

            detail = {
                "color": color,
                "garment": garment,
                "styleTags": style_tags,
                "description": description,
            }

            keywords: List[str] = []
            for value in (color, garment):
                if isinstance(value, str) and value and value not in keywords:
                    keywords.append(value)
            for tag in style_tags:
                if tag not in keywords:
                    keywords.append(tag)
            if description:
                keywords.extend([word for word in description.split() if word and word not in keywords])

            return keywords, detail
    except Exception as exc:
        print(f"❌ clothing analysis parse failed: {exc}")

    # fallback: treat as whitespace-separated tokens, keep raw description only
    return cleaned.split(), {"description": cleaned}


def _merge_slot_analysis(
    analysis: Dict[str, Any],
    slot: str,
    tokens: List[str],
    detail: Dict[str, Any] | None = None,
) -> None:
    keywords: List[str] = []
    for token in tokens:
        t = str(token).strip()
        if t and t not in keywords:
            keywords.append(t)

    if detail:
        for extra in (detail.get("garment"), detail.get("color")):
            if isinstance(extra, str):
                value = extra.strip()
                if value and value not in keywords:
                    keywords.append(value)
        for tag in detail.get("styleTags") or []:
            value = str(tag).strip()
            if value and value not in keywords:
                keywords.append(value)

    analysis[slot] = keywords

    if not detail:
        return

    slot_details = analysis.setdefault("slotDetails", {})
    if isinstance(slot_details, dict):
        slot_details[slot] = detail

    color = detail.get("color")
    if isinstance(color, str) and color.strip():
        colors = analysis.setdefault("colors", [])
        if isinstance(colors, list) and color not in colors:
            colors.append(color)

    style_tags = detail.get("styleTags") or []
    if style_tags:
        tags = analysis.setdefault("tags", [])
        if isinstance(tags, list):
            for tag in style_tags:
                tag_str = str(tag).strip()
                if tag_str and tag_str not in tags:
                    tags.append(tag_str)

    description = detail.get("description")
    if isinstance(description, str) and description.strip():
        captions = analysis.setdefault("captions", [])
        if isinstance(captions, list) and description not in captions:
            captions.append(description)


def _format_db_recommendations(db_recs: list) -> dict:
    """
    DB 추천 결과를 카테고리별로 포맷팅
    """
    formatted = {"top": [], "pants": [], "shoes": [], "outer": [], "accessories": []}

    for rec in db_recs:
        category = rec.get("category", "top").lower()
        if category in formatted:
            formatted[category].append(rec)

    return formatted


def _requested_slots(
    clothing: ClothingItems | None = None,
    selected_ids: dict[str, str] | None = None,
) -> set[str]:
    slots: set[str] = set()
    if clothing:
        # clothing이 딕셔너리인 경우 처리
        if isinstance(clothing, dict):
            for key in ["top", "pants", "shoes", "outer"]:
                item = clothing.get(key)
                if item and item.get("base64"):
                    slots.add(key)
        else:
            # Pydantic 모델인 경우 처리
            top = getattr(clothing, "top", None)
            if top is not None and getattr(top, "base64", ""):
                slots.add("top")
            pants = getattr(clothing, "pants", None)
            if pants is not None and getattr(pants, "base64", ""):
                slots.add("pants")
            shoes = getattr(clothing, "shoes", None)
            if shoes is not None and getattr(shoes, "base64", ""):
                slots.add("shoes")
            outer = getattr(clothing, "outer", None)
            if outer is not None and getattr(outer, "base64", ""):
                slots.add("outer")
    if selected_ids:
        for cat, val in selected_ids.items():
            if val is None or not str(val).strip():
                continue
            slots.add(_normalize_category(cat))
    return slots


def _normalize_category(raw: str | None) -> str:
    value = (raw or "").strip().lower()
    if not value:
        return "unknown"
    if "outer" in value or "jacket" in value or "coat" in value:
        return "outer"
    if "top" in value or "shirt" in value or "tee" in value or "상의" in value:
        return "top"
    if (
        "pant" in value
        or "bottom" in value
        or "하의" in value
        or "denim" in value
        or "skirt" in value
    ):
        return "pants"
    if "shoe" in value or "sneaker" in value or "신발" in value:
        return "shoes"
    if "access" in value:
        return "accessories"
    return value


def _infer_slots_from_analysis(analysis: dict | None) -> set[str]:
    slots: set[str] = set()
    if not analysis:
        return slots
    # explicit categories list
    cats = analysis.get("categories") if isinstance(analysis, dict) else None
    if isinstance(cats, list):
        for c in cats:
            slots.add(_normalize_category(str(c)))
    # presence of per-slot keys with any content
    for key in ("top", "pants", "shoes", "outer"):
        val = analysis.get(key) if isinstance(analysis, dict) else None
        if val:
            slots.add(key)
    # keep only known slots
    return {s for s in slots if s in {"top", "pants", "shoes", "outer"}}


def _embedded_recommendations(
    selected_ids: dict[str, str], max_per_category: int
) -> dict[str, list[dict]]:
    # 우선순위: IndexOnlyRecommender -> DbPosRecommender
    if not selected_ids:
        return {}

    by_category: dict[str, list[dict]] = {
        "top": [],
        "pants": [],
        "shoes": [],
        "outer": [],
        "accessories": [],
    }

    for slot_cat, value in selected_ids.items():
        slot_norm = _normalize_category(slot_cat)
        if slot_norm not in by_category:
            continue
        try:
            pos = int(value)
        except (TypeError, ValueError):
            continue

        pool_size = max_per_category * 6 if max_per_category > 0 else 18

        # 1. IndexOnlyRecommender 시도
        pool = None
        if index_only_recommender and index_only_recommender.available():
            try:
                pool = index_only_recommender.recommend(
                    positions=[pos],
                    top_k=pool_size,
                    w_dense=0.5,
                    w_maxsim=0.5,
                    same_category=True,
                )
            except Exception:
                pass

        # 2. DbPosRecommender 시도
        if pool is None and db_pos_recommender.available():
            try:
                pool = db_pos_recommender.recommend(positions=[pos], top_k=pool_size)
            except Exception:
                continue

        if not pool:
            continue

        for item in pool:
            cat = _normalize_category(item.get("category"))
            if cat != slot_norm:
                continue
            item_id = (
                str(item.get("id"))
                if item.get("id") is not None
                else str(item.get("pos"))
            )
            if item_id == str(pos):
                continue
            if len(by_category[slot_norm]) >= max_per_category:
                break
            by_category[slot_norm].append(item)

    return {k: v for k, v in by_category.items() if v}


def _db_products() -> list[dict] | None:
    if not db_pos_recommender.available():
        return None
    db_products = list(db_pos_recommender.products)
    return db_products if db_products else None


def _candidate_kwargs(opts: RecommendationOptions) -> dict:
    return {
        "max_per_category": _candidate_budget(opts),
        "include_score": True,
        "min_price": opts.minPrice,
        "max_price": opts.maxPrice,
        "exclude_tags": opts.excludeTags,
    }


def _build_candidates(
    analysis: dict,
    svc,
    opts: RecommendationOptions,
) -> dict[str, list[dict]]:
    kwargs = _candidate_kwargs(opts)
    products_override = _db_products()
    if products_override is None:
        # Fallback to catalog.json when DB is not available
        try:
            return svc.find_similar(
                analysis,
                products=None,  # use internal catalog dataset
                **kwargs,
            )
        except Exception:
            # last resort – empty buckets
            return {c: [] for c in svc.config.categories}

    return svc.find_similar(
        analysis,
        products=products_override,
        **kwargs,
    )


@router.get("/status")
def status():
    stats = get_catalog_service().stats()
    return {
        "aiService": {
            "azureOpenAI": {
                "available": azure_openai_service.available(),
                "deploymentId": getattr(azure_openai_service, "deployment_id", None),
                "apiVersion": getattr(azure_openai_service, "api_version", None),
            },
            "llmReranker": {
                "available": llm_ranker.available(),
                "deploymentId": getattr(llm_ranker, "deployment_id", None),
            },
        },
        "catalogService": {
            "available": stats.get("totalProducts", 0) > 0,
            "productCount": stats.get("totalProducts", 0),
        },
    }


@router.get("/catalog")
def catalog_stats():
    return get_catalog_service().stats()


@router.get("/random")
def random_products(
    limit: int = 18,
    category: str | None = None,
    gender: str | None = None,
    response: Response = None,
):
    # Prefer DB-backed products when available, otherwise use catalog JSON
    if db_pos_recommender.available():
        products = list(db_pos_recommender.products)
        source = "db"
    else:
        svc = get_catalog_service()
        products = svc.get_all()
        source = "catalog"

    def norm_slot(s: str) -> str:
        c = (s or "").strip().lower()
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

    if category:
        req_slot = norm_slot(category)
        products = [
            p for p in products if norm_slot(str(p.get("category") or "")) == req_slot
        ]

    if gender:
        gq = (gender or "").strip().lower()
        want = _db_norm_gender(gq)

        # 남/여 요청 시 공용도 함께 포함
        def match(prod_gender: str) -> bool:
            ng = _db_norm_gender(prod_gender)
            if want == "male":
                return ng in {"male", "unisex"}
            if want == "female":
                return ng in {"female", "unisex"}
            return ng == want

        products = [p for p in products if match(str(p.get("gender") or ""))]

    import random

    random.shuffle(products)
    result = []
    for p in products[: min(max(limit, 1), 100)]:
        item = {
            "id": str(p.get("id")),
            "title": p.get("title") or "",
            "price": int(p.get("price") or 0),
            "imageUrl": p.get("imageUrl"),
            "productUrl": p.get("productUrl"),
            "tags": p.get("tags") or [],
            "category": p.get("category") or "top",
        }
        # propagate pos if available; otherwise derive from numeric id
        if p.get("pos") is not None:
            try:
                item["pos"] = int(p.get("pos"))
            except Exception:
                pass
        else:
            try:
                item["pos"] = int(item["id"]) if item.get("id") else None
            except Exception:
                pass
        result.append(item)
    try:
        if response is not None:
            response.headers["X-Rec-Source"] = source
            if gender:
                response.headers["X-Rec-Gender"] = (gender or "").lower()
    except Exception:
        pass
    return result


@router.post("")
def recommend_from_upload(req: RecommendationRequest) -> RecommendationResponse:
    # 디버깅: 요청 데이터 확인
    print("🚀🚀🚀 recommend_from_upload 함수 시작! 🚀🚀🚀")
    print("🔍 추천 요청 데이터 확인:")
    print("🔍🔍🔍 외부 데이터인지 확인 중... 🔍🔍🔍")
    print(f"  - person: {req.person is not None}")
    print(f"  - clothingItems: {req.clothingItems is not None}")
    if req.clothingItems:
        print(f"  - top: {getattr(req.clothingItems, 'top', None) is not None}")
        print(f"  - pants: {getattr(req.clothingItems, 'pants', None) is not None}")
        print(f"  - shoes: {getattr(req.clothingItems, 'shoes', None) is not None}")
        print(f"  - outer: {getattr(req.clothingItems, 'outer', None) is not None}")

    # Analyze style: 각 슬롯별로 개별 분석
    analysis = {}
    analysis_method = "fallback"
    if azure_openai_service.available():
        print("🔍 Azure OpenAI 서비스 사용 가능, 각 슬롯별 스타일 분석 시작...")
        try:
            # 각 슬롯별로 개별 분석
            if req.clothingItems:
                for slot_name in ("top", "pants", "shoes", "outer"):
                    item = getattr(req.clothingItems, slot_name, None)
                    if item:
                        print(f"🔍 {slot_name} 슬롯 분석 중...")
                        print(
                            f"🔍 Azure OpenAI 서비스 상태: {azure_openai_service.available()}"
                        )
                        try:
                            description = azure_openai_service.analyze_clothing_item(
                                item
                            )
                            print(f"🎯 {slot_name} 분석 결과: {description}")
                            print(f"🤖 GPT 설명 ({slot_name}): {description}")
                            tokens, detail = _parse_clothing_analysis(description)
                            _merge_slot_analysis(analysis, slot_name, tokens, detail)
                        except Exception as e:
                            print(f"❌ {slot_name} 분석 실패: {e}")
                            fallback_detail = {
                                "color": None,
                                "garment": slot_name,
                                "styleTags": [slot_name, "basic", "casual"],
                                "description": f"{slot_name} basic garment.",
                            }
                            _merge_slot_analysis(
                                analysis,
                                slot_name,
                                fallback_detail["styleTags"],
                                fallback_detail,
                            )

            # 전체 스타일 분석도 시도
            if req.person:
                try:
                    overall_analysis = azure_openai_service.analyze_style_from_images(
                        req.person, req.clothingItems
                    )
                    print(f"🤖 전체 스타일 분석 결과: {overall_analysis}")
                    print(f"🤖 GPT 전체 스타일 설명: {overall_analysis}")
                    analysis.update(overall_analysis)
                except Exception as e:
                    print(f"❌ 전체 스타일 분석 실패: {e}")

            analysis_method = "ai"
        except Exception as e:
            print(f"❌ GPT-4.1 Mini 분석 실패: {e}")
            analysis = {}
            analysis_method = "fallback"
    else:
        print("❌ Azure OpenAI 서비스 사용 불가")

    if not analysis:
        print("🔍 분석 결과 없음, fallback 사용")
        if req.person:
            analysis["overall_style"] = ["casual", "everyday"]
        if req.clothingItems:
            for k in ("top", "pants", "shoes", "outer"):
                if getattr(req.clothingItems, k) is not None:
                    fallback_detail = {
                        "color": None,
                        "garment": k,
                        "styleTags": [k, "basic", "casual"],
                        "description": f"Simple {k} garment with casual styling.",
                    }
                    _merge_slot_analysis(
                        analysis,
                        k,
                        fallback_detail["styleTags"],
                        fallback_detail,
                    )
        print(f"🔍 Fallback 분석 결과: {analysis}")

    svc = get_catalog_service()
    opts = req.options if req.options is not None else RecommendationOptions()

    # 1. 임베딩 서버 연동으로 벡터 기반 추천 시도
    candidate_recs = {}
    if embedding_client.available() and analysis:
        try:
            print("🔍 임베딩 서버로 분석 결과 전송 중...")
            # 분석 결과를 텍스트로 변환
            analysis_text = _convert_analysis_to_text(analysis)
            print(f"📝 분석 텍스트: {analysis_text}")

            # 임베딩 서버에서 Dense + ColBERT 벡터 생성
            colbert_response = embedding_client.get_colbert_embedding(analysis_text)
            dense_vector = colbert_response["dense_embedding"]
            colbert_vectors = colbert_response["colbert_embeddings"]
            print(
                f"✅ 하이브리드 벡터 생성 완료 - Dense: {len(dense_vector)}차원, ColBERT: {len(colbert_vectors)}토큰"
            )

            # 우선순위: IndexOnlyRecommender -> DbPosRecommender
            if index_only_recommender and index_only_recommender.available():
                print("🔍 IndexOnlyRecommender로 하이브리드 추천 중...")
                # IndexOnlyRecommender 사용 (Dense + ColBERT)
                db_recs = index_only_recommender.recommend_by_embedding(
                    query_embedding=dense_vector,
                    query_colbert=colbert_vectors,
                    top_k=opts.maxPerCategory or 3,
                    w_dense=0.5,
                    w_maxsim=0.5,
                )
                print(f"📊 IndexOnlyRecommender 하이브리드 추천 결과: {len(db_recs)}개")
                candidate_recs = _format_db_recommendations(db_recs)
            elif db_pos_recommender.available():
                print("🗄️ DB에서 Dense 임베딩 기반 추천 중...")
                # Dense 임베딩 기반 추천 사용 (ColBERT는 지원하지 않음)
                db_recs = db_pos_recommender.recommend_by_embedding(
                    query_embedding=dense_vector,
                    top_k=opts.maxPerCategory or 3,
                    alpha=0.7,  # 임베딩 가중치
                    w1=0.8,  # 스타일 가중치
                    w2=0.2,  # 가격 가중치
                )
                print(f"📊 DB Dense 추천 결과: {len(db_recs)}개")
                candidate_recs = _format_db_recommendations(db_recs)
            else:
                print("⚠️ 추천 서비스 사용 불가, CatalogService로 폴백")
                candidate_recs = _build_candidates(analysis, svc, opts)
        except Exception as e:
            print(f"❌ 임베딩 서버 연동 실패: {e}")
            print("🔄 CatalogService로 폴백")
            candidate_recs = _build_candidates(analysis, svc, opts)
    else:
        print("⚠️ 임베딩 서버 사용 불가 또는 분석 결과 없음, CatalogService 사용")
        candidate_recs = _build_candidates(analysis, svc, opts)

    selected_ids = dict(req.selectedProductIds or {})
    active_slots = _requested_slots(
        req.clothingItems, selected_ids if selected_ids else None
    )

    # 기존 임베딩 추천도 시도 (selectedProductIds가 있는 경우)
    embed_recs = _embedded_recommendations(
        selected_ids,
        opts.maxPerCategory or 3,
    )
    if embed_recs:
        for cat, items in embed_recs.items():
            if items:
                candidate_recs[cat] = items
    # Strict slot gating: only return categories the user actually provided
    # If no slots are active, suppress all category recommendations
    for cat in list(candidate_recs.keys()):
        if not active_slots or cat not in active_slots:
            candidate_recs[cat] = []

    # Optional LLM rerank (default to Azure OpenAI when configured)
    max_k = opts.maxPerCategory or 3
    user_llm_pref = opts.useLLMRerank
    use_llm = user_llm_pref if user_llm_pref is not None else llm_ranker.available()
    if use_llm and llm_ranker.available():
        ids = llm_ranker.rerank(analysis, candidate_recs, top_k=max_k)
        if ids:
            # reorder by ids
            recs = {cat: [] for cat in candidate_recs.keys()}
            for cat in candidate_recs.keys():
                # map id->item
                idx = {str(p["id"]): p for p in candidate_recs[cat]}
                for _id in ids.get(cat, []):
                    if _id in idx:
                        recs[cat].append(idx[_id])
            # fill if not enough
            for cat in recs.keys():
                if len(recs[cat]) < max_k:
                    for p in candidate_recs[cat]:
                        if p not in recs[cat]:
                            recs[cat].append(p)
                        if len(recs[cat]) >= max_k:
                            break
        else:
            recs = {cat: (candidate_recs[cat][:max_k]) for cat in candidate_recs.keys()}
    else:
        recs = {cat: (candidate_recs[cat][:max_k]) for cat in candidate_recs.keys()}

    # Convert lists of dicts to CategoryRecommendations model
    as_model = CategoryRecommendations(
        top=[RecommendationItem(**p) for p in recs.get("top", [])],
        pants=[RecommendationItem(**p) for p in recs.get("pants", [])],
        shoes=[RecommendationItem(**p) for p in recs.get("shoes", [])],
        outer=[RecommendationItem(**p) for p in recs.get("outer", [])],
        accessories=[RecommendationItem(**p) for p in recs.get("accessories", [])],
    )

    return RecommendationResponse(
        recommendations=as_model,
        analysisMethod=analysis_method,
        styleAnalysis=analysis if analysis_method == "ai" else None,
        requestId=f"req_{int(datetime.utcnow().timestamp())}",
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


# from-fitting API 제거됨 - recommend API로 통합
