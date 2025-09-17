from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException, Response

from ..models import (
    CategoryRecommendations,
    RecommendationFromFittingRequest,
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
)
from ..services.azure_openai_service import azure_openai_service
from ..services.product_index import product_index
from ..services.llm_ranker import llm_ranker

router = APIRouter(prefix="/api/recommend", tags=["Recommendations"])


@router.get("/status")
def status():
    catalog_available = product_index.available()
    stats = product_index.stats() if catalog_available else {
        "totalProducts": 0,
        "categories": {},
        "priceRange": {"min": 0, "max": 0, "average": 0},
    }
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
            "available": catalog_available,
            "productCount": stats.get("totalProducts", 0),
            "categories": stats.get("categories", {}),
            "priceRange": stats.get("priceRange", {}),
        },
    }

@router.get("/catalog")
def catalog_stats():
    if not product_index.available():
        raise HTTPException(status_code=503, detail="Product catalog unavailable")
    return product_index.stats()

@router.get("/random")
def random_products(
    limit: int = 18,
    category: str | None = None,
    gender: str | None = None,
    *,
    response: Response,
):
    if not product_index.available():
        raise HTTPException(status_code=503, detail="Product catalog unavailable")

    try:
        items = product_index.random_products(limit=limit, category=category, gender=gender)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    response.headers["X-Rec-Source"] = "db"
    if gender:
        response.headers["X-Rec-Gender"] = (gender or "").lower()

    return items

@router.post("")
def recommend_from_upload(req: RecommendationRequest) -> RecommendationResponse:
    # Analyze style: prefer Azure OpenAI if available
    analysis = {}
    analysis_method = "fallback"
    if azure_openai_service.available():
        try:
            analysis = azure_openai_service.analyze_style_from_images(req.person, req.clothingItems)
            analysis_method = "ai"
        except Exception:
            analysis = {}
            analysis_method = "fallback"
    if not analysis:
        if req.person:
            analysis["overall_style"] = ["casual", "everyday"]
        if req.clothingItems:
            for k in ("top", "pants", "shoes"):
                if getattr(req.clothingItems, k) is not None:
                    analysis.setdefault(k, []).extend([k, "basic", "casual"])

    if not product_index.available():
        raise HTTPException(status_code=503, detail="Product catalog unavailable")
    opts = req.options or {}
    try:
        candidate_recs = product_index.find_similar(
            analysis,
            max_per_category=(opts.maxPerCategory or 3) * 4 if hasattr(opts, "maxPerCategory") else 12,
            include_score=True,
            min_price=getattr(opts, "minPrice", None),
            max_price=getattr(opts, "maxPrice", None),
            exclude_tags=getattr(opts, "excludeTags", None),
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    # Optional LLM rerank (default to Azure OpenAI when configured)
    max_k = (opts.maxPerCategory or 3) if hasattr(opts, "maxPerCategory") else 3
    user_llm_pref = getattr(opts, "useLLMRerank", None)
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
        accessories=[RecommendationItem(**p) for p in recs.get("accessories", [])],
    )

    return RecommendationResponse(
        recommendations=as_model,
        analysisMethod=analysis_method,
        styleAnalysis=analysis if analysis_method == "ai" else None,
        requestId=f"req_{int(datetime.utcnow().timestamp())}",
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@router.post("/from-fitting")
def recommend_from_fitting(req: RecommendationFromFittingRequest) -> RecommendationResponse:
    # For fitting: prefer Azure analysis on generated image
    analysis_method = "fallback"
    analysis = {"overall_style": ["casual", "relaxed"], "categories": ["top", "pants", "shoes"]}
    if azure_openai_service.available() and req.generatedImage:
        try:
            analysis = azure_openai_service.analyze_virtual_try_on(req.generatedImage)
            analysis_method = "ai"
        except Exception:
            analysis_method = "fallback"
    if not product_index.available():
        raise HTTPException(status_code=503, detail="Product catalog unavailable")
    opts = req.options or {}
    try:
        candidate_recs = product_index.find_similar(
            analysis,
            max_per_category=(opts.maxPerCategory or 3) * 4 if hasattr(opts, "maxPerCategory") else 12,
            include_score=True,
            min_price=getattr(opts, "minPrice", None),
            max_price=getattr(opts, "maxPrice", None),
            exclude_tags=getattr(opts, "excludeTags", None),
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    max_k = (opts.maxPerCategory or 3) if hasattr(opts, "maxPerCategory") else 3
    user_llm_pref = getattr(opts, "useLLMRerank", None)
    use_llm = user_llm_pref if user_llm_pref is not None else llm_ranker.available()
    if use_llm and llm_ranker.available():
        ids = llm_ranker.rerank(analysis, candidate_recs, top_k=max_k)
        if ids:
            recs = {cat: [] for cat in candidate_recs.keys()}
            for cat in candidate_recs.keys():
                idx = {str(p["id"]): p for p in candidate_recs[cat]}
                for _id in ids.get(cat, []):
                    if _id in idx:
                        recs[cat].append(idx[_id])
                for p in candidate_recs[cat]:
                    if len(recs[cat]) >= max_k:
                        break
                    if p not in recs[cat]:
                        recs[cat].append(p)
        else:
            recs = {cat: (candidate_recs[cat][:max_k]) for cat in candidate_recs.keys()}
    else:
        recs = {cat: (candidate_recs[cat][:max_k]) for cat in candidate_recs.keys()}

    as_model = CategoryRecommendations(
        top=[RecommendationItem(**p) for p in recs.get("top", [])],
        pants=[RecommendationItem(**p) for p in recs.get("pants", [])],
        shoes=[RecommendationItem(**p) for p in recs.get("shoes", [])],
        accessories=[RecommendationItem(**p) for p in recs.get("accessories", [])],
    )

    return RecommendationResponse(
        recommendations=as_model,
        analysisMethod=analysis_method,
        styleAnalysis=analysis if analysis_method == "ai" else None,
        requestId=f"req_{int(datetime.utcnow().timestamp())}",
        timestamp=datetime.utcnow().isoformat() + "Z",
    )

