from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .db_recommender import db_pos_recommender

try:
    from .db_recommender import _normalize_slot as _db_normalize_slot  # type: ignore
    from .db_recommender import _normalize_gender as _db_normalize_gender  # type: ignore
except ImportError:  # pragma: no cover - fallback definitions
    def _db_normalize_slot(raw: Optional[str]) -> str:
        value = (str(raw or "").strip().lower())
        if not value:
            return "unknown"
        return value

    def _db_normalize_gender(raw: Optional[str]) -> str:
        value = (str(raw or "").strip().lower())
        if not value:
            return "unknown"
        return value


@dataclass
class ProductIndexConfig:
    categories: Tuple[str, ...] = ("top", "pants", "shoes", "outer", "accessories")
    exact_weight: float = float(os.getenv("PRODUCT_INDEX_EXACT_WEIGHT", "1.0"))
    partial_weight: float = float(os.getenv("PRODUCT_INDEX_PARTIAL_WEIGHT", "0.5"))
    score_threshold: float = float(os.getenv("PRODUCT_INDEX_SCORE_THRESHOLD", "0.0"))
    max_recommendations: int = int(os.getenv("PRODUCT_INDEX_MAX_PER_CATEGORY", "10"))


class ProductIndex:
    """In-memory product index backed by the DB recommender cache."""

    def __init__(self, config: Optional[ProductIndexConfig] = None) -> None:
        self.config = config or ProductIndexConfig()

    def available(self) -> bool:
        return db_pos_recommender.available()

    def _products(self) -> List[Dict]:
        if not self.available():
            raise RuntimeError("DB product catalog is unavailable")
        return list(db_pos_recommender.products)

    def stats(self) -> Dict:
        products = self._products()
        total = len(products)
        categories: Dict[str, int] = {}
        min_price = float("inf")
        max_price = 0.0
        total_price = 0.0

        for item in products:
            cat = str(item.get("category", "unknown"))
            categories[cat] = categories.get(cat, 0) + 1
            price = float(item.get("price") or 0)
            total_price += price
            min_price = min(min_price, price)
            max_price = max(max_price, price)

        avg_price = int(round(total_price / total, 0)) if total > 0 else 0
        if min_price == float("inf"):
            min_price = 0.0

        return {
            "totalProducts": total,
            "categories": categories,
            "priceRange": {
                "min": int(min_price),
                "max": int(max_price),
                "average": int(avg_price),
            },
        }

    def random_products(
        self,
        *,
        limit: int = 18,
        category: Optional[str] = None,
        gender: Optional[str] = None,
    ) -> List[Dict]:
        products = self._products()
        if category:
            normalized_category = _db_normalize_slot(category)
            products = [p for p in products if _db_normalize_slot(p.get("category")) == normalized_category]
        if gender:
            normalized_gender = _db_normalize_gender(gender)
            products = [p for p in products if _db_normalize_gender(p.get("gender")) == normalized_gender]
        random.shuffle(products)
        limit = max(1, min(int(limit), 100))
        result: List[Dict] = []
        for product in products[:limit]:
            result.append(
                {
                    "id": str(product.get("id")),
                    "title": product.get("title") or "",
                    "price": int(product.get("price") or 0),
                    "imageUrl": product.get("imageUrl"),
                    "productUrl": product.get("productUrl"),
                    "tags": product.get("tags") or [],
                    "category": product.get("category") or "top",
                }
            )
        return result

    def _score_product(self, product: Dict, keywords: List[str]) -> float:
        item_text = f"{product.get('title', '')} {' '.join(product.get('tags', []))}".lower()
        score = 0.0
        for kw in keywords:
            query = kw.lower()
            if query in item_text:
                score += self.config.exact_weight
            else:
                tokens = [tok for tok in query.split() if tok]
                if any(token in item_text for token in tokens):
                    score += self.config.partial_weight
        return score

    def find_similar(
        self,
        analysis: Dict,
        *,
        max_per_category: int = 3,
        include_score: bool = True,
        min_price: Optional[int] = None,
        max_price: Optional[int] = None,
        exclude_tags: Optional[List[str]] = None,
    ) -> Dict[str, List[Dict]]:
        products = self._products()
        categories = {cat: [] for cat in self.config.categories}

        keywords: List[str] = []
        for key in (
            "tags",
            "captions",
            "top",
            "pants",
            "shoes",
            "outer",
            "accessories",
            "overall_style",
            "detected_style",
            "colors",
            "categories",
        ):
            value = analysis.get(key)
            if isinstance(value, list):
                keywords.extend([str(v) for v in value if v])

        exclude: set[str] = set()
        if exclude_tags:
            exclude = {str(tag).lower() for tag in exclude_tags if tag}

        score_threshold = float(self.config.score_threshold)
        for product in products:
            cat = str(product.get("category") or "").lower() or "unknown"
            if cat not in categories:
                continue

            score = self._score_product(product, keywords)
            if score <= score_threshold:
                continue

            price_val = int(product.get("price") or 0)
            if min_price is not None and price_val < int(min_price):
                continue
            if max_price is not None and price_val > int(max_price):
                continue

            if exclude and exclude.intersection({str(t).lower() for t in product.get("tags") or []}):
                continue

            item = dict(product)
            if include_score:
                item["score"] = float(score)
            else:
                item.pop("score", None)
            categories[cat].append(item)

        for cat, items in categories.items():
            items.sort(key=lambda it: float(it.get("score", 0.0)), reverse=True)
            categories[cat] = items[: max_per_category]

            if not include_score:
                for entry in categories[cat]:
                    entry.pop("score", None)

        return categories


product_index = ProductIndex()

