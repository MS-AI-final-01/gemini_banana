from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_CATALOG_PATH = ROOT_DIR / "data" / "catalog.json"


@dataclass
class CatalogServiceConfig:
    catalog_path: Path = Path(os.getenv("CATALOG_PATH", str(DEFAULT_CATALOG_PATH)))
    max_recommendations: int = 10
    score_threshold: float = 0.0
    categories: Tuple[str, ...] = ("top", "pants", "shoes", "accessories")


class CatalogService:
    def __init__(self, config: Optional[CatalogServiceConfig] = None) -> None:
        self.config = config or CatalogServiceConfig()
        self._catalog: List[Dict] = []
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(Path(self.config.catalog_path).read_text(encoding="utf-8"))
            # Basic normalization
            for p in data:
                p.setdefault("tags", [])
                p.setdefault("title", "")
                p.setdefault("category", "")
            self._catalog = data
            print(f"[CatalogService] Loaded {len(self._catalog)} products from {self.config.catalog_path}")
        except Exception as e:
            print(f"[CatalogService] Failed to load catalog: {e}")
            self._catalog = []

    def reload(self) -> bool:
        try:
            self._load()
            return True
        except Exception:
            return False

    def get_all(self) -> List[Dict]:
        return list(self._catalog)

    def stats(self) -> Dict:
        total = len(self._catalog)
        cats: Dict[str, int] = {}
        min_price = float("inf")
        max_price = 0.0
        total_price = 0.0

        for p in self._catalog:
            cats[p.get("category", "unknown")] = cats.get(p.get("category", "unknown"), 0) + 1
            price = float(p.get("price", 0))
            total_price += price
            min_price = min(min_price, price)
            max_price = max(max_price, price)

        avg_price = int(round(total_price / total, 0)) if total > 0 else 0
        if min_price == float("inf"):
            min_price = 0

        return {
            "totalProducts": total,
            "categories": cats,
            "priceRange": {"min": int(min_price), "max": int(max_price), "average": int(avg_price)},
        }

    def _score_product(self, product: Dict, keywords: List[str]) -> float:
        item_text = f"{product.get('title','') } {' '.join(product.get('tags', []))}".lower()
        score = 0.0
        for kw in keywords:
            kw_l = kw.lower()
            if kw_l in item_text:
                score += 1.0
            else:
                # partial match on token
                if any(tok for tok in kw_l.split() if tok and tok in item_text):
                    score += 0.5
        return score

    def search(self, keywords: List[str], *, categories: Optional[List[str]] = None, max_results: int = 10,
               score_threshold: float = 0.0) -> List[Dict]:
        categories = categories or list(self.config.categories)
        normalized = [k.strip().lower() for k in keywords if k and k.strip()]
        results: List[Tuple[float, Dict]] = []
        for product in self._catalog:
            if product.get("category") not in categories:
                continue
            s = self._score_product(product, normalized)
            if s > score_threshold:
                copy = dict(product)
                copy["score"] = s
                results.append((s, copy))
        results.sort(key=lambda t: t[0], reverse=True)
        return [p for _, p in results[:max_results]]

    def find_similar(self, analysis: Dict, *, max_per_category: int = 3, include_score: bool = True,
                      min_price: Optional[int] = None, max_price: Optional[int] = None,
                      exclude_tags: Optional[List[str]] = None) -> Dict[str, List[Dict]]:
        keywords: List[str] = []
        # collect keywords from analysis structure
        for key in ("tags", "captions", "top", "pants", "shoes", "overall_style", "detected_style", "colors", "categories"):
            val = analysis.get(key)
            if isinstance(val, list):
                keywords.extend([str(v) for v in val])

        recs = {c: [] for c in self.config.categories}
        for cat in self.config.categories:
            cat_products = self.search(keywords, categories=[cat], max_results=max_per_category * 3)
            # filters
            if min_price is not None or max_price is not None:
                cat_products = [p for p in cat_products if (min_price or 0) <= int(p.get("price", 0)) <= (max_price or 1_000_000_000)]
            if exclude_tags:
                ex = set(t.lower() for t in exclude_tags)
                cat_products = [p for p in cat_products if not ex.intersection({t.lower() for t in p.get("tags", [])})]

            cat_products = cat_products[:max_per_category]
            if not include_score:
                for p in cat_products:
                    p.pop("score", None)
            recs[cat] = cat_products

        return recs


@lru_cache(maxsize=1)
def get_catalog_service() -> CatalogService:
    return CatalogService()

