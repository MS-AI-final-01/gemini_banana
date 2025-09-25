# db_recommender.py (boot-source logging + FAISS bytes/ndarray compat)
from __future__ import annotations

import logging
import os
import pickle
import re
import zlib
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

try:
    import redis  # type: ignore
except Exception:
    redis = None  # type: ignore

try:
    import faiss  # type: ignore
except Exception:
    faiss = None  # type: ignore

try:
    from sqlalchemy import create_engine, text  # type: ignore
    from sqlalchemy.engine import Engine  # type: ignore
except Exception:
    create_engine = None  # type: ignore
    text = None  # type: ignore

    class Engine:  # type: ignore
        pass


# -----------------------------
# Regex utils
# -----------------------------
RE_UNISEX = re.compile(
    r"(?:\buni(?:sex)?\b|男女|공용|유니섹스|남녀|남여|공용/유니섹스|all\s*genders?)",
    re.I,
)
RE_KIDS = re.compile(
    r"(?:\bkid(?:s)?\b|\bchild(?:ren)?\b|\byouth\b|\bjunior\b|boys?\s*&\s*girls?|아동|키즈)",
    re.I,
)
RE_FEMALE = re.compile(
    r"(?:\bwomen\b|\bwoman\b|\bfemale\b|\bladies\b|\blady\b|\bgirls?\b|여성|여자|우먼)",
    re.I,
)
RE_MALE = re.compile(
    r"(?:\bmen\b|\bman\b|\bmale\b|\bboys?\b|\bmens\b|\bman's\b|\bmans\b|남성|남자|맨)",
    re.I,
)


# -----------------------------
# Configs
# -----------------------------
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
        flag = os.getenv("DB_RECO_ENABLED", "").strip().lower()
        if flag in {"0", "false", "off", "no"}:
            return ""
        if not (self.host and self.user):
            return ""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}?sslmode={self.sslmode}"


@dataclass
class RedisConfig:
    host: str = os.getenv("REDIS_HOST", "localhost")
    port: int = int(os.getenv("REDIS_PORT", "6379"))
    password: str = os.getenv("REDIS_PASSWORD", "")
    db: int = int(os.getenv("REDIS_DB", "0"))
    ssl: bool = os.getenv("REDIS_SSL", "false").lower() == "true"

    @property
    def url(self) -> str:
        scheme = "rediss" if self.ssl else "redis"
        auth = f":{self.password}@" if self.password else ""
        return f"{scheme}://{auth}{self.host}:{self.port}/{self.db}"


# -----------------------------
# Redis helpers
# -----------------------------
REDIS_KEYS = {
    "dense": "embeddings:dense_vectors",
    "colbert": "embeddings:colbert_data",
    "faiss": "embeddings:faiss_index",
    "meta": "embeddings:metadata",
    "colbert_loaded": "embeddings:colbert_loaded",
    "colbert_dim": "embeddings:colbert_dim",
}
REDIS_CHUNK_BYTES = int(os.getenv("REDIS_CHUNK_BYTES", str(8 * 1024 * 1024)))  # 8MB
REDIS_ENABLED = os.getenv("RECO_USE_REDIS", "true").lower() not in {
    "0",
    "false",
    "no",
    "off",
}


def _r_set_large(r, base_key: str, raw: bytes, *, compress=True) -> None:
    if compress:
        raw = zlib.compress(raw, level=3)
    n = len(raw)
    parts = (n + REDIS_CHUNK_BYTES - 1) // REDIS_CHUNK_BYTES
    pipe = r.pipeline()
    pipe.delete(base_key + ":parts")
    for i in range(parts):
        seg = raw[i * REDIS_CHUNK_BYTES : (i + 1) * REDIS_CHUNK_BYTES]
        pipe.set(f"{base_key}:part:{i}", seg)
    pipe.set(base_key + ":parts", str(parts).encode())
    pipe.execute()


def _r_get_large(r, base_key: str, *, decompress=True) -> Optional[bytes]:
    parts_b = r.get(base_key + ":parts")
    if not parts_b:
        return None
    parts = int(parts_b.decode())
    pipe = r.pipeline()
    for i in range(parts):
        pipe.get(f"{base_key}:part:{i}")
    blobs = pipe.execute()
    if any(b is None for b in blobs):
        return None
    data = b"".join(blobs)
    if decompress:
        data = zlib.decompress(data)
    return data


def purge_redis_embeddings(r) -> None:
    ks = [
        REDIS_KEYS[k]
        for k in ("dense", "colbert", "faiss", "meta", "colbert_loaded", "colbert_dim")
    ]
    part_keys = []
    for k in ks:
        part_keys.append(k + ":parts")
        for i in range(0, 4096):  # up to 4096 chunks safety
            part_keys.append(f"{k}:part:{i}")
    r.delete(*ks, *part_keys)


# -----------------------------
# Normalizers
# -----------------------------


def _normalize_slot(raw: Optional[str]) -> str:
    c = str(raw or "").strip().lower()
    if not c:
        return "unknown"
    if c in ["man_outer", "woman_outer"]:
        return "outer"
    elif c in ["man_top", "woman_top"]:
        return "top"
    elif c in ["man_bottom", "woman_bottom"]:
        return "pants"
    elif c in ["man_shoes", "woman_shoes"]:
        return "shoes"
    elif c == "woman_dress_skirt":
        return "pants"
    return c


def _normalize_gender(raw: Optional[str]) -> str:
    g = str(raw or "").strip()
    if not g:
        return "unknown"
    g = re.sub(r"[_\-/]+", " ", g)
    if RE_UNISEX.search(g):
        return "unisex"
    if RE_KIDS.search(g):
        return "kids"
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
    if not price_raw:
        return 0
    s = re.sub(r"[^0-9]", "", str(price_raw).strip())
    if not s:
        return 0
    try:
        return int(s)
    except ValueError:
        return 0


# -----------------------------
# IndexOnlyRecommender
# -----------------------------
class IndexOnlyRecommender:
    """
    DB 기반 하이브리드 추천 (Dense + ColBERT) with Redis cache
    - Redis 로드 실패/부분로딩 시 → DB 로드 (검증 OK일 때만 Redis 저장)
    """

    def __init__(
        self,
        cfg: Optional[DbConfig] = None,
        w_dense: float = 0.5,
        w_maxsim: float = 0.5,
    ):
        self.logger = logging.getLogger(__name__)
        self.cfg = cfg or DbConfig()
        self.w_dense = w_dense
        self.w_maxsim = w_maxsim

        # Handles
        self.engine: Optional[Engine] = None
        self.redis_client = None
        self.redis_available = False

        # Data
        self.dense_index = None  # FAISS
        self.metadata: Optional[pd.DataFrame] = None
        self.dense_vectors: Optional[np.ndarray] = None
        self.colbert_data: Dict[int, Dict[str, np.ndarray]] = {}
        self.colbert_loaded = False
        self.colbert_dim: Optional[int] = None
        self.colbert_cache: Dict[int, np.ndarray] = {}
        self.cache_limit = 1000

        # Boot-source tracking
        self.boot_source = "unknown"

        # Init
        self._init_redis()
        self._init_db_and_load()

    # ---------- init ----------
    def _init_redis(self):
        if not REDIS_ENABLED or redis is None:
            self.logger.info("[Reco] Redis disabled or not installed")
            return
        try:
            rcfg = RedisConfig()
            params = {}
            if rcfg.ssl:
                params.update({"ssl_cert_reqs": None, "ssl_check_hostname": False})
            self.redis_client = redis.from_url(rcfg.url, decode_responses=False, **params)  # type: ignore
            self.redis_client.ping()  # type: ignore
            self.redis_available = True
            self.logger.info(f"[Reco] Redis connected: {rcfg.url}")
        except Exception as e:
            self.logger.warning(f"[Reco] Redis connect failed: {e}")
            self.redis_available = False
            self.redis_client = None

    def _init_db_and_load(self):
        if not (self.cfg.url and create_engine is not None and text is not None):
            self.logger.warning("[Reco] DB URL not provided or SQLAlchemy missing")
            return
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
            self.logger.info("[Reco] DB connected")
            self._load_all()
        except Exception as exc:
            self.logger.exception(f"[Reco] DB init failed: {exc}")
            self.engine = None

    # ---------- load pipeline ----------
    def _load_all(self) -> None:
        try:
            # 1) Try Redis
            if self._load_from_redis():
                self.boot_source = "redis"
                self.logger.info(
                    "[Reco] BOOT_SOURCE=redis (loaded everything from Redis)"
                )
                try:
                    if self.redis_available and self.redis_client:
                        self.redis_client.set(
                            "embeddings:boot_source", b"redis", ex=3600
                        )
                except Exception:
                    pass
                return

            # 2) Load from DB
            self.logger.info("[Reco] Redis miss → loading from DB…")
            self._load_products_from_db()
            self._load_embeddings_from_db()
            self._build_faiss_index()
            self._build_colbert_index()

            # 3) Validate before saving to Redis
            if self._is_data_consistent():
                if self.redis_available:
                    self._save_to_redis()
                    self.boot_source = "db->redis"
                    self.logger.info(
                        "[Reco] BOOT_SOURCE=db->redis (loaded from DB, saved to Redis)"
                    )
                    try:
                        if self.redis_client:
                            self.redis_client.set(
                                "embeddings:boot_source", b"db->redis", ex=3600
                            )
                    except Exception:
                        pass
                else:
                    self.boot_source = "db"
                    self.logger.info(
                        "[Reco] BOOT_SOURCE=db (loaded from DB, Redis disabled)"
                    )
            else:
                self.boot_source = "db_inconsistent"
                self.logger.warning(
                    "[Reco] BOOT_SOURCE=db_inconsistent (loaded from DB but inconsistent; skipped Redis save)"
                )
        except Exception as e:
            self.boot_source = "error"
            self.logger.error(f"[Reco] BOOT_SOURCE=error (load failed): {e}")
            raise

    def _is_data_consistent(self) -> bool:
        try:
            n_meta = (
                len(self.metadata) if isinstance(self.metadata, pd.DataFrame) else 0
            )
            n_dense = (
                int(self.dense_vectors.shape[0])
                if isinstance(self.dense_vectors, np.ndarray)
                and self.dense_vectors.ndim == 2
                else 0
            )
            n_faiss = (
                int(getattr(self.dense_index, "ntotal", 0))
                if self.dense_index is not None
                else 0
            )

            if not (n_meta and n_dense and n_faiss):
                self.logger.warning(
                    f"[Reco] Consistency fail (non-positive): meta={n_meta}, dense={n_dense}, faiss={n_faiss}"
                )
                return False
            if not (n_meta == n_dense == n_faiss):
                self.logger.warning(
                    f"[Reco] Count mismatch: meta={n_meta}, dense={n_dense}, faiss={n_faiss}"
                )
                return False
            return True
        except Exception as e:
            self.logger.warning(f"[Reco] Consistency check error: {e}")
            return False

    # ---------- Redis I/O ----------
    def _load_from_redis(self) -> bool:
        if not self.redis_available or self.redis_client is None:
            return False
        try:
            dense_b = _r_get_large(self.redis_client, REDIS_KEYS["dense"])
            cdata_b = _r_get_large(self.redis_client, REDIS_KEYS["colbert"])
            faiss_b = _r_get_large(self.redis_client, REDIS_KEYS["faiss"])
            meta_b = _r_get_large(self.redis_client, REDIS_KEYS["meta"])

            if not all([dense_b, cdata_b, faiss_b, meta_b]):
                self.logger.info("[Reco] Redis missing some parts → fallback DB")
                return False

            # 1) dense vectors
            try:
                self.dense_vectors = pickle.loads(dense_b)
                if (
                    not isinstance(self.dense_vectors, np.ndarray)
                    or self.dense_vectors.ndim != 2
                ):
                    raise TypeError(
                        "Dense vectors invalid type or shape after unpickling"
                    )
            except Exception as e:
                self.logger.warning(f"[Reco] Failed to unpickle dense vectors: {e}")
                return False

            # 2) ColBERT
            try:
                self.colbert_data = pickle.loads(cdata_b)
                if not isinstance(self.colbert_data, dict):
                    raise TypeError("Colbert data invalid type after unpickling")
            except Exception as e:
                self.logger.warning(f"[Reco] Failed to unpickle colbert data: {e}")
                return False

            # 3) FAISS
            try:
                if faiss is None:
                    raise RuntimeError("FAISS not available")
                if not isinstance(faiss_b, (bytes, bytearray, memoryview, np.ndarray)):
                    raise ValueError(f"Invalid FAISS buffer type: {type(faiss_b)}")
                buf_for_load = (
                    np.frombuffer(faiss_b, dtype=np.uint8)
                    if isinstance(faiss_b, (bytes, bytearray, memoryview))
                    else faiss_b
                )
                self.dense_index = faiss.deserialize_index(buf_for_load)
                if self.dense_index is None:
                    raise ValueError("FAISS index is None after deserialization")
                ntotal = getattr(self.dense_index, "ntotal", 0)
                if ntotal <= 0:
                    raise ValueError(f"FAISS index invalid ntotal: {ntotal}")
                if (
                    self.dense_vectors is not None
                    and ntotal != self.dense_vectors.shape[0]
                ):
                    raise ValueError(
                        f"FAISS ntotal ({ntotal}) != dense_vectors rows ({self.dense_vectors.shape[0]})"
                    )
            except Exception as e:
                self.logger.warning(f"[Reco] Failed to deserialize FAISS index: {e}")
                try:
                    self.logger.info("[Reco] Purging corrupted Redis cache...")
                    purge_redis_embeddings(self.redis_client)
                except Exception as purge_e:
                    self.logger.warning(f"[Reco] Failed to purge Redis: {purge_e}")
                return False

            # 4) metadata
            try:
                self.metadata = pickle.loads(meta_b)
                if not isinstance(self.metadata, pd.DataFrame):
                    raise TypeError("Metadata invalid type after unpickling")
            except Exception as e:
                self.logger.warning(f"[Reco] Failed to unpickle metadata: {e}")
                return False

            # flags
            cl = self.redis_client.get(REDIS_KEYS["colbert_loaded"]) or None
            if cl:
                self.colbert_loaded = pickle.loads(cl)
            cd = self.redis_client.get(REDIS_KEYS["colbert_dim"]) or None
            if cd:
                self.colbert_dim = pickle.loads(cd)

            if not self._is_data_consistent():
                self.logger.warning("[Reco] Redis data inconsistent → reject & purge")
                try:
                    purge_redis_embeddings(self.redis_client)
                except Exception as purge_e:
                    self.logger.warning(f"[Reco] Failed to purge Redis: {purge_e}")
                return False

            self.logger.info("[Reco] Redis load OK")
            return True

        except Exception as e:
            self.logger.error(f"[Reco] Final Redis load failed: {e}")
            if self.redis_available and self.redis_client is not None:
                try:
                    self.logger.info(
                        "[Reco] Purging Redis cache due to load failure..."
                    )
                    purge_redis_embeddings(self.redis_client)
                except Exception as purge_e:
                    self.logger.warning(f"[Reco] Failed to purge Redis: {purge_e}")
            return False

    def _save_to_redis(self) -> None:
        if not self.redis_available or self.redis_client is None:
            return
        try:
            if not self._is_data_consistent():
                self.logger.warning("[Reco] Skip Redis save due to inconsistent data")
                return

            # dense
            if isinstance(self.dense_vectors, np.ndarray):
                _r_set_large(
                    self.redis_client,
                    REDIS_KEYS["dense"],
                    pickle.dumps(self.dense_vectors, protocol=pickle.HIGHEST_PROTOCOL),
                )

            # colbert
            if isinstance(self.colbert_data, dict) and self.colbert_data:
                _r_set_large(
                    self.redis_client,
                    REDIS_KEYS["colbert"],
                    pickle.dumps(self.colbert_data, protocol=pickle.HIGHEST_PROTOCOL),
                )

            # faiss
            if self.dense_index is not None and faiss is not None:
                try:
                    buf = faiss.serialize_index(self.dense_index)
                    if isinstance(buf, np.ndarray):
                        if buf.dtype != np.uint8:
                            raise ValueError(f"Invalid FAISS buffer dtype: {buf.dtype}")
                        faiss_bytes = buf.tobytes()
                    elif isinstance(buf, (bytes, bytearray, memoryview)):
                        faiss_bytes = bytes(buf)
                    else:
                        raise ValueError(
                            f"Invalid FAISS serialization result type: {type(buf)}"
                        )
                    if len(faiss_bytes) < 100:
                        raise ValueError(
                            f"Invalid FAISS serialization length: {len(faiss_bytes)}"
                        )
                    # verify
                    buf_for_verify = (
                        np.frombuffer(faiss_bytes, dtype=np.uint8)
                        if isinstance(faiss_bytes, (bytes, bytearray, memoryview))
                        else faiss_bytes
                    )
                    test_index = faiss.deserialize_index(buf_for_verify)
                    if getattr(test_index, "ntotal", 0) != getattr(
                        self.dense_index, "ntotal", 0
                    ):
                        raise ValueError("FAISS serialization verification failed")
                    _r_set_large(self.redis_client, REDIS_KEYS["faiss"], faiss_bytes)
                    self.logger.info(
                        f"[Reco] FAISS index saved to Redis (size: {len(faiss_bytes):,} bytes)"
                    )
                except Exception as e:
                    self.logger.error(f"[Reco] FAISS save failed: {e}")

            # metadata
            if isinstance(self.metadata, pd.DataFrame):
                _r_set_large(
                    self.redis_client,
                    REDIS_KEYS["meta"],
                    pickle.dumps(self.metadata, protocol=pickle.HIGHEST_PROTOCOL),
                )

            # flags
            self.redis_client.set(
                REDIS_KEYS["colbert_loaded"], pickle.dumps(self.colbert_loaded)
            )
            self.redis_client.set(
                REDIS_KEYS["colbert_dim"], pickle.dumps(self.colbert_dim)
            )

            self.logger.info("[Reco] Saved to Redis (chunked)")
        except Exception as e:
            self.logger.error(f"[Reco] Redis save failed: {e}")

    # ---------- DB loaders ----------
    def _load_products_from_db(self) -> None:
        assert self.engine is not None and text is not None
        with self.engine.begin() as conn:
            rows = (
                (
                    conn.execute(
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
                    )
                )
                .mappings()
                .all()
            )
        data = []
        for r in rows:
            data.append(
                {
                    "pos": int(r.get("pos")),
                    "Product_U": r.get("Product_U"),
                    "Product_img_U": r.get("Product_img_U"),
                    "Product_N": r.get("Product_N"),
                    "Product_Desc": r.get("Product_Desc"),
                    "Product_P": _parse_price(r.get("Product_P")),
                    "Category": r.get("Category"),
                    "Product_B": r.get("Product_B"),
                    "Product_G": r.get("Product_G"),
                    "Image_P": r.get("Image_P"),
                }
            )
        self.metadata = pd.DataFrame(data)
        self.logger.info(f"[Reco] products2 loaded: {len(self.metadata):,} rows")

    def _load_embeddings_from_db(self) -> None:
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
            self.logger.error("[Reco] embeddings2 empty")
            return

        dense_list: List[np.ndarray] = []
        colbert_vecs_list: List[Optional[np.ndarray]] = []
        colbert_offsets_list: List[Optional[np.ndarray]] = []

        for pos, dense_bytes, colbert_vecs_bytes, colbert_offsets_bytes in data:
            if dense_bytes:
                dense_vec = np.frombuffer(dense_bytes, dtype=np.float32)
                dense_list.append(dense_vec)
            else:
                dense_list.append(None)  # type: ignore

            if colbert_vecs_bytes:
                v1d = np.frombuffer(colbert_vecs_bytes, dtype=np.float32)
                if len(v1d) > 0 and len(v1d) % 1024 == 0:
                    num_tok = len(v1d) // 1024
                    v = v1d.reshape(num_tok, 1024)
                else:
                    v = v1d.reshape(-1, 1) if len(v1d) > 0 else None
                colbert_vecs_list.append(v)
            else:
                colbert_vecs_list.append(None)

            if colbert_offsets_bytes:
                offs = np.frombuffer(colbert_offsets_bytes, dtype=np.int64)
                colbert_offsets_list.append(offs)
            else:
                colbert_offsets_list.append(None)

        valid_dense = [v for v in dense_list if isinstance(v, np.ndarray)]
        if valid_dense:
            self.dense_vectors = np.vstack(valid_dense).astype(np.float32, copy=False)
            self.logger.info(f"[Reco] Dense loaded: {self.dense_vectors.shape}")
        else:
            self.dense_vectors = None
            self.logger.warning("[Reco] No valid dense vectors")

        self.colbert_data = {}
        for i, (vecs, offs) in enumerate(zip(colbert_vecs_list, colbert_offsets_list)):
            if (
                isinstance(vecs, np.ndarray)
                and isinstance(offs, np.ndarray)
                and len(offs) >= 2
            ):
                self.colbert_data[i] = {
                    "vecs": vecs,
                    "start_offset": int(offs[0]),
                    "end_offset": int(offs[1]),
                }
        self.colbert_loaded = bool(self.colbert_data)
        if self.colbert_loaded:
            first_key = next(iter(self.colbert_data))
            fv = self.colbert_data[first_key]["vecs"]
            self.colbert_dim = int(fv.shape[1]) if fv.ndim == 2 else int(fv.shape[0])
        self.logger.info(
            f"[Reco] ColBERT loaded: items={len(self.colbert_data)}, d={self.colbert_dim}"
        )

    def _build_faiss_index(self) -> None:
        if self.dense_vectors is None:
            self.logger.error("[Reco] No dense vectors to build FAISS")
            return
        if faiss is None:
            self.logger.error("[Reco] faiss not installed")
            return
        d = int(self.dense_vectors.shape[1])
        index = faiss.IndexFlatIP(d)  # type: ignore
        index.add(self.dense_vectors.astype(np.float32))  # type: ignore
        self.dense_index = index
        self.logger.info(
            f"[Reco] FAISS built: n={getattr(self.dense_index,'ntotal',0):,}, d={d}"
        )

    def _build_colbert_index(self) -> None:
        if not self.colbert_data:
            self.logger.warning("[Reco] No ColBERT data; skip")
            self.colbert_loaded = False
            return
        self.colbert_loaded = True
        self.colbert_cache = {}
        self.cache_limit = 1000
        self.logger.info(
            f"[Reco] ColBERT ready: items={len(self.colbert_data)}, d={self.colbert_dim}"
        )

    # ---------- helpers ----------
    def available(self) -> bool:
        return (
            self.engine is not None
            and self.dense_index is not None
            and self.metadata is not None
        )

    @staticmethod
    def _minmax(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype="float32")
        if x.size == 0:
            return x
        x_min, x_max = float(np.min(x)), float(np.max(x))
        if x_max - x_min < 1e-12:
            return np.zeros_like(x, dtype="float32")
        return (x - x_min) / (x_max - x_min)

    def _get_doc_tokens(self, product_id: int) -> Optional[np.ndarray]:
        if product_id in self.colbert_cache:
            return self.colbert_cache[product_id]
        data = self.colbert_data.get(product_id)
        if not data:
            return None
        vecs = data["vecs"]
        if len(self.colbert_cache) >= self.cache_limit:
            self.colbert_cache.pop(next(iter(self.colbert_cache)))
        self.colbert_cache[product_id] = vecs
        return vecs

    def compute_maxsim(
        self, q_tok: np.ndarray, d_tok: np.ndarray, per_token_mode: str = "raw"
    ) -> float:
        if q_tok.size == 0 or d_tok.size == 0:
            return 0.0
        if q_tok.ndim != 2 or d_tok.ndim != 2 or q_tok.shape[1] != d_tok.shape[1]:
            raise RuntimeError("MaxSim shape mismatch")
        q = (q_tok.T / (np.linalg.norm(q_tok, axis=1) + 1e-8)).T
        d = (d_tok.T / (np.linalg.norm(d_tok, axis=1) + 1e-8)).T
        sims = np.dot(q, d.T)
        per_tok = np.max(sims, axis=1)
        if per_token_mode == "relu":
            per_tok = np.maximum(per_tok, 0.0)
        elif per_token_mode == "shift01":
            per_tok = (per_tok + 1.0) * 0.5
        elif per_token_mode != "raw":
            raise ValueError("per_token_mode ∈ {raw,relu,shift01}")
        return float(per_tok.sum())

    # ---------- recommend ----------
    def recommend_indices(
        self,
        self_idx: int,
        top_n: int = 5,
        cand_k: int = 50,
        same_category: bool = True,
        w_dense: Optional[float] = None,
        w_maxsim: Optional[float] = None,
    ) -> List[int]:
        if not self.available():
            raise RuntimeError("IndexOnlyRecommender unavailable")
        w_dense = w_dense or self.w_dense
        w_maxsim = w_maxsim or self.w_maxsim
        if not (0 <= self_idx < len(self.metadata)):
            raise ValueError(f"Invalid self_idx: {self_idx}")
        query_vec = self.dense_vectors[self_idx].reshape(1, -1)
        dense_scores, cand = self.dense_index.search(query_vec, cand_k)  # type: ignore
        dense_scores = dense_scores[0]
        cand_ids = cand[0].astype(int)
        keep_mask = cand_ids != self_idx
        cand_ids = cand_ids[keep_mask]
        dense_scores = dense_scores[keep_mask]
        if same_category:
            target_category = str(self.metadata.iloc[self_idx].get("Category", ""))
            if target_category:
                norm_target = _normalize_slot(target_category)
                meta_cats = (
                    self.metadata.loc[cand_ids, "Category"].map(_normalize_slot).values
                )
                mask = meta_cats == norm_target
                cand_ids = cand_ids[mask]
                dense_scores = dense_scores[mask]
        if cand_ids.size == 0:
            return []
        maxsim_scores = None
        if self.colbert_loaded and self_idx in self.colbert_data:
            q_tok = self.colbert_data[self_idx]["vecs"]
            maxsim_list = []
            for pid in cand_ids:
                doc_tok = self._get_doc_tokens(pid)
                if doc_tok is None or doc_tok.size == 0:
                    maxsim_list.append(float("-inf"))
                else:
                    maxsim_list.append(self.compute_maxsim(q_tok, doc_tok))
            maxsim_scores = np.array(maxsim_list, dtype="float32")
        if maxsim_scores is not None:
            dense_norm = self._minmax(dense_scores)
            maxsim_norm = self._minmax(maxsim_scores)
            combined = w_dense * dense_norm + w_maxsim * maxsim_norm
        else:
            combined = dense_scores
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
        same_category: bool = True,
    ) -> List[Dict]:
        if not self.available():
            raise RuntimeError("IndexOnlyRecommender unavailable")
        if not positions:
            return []
        all_recommendations: List[Dict] = []
        for pos in positions:
            if 0 <= pos < len(self.metadata):
                rec_results = self.recommend_indices(
                    self_idx=pos,
                    top_n=top_k,
                    cand_k=50,
                    same_category=same_category,
                    w_dense=w_dense,
                    w_maxsim=w_maxsim,
                )
                for idx, score in rec_results:
                    if 0 <= idx < len(self.metadata):
                        row = self.metadata.iloc[idx]
                        all_recommendations.append(
                            {
                                "id": str(idx),
                                "pos": int(idx),
                                "title": str(row.get("Product_N", "")),
                                "price": int(row.get("Product_P", 0)),
                                "tags": [
                                    str(row.get("Product_B", "")),
                                    str(row.get("Product_G", "")),
                                ],
                                "category": _normalize_slot(row.get("Category", "")),
                                "gender": _normalize_gender(row.get("Product_G", "")),
                                "imageUrl": str(row.get("Product_img_U", "")),
                                "productUrl": str(row.get("Product_U", "")),
                                "score": float(score),
                            }
                        )
        seen = set()
        uniq = []
        for rec in all_recommendations:
            if rec["pos"] not in seen:
                seen.add(rec["pos"])
                uniq.append(rec)
                if len(uniq) >= top_k:
                    break
        return uniq

    def recommend_by_embedding(
        self,
        query_embedding: List[float],
        *,
        query_colbert: Optional[List[List[float]]] = None,
        category: Optional[str] = None,
        top_k: int = 5,
        w_dense: Optional[float] = None,
        w_maxsim: Optional[float] = None,
    ) -> List[Dict]:
        if not self.available():
            raise RuntimeError("IndexOnlyRecommender unavailable")
        w_dense = w_dense or self.w_dense
        w_maxsim = w_maxsim or self.w_maxsim
        qv = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        dense_scores, cand = self.dense_index.search(qv, top_k * 10)  # type: ignore
        dense_scores = dense_scores[0]
        cand_ids = cand[0].astype(int)
        if category:
            norm_c = _normalize_slot(category)
            meta_cats = (
                self.metadata.loc[cand_ids, "Category"].map(_normalize_slot).values
            )
            mask = meta_cats == norm_c
            cand_ids = cand_ids[mask]
            dense_scores = dense_scores[mask]
        if cand_ids.size == 0:
            return []
        combined = dense_scores
        if query_colbert is not None and self.colbert_loaded:
            q_tok = np.array(query_colbert, dtype=np.float32)
            maxsim_list = []
            for pid in cand_ids:
                doc_tokens = self.colbert_data.get(pid, {}).get("vecs")
                if isinstance(doc_tokens, np.ndarray) and doc_tokens.size > 0:
                    maxsim_list.append(self.compute_maxsim(q_tok, doc_tokens))
                else:
                    maxsim_list.append(float("-inf"))
            maxsim_scores = np.array(maxsim_list, dtype=np.float32)
            d_used = self._minmax(dense_scores)
            m_used = self._minmax(maxsim_scores)
            combined = w_dense * d_used + w_maxsim * m_used
        order = np.argsort(combined)[::-1]
        final_ids = cand_ids[order][:top_k]
        results = []
        for rank, idx in enumerate(final_ids, 1):
            if 0 <= idx < len(self.metadata):
                row = self.metadata.iloc[idx]
                results.append(
                    {
                        "id": str(idx),
                        "pos": int(idx),
                        "title": str(row.get("Product_N", "")),
                        "price": int(row.get("Product_P", 0)),
                        "tags": [
                            str(row.get("Product_B", "")),
                            str(row.get("Product_G", "")),
                        ],
                        "category": _normalize_slot(row.get("Category", "")),
                        "gender": _normalize_gender(row.get("Product_G", "")),
                        "imageUrl": str(row.get("Product_img_U", "")),
                        "productUrl": str(row.get("Product_U", "")),
                        "score": float(combined[order][:top_k][rank - 1]),
                    }
                )
        return results


# -----------------------------
# DbPosRecommender (legacy dense)
# -----------------------------
class DbPosRecommender:
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
                        "keepalives": 1,
                        "keepalives_idle": 30,
                        "keepalives_interval": 10,
                        "keepalives_count": 5,
                        "connect_timeout": int(os.getenv("DB_CONNECT_TIMEOUT", "5")),
                    },
                )
                with self.engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                self.logger.info("[DbPos] DB connected")
                self._load_all()
                if self.available():
                    self.logger.info(f"[DbPos] Loaded {len(self.products)} products")
            except Exception as exc:
                self.logger.exception(f"[DbPos] init failed: {exc}")
                self.engine = None

    def _load_all(self) -> None:
        assert self.engine is not None and text is not None
        with self.engine.begin() as conn:
            rows = (
                (
                    conn.execute(
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
                    )
                )
                .mappings()
                .all()
            )
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
            norm_cat = _normalize_slot(r.get("Category"))
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
            self.logger.error("[DbPos] embeddings2 empty")
            self.products = []
            self.emb = None
            return
        dense_vectors = []
        for row in data:
            if row[1]:
                dense_vec = np.frombuffer(row[1], dtype=np.float32)
                dense_vectors.append(dense_vec)
        if not dense_vectors:
            self.logger.error("[DbPos] No valid dense vectors")
            self.products = []
            self.emb = None
            return
        mat = np.vstack(dense_vectors).astype(np.float32, copy=False)
        if len(self.products) != mat.shape[0]:
            self.logger.error(
                f"[DbPos] count mismatch: products={len(self.products)}, emb_rows={mat.shape[0]}"
            )
            self.products = []
            self.emb = None
            self.emb_norm = None
            self.prices = None
            return
        self.emb = mat
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
        w2: float = 0.03,
    ) -> np.ndarray:
        emb_norm = self.emb_norm  # type: ignore
        prices = self.prices  # type: ignore
        sim = emb_norm @ query_vec
        avg_price = float(prices.mean())
        clog = np.log1p(prices)
        qlog = np.log1p(avg_price)
        price_score = np.exp(-alpha * np.abs(clog - qlog))
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
        q = self.emb_norm[positions].mean(axis=0)  # type: ignore
        qn = np.linalg.norm(q)
        if qn == 0:
            qn = 1e-8
        q = q / qn
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
        if not self.available():
            raise RuntimeError("DbPosRecommender unavailable")
        n = len(self.products)
        k = max(1, min(int(top_k), n))
        query_vec = np.array(query_embedding, dtype=np.float32)
        qn = np.linalg.norm(query_vec)
        if qn == 0:
            qn = 1e-8
        query_vec = query_vec / qn
        total = self._calculate_similarity_scores(query_vec, alpha=alpha, w1=w1, w2=w2)
        if category:
            norm_c = _normalize_slot(category)
            filt = np.full(n, -np.inf, dtype=np.float32)
            for i, product in enumerate(self.products):
                if _normalize_slot(product.get("category", "")) == norm_c:
                    filt[i] = total[i]
            total = filt
        if k >= n:
            top_idx = np.argsort(-total)
        else:
            part = np.argpartition(-total, kth=k - 1)[:k]
            top_idx = part[np.argsort(-total[part])]
        out: List[Dict] = []
        for i in top_idx.tolist():
            if total[i] == -np.inf:
                continue
            p = dict(self.products[i])
            p["score"] = float(total[i])
            out.append(p)
        return out


# -----------------------------
# Module-level singletons
# -----------------------------
_flag = os.getenv("DB_RECO_ENABLED", "").strip().lower()
_db_enabled = False if _flag in {"0", "false", "off", "no"} else True

db_pos_recommender = (
    DbPosRecommender() if _db_enabled else DbPosRecommender(DbConfig(host="", user=""))
)

if _db_enabled:
    try:
        index_only_recommender = IndexOnlyRecommender()
    except Exception as e:
        logging.getLogger(__name__).warning(f"IndexOnlyRecommender init failed: {e}")
        index_only_recommender = None
else:
    index_only_recommender = None
