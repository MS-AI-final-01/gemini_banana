# 추천 알고리즘 개요 (Recommendation Algorithm)

FastAPI 백엔드에서 동작하는 추천 파이프라인과 LLm 재랭킹 구조를 요약했습니다. 운영자가 빠르게 구조를 이해하고 기능을 확장할 수 있도록 DB-only 구조에 맞춰 최신 정보를 정리했습니다.

## TL;DR
- 두 축의 추천을 제공합니다.
  1. **스타일 분석 + 상품 검색 + (옵션) LLM 재랭킹**: 인물/착용 이미지에서 추출한 태그·색상·핏 정보를 기반으로 DB에서 검색 후보를 만들고, 필요할 때 Azure OpenAI로 재정렬합니다.
  2. **포지션(임베딩) 기반 추천**: 이미 선택한 상품의 `positions` 인덱스를 입력받아, 임베딩 평균과 가격 가중치를 이용해 유사 상품을 반환합니다. 모든 데이터는 PostgreSQL에서 로드됩니다.
- 로컬 JSON/`real_data` 폴더는 더 이상 사용하지 않습니다. DB가 준비되지 않으면 관련 API는 503을 반환합니다.

---

## 서비스 구성 (코드 기준)
- 주요 서비스 모듈
  - DB 기반 상품 인덱스/검색: `backend_py/app/services/product_index.py`
  - 포지션 임베딩 추천: `backend_py/app/services/db_recommender.py`
  - LLM 재랭킹(옵션): `backend_py/app/services/llm_ranker.py`
  - 스타일 분석(Azure OpenAI Vision): `backend_py/app/services/azure_openai_service.py`
- 주요 라우트
  - 스타일 기반 추천/랜덤 피드: `backend_py/app/routes/recommend.py`
  - 포지션 기반 추천: `backend_py/app/routes/recommend_positions.py`

---

## 데이터 소스
- PostgreSQL (필수)
  - `public.products(pos, "Product_U", "Product_img_U", "Product_N", "Product_Desc", "Product_P", "Category", "Product_B", "Product_G", "Image_P")`
  - `public.embeddings` (vector 컬럼 `col_0...` 또는 JSON/배열 `value`)
- 메모리 캐시/인덱스 흐름
  - `DbPosRecommender`가 products/embeddings를 한 번에 로드하고 정규화합니다.
  - `ProductIndex`는 `DbPosRecommender.products` 리스트를 사용해 검색, 통계, 랜덤 피드를 제공합니다.

---

## API 개요
- 스타일 기반 (이미지/분석)
  - `POST /api/recommend`: 카테고리별(top, pants, shoes, accessories) 추천 묶음 반환
  - `POST /api/recommend/from-fitting`: 가상 착장 결과 이미지를 기반으로 추천
- 포지션 기반 (임베딩)
  - `POST /api/recommend/by-positions`: `positions: number[]`을 입력받아 유사 상품 반환
- 보조
  - `GET /api/recommend/random`: 카테고리/성별 조건으로 랜덤 상품 추출 (DB 필수)

요청/응답 모델은 `backend_py/app/models.py` 참고. 주요 옵션:
- `RecommendationOptions.maxPerCategory`, `minPrice`, `maxPrice`, `excludeTags`, `useLLMRerank`
- 포지션 기반: `top_k`, `alpha`, `w1`, `w2`

---

## 알고리즘 상세

### 1) 스타일 분석 + 상품 검색 + LLM 재랭킹
1. **스타일 분석**
   - 우선순위: Azure OpenAI Vision (`azure_openai_service.analyze_style_from_images` / `analyze_virtual_try_on`).
   - 입력: 인물 이미지(`person`), 의류 조각(`clothingItems`), 가상 착장 이미지(`generatedImage`).
   - 출력(JSON): `overall_style`, `detected_style`, `colors`, `categories`, `fit`, `silhouette` 등.
   - 미설정 시 안전한 기본 태그를 부여합니다.

2. **상품 후보 생성 (`ProductIndex.find_similar`)**
   - 분석에서 추출한 키워드(`tags`, `captions`, `overall_style`, `colors`, `categories` 등)를 모읍니다.
   - DB에서 가져온 상품 리스트를 순회하며 제목/태그 문자열에 키워드가 정확히 포함되면 `exact_weight`, 토큰 단위로 포함되면 `partial_weight`를 더합니다.
   - `minPrice`, `maxPrice`, `excludeTags` 옵션으로 후보를 필터링하고 카테고리별 최상위 N개를 반환합니다.

3. **LLM 재랭킹 (옵션)**
   - `useLLMRerank`가 true이거나 Azure OpenAI 설정이 감지되면 실행합니다.
   - 후보 리스트와 분석 JSON을 전달하여 카테고리별 선호 id 목록을 받습니다.
   - LLM이 반환하지 않은 항목은 원래 점수 순으로 채워넣습니다.

4. **응답 변환**
   - 각 카테고리별 `maxPerCategory` 만큼 `RecommendationItem` 모델로 직렬화합니다.

### 2) 포지션 기반 임베딩 추천 (`DbPosRecommender.recommend`)
1. 입력 `positions`의 임베딩을 평균하여 쿼리 벡터를 구성하고 정규화합니다.
2. 전체 상품 임베딩과 코사인 유사도를 계산합니다.
3. 평균 가격과 로그 스케일을 이용해 가격 유사도 가중치를 계산합니다.
4. 총점 = `w1 * cosine + w2 * price_score` 로 결합하고, 입력 포지션은 제외합니다.
5. 상위 `top_k` 개를 점수순으로 반환하여 `RecommendationItem`으로 매핑합니다.

---

## 장애/폴백 동작
- DB가 준비되지 않으면 `GET /api/recommend/random`, `POST /api/recommend`, `POST /api/recommend/by-positions` 모두 503을 반환합니다.
- Azure OpenAI 미설정 시 스타일 분석/LLM 재랭킹은 자동으로 규칙 기반/스킵 모드로 전환됩니다.

이 문서는 DB-only 구조를 기준으로 하며, 로컬 CSV(`real_data`) 기반 도구는 완전히 제거되었습니다.
