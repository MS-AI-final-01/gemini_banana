# SDD: AI Virtual Try‑On (FastAPI + React/Vite)

Purpose: Single source of truth for scope, requirements, design, and tasks. Updated to reflect current Phase‑1 (Could → 1차) progress.

---

## 0) Agent Instructions (for coding assistant)

Role: code generation/refactoring assistant working within this repo.

Workflow
1) Read sections 1–6 to extract goals/constraints/acceptance.
2) Plan before code; ask only blocking questions.
3) Implement minimal, incremental diffs tied to tasks.
4) Validate and document changes.

Guardrails
- Keep changes scoped, avoid drive‑by refactors.
- Follow project conventions and environment constraints below.

---

## 1) What & Why

### 1.1 Problem Statement
사용자가 인물 사진(또는 모델)과 의류 이미지를 조합해 자연스러운 가상 착용 이미지를 생성하고, 결과 기반 추천/스타일 팁/평가 및 공유까지 한 화면에서 경험하도록 한다.

### 1.2 Objectives / Success Metrics
- O1: 스타일 팁/평가 기능 실패율 < 2% (키 미설정 시 Fallback 제공)
- O2: 추천 응답(카탈로그/LLM 재랭크 포함) p95 < 1.5s 목표
- O3: SNS 공유 PNG(≥1024px) 생성 성공률 > 99%

### 1.3 Target Users & Use Cases
- U1: 패션 소비자 – 업로드/모델 선택 → 합성 → 추천 탐색 → 공유
- U2: MD/운영 – 카탈로그 인입 및 배경 없는 이미지 선별로 품질 관리

### 1.4 Non‑Goals / Out of Scope
- 결제/주문/재고 통합, 인증/권한 모델

---

## 2) Scope & Interfaces

### 2.1 In‑Scope
- S1: 이미지 합성 `/api/generate`
- S2: 추천 `/api/recommend`, `/api/recommend/from-fitting`
- S3: 스타일 팁 `/api/tips` (Azure OpenAI 우선, Fallback 제공)
- S4: 결과 평가 `/api/evaluate` (0–100 점수)
- S5: 외부 추천기 브릿지 `/api/recommend/by-positions` (선택)
- S7: 프론트 UI: 업로드/히스토리/추천/공유/평가

### 2.2 Out‑of‑Scope
- 사용자 계정/권한, 결제/주문

### 2.3 External Interfaces
- Azure OpenAI Chat Completions (선택)
- Gemini (선택; 이미지/프롬프트)
- 외부 추천기(FastAPI; `RECOMMENDER_URL` 로 연동)

---

## 3) Constraints & Assumptions
- Runtime: Python 3.11+, Node 18+
- Platform: FastAPI + Vite SPA
- Data: PII 최소화(이미지 외 메타 없음), 로컬 저장은 브라우저 localStorage
- Security: 민감키 .env 주입, LLM 출력 JSON 파싱 가드, 인물 식별/속성 추론 금지
- Performance: Tips/Evaluate warm < 1.2s, 추천 < 1.5s 목표
- Assumptions: 카탈로그 JSON 또는 외부 추천기 중 하나는 사용 가능

---

## 4) Requirements (Acceptance Criteria)

### 4.1 Functional
- R1 Generate: 업로드된 인물(+의류) 또는 의류 3종만으로 합성 이미지 생성
  - Given 입력 유효, When `/api/generate`, Then data URI 반환
- R2 Recommend: 업로드 또는 합성 결과 기반 추천 3개/카테고리
  - Given 유효 입력, When `/api/recommend` 또는 `/from-fitting`, Then 카테고리별 리스트 반환
- R3 Style Tips: 3–6개 짧은 팁, 키 미설정 시 Fallback 동작
  - Given generatedImage, When `/api/tips`, Then `{tips:[…]}` 비공백
- R4 Evaluate: 선택 이미지들 0–100 점수
  - Given images[], When `/api/evaluate`, Then `{results:[{index,score}]}`
- R5 Share: 클라이언트에서 1080+ PNG 생성 및 다운로드
- R6 History: 입력/출력 히스토리 중복 방지(디듀프 + UI 가드)

### 4.2 Non‑Functional
- NFR‑S: PII 미전송, 비정상 LLM 출력 가드, 키/비밀은 서버측 .env
- NFR‑P: Tips/Evaluate warm < 1.2s, 추천 < 1.5s
- NFR‑O: 간단 콘솔 타이밍과 카운터, 오류 로그

---

## 5) Design (Proposed)

### 5.1 Architecture
- Monolith (FastAPI) + SPA (React/Vite)
- Optional microservice: 외부 추천기(FastAPI; positions 기반)

### 5.2 Key Modules
- Services: `azure_openai_service`, `llm_ranker`, `product_index`, `db_recommender`
- Routes: `generate`, `recommend`, `tips`, `evaluate`, `recommend_positions`

### 5.3 API Sketch
- POST `/api/tips` → `{ tips: string[], source, score?, requestId, timestamp }`
- POST `/api/evaluate` → `{ results: [{index, score, reasoning?}], source }`
- POST `/api/recommend/by-positions` → `[RecommendationItem…]`

### 5.4 Risks & Mitigations
- RSK‑1: LLM JSON 파싱 실패 → JSON‑only 지시 + 엄격 파싱/타임아웃/재시도
- RSK‑2: 히스토리 용량 증가 → 썸네일 다운샘플링 도입 검토
- RSK‑3: 이벤트 중복 → 전역 디듀프 + stopPropagation + 로딩 락

---

## 6) Tasks (WBS) & Traceability

| Task ID | Title | Maps to | DOD | Status |
|---|---|---|---|---|
| T1.1 | Tips API + FE 카드 | R3, NFR‑P | `/api/tips` + FE `StyleTipsCard` | Done |
| T1.2 | Evaluate API + FE | R4, NFR‑P | `/api/evaluate` + 점수 배지 | Done |
| T1.3 | Share PNG 유틸 | R5 | html2canvas, 다중 사이즈 | Done |
| T2.1 | 히스토리 중복 가드 | R6 | 전역 디듀프 + 가드 | Done |
| T3.1 | 외부 추천기 브릿지 | R2 | `/api/recommend/by-positions` | Done |
| T4.1 | DB-only 추천 전환 | R2 | 파일 기반 경로 제거, DB 의존 고정 | Done |

Milestones
- M1 Phase‑1 Could: Tips/Evaluate/Share + 안정화 (완료)

---

## 7) Ops & Rollout
- Feature flags (FE): `VITE_FEATURE_TIPS|EVALUATE|SHARE` (기본 ON)
- Docker: dev/prod compose 지원
- Azure: App Service/Container 배포 가능
- Telemetry: 콘솔 타이밍/카운터

---

## 8) Notes
- 프론트 Node 워크스페이스는 `frontend/`로 통합됨. 루트 Node 워크스페이스 제거
- Python venv는 `backend_py/.venv`만 사용
- 선택: `RECOMMENDER_URL` 설정 시 `/api/recommend/by-positions` 활성화

