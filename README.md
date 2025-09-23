<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# AI Virtual Try-On 모놀리포

Python FastAPI 백엔드(`backend_py`)와 React + Vite + TypeScript 프런트엔드(`frontend`)가 함께 들어 있는 저장소입니다. Azure OpenAI 기반 스타일 분석과 LLM 점수 계산, 가상 피팅(VTO) 워크플로, SNS 공유용 이미지 생성까지 한 번에 개발할 수 있도록 구성되어 있습니다.

- 데모: https://ai.studio/apps/drive/1ORGriwJMQVw1Sd-cSjddK7sGBrrm_B6D
- 목표: 1) 스타일 팁 자동 생성 2) 히스토리 스코어링 3) 공유 이미지 생성 (기본 활성화된 플래그로 제어)

## 주요 기능
- **AI 스타일링 팁**: `/api/tips` 엔드포인트에서 Azure OpenAI 비전 모델 또는 규칙 기반 fallback으로 최대 6개의 팁을 반환.
- **히스토리 LLM 평가**: `/api/evaluate`가 히스토리 이미지에 대한 0~100 점수와 근거를 제공하며, 프런트엔드에서 배지로 표기.
- **가상 피팅 파이프라인**: `/api/generate`, `/api/recommend` 등 기존 추천 및 합성 API 유지.
- **SNS 공유 이미지**: 프런트엔드 `html2canvas` 유틸을 이용한 1080/1920/630 사이즈 PNG 저장.
- **카탈로그/추천 서비스 연계**: `catalog.py`, `db_recommender.py`, `external_recommender.py` 모듈을 통한 다양한 소스 활용.

## 디렉터리 구조
```text
.
├─backend_py/            # FastAPI 백엔드
│  ├─app/
│  │  ├─routes/          # generate, recommend, tips, evaluate 등 API 라우터
│  │  ├─services/        # Azure OpenAI, Gemini, 카탈로그, LLM 랭커 서비스
│  │  ├─middleware/      # 요청/응답 로깅 미들웨어
│  │  ├─main.py          # FastAPI 엔트리포인트 및 라우터 등록
│  │  └─settings.py      # 환경 변수 로딩
│  ├─tests/              # 단위 테스트 (예: Vertex 비디오 서비스)
│  ├─tools/              # CSV 인입/인코딩 도구
│  └─requirements.txt    # 백엔드 의존성
├─frontend/              # React 19 + Vite 6 + Tailwind UI
│  ├─src/components/     # Try-on UI, StyleTipsCard, HistoryEvaluator 등
│  ├─src/services/       # API 래퍼, VTO 서비스, 히스토리 관리
│  ├─src/utils/shareImage.ts # SNS 공유 이미지 유틸
│  └─vite.config.ts      # API 프록시 및 기능 플래그 설정
├─docs/                  # 데이터 준비 등 참고 문서
├─scripts/               # PowerShell/Node 빠른 실행 스크립트
├─docker-compose*.yml    # 로컬/프로덕션 Docker 설정
└─README.md              # 이 문서
```

## 빠른 시작
### 1. 백엔드 (FastAPI)
```powershell
cd backend_py
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 3001
```

### 2. 프런트엔드 (Vite)
```powershell
cd frontend
npm install
npm run dev  # 기본 포트 5173
```

> `scripts/quickstart.ps1`를 실행하면 백엔드 가상환경 생성, 의존성 설치, CSV 인입, 서버 실행까지 자동화할 수 있습니다.

## 환경 변수
`backend_py/.env.example`을 복사해 `backend_py/.env`를 만들고 필요한 키만 채워도 됩니다.

## Configuration
Copy `backend_py/.env.example` to `backend_py/.env` and set keys as needed.
- CORS: `FRONTEND_URL`
- Azure OpenAI (optional): `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_KEY`, `AZURE_OPENAI_DEPLOYMENT_ID`
- Gemini (optional): `GEMINI_API_KEY`, `GEMINI_FIXED_PROMPT`, `GEMINI_TEMPERATURE`
- Vertex AI video generation (optional): `VERTEX_PROJECT_ID`, `VERTEX_LOCATION`, `VERTEX_MODEL_ID`, `VERTEX_API_ENDPOINT`, `GOOGLE_APPLICATION_CREDENTIALS`
- External recommender (optional): `RECOMMENDER_URL`, `RECOMMENDER_TIMEOUT`

Frontend dev uses `VITE_API_URL` when provided; otherwise defaults to `http://localhost:3001` via `vite.config.ts`.
Optional frontend flags: `VITE_FEATURE_VIDEO`, `VITE_VIDEO_PROMPT`, `VITE_VIDEO_ASPECT`, `VITE_VIDEO_DURATION`, `VITE_VIDEO_RESOLUTION`.

| 그룹 | 키 | 설명 |
| --- | --- | --- |
| 공통 | `FRONTEND_URL`, `HOST`, `PORT` | CORS 도메인 및 서버 포트 |
| Azure OpenAI (선택) | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_KEY`, `AZURE_OPENAI_DEPLOYMENT_ID`, `AZURE_OPENAI_API_VERSION` | 비전 챗 모델 호출용. 미설정 시 규칙 기반 fallback 사용 |
| Gemini (선택) | `GEMINI_API_KEY`, `GEMINI_FIXED_PROMPT`, `GEMINI_TEMPERATURE` | 이미지 변형/보조 기능 |
| Vertex 비디오 (선택) | `VERTEX_PROJECT_ID`, `VERTEX_LOCATION`, `VERTEX_MODEL_ID`, `VERTEX_API_ENDPOINT`, `GOOGLE_APPLICATION_CREDENTIALS` | 동영상 생성 파이프라인 |
| 외부 추천기 (선택) | `RECOMMENDER_URL`, `RECOMMENDER_TIMEOUT` | 외부 서비스 프록시 |

프런트엔드는 `.env` 또는 실행 시 환경 변수로 아래 키를 읽습니다.

| 키 | 기본 | 역할 |
| --- | --- | --- |
| `VITE_API_URL` | `http://localhost:3001` | 백엔드 프록시 경로 |
| `VITE_FEATURE_TIPS` | `true` | 스타일 팁 카드 표시 |
| `VITE_FEATURE_EVALUATE` | `true` | 히스토리 점수화 UI |
| `VITE_FEATURE_SHARE` | `true` | SNS 공유 다이얼로그 |
| `VITE_FEATURE_VIDEO` 등 | 필요 시 | 비디오 생성 관련 옵션 |

## API 개요
| 메서드 | 경로 | 설명 |
| --- | --- | --- |
| `GET` | `/health` | 헬스체크 |
| `POST` | `/api/generate` | 가상 피팅 이미지 생성 |
| `POST` | `/api/recommend` | 카탈로그 기반 추천 |
| `POST` | `/api/recommend/from-fitting` | 피팅 결과 기반 추천 |
| `POST` | `/api/recommend/by-positions` | 좌표 기반 추천 (옵션) |
| `POST` | `/api/tips` | 스타일 팁 생성 (AI/규칙 fallback) |
| `POST` | `/api/evaluate` | 히스토리 이미지 점수 계산 |
| `POST` | `/api/try-on/video` | 비디오 생성 요청 |
| `POST` | `/api/try-on/video/status` | 비디오 상태 조회 |

각 AI 경로는 JSON-only 응답을 강제하며 키 미설정 시 즉시 규칙 기반 결과를 반환하도록 방어 로직이 포함되어 있습니다.

## 빌드 & 배포
- **Docker 개발용**: `docker compose -f docker-compose.dev.yml up`
- **Docker 프로덕션용**: `docker compose up -d`
- 정적 호스팅 설정은 `staticwebapp.config.json` 참고

## 데이터 준비 도구
- `python backend_py/tools/ingest_csv_to_catalog.py`: 상품 CSV를 JSON 카탈로그로 변환
- `python backend_py/tools/convert_csv_encoding.py`: CSV 인코딩 변환
- `python backend_py/tools/select_transparent_images.py`: 투명 배경 이미지 필터 (`docs/data-prep-transparent-images.md` 참고)

## 자주 묻는 질문
- **Vite 명령이 동작하지 않나요?** → `cd frontend && npm install`
- **`ModuleNotFoundError: app` 오류가 발생하나요?** → `backend_py` 폴더에서 실행하거나 `uvicorn backend_py.app.main:app` 형태로 실행
- **윈도우에서 줄바꿈 경고가 나오나요?** → `.gitattributes`가 자동으로 처리하므로 무시해도 됩니다.
- **Azure 키가 없는데도 동작하나요?** → `/api/tips`, `/api/evaluate`는 규칙 기반 fallback을 통해 안전한 기본 응답을 제공합니다.

## 추가 참고
- `agents.md`, `sdd.md`: 시스템 디자인 및 프롬프트 흐름 정리
- `readme_recommend.md`: 추천 엔진 상세
- `taskmd/` 디렉터리: 작업 이력 및 메모

필요한 부분만 선택적으로 설정해도 전체 워크플로가 동작하도록 구성되어 있으므로, 우선 백엔드/프런트 기본 의존성을 설치한 뒤 기능 플래그를 사용해 점진적으로 확장하면 됩니다.
