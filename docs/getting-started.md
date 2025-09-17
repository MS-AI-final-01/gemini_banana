# 실행 가이드 (Python FastAPI + React)

## 빠른 시작

1) 의존성 설치

```
npm install
```

2) 환경 변수 설정

- 백엔드: `backend_py/.env` 파일에서 `GEMINI_API_KEY` 등 값을 설정합니다.
- 프론트엔드: 개발 실행 시 `VITE_API_URL`은 자동으로 `http://localhost:3001`로 설정됩니다.

3) 실행 (FastAPI 3001 + Vite 5173)

```
npm run dev:py
```

4) 접속

- 앱: http://localhost:5173
- API: http://localhost:3001

## 주요 엔드포인트

- `GET /health`
- `POST /api/generate`, `GET /api/generate/status`
- `POST /api/recommend`, `POST /api/recommend/from-fitting`, `GET /api/recommend/status`, `GET /api/recommend/catalog`

## 예시 호출 (PowerShell)

```powershell
$body = @{
  person = @{ base64 = "base64_image"; mimeType = "image/jpeg" }
  clothingItems = @{ top = @{ base64 = "base64_image"; mimeType = "image/jpeg" } }
} | ConvertTo-Json -Depth 3

Invoke-RestMethod -Method Post -Uri 'http://localhost:3001/api/recommend' -ContentType 'application/json' -Body $body
```

## 프로젝트 구조 (요약)

```
backend_py/                 # FastAPI 백엔드
  app/
    routes/                 # API 라우트 (generate, recommend, health, proxy)
    services/               # Gemini/Azure/OpenAI 연계 서비스
frontend/                   # React 프론트엔드
docs/                       # 문서
```

## 문제 해결

- 포트 점유 확인: 3001(FastAPI), 5173(Vite)가 사용 중인지 확인
- `.env` 값 확인: `backend_py/.env`의 키/엔드포인트가 올바른지 확인
- 네트워크/방화벽: 로컬 접근이 차단되지 않았는지 확인

