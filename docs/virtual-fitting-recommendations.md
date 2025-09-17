# 가상 피팅 기반 상품 추천 시스템

## 🎯 새로운 기능

### 가상 피팅 완료 후 자동 추천
1. **사용자가 인물 사진 + 의류 사진을 업로드**
2. **AI가 가상 피팅 이미지를 생성**
3. **생성된 이미지를 자동으로 분석하여 유사한 상품 추천**
4. **카테고리별(상의/하의/신발/액세서리)로 정리된 추천 결과 표시**

## 🚀 실행 방법

### 1. 환경 설정
```bash
# 의존성 설치
npm install

# 환경 변수 설정 (.env 파일)
GEMINI_API_KEY=your_gemini_api_key_here

# Azure OpenAI (선택사항 - 현재는 Mock으로 동작)
AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
AZURE_OPENAI_KEY=your_azure_openai_key
AZURE_OPENAI_DEPLOYMENT_ID=your_deployment_id
```

### 2. 서버 실행
```bash
# 터미널 1: API 서버
npm run server

# 터미널 2: 프론트엔드
npm run dev
```

### 3. 사용 방법
1. 브라우저에서 `http://localhost:5173/` 접속
2. 인물 사진 업로드 (전신 사진 권장)
3. 의류 사진 업로드 (상의, 하의, 신발 중 최소 1개)
4. "Generate Virtual Try-On" 버튼 클릭
5. 가상 피팅 결과 확인
6. **자동으로 표시되는 추천 상품 확인**

## 🔧 API 엔드포인트

### 기존 기능
- `POST /api/generate`: 가상 피팅 이미지 생성

### 새로운 추천 기능
- `POST /api/recommend`: 업로드 이미지 기반 추천
- `POST /api/recommend-from-fitting`: **가상 피팅 결과 기반 추천** (핵심 기능)

#### 가상 피팅 기반 추천 API 예시
```javascript
// 요청
{
  "generatedImage": "data:image/jpeg;base64,/9j/4AAQ...", // 생성된 피팅 이미지
  "mimeType": "image/jpeg",
  "originalClothingItems": {
    "top": { "base64": "...", "mimeType": "image/jpeg" },
    "pants": { "base64": "...", "mimeType": "image/jpeg" }
  }
}

// 응답
{
  "recommendations": {
    "top": [
      {
        "id": "p001",
        "title": "블랙 오버사이즈 후드티",
        "price": 42000,
        "score": 3,
        "category": "top"
      }
    ],
    "pants": [...],
    "shoes": [...],
    "accessories": [...]
  }
}
```

## 🏗️ 아키텍처

```
사용자 업로드 이미지
        ↓
   가상 피팅 생성 (Gemini API)
        ↓
   생성된 이미지 분석 (Azure OpenAI - 현재 Mock)
        ↓
   카탈로그 검색 및 매칭
        ↓
   카테고리별 추천 결과 반환
```

## 📁 주요 파일

### 새로 추가된 파일
- `backend_py/app/routes/recommend.py`: 추천 로직 (가상 피팅 기반 분석 포함)
- `components/RecommendationDisplay.tsx`: 추천 상품 표시 컴포넌트

### 수정된 파일
- 루트 `server.js`: 레거시(제거됨). FastAPI가 `/api/recommend-from-fitting` 포함 추천 관련 엔드포인트를 제공합니다.
- `services/geminiService.ts`: `getRecommendationsFromFitting()` 함수 추가
- `components/VirtualTryOnUI.tsx`: 추천 기능 통합

## 🎨 UI 플로우

1. **업로드 섹션**: 인물 + 의류 이미지 업로드
2. **피팅 생성**: 버튼 클릭으로 가상 피팅 실행
3. **결과 표시**: 생성된 이미지 표시
4. **추천 로딩**: "추천 상품을 찾고 있습니다..." 표시
5. **추천 결과**: 카테고리별 상품 카드 형태로 표시

## 🔄 다음 단계

### 현재 상태 (MVP)
- ✅ 가상 피팅 생성
- ✅ 피팅 결과 기반 자동 추천
- ✅ 카테고리별 추천 표시
- ✅ Mock 키워드 매칭

### 향후 개선 사항
1. **Azure OpenAI Vision 연동**: 실제 이미지 분석으로 정확도 향상
2. **벡터 검색**: Azure Cognitive Search로 시각적 유사성 검색
3. **추천 알고리즘 고도화**: 개인화, 시즌, 트렌드 반영
4. **상품 상세 연결**: 추천 상품 클릭 시 상세 페이지 연동

## 💡 테스트 팁

1. **피팅 테스트**: 명확한 전신 사진과 단품 의류 사진 사용
2. **추천 확인**: 생성된 이미지의 색상/스타일과 추천 상품 비교
3. **카테고리 분류**: 상의/하의/신발이 올바르게 분류되는지 확인

현재 시스템은 **가상 피팅 → 자동 추천**의 완전한 플로우를 제공합니다!
