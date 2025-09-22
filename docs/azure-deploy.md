Azure App Service 배포(요약)

선택지
- 단일 컨테이너 2앱(권장): 프런트용 웹앱 1개 + 백엔드용 웹앱 1개
- 멀티 컨테이너(옵션): docker‑compose.azure.yml로 한 앱에 프런트+백엔드 구성

사전 준비
1) Azure Container Registry(ACR) 생성: `az acr create -n <ACR_NAME> -g <RG> --sku Basic`
2) 두 웹앱 생성(단일 컨테이너, Linux, S1 이상 권장)
3) 각 웹앱에 시스템 할당 ID 켜기 → ACR에 `AcrPull` 역할 부여

이미지 빌드/푸시(로컬 예시)
```
# 백엔드
docker build -t <ACR>.azurecr.io/avto-backend:1.0.0 ./backend_py
docker push <ACR>.azurecr.io/avto-backend:1.0.0

# 프런트: 백엔드 외부 URL 주입
docker build \
  --build-arg NGINX_CONF=nginx.prod.conf \
  --build-arg VITE_API_URL=https://<API_APP>.azurewebsites.net \
  -t <ACR>.azurecr.io/avto-frontend:1.0.0 ./frontend
docker push <ACR>.azurecr.io/avto-frontend:1.0.0
```

웹앱 설정(App Settings)
- API 앱: `WEBSITES_PORT=3001`, `PORT=3001`, `HOST=0.0.0.0`, `NODE_ENV=production`, `FRONTEND_URL=https://<WEB_APP>.azurewebsites.net`
- WEB 앱: `WEBSITES_PORT=8080`
- DB/키(선택): `DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD/DB_SSLMODE=require` 등

GitHub Actions로 자동화
- `.github/workflows/azure-acr-2apps.yml` 사용
- 저장소 시크릿 설정
  - `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`
  - `ACR_NAME`, `AZURE_RG`, `AZURE_WEBAPP_API`, `AZURE_WEBAPP_WEB`
  - `BACKEND_URL` = `https://<API_APP>.azurewebsites.net`
- main에 push 또는 수동 실행 → ACR 빌드/푸시 → 두 웹앱 이미지 갱신/재시작

멀티 컨테이너(옵션)
- `docker-compose.azure.yml` 템플릿의 `<ACR_NAME>`, `<TAG>` 교체
- CLI 적용: `az webapp config container set --multicontainer-config-type compose --multicontainer-config-file docker-compose.azure.yml -g <RG> -n <APP>`

헬스체크
- WEB: `/health` (Nginx가 200)
- API: `/health` (FastAPI)

문제 해결
- ACR Pull 실패: 웹앱 관리형 ID에 ACR `AcrPull` 역할 확인
- 502/404: 프런트 이미지 빌드시 `VITE_API_URL`이 올바른지 확인
- DB 연결 실패: Outbound IP 화이트리스트, `sslmode=require` 값 확인

