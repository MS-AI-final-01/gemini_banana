# ---------- Build stage ----------
# Debian 기반 이미지를 사용하여 glibc 환경에서 빌드(rollup optional binary 호환).
FROM node:20-bullseye-slim AS builder
WORKDIR /app

# 빌드 타임 API URL 주입 (Vite는 VITE_* 환경변수만 사용)
ARG VITE_API_URL
ENV VITE_API_URL=${VITE_API_URL}

# 의존성 설치 (실제 package.json/lock 사용)
COPY package.json package-lock.json .npmrc ./
RUN npm ci --no-audit --no-fund

# 소스 복사 및 빌드
COPY . .
RUN npm run build

# ---------- Runtime stage ----------
FROM nginx:alpine AS production

# Nginx 설정 선택 (prod/dev/local 중 하나)
ARG NGINX_CONF=nginx.prod.conf
COPY ${NGINX_CONF} /etc/nginx/nginx.conf

# 정적 파일 복사
COPY --from=builder /app/dist /usr/share/nginx/html

# 비루트 워커용 사용자 생성(마스터 프로세스는 root 유지)
RUN addgroup -g 1001 -S frontend \
    && adduser -S frontend -u 1001 -G frontend \
    && mkdir -p /var/cache/nginx /var/log/nginx /tmp/nginx \
    && chown -R frontend:frontend /usr/share/nginx/html /var/cache/nginx /var/log/nginx /tmp/nginx
EXPOSE 8080
CMD ["nginx", "-g", "daemon off;"]
