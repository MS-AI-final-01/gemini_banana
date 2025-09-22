#!/usr/bin/env bash
set -euo pipefail

# Deploy two single-container Web Apps (API + WEB) on Azure App Service
# Requires: az CLI logged in, Docker if BUILD_PUSH=true

# ===== Config =====
: "${SUBSCRIPTION_ID:?Set SUBSCRIPTION_ID}"
: "${LOCATION:=koreacentral}"
: "${RESOURCE_GROUP:?Set RESOURCE_GROUP}"
: "${ACR_NAME:?Set ACR_NAME (lowercase, unique)}"
: "${PLAN_NAME:?Set PLAN_NAME}"
: "${API_APP:?Set API_APP (webapp name for backend)}"
: "${WEB_APP:?Set WEB_APP (webapp name for frontend)}"
: "${BACKEND_IMAGE:=avto-backend}"
: "${FRONTEND_IMAGE:=avto-frontend}"
: "${IMAGE_TAG:=latest}"
: "${BACKEND_URL:=https://${API_APP}.azurewebsites.net}"
: "${BUILD_PUSH:=false}"

echo "==> Using subscription: $SUBSCRIPTION_ID"
az account set -s "$SUBSCRIPTION_ID"

echo "==> Create resource group"
az group create -n "$RESOURCE_GROUP" -l "$LOCATION" -o table

echo "==> Create ACR ($ACR_NAME)"
az acr create -n "$ACR_NAME" -g "$RESOURCE_GROUP" --sku Basic -o table || true

echo "==> Create App Service plan ($PLAN_NAME)"
az appservice plan create -n "$PLAN_NAME" -g "$RESOURCE_GROUP" --is-linux --sku S1 -o table || true

echo "==> Create Web Apps (container)"
# Create with temporary public image; we will set ACR image later
az webapp create -n "$API_APP" -g "$RESOURCE_GROUP" --plan "$PLAN_NAME" \
  --deployment-container-image-name mcr.microsoft.com/azuredocs/aci-helloworld:latest -o table || true
az webapp create -n "$WEB_APP" -g "$RESOURCE_GROUP" --plan "$PLAN_NAME" \
  --deployment-container-image-name mcr.microsoft.com/azuredocs/aci-helloworld:latest -o table || true

echo "==> Enable system-assigned managed identity"
az webapp identity assign -g "$RESOURCE_GROUP" -n "$API_APP" -o table || true
az webapp identity assign -g "$RESOURCE_GROUP" -n "$WEB_APP" -o table || true

API_MI=$(az webapp show -g "$RESOURCE_GROUP" -n "$API_APP" --query identity.principalId -o tsv)
WEB_MI=$(az webapp show -g "$RESOURCE_GROUP" -n "$WEB_APP" --query identity.principalId -o tsv)
ACR_ID=$(az acr show -n "$ACR_NAME" -g "$RESOURCE_GROUP" --query id -o tsv)

echo "==> Grant AcrPull to web apps"
az role assignment create --assignee "$API_MI" --role AcrPull --scope "$ACR_ID" >/dev/null || true
az role assignment create --assignee "$WEB_MI" --role AcrPull --scope "$ACR_ID" >/dev/null || true

REGISTRY="${ACR_NAME}.azurecr.io"

if [[ "$BUILD_PUSH" == "true" ]]; then
  echo "==> Build & push images to $REGISTRY"
  az acr login -n "$ACR_NAME"
  docker build -t "$REGISTRY/${BACKEND_IMAGE}:${IMAGE_TAG}" ./backend_py
  docker push "$REGISTRY/${BACKEND_IMAGE}:${IMAGE_TAG}"
  docker build --build-arg NGINX_CONF=nginx.prod.conf --build-arg VITE_API_URL="$BACKEND_URL" \
    -t "$REGISTRY/${FRONTEND_IMAGE}:${IMAGE_TAG}" ./frontend
  docker push "$REGISTRY/${FRONTEND_IMAGE}:${IMAGE_TAG}"
fi

echo "==> Configure container images"
az webapp config container set -g "$RESOURCE_GROUP" -n "$API_APP" \
  --docker-custom-image-name "$REGISTRY/${BACKEND_IMAGE}:${IMAGE_TAG}" \
  --docker-registry-server-url "https://${REGISTRY}" -o table

az webapp config container set -g "$RESOURCE_GROUP" -n "$WEB_APP" \
  --docker-custom-image-name "$REGISTRY/${FRONTEND_IMAGE}:${IMAGE_TAG}" \
  --docker-registry-server-url "https://${REGISTRY}" -o table

echo "==> App Settings"
az webapp config appsettings set -g "$RESOURCE_GROUP" -n "$API_APP" --settings \
  WEBSITES_PORT=3001 PORT=3001 HOST=0.0.0.0 NODE_ENV=production \
  FRONTEND_URL="https://${WEB_APP}.azurewebsites.net" -o table

az webapp config appsettings set -g "$RESOURCE_GROUP" -n "$WEB_APP" --settings \
  WEBSITES_PORT=8080 -o table

echo "==> Health check paths"
az webapp config set -g "$RESOURCE_GROUP" -n "$API_APP" --health-check-path /health -o table
az webapp config set -g "$RESOURCE_GROUP" -n "$WEB_APP" --health-check-path /health -o table

echo "==> Restart apps"
az webapp restart -g "$RESOURCE_GROUP" -n "$API_APP"
az webapp restart -g "$RESOURCE_GROUP" -n "$WEB_APP"

echo "\n[Done]"
echo "API: https://${API_APP}.azurewebsites.net/health"
echo "WEB: https://${WEB_APP}.azurewebsites.net"

