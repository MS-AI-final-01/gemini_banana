#!/usr/bin/env bash
set -euo pipefail

# Install Azure CLI on Debian/Ubuntu if not present.

if command -v az >/dev/null 2>&1; then
  echo "[ok] az already installed: $(az version --query azure-cli -o tsv 2>/dev/null || echo)"
  exit 0
fi

if [[ $(id -u) -ne 0 ]]; then
  echo "Please run as root (sudo) to install Azure CLI" >&2
  exit 1
fi

echo "Installing Azure CLI (Microsoft repo) ..."
curl -sL https://aka.ms/InstallAzureCLIDeb | bash
echo "[done] Azure CLI installed"

