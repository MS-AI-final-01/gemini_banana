$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Write-Step($msg) { Write-Host "[SETUP] $msg" -ForegroundColor Cyan }
function Write-Ok($msg) { Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Warning $msg }

# 1) Prerequisites
Write-Step "Checking prerequisites (Node, Python)"
if (-not (Get-Command node -ErrorAction SilentlyContinue)) { throw 'Node.js is required' }
if (-not (Get-Command python -ErrorAction SilentlyContinue)) { throw 'Python is required' }
Write-Ok "Node $(node -v) / Python $((python -V).Split()[1])"

# 2) NPM install (frontend only)
Write-Step "Installing frontend Node dependencies"
Push-Location frontend
try {
  if (Test-Path 'package-lock.json') { npm ci } else { npm install }
  Write-Ok "Frontend dependencies installed"
} finally {
  Pop-Location
}

# 3) Python venv + deps via JS runner (ensures pip install)
Write-Step "Ensuring Python venv and dependencies for backend_py"
node scripts/run_backend_py.js --no-start 2>$null 1>$null
# scripts/run_backend_py.js will install deps when starting; if --no-start is passed, ignore errors
Write-Ok "Python environment prepared"

# 4) Start Python backend (port 3001) + Frontend (5173) concurrently
Write-Step "Starting Python backend (3001) and Frontend (5173)"
npx concurrently --names "PY,FRONTEND" --prefix-colors "magenta,green" `
  "node scripts/run_backend_py.js" `
  "cd frontend && cross-env VITE_API_URL=http://localhost:3001 npm run dev"

