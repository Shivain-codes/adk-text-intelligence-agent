#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#  deploy.sh — Build & deploy ADK Text Intelligence Agent to Cloud Run
#  Usage: ./deploy.sh [PROJECT_ID] [REGION]
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ID="${1:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${2:-us-central1}"
SERVICE_NAME="text-intelligence-agent"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest"

if [[ -z "$PROJECT_ID" ]]; then
  echo "❌  ERROR: No PROJECT_ID found. Set it with: gcloud config set project YOUR_PROJECT_ID"
  exit 1
fi

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   ADK Text Intelligence Agent — Cloud Run Deploy     ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Project : ${PROJECT_ID}"
echo "║  Region  : ${REGION}"
echo "║  Service : ${SERVICE_NAME}"
echo "║  Image   : ${IMAGE_NAME}"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── Pre-flight checks ─────────────────────────────────────────────────────────
echo "🔍  Checking prerequisites..."
command -v gcloud >/dev/null 2>&1 || { echo "❌  gcloud CLI not found. Install it: https://cloud.google.com/sdk/docs/install"; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "❌  Docker not found. Install Docker Desktop."; exit 1; }

# Check GEMINI_API_KEY
if [[ -z "${GEMINI_API_KEY:-}" ]]; then
  echo "❌  ERROR: GEMINI_API_KEY environment variable not set."
  echo "    Get your key at: https://aistudio.google.com/app/apikey"
  echo "    Then run: export GEMINI_API_KEY=your_key_here"
  exit 1
fi

# ── Enable required GCP APIs ──────────────────────────────────────────────────
echo "🔧  Enabling required GCP APIs..."
gcloud services enable \
  run.googleapis.com \
  containerregistry.googleapis.com \
  cloudbuild.googleapis.com \
  --project="${PROJECT_ID}" --quiet

# ── Build Docker image ────────────────────────────────────────────────────────
echo ""
echo "🐳  Building Docker image..."
gcloud builds submit \
  --tag="${IMAGE_NAME}" \
  --project="${PROJECT_ID}" \
  --timeout=600s \
  .

echo ""
echo "✅  Image built: ${IMAGE_NAME}"

# ── Deploy to Cloud Run ───────────────────────────────────────────────────────
echo ""
echo "🚀  Deploying to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
  --image="${IMAGE_NAME}" \
  --platform=managed \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --allow-unauthenticated \
  --port=8080 \
  --memory=1Gi \
  --cpu=1 \
  --min-instances=0 \
  --max-instances=10 \
  --timeout=60s \
  --set-env-vars="GEMINI_API_KEY=${GEMINI_API_KEY},GOOGLE_GENAI_USE_VERTEXAI=FALSE" \
  --quiet

# ── Get service URL ───────────────────────────────────────────────────────────
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --format="value(status.url)")

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║              🎉  DEPLOYMENT SUCCESSFUL!              ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Service URL: ${SERVICE_URL}"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Endpoints:"
echo "║    GET  ${SERVICE_URL}/"
echo "║    GET  ${SERVICE_URL}/health"
echo "║    POST ${SERVICE_URL}/summarize"
echo "║    POST ${SERVICE_URL}/classify"
echo "║    POST ${SERVICE_URL}/chat"
echo "║    GET  ${SERVICE_URL}/docs  (Swagger UI)"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "📋  Quick test:"
echo "    curl -X POST ${SERVICE_URL}/summarize \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"text\": \"Your text here...\", \"style\": \"concise\"}'"
echo ""
