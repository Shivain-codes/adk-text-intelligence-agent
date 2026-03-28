# 🤖 ADK Text Intelligence Agent

> A production-ready AI agent built with **Google ADK** + **gemini-2.5-flash-preview-04-17**, deployable to **Cloud Run** in minutes.

[![Google ADK](https://img.shields.io/badge/Google%20ADK-1.2.1-4285F4?logo=google&logoColor=white)](https://google.github.io/adk-docs/)
[![Gemini](https://img.shields.io/badge/Gemini-2.0%20Flash-34A853?logo=google&logoColor=white)](https://ai.google.dev/)
[![Cloud Run](https://img.shields.io/badge/Cloud%20Run-Ready-4285F4?logo=google-cloud&logoColor=white)](https://cloud.google.com/run)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)

---

## 📋 Project Overview

| Property | Value |
|----------|-------|
| **Framework** | Google Agent Development Kit (ADK) v1.2.1 |
| **Model** | gemini-2.5-flash-preview-04-17 (via LiteLLM) |
| **Capability** | Text Summarization + Classification |
| **Server** | FastAPI + Uvicorn |
| **Deployment** | Google Cloud Run (containerized) |

---

## 🧠 Agent Architecture

```
HTTP Request
     │
     ▼
┌─────────────────────────────────┐
│         FastAPI Server          │  ← main.py
│  /summarize  /classify  /chat   │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│         ADK Runner              │  ← google.adk.runners.Runner
│   (manages sessions & turns)    │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│      root_agent (Agent)         │  ← agent.py
│   name: text_intelligence_agent │
│   model:gemini-2.5-flash-preview-04-17       │
│   tools: [summarize, classify]  │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│     Gemini 2.0 Flash (LLM)      │  ← via LiteLLM
│   Inference + tool orchestration│
└─────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- [Gemini API Key](https://aistudio.google.com/app/apikey) (free tier available)
- Docker (for containerized run)
- `gcloud` CLI (for Cloud Run deployment)

### 1. Clone & Configure

```bash
git clone https://github.com/YOUR_USERNAME/adk-summarizer-agent.git
cd adk-summarizer-agent

cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 2. Run Locally (Python)

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here
export GOOGLE_GENAI_USE_VERTEXAI=FALSE
python main.py
```

### 3. Run with Docker

```bash
export GEMINI_API_KEY=your_key_here
docker compose up --build
```

The agent will be live at **http://localhost:8080**

---

## 🌐 API Reference

### `GET /health`
Health check endpoint.

```bash
curl http://localhost:8080/health
# → {"status": "healthy", "agent": "text_intelligence_agent"}
```

---

### `POST /summarize`
Summarize any text.

**Request Body:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | ✅ | Text to summarize (min 10 chars) |
| `style` | string | ❌ | `concise` (default) \| `detailed` \| `bullets` |

```bash
curl -X POST http://localhost:8080/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Artificial intelligence is transforming industries worldwide. Machine learning algorithms can now diagnose diseases, predict market movements, and even generate creative content. However, concerns about job displacement and algorithmic bias remain significant challenges that society must address thoughtfully.",
    "style": "bullets"
  }'
```

**Response:**
```json
{
  "response": "• AI is transforming multiple industries globally\n• ML algorithms enable medical diagnosis, market prediction, and content generation\n• Key concerns include job displacement and algorithmic bias",
  "session_id": "summarize-...",
  "agent_name": "text_intelligence_agent"
}
```

---

### `POST /classify`
Classify the topic of any text.

```bash
curl -X POST http://localhost:8080/classify \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The Federal Reserve raised interest rates by 25 basis points citing persistent inflation concerns."
  }'
```

**Response:**
```json
{
  "response": "{\"category\": \"Business\", \"confidence\": \"high\", \"reasoning\": \"The text discusses Federal Reserve monetary policy and inflation, which are financial/business topics.\"}",
  "session_id": "classify-...",
  "agent_name": "text_intelligence_agent"
}
```

---

### `POST /chat`
Free-form conversation with the agent.

```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Summarize this in detail: The James Webb Space Telescope has captured unprecedented images of the early universe, revealing galaxies that formed just 300 million years after the Big Bang."
  }'
```

---

### `GET /docs`
Interactive Swagger UI — test all endpoints in your browser.

---

## ☁️ Deploy to Cloud Run

### One-Command Deployment

```bash
export GEMINI_API_KEY=your_key_here
chmod +x deploy.sh
./deploy.sh YOUR_GCP_PROJECT_ID us-central1
```

### Manual Step-by-Step

```bash
# 1. Set your project
export PROJECT_ID=your-gcp-project-id
export REGION=us-central1
export SERVICE_NAME=text-intelligence-agent

# 2. Authenticate
gcloud auth login
gcloud config set project $PROJECT_ID

# 3. Enable APIs
gcloud services enable run.googleapis.com containerregistry.googleapis.com cloudbuild.googleapis.com

# 4. Build image with Cloud Build
gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME:latest .

# 5. Deploy to Cloud Run
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$SERVICE_NAME:latest \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --port 8080 \
  --memory 1Gi \
  --cpu 1 \
  --set-env-vars "GEMINI_API_KEY=$GEMINI_API_KEY,GOOGLE_GENAI_USE_VERTEXAI=FALSE"
```

### After Deployment

Your agent will be live at:
```
https://text-intelligence-agent-XXXXXXXX-uc.a.run.app
```

Test it immediately:
```bash
curl https://YOUR_CLOUD_RUN_URL/health
```

---

## 📁 Project Structure

```
adk-summarizer-agent/
├── agent.py          # ADK Agent definition + tool functions
├── main.py           # FastAPI server + ADK Runner
├── requirements.txt  # Python dependencies
├── Dockerfile        # Multi-stage Docker build
├── docker-compose.yml# Local development
├── deploy.sh         # Cloud Run deployment script
├── .env.example      # Environment variables template
├── .gitignore
└── README.md
```

---

## 🔧 Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `GEMINI_API_KEY` | Google AI Studio API key | **Required** |
| `GOOGLE_GENAI_USE_VERTEXAI` | Use Vertex AI instead of API key | `FALSE` |
| `PORT` | HTTP server port | `8080` |

---

## 🧪 Sample Inputs to Test

**Summarize (concise):**
```json
{"text": "The Python programming language was created by Guido van Rossum and first released in 1991. Python emphasizes code readability and simplicity, making it one of the most popular programming languages in the world. It supports multiple programming paradigms including procedural, object-oriented, and functional programming.", "style": "concise"}
```

**Summarize (bullets):**
```json
{"text": "Climate change is causing rising sea levels, more frequent extreme weather events, and disruptions to ecosystems. The primary driver is greenhouse gas emissions from human activities like burning fossil fuels. International agreements like the Paris Accord aim to limit warming to 1.5 degrees Celsius.", "style": "bullets"}
```

**Classify:**
```json
{"text": "The quarterback threw three touchdown passes as the home team dominated the second half."}
```

---

## 🏗️ Built With

- [Google ADK](https://google.github.io/adk-docs/) — Agent Development Kit
- [Gemini 2.0 Flash](https://ai.google.dev/gemini-api/docs/models/gemini) — Fast, capable LLM
- [FastAPI](https://fastapi.tiangolo.com/) — High-performance Python web framework
- [LiteLLM](https://docs.litellm.ai/) — Unified LLM interface for ADK
- [Cloud Run](https://cloud.google.com/run) — Serverless container platform

---

## 📝 License

MIT License — free to use, modify, and deploy.
