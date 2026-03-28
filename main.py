"""
FastAPI HTTP server for the ADK Text Intelligence Agent.
Exposes /summarize and /classify endpoints callable via HTTP.
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from agent import root_agent

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Request / Response Schemas ─────────────────────────────────────────────────

class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=5, description="Text to summarize")
    style: Optional[str] = Field(
        "concise",
        description="Summary style: 'concise', 'detailed', or 'bullets'"
    )

class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=5, description="Text to classify")

class AgentRequest(BaseModel):
    message: str = Field(..., description="Free-form message to the agent")
    session_id: Optional[str] = Field(None, description="Optional session ID")

class AgentResponse(BaseModel):
    response: str
    session_id: str
    agent_name: str

# ── ADK Setup ──────────────────────────────────────────────────────────────────

APP_NAME = "text-intelligence-agent"
USER_ID  = "http-user"

session_service = InMemorySessionService()
runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=session_service,
)

# ── Core Agent Runner ──────────────────────────────────────────────────────────

async def run_agent(message: str, session_id: str) -> str:
    """Run the ADK agent and return the final text response."""

    # Always create a fresh session — never check if it exists first
    session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session_id,
    )

    content = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=message)],
    )

    final_response = ""
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=content,
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_response = "".join(
                    p.text for p in event.content.parts
                    if hasattr(p, "text") and p.text
                )

    return final_response.strip() or "No response generated."

# ── FastAPI App ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Agent starting up...")
    yield
    logger.info("Agent shutting down.")

app = FastAPI(
    title="ADK Text Intelligence Agent",
    description="AI-powered text summarization and classification using Google ADK + Gemini",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {
        "agent": root_agent.name,
        "status": "running",
        "model": "gemini-2.0-flash",
        "endpoints": {
            "summarize": "POST /summarize",
            "classify":  "POST /classify",
            "chat":      "POST /chat",
            "health":    "GET /health",
            "docs":      "GET /docs",
        },
    }

@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy", "agent": root_agent.name}

@app.post("/summarize", response_model=AgentResponse, tags=["Agent"])
async def summarize(req: SummarizeRequest):
    """Summarize any text. Style: concise | detailed | bullets"""
    import uuid
    session_id = f"s-{uuid.uuid4().hex}"
    message = (
        f"Please summarize the following text in a '{req.style}' style:\n\n{req.text}"
    )
    try:
        response = await run_agent(message, session_id)
        return AgentResponse(
            response=response,
            session_id=session_id,
            agent_name=root_agent.name,
        )
    except Exception as e:
        logger.error(f"Summarize error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify", response_model=AgentResponse, tags=["Agent"])
async def classify(req: ClassifyRequest):
    """Classify the topic/domain of any text."""
    import uuid
    session_id = f"c-{uuid.uuid4().hex}"
    message = (
        f"Please classify the following text and tell me its topic/category:\n\n{req.text}"
    )
    try:
        response = await run_agent(message, session_id)
        return AgentResponse(
            response=response,
            session_id=session_id,
            agent_name=root_agent.name,
        )
    except Exception as e:
        logger.error(f"Classify error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=AgentResponse, tags=["Agent"])
async def chat(req: AgentRequest):
    """Free-form conversation with the agent."""
    import uuid
    session_id = req.session_id or f"ch-{uuid.uuid4().hex}"
    try:
        response = await run_agent(req.message, session_id)
        return AgentResponse(
            response=response,
            session_id=session_id,
            agent_name=root_agent.name,
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
