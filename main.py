"""
FastAPI HTTP server for the ADK Text Intelligence Agent.
Uses await for all async ADK session calls.
"""

import os
import uuid
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Schemas ────────────────────────────────────────────────────────────────────

class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=5)
    style: Optional[str] = Field("concise")

class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=5)

class AgentRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class AgentResponse(BaseModel):
    response: str
    session_id: str
    agent_name: str

# ── Core runner ────────────────────────────────────────────────────────────────

async def run_agent(message: str) -> str:
    """Fresh session + runner per request. All ADK calls awaited."""
    session_id = uuid.uuid4().hex
    app_name   = "agent-app"
    user_id    = "user"

    svc = InMemorySessionService()

    # create_session is async — must be awaited
    await svc.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )

    r = Runner(
        agent=root_agent,
        app_name=app_name,
        session_service=svc,
    )

    content = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=message)],
    )

    final_response = ""
    async for event in r.run_async(
        user_id=user_id,
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

# ── App ────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("ADK Text Intelligence Agent starting...")
    yield

app = FastAPI(
    title="ADK Text Intelligence Agent",
    description="Text summarization and classification via Google ADK + Gemini-2.5-flash",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "agent": root_agent.name,
        "status": "running",
        "model": "gemini-2.5-flash",
        "endpoints": {
            "summarize": "POST /summarize",
            "classify":  "POST /classify",
            "chat":      "POST /chat",
            "health":    "GET /health",
            "docs":      "GET /docs",
        },
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "agent": root_agent.name}

@app.post("/summarize", response_model=AgentResponse)
async def summarize(req: SummarizeRequest):
    """Summarize text. Style: concise | detailed | bullets"""
    message = f"Summarize the following text in '{req.style}' style:\n\n{req.text}"
    try:
        response = await run_agent(message)
        return AgentResponse(
            response=response,
            session_id=uuid.uuid4().hex,
            agent_name=root_agent.name,
        )
    except Exception as e:
        logger.error(f"Summarize error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify", response_model=AgentResponse)
async def classify(req: ClassifyRequest):
    """Classify the topic/domain of text."""
    message = f"Classify the topic of this text:\n\n{req.text}"
    try:
        response = await run_agent(message)
        return AgentResponse(
            response=response,
            session_id=uuid.uuid4().hex,
            agent_name=root_agent.name,
        )
    except Exception as e:
        logger.error(f"Classify error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=AgentResponse)
async def chat(req: AgentRequest):
    """Free-form conversation with the agent."""
    try:
        response = await run_agent(req.message)
        return AgentResponse(
            response=response,
            session_id=uuid.uuid4().hex,
            agent_name=root_agent.name,
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ── Entry ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
