from contextlib import asynccontextmanager
import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from app.agent import run_agent
from app.gateway import gateway
from app.schemas import AgentRequest

load_dotenv()

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="Enterprise MCP Agent",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "enterprise-mcp-agent",
    }


@app.get("/api/v1/mcp/tools")
async def list_tools(role: str = "sales"):
    tools = await gateway.discover_tools(role)

    return {
        "role": role,
        "tools": tools,
    }


@app.post("/api/v1/agent/chat")
async def chat(request: AgentRequest):
    try:
        return await run_agent(
            question=request.question,
            role=request.role,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@app.get("/api/v1/mcp/traces")
async def list_traces():
    return {
        "items": gateway.traces[-100:],
    }


@app.get("/api/v1/mcp/approvals")
async def list_approvals():
    return {
        "items": gateway.approvals[-100:],
    }