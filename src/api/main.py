from fastapi import FastAPI 
from src.config.logging import setup_logging
setup_logging() 
from .endpoints.health import health_router 
from .endpoints.ingestion.routes import document_router 
from contextlib import asynccontextmanager 
from src.agents.agent_singleton import coordinator_agent

from src.api.endpoints.coordinator.routes import router as coordinator_router
from src.api.endpoints.mcp.routes import mcp_router
from src.api.endpoints.retrieval.routes import router as retrieval_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the single coordinator agent instance
    await coordinator_agent.initialize()
    await coordinator_agent.start()
    await coordinator_agent.start_all_agents()
    yield

app = FastAPI(
    title = "Agentic RAG API",
    version = "0.1.0",
    lifespan = lifespan
)
# app.include_router(health_router, tags=["Health"])
# app.include_router(document_router, tags=["Ingestion"])
app.include_router(coordinator_router) 
app.include_router(mcp_router, tags=["MCP"])
# app.include_router(retrieval_router, tags=["Retrieval"])