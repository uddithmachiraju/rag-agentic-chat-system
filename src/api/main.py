from fastapi import FastAPI 
from src.config.logging import setup_logging
setup_logging() 
from .endpoints.health import health_router 
from .endpoints.documents import document_router 
from contextlib import asynccontextmanager 
from src.agents.coordinator import CoordinatorAgent 
from src.api.endpoints.coordinator.routes import router as coordinator_router
from src.api.endpoints.mcp.routes import mcp_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the coordinator agent
    coordinator_agent = CoordinatorAgent()
    await coordinator_agent._initialize_agent()
    yield

app = FastAPI(
    title = "Agentic RAG API",
    version = "0.1.0",
    lifespan = lifespan
)
app.include_router(health_router) 
app.include_router(document_router) 
app.include_router(coordinator_router) 
app.include_router(mcp_router) 