from fastapi import FastAPI 
from src.config.logging import setup_logging
setup_logging() 
from .endpoints.health import health_router 
from .endpoints.documents import document_router 

app = FastAPI(
    title = "Agentic RAG API",
    version = "0.1.0" 
)
app.include_router(health_router) 
app.include_router(document_router) 