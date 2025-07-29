from fastapi import FastAPI 
from .endpoints.health import health_router 

app = FastAPI(
    title = "Agentic RAG API",
    version = "0.1.0" 
)
app.include_router(health_router) 