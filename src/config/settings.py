import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    app_name: str = "Agentic RAG Chatbot"
    app_version: str = "0.1.0"
    debug: bool = Field(False, env = "DEBUG") 

    # API configuration
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    api_reload: bool = Field(True, env="API_RELOAD") 
    
    # Google Gemini Configuration
    google_gemini_api_key: Optional[str] = Field(None, env="GOOGLE_GEMINI_API_KEY")
    google_gemini_model: str = Field("gemini-2.5-pro", env="GOOGLE_GEMINI_MODEL") 
    gemini_embedding_model: str = Field("models/embedding-001", env="GEMINI_EMBEDDING_MODEL")
    google_gemini_max_tokens: int = Field(4096, env="GOOGLE_GEMINI_MAX_TOKENS")
    google_gemini_temperature: float = Field(0.2, env="GOOGLE_GEMINI_TEMPERATURE")

    # Hugging Face Configuration
    huggingface_api_key: Optional[str] = Field(None, env="HUGGINGFACE_API_KEY")
    huggingface_model: str = Field("sentence-transformers/all-MiniLM-L6-v2", env="HUGGINGFACE_MODEL")
    huggingface_max_tokens: int = Field(2048, env="HUGGINGFACE_MAX_TOKENS")
    huggingface_temperature: float = Field(0.7, env="HUGGINGFACE_TEMPERATURE")

    # chunking configuration
    chunk_size: int = Field(500, env="CHUNK_SIZE")
    chunk_overlap: int = Field(50, env="CHUNK_OVERLAP")

    # Vector database configuration
    vector_db_type: str = Field("chroma", env="VECTOR_DB_TYPE")
    vector_db_path: str = Field("./vector_db", env="VECTOR_DB_PATH")
    chroma_db_path: str = Field("./chroma_db", env="CHROMA_DB_PATH")

    # Storage paths
    upload_directory: str = Field("./uploads", env="UPLOAD_DIRECTORY")
    logs_directory: str = Field("./logs", env="LOGS_DIRECTORY")

    # MCP Configuration
    mcp_timeout: int = Field(60, env="MCP_TIMEOUT")
    mcp_max_retries: int = Field(3, env="MCP_MAX_RETRIES")

@lru_cache()
def get_settings() -> Settings:
    """Get cached application instance."""
    return Settings() 

def ensure_directories():
    """Ensure necessary directories exist."""
    settings = Settings()
    os.makedirs(settings.upload_directory, exist_ok=True)
    os.makedirs(settings.logs_directory, exist_ok=True)
    os.makedirs(settings.vector_db_path, exist_ok=True) 