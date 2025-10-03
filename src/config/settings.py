import os 
from functools import lru_cache
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

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

    # Vector database configuration
    vector_db_type: str = Field("chroma", env="VECTOR_DB_TYPE")
    vector_db_path: str = Field("./vector_db", env="VECTOR_DB_PATH")
    chroma_db_path: str = Field("./chroma_db", env="CHROMA_DB_PATH")

    # Storage paths
    upload_directory: str = Field("./uploads", env="UPLOAD_DIRECTORY")
    logs_directory: str = Field("./logs", env="LOGS_DIRECTORY")

    # MCP Configuration
    mcp_timeout: int = Field(30, env="MCP_TIMEOUT")
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