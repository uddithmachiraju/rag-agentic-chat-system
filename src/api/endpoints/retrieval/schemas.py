from pydantic import BaseModel, Field
from typing import List, Dict, Any, Union, Optional 
from src.models.query import (
    QueryType, RetrievalStrategy
)

# ------------------------------------ /search ------------------------------------
class SearchRequest(BaseModel):
    """Request Model for search"""
    query: str = Field(..., description = "Search query text")
    query_type: Optional[QueryType] = Field(QueryType.SEMANTIC, description = "Type of query")
    retrieval_strategy: Optional[RetrievalStrategy] =Field(RetrievalStrategy.SIMILARITY, description = "Retrieval strategy to use")
    max_results: Optional[int] = Field(10, ge = 1, le = 100, description = "Max number of results")
    similarity_threshold: Optional[float] = Field(0.3, ge = 0.0, le = 1.0, description = "Minimum similarity threshold")
    document_ids: Optional[List[str]] = Field(None, description = "Filter by document ID")
    chunk_types: Optional[List[str]] = Field(None, description = "Filter by chunk types")
    user_id: Optional[str] = Field(None, description = "User ID") 
    session_id: Optional[str] = Field(None, description = "Session ID for context")

class SearchResult(BaseModel):
    """Search result model"""
    chunk_id: str 
    document_id: str 
    content: str 
    highlightes_content: Optional[str] = None 
    similarity_score: float
    metadata: Dict[str, Any] 

class SearchResponse(BaseModel):
    """Response model for search request"""
    query_id: str 
    results: List[Dict] 
    total_results: int 
    retrieval_time: float 
    processing_time: Optional[float] = None 
    strategy_used: str 
    cached: bool = False 