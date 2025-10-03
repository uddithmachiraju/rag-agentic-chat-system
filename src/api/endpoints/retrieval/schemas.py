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
    retrieval_strategy: Optional[RetrievalStrategy] =Field(RetrievalStrategy.MMR, description = "Retrieval strategy to use")
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

class DocumentChunk(BaseModel):
    chunk_id: str = Field(..., description = "Unique chunk id")
    document_id: str = Field(..., description = "parent document identifier")
    content: str = Field(..., description = "chunk content")
    chunk_type: str = Field(..., description = "Type of chunk") 
    start_index: int = Field(..., description = "Start Position in document") 
    end_index: int = Field(..., description = "End position in document") 
    metadata: Optional[Dict[str, Any]] = Field(default_factory = dict, description = "chunk metadata")

class AddDocumentsRequest(BaseModel):
    chunks: List[DocumentChunk] = Field(..., description="List of document chunks to add")