"""Query data models for search and retrieval operations."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, validator


class QueryType(str, Enum):
    """Types of queries supported by the system."""
    SIMPLE = "simple"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    COMPLEX = "complex"
    MULTI_TURN = "multi_turn"


class SearchScope(str, Enum):
    """Scope of search operation."""
    ALL_DOCUMENTS = "all_documents"
    SPECIFIC_DOCUMENTS = "specific_documents"
    DOCUMENT_COLLECTION = "document_collection"
    DATE_RANGE = "date_range"


class RetrievalStrategy(str, Enum):
    """Retrieval strategies for document search."""
    SIMILARITY = "similarity"
    MMR = "mmr"  # Maximal Marginal Relevance
    DIVERSITY = "diversity"
    RERANK = "rerank"


class QueryFilters(BaseModel):
    """Filters for query execution."""
    
    document_ids: Optional[List[str]] = None
    document_formats: Optional[List[str]] = None
    chunk_types: Optional[List[str]] = None
    date_range: Optional[Dict[str, datetime]] = None
    metadata_filters: Dict[str, Any] = Field(default_factory=dict)
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0)
    max_results: int = Field(5, ge=1, le=50)
    
    @validator('date_range')
    def validate_date_range(cls, v):
        if v and 'start' in v and 'end' in v:
            if v['start'] > v['end']:
                raise ValueError('Start date must be before end date')
        return v


class ConversationContext(BaseModel):
    """Context from previous conversation turns."""
    
    conversation_id: str = Field(default_factory=lambda: str(uuid4()))
    previous_queries: List[str] = Field(default_factory=list)
    previous_responses: List[str] = Field(default_factory=list)
    relevant_chunks: List[str] = Field(default_factory=list)  # chunk_ids
    context_summary: Optional[str] = None
    
    def add_turn(self, query: str, response: str, relevant_chunks: List[str] = None) -> None:
        """Add a conversation turn."""
        self.previous_queries.append(query)
        self.previous_responses.append(response)
        if relevant_chunks:
            self.relevant_chunks.extend(relevant_chunks)
    
    def get_recent_context(self, turns: int = 3) -> Dict[str, List[str]]:
        """Get recent conversation context."""
        return {
            "queries": self.previous_queries[-turns:],
            "responses": self.previous_responses[-turns:]
        }


class Query(BaseModel):
    """Main query model for document search and retrieval."""
    
    query_id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    query_type: QueryType = QueryType.SEMANTIC
    search_scope: SearchScope = SearchScope.ALL_DOCUMENTS
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.SIMILARITY
    
    # Query configuration
    filters: QueryFilters = Field(default_factory=QueryFilters)
    processing_options: Dict[str, Any] = Field(default_factory=dict)
    
    # Conversation context
    conversation_context: Optional[ConversationContext] = None
    
    # Query metadata
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Query processing information
    processed_query: Optional[str] = None  # After preprocessing
    query_embedding: Optional[List[float]] = None
    processing_time: Optional[float] = None
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Query text cannot be empty')
        return v.strip()
    
    def preprocess_query(self) -> str:
        """Preprocess the query text."""
        # Basic preprocessing - can be extended
        processed = self.text.strip().lower()
        
        # Store processed version
        self.processed_query = processed
        return processed
    
    def add_conversation_context(self, context: ConversationContext) -> None:
        """Add conversation context to the query."""
        self.conversation_context = context
        self.query_type = QueryType.MULTI_TURN
    
    def is_multi_turn(self) -> bool:
        """Check if this is a multi-turn query."""
        return self.conversation_context is not None
    
    def get_context_summary(self) -> Optional[str]:
        """Get conversation context summary."""
        if self.conversation_context:
            recent = self.conversation_context.get_recent_context()
            if recent["queries"]:
                return f"Previous queries: {'; '.join(recent['queries'][-2:])}"
        return None


class QueryExpansion(BaseModel):
    """Query expansion and refinement."""
    
    original_query: str
    expanded_terms: List[str] = Field(default_factory=list)
    synonyms: List[str] = Field(default_factory=list)
    related_concepts: List[str] = Field(default_factory=list)
    
    # Expansion metadata
    expansion_method: str = "semantic"
    confidence_scores: Dict[str, float] = Field(default_factory=dict)
    
    def get_expanded_query(self) -> str:
        """Get the expanded query string."""
        all_terms = [self.original_query] + self.expanded_terms + self.synonyms
        return " ".join(all_terms)


class SearchResult(BaseModel):
    """Individual search result."""
    
    chunk_id: str
    document_id: str
    content: str
    similarity_score: float
    
    # Result metadata
    chunk_metadata: Dict[str, Any] = Field(default_factory=dict)
    document_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Position information
    page_number: Optional[int] = None
    chunk_index: Optional[int] = None
    
    # Highlighting information
    highlighted_content: Optional[str] = None
    match_positions: List[Dict[str, int]] = Field(default_factory=list)


class RetrievalResults(BaseModel):
    """Results from document retrieval."""
    
    query_id: str
    results: List[SearchResult] = Field(default_factory=list)
    
    # Result statistics
    total_results: int = 0
    max_similarity: float = 0.0
    min_similarity: float = 1.0
    avg_similarity: float = 0.0
    
    # Performance metrics
    retrieval_time: float
    post_processing_time: float = 0.0
    
    # Result metadata
    retrieval_strategy: RetrievalStrategy
    filters_applied: QueryFilters
    
    def add_result(self, result: SearchResult) -> None:
        """Add a search result."""
        self.results.append(result)
        self._update_statistics()
    
    def _update_statistics(self) -> None:
        """Update result statistics."""
        if not self.results:
            return
        
        self.total_results = len(self.results)
        scores = [r.similarity_score for r in self.results]
        self.max_similarity = max(scores)
        self.min_similarity = min(scores)
        self.avg_similarity = sum(scores) / len(scores)
    
    def get_top_results(self, n: int = 5) -> List[SearchResult]:
        """Get top N results by similarity score."""
        sorted_results = sorted(
            self.results, 
            key=lambda x: x.similarity_score, 
            reverse=True
        )
        return sorted_results[:n]
    
    def filter_by_similarity(self, threshold: float) -> List[SearchResult]:
        """Filter results by similarity threshold."""
        return [r for r in self.results if r.similarity_score >= threshold]
