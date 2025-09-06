"""Data models package for the agentic RAG chatbot."""

from .document import (
    Document,
    DocumentChunk,
    DocumentCollection,
    DocumentFormat,
    DocumentMetadata,
    ProcessingOptions,
    ProcessingStatus,
    ChunkType
)
from .query import (
    Query,
    QueryFilters,
    QueryType,
    ConversationContext,
    SearchScope,
    RetrievalStrategy,
    RetrievalResults,
    SearchResult,
    QueryExpansion
)
from .response import (
    Response,
    ResponseType,
    ConfidenceLevel,
    SourceAttribution,
    ResponseMetrics,
    ConversationTurn,
    Conversation
)

__all__ = [
    # Document models
    "Document",
    "DocumentChunk", 
    "DocumentCollection",
    "DocumentFormat",
    "DocumentMetadata",
    "ProcessingOptions",
    "ProcessingStatus",
    "ChunkType",
    
    # Query models
    "Query",
    "QueryFilters",
    "QueryType",
    "ConversationContext", 
    "SearchScope",
    "RetrievalStrategy",
    "RetrievalResults",
    "SearchResult",
    "QueryExpansion",
    
    # Response models
    "Response",
    "ResponseType",
    "ConfidenceLevel",
    "SourceAttribution",
    "ResponseMetrics",
    "ConversationTurn",
    "Conversation"
]
