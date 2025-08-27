"""Response data models for LLM-generated answers."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, validator


class ResponseType(str, Enum):
    """Types of responses."""
    DIRECT_ANSWER = "direct_answer"
    SUMMARIZED = "summarized"
    COMPARATIVE = "comparative"
    INSTRUCTIONAL = "instructional"
    CLARIFICATION = "clarification"
    NO_ANSWER = "no_answer"


class ConfidenceLevel(str, Enum):
    """Confidence levels for responses."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"


class SourceAttribution(BaseModel):
    """Attribution information for response sources."""
    
    chunk_id: str
    document_id: str
    document_name: str
    page_number: Optional[int] = None
    chunk_index: int
    
    # Content information
    relevant_content: str
    similarity_score: float
    
    # Citation information
    citation_text: Optional[str] = None
    citation_number: Optional[int] = None
    
    def get_citation(self) -> str:
        """Get formatted citation."""
        citation = f"{self.document_name}"
        if self.page_number:
            citation += f", page {self.page_number}"
        return citation


class ResponseMetrics(BaseModel):
    """Metrics for response generation."""
    
    # Performance metrics
    retrieval_time: float
    generation_time: float
    total_response_time: float
    
    # Content metrics
    source_documents_count: int
    chunks_processed: int
    tokens_generated: int
    
    # Quality metrics
    avg_source_similarity: float
    confidence_score: float
    
    # Resource usage
    embedding_calls: int = 0
    llm_api_calls: int = 1
    tokens_consumed: int = 0


class Response(BaseModel):
    """Main response model for LLM-generated answers."""
    
    response_id: str = Field(default_factory=lambda: str(uuid4()))
    query_id: str
    
    # Response content
    answer: str
    response_type: ResponseType = ResponseType.DIRECT_ANSWER
    confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM
    
    # Source information
    sources: List[SourceAttribution] = Field(default_factory=list)
    source_summary: Optional[str] = None
    
    # Response metadata
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    model_used: Optional[str] = None
    temperature: Optional[float] = None
    
    # Quality indicators
    factual_accuracy: Optional[float] = None  # 0-1 score
    completeness: Optional[float] = None      # 0-1 score
    relevance: Optional[float] = None         # 0-1 score
    
    # Performance metrics
    metrics: ResponseMetrics
    
    # Follow-up suggestions
    follow_up_questions: List[str] = Field(default_factory=list)
    related_topics: List[str] = Field(default_factory=list)
    
    # User feedback
    user_rating: Optional[int] = None  # 1-5 rating
    user_feedback: Optional[str] = None
    
    @validator('answer')
    def answer_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Response answer cannot be empty')
        return v.strip()
    
    @validator('user_rating')
    def rating_in_range(cls, v):
        if v is not None and (v < 1 or v > 5):
            raise ValueError('User rating must be between 1 and 5')
        return v
    
    def add_source(self, source: SourceAttribution) -> None:
        """Add a source attribution."""
        # Assign citation number
        source.citation_number = len(self.sources) + 1
        self.sources.append(source)
    
    def get_formatted_sources(self) -> List[str]:
        """Get formatted source citations."""
        return [
            f"[{source.citation_number}] {source.get_citation()}"
            for source in self.sources
        ]
    
    def update_confidence(self) -> None:
        """Update confidence level based on metrics."""
        if self.metrics.avg_source_similarity >= 0.8 and len(self.sources) >= 2:
            self.confidence_level = ConfidenceLevel.HIGH
        elif self.metrics.avg_source_similarity >= 0.6 and len(self.sources) >= 1:
            self.confidence_level = ConfidenceLevel.MEDIUM
        elif self.metrics.avg_source_similarity >= 0.4:
            self.confidence_level = ConfidenceLevel.LOW
        else:
            self.confidence_level = ConfidenceLevel.UNCERTAIN
    
    def add_follow_up_question(self, question: str) -> None:
        """Add a follow-up question suggestion."""
        if question and question not in self.follow_up_questions:
            self.follow_up_questions.append(question)
    
    def get_response_summary(self) -> Dict[str, Any]:
        """Get response summary for logging/analytics."""
        return {
            "response_id": self.response_id,
            "query_id": self.query_id,
            "response_type": self.response_type,
            "confidence_level": self.confidence_level,
            "sources_count": len(self.sources),
            "answer_length": len(self.answer),
            "generated_at": self.generated_at,
            "total_response_time": self.metrics.total_response_time
        }
    
    def set_user_feedback(self, rating: Optional[int], feedback: Optional[str]) -> None:
        """Set user feedback for the response."""
        self.user_rating = rating
        self.user_feedback = feedback


class ConversationTurn(BaseModel):
    """A single turn in a conversation."""
    
    turn_id: str = Field(default_factory=lambda: str(uuid4()))
    query: str
    response: Response
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Turn metadata
    turn_number: int
    processing_successful: bool = True
    error_message: Optional[str] = None


class Conversation(BaseModel):
    """Complete conversation with multiple turns."""
    
    conversation_id: str = Field(default_factory=lambda: str(uuid4()))
    turns: List[ConversationTurn] = Field(default_factory=list)
    
    # Conversation metadata
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    started_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Conversation statistics
    total_turns: int = 0
    successful_turns: int = 0
    
    def add_turn(self, query: str, response: Response) -> ConversationTurn:
        """Add a new turn to the conversation."""
        turn = ConversationTurn(
            query=query,
            response=response,
            turn_number=len(self.turns) + 1
        )
        
        self.turns.append(turn)
        self.total_turns = len(self.turns)
        self.successful_turns = sum(1 for t in self.turns if t.processing_successful)
        self.last_activity_at = datetime.utcnow()
        
        return turn
    
    def get_recent_turns(self, n: int = 3) -> List[ConversationTurn]:
        """Get the most recent N turns."""
        return self.turns[-n:]
    
    def get_conversation_context(self) -> str:
        """Get conversation context for multi-turn queries."""
        recent_turns = self.get_recent_turns(3)
        context_parts = []
        
        for turn in recent_turns:
            context_parts.append(f"Q: {turn.query}")
            context_parts.append(f"A: {turn.response.answer[:200]}...")
        
        return "\n".join(context_parts)
