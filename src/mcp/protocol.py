import uuid
from datetime import UTC, datetime 
from enum import Enum 
from typing import Any, Dict, Optional, Union 
from pydantic import BaseModel, Field

class MessageType(str, Enum):
    REQUEST = "REQUEST"
    RESPONSE = "RESPONSE"
    NOTIFICATION = "NOTIFICATION"
    ERROR = "ERROR"
    CONTEXT_REQUEST = "CONTEXT_REQUEST"
    CONTEXT_RESPONSE = "CONTEXT_RESPONSE"
    DOCUMENT_PROCESSED = "DOCUMENT_PROCESSED"
    RETRIEVAL_REQUEST = "RETRIEVAL_REQUEST"
    RETRIEVAL_RESULT = "RETRIEVAL_RESULT"
    LLM_REQUEST = "LLM_REQUEST"
    LLM_RESPONSE = "LLM_RESPONSE"

class AgentType(str, Enum):
    """Agent types in the system."""
    COORDINATOR = "CoordinatorAgent"
    INGESTION = "IngestionAgent"
    RETRIEVAL = "RetrievalAgent"
    LLM_RESPONSE = "LLMResponseAgent"

class MessageStatus(str, Enum):
    """Message processing status."""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"

class MCPMessage(BaseModel):
    message_id: str = Field(default_factory = lambda: str(uuid.uuid4()))
    trace_id: str = Field(default_factory = lambda: str(uuid.uuid4())) 
    timestamp: datetime = Field(default_factory = lambda: datetime.now(UTC)) 
    sender: AgentType 
    receiver: AgentType
    message_type: MessageType 
    status: MessageStatus = MessageStatus.PENDING 
    payload: Dict[str, Any] = Field(default_factory = dict)
    metadata: Dict[str, Any] = Field(default_factory = dict) 

    class ConfigDict:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class DocumentProcessingRequest(BaseModel):
    """Request for document processing."""
    
    file_path: str
    file_name: str
    file_type: str
    user_id: Optional[str] = None
    processing_options: Dict[str, Any] = Field(default_factory=dict)

class DocumentProcessingResponse(BaseModel):
    """Response from document processing."""
    
    document_id: str
    chunks_created: int
    metadata: Dict[str, Any]
    processing_time: float
    success: bool
    error_message: Optional[str] = None

def create_mcp_message(sender: AgentType, receiver: AgentType, message_type: MessageType, 
                       payload: Dict[str, Any], trace_id: Optional[str] = None, 
                       metadata: Optional[Dict[str, Any]] = None):
    return MCPMessage(
        sender = sender, 
        receiver = receiver, 
        message_type = message_type,
        payload = payload, 
        trace_id = trace_id or str(uuid.uuid4()),
        metadata = metadata or {} 
    ) 

def create_response_message(original_message: MCPMessage, response_payload: Dict[str, Any],status: MessageStatus = MessageStatus.SUCCESS) -> MCPMessage:
    """Create a response message for an original request."""
    
    return MCPMessage(
        sender = original_message.receiver,
        receiver = original_message.sender,
        message_type = MessageType.RESPONSE,
        payload = response_payload,
        trace_id = original_message.trace_id,
        status = status,
        metadata = {
            "original_message_id": original_message.message_id,
            "response_to": original_message.message_type
        }
    )


def create_error_message(original_message: MCPMessage, error_message: str, error_code: Optional[str] = None) -> MCPMessage:
    """Create an error response message."""
    
    return MCPMessage(
        sender = original_message.receiver,
        receiver = original_message.sender,
        message_type = MessageType.ERROR,
        status = MessageStatus.ERROR,
        payload = {
            "error_message": error_message,
            "error_code": error_code,
            "original_payload": original_message.payload
        },
        trace_id = original_message.trace_id,
        metadata = {
            "original_message_id": original_message.message_id,
            "error_timestamp": datetime.now(UTC).isoformat()
        }
    )