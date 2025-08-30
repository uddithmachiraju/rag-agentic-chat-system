from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from src.mcp.protocol import (
    AgentType, MessageType
)

class SendMessageRequest(BaseModel):
    sender: AgentType
    receiver: AgentType
    message_type: MessageType
    payload: Dict[str, Any] 
    metadata: Optional[Dict[str, Any]] = None
    trace_id: Optional[str] = None 

class BroadcastMessageRequest(BaseModel):
    sender: AgentType
    target_agents: List[AgentType]
    message_type: MessageType
    payload: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None 

class MessageResponse(BaseModel):
    success: bool
    message_id: Optional[str] = None 
    trace_id: Optional[str] = None
    error: Optional[str] = None 

class AgentSubscriptionRequest(BaseModel):
    agent_type: AgentType

class RequestResponseModel(BaseModel):
    sender: AgentType
    receiver: AgentType
    message_type: MessageType
    payload: Dict[str, Any]
    timeout: Optional[float] = 30.0
    metadata: Optional[Dict[str, Any]] = None