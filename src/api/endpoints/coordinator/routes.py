from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional
import asyncio

from src.agents.coordinator import CoordinatorAgent
from src.mcp.protocol import MCPMessage, MessageType, AgentType

router = APIRouter()
coordinator = CoordinatorAgent()

class CoordinatorRequest(BaseModel): 
    action: str
    payload: Optional[Dict[str, Any]] = {}


class CoordinatorResponse(BaseModel):
    success: bool
    payload: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@router.post("/coordinator/message", response_model=CoordinatorResponse)
async def coordinator_message(request: CoordinatorRequest):
    """
    Generic endpoint for sending messages to CoordinatorAgent.
    The 'action' field determines the route to execute.
    """
    try:
        # Build MCPMessage
        mcp_message = MCPMessage(
            sender=AgentType.COORDINATOR,
            receiver=AgentType.COORDINATOR,
            message_type=MessageType.REQUEST,
            payload={"action": request.action, **(request.payload or {})}
        )

        # Handle message via CoordinatorAgent
        response = await coordinator._handle_message(mcp_message)

        if not response:
            raise HTTPException(status_code=500, detail="No response from coordinator")

        # Convert MCP response to API response
        if response.message_type == MessageType.ERROR:
            return CoordinatorResponse(success=False, error=response.payload.get("error"))

        return CoordinatorResponse(success=True, payload=response.payload)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Check Coordinator and agent health.
@router.get("/coordinator/health", response_model=CoordinatorResponse)
async def coordinator_health():
    try:
        health = await coordinator._route_health(MCPMessage(
            sender=AgentType.COORDINATOR,
            receiver=AgentType.COORDINATOR,
            message_type=MessageType.REQUEST,
            payload={"action": "health"}
        ))
        return CoordinatorResponse(success=True, payload=health.payload)
    except Exception as e:
        return CoordinatorResponse(success=False, error=str(e))
