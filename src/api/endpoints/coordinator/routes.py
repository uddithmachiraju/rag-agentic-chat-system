from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uuid
import shutil
from pathlib import Path

from src.api.endpoints.coordinator.schemas import CoordinatorRequest, CoordinatorResponse
from src.agents.agent_singleton import coordinator_agent
from src.mcp.protocol import MCPMessage, MessageType, AgentType

router = APIRouter()
coordinator = coordinator_agent

# Directory to store uploaded files (same as ingestion)
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@router.post("/ingestion/upload")
async def coordinator_upload(file: UploadFile = File(...), user_id: str = Form("anonymous")):
    """Upload a document for ingestion via coordinator."""
    file_id = str(uuid.uuid4().hex[:6])
    file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
    try:
        with open(file_path, "wb") as dest:
            shutil.copyfileobj(file.file, dest)
        # Route to coordinator for processing
        mcp_message = MCPMessage(
            sender=AgentType.COORDINATOR,
            receiver=AgentType.COORDINATOR,
            message_type=MessageType.REQUEST,
            payload={"action": "process_document", "file_path": str(file_path), "user_id": user_id}
        )
        response = await coordinator._handle_message(mcp_message)
        if not response or response.message_type == MessageType.ERROR:
            raise HTTPException(status_code=500, detail=response.payload.get("error", "Ingestion failed"))
        return response.payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ingestion/documents/{user_id}")
async def coordinator_list_documents(user_id: str):
    """List all documents for a user via coordinator."""
    mcp_message = MCPMessage(
        sender=AgentType.COORDINATOR,
        receiver=AgentType.COORDINATOR,
        message_type=MessageType.REQUEST,
        payload={"action": "list_documents", "user_id": user_id}
    )
    response = await coordinator._handle_message(mcp_message)
    if not response or response.message_type == MessageType.ERROR:
        raise HTTPException(status_code=500, detail=response.payload.get("error", "List failed"))
    return response.payload

@router.get("/ingestion/document/{document_id}")
async def coordinator_get_document(document_id: str):
    """Get document details and chunks via coordinator."""
    mcp_message = MCPMessage(
        sender=AgentType.COORDINATOR,
        receiver=AgentType.COORDINATOR,
        message_type=MessageType.REQUEST,
        payload={"action": "get_document", "document_id": document_id}
    )
    response = await coordinator._handle_message(mcp_message)
    if not response or response.message_type == MessageType.ERROR:
        raise HTTPException(status_code=500, detail=response.payload.get("error", "Get failed"))
    # Return the full MCPMessage as a dict (model_dump), not just the payload
    return response.model_dump()

@router.get("/ingestion/document/{document_id}/status")
async def coordinator_get_document_status(document_id: str):
    """Get processing status for a document via coordinator."""
    mcp_message = MCPMessage(
        sender=AgentType.COORDINATOR,
        receiver=AgentType.COORDINATOR,
        message_type=MessageType.REQUEST,
        payload={"action": "get_processing_status", "document_id": document_id}
    )
    response = await coordinator._handle_message(mcp_message)
    if not response or response.message_type == MessageType.ERROR:
        raise HTTPException(status_code=500, detail=response.payload.get("error", "Status failed"))
    return response.payload

@router.delete("/ingestion/document/{document_id}")
async def coordinator_delete_document(document_id: str):
    """Delete a document via coordinator."""
    mcp_message = MCPMessage(
        sender=AgentType.COORDINATOR,
        receiver=AgentType.COORDINATOR,
        message_type=MessageType.REQUEST,
        payload={"action": "delete_document", "document_id": document_id}
    )
    response = await coordinator._handle_message(mcp_message)
    if not response or response.message_type == MessageType.ERROR:
        raise HTTPException(status_code=500, detail=response.payload.get("error", "Delete failed"))
    return response.payload

@router.get("/ingestion/ingestion/stats")
async def coordinator_get_ingestion_stats():
    """Get ingestion statistics via coordinator."""
    mcp_message = MCPMessage(
        sender=AgentType.COORDINATOR,
        receiver=AgentType.COORDINATOR,
        message_type=MessageType.REQUEST,
        payload={"action": "get_ingestion_stats"}
    )
    response = await coordinator._handle_message(mcp_message)
    if not response or response.message_type == MessageType.ERROR:
        raise HTTPException(status_code=500, detail=response.payload.get("error", "Stats failed"))
    return response.payload

@router.get("/ingestion/debug/documents")
async def coordinator_debug_documents():
    """Debug: List all active and stored documents via coordinator."""
    mcp_message = MCPMessage(
        sender=AgentType.COORDINATOR,
        receiver=AgentType.COORDINATOR,
        message_type=MessageType.REQUEST,
        payload={"action": "debug_documents"}
    )
    response = await coordinator._handle_message(mcp_message)
    if not response or response.message_type == MessageType.ERROR:
        raise HTTPException(status_code=500, detail=response.payload.get("error", "Debug failed"))
    return response.payload

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

# List all built-in coordinator routes
@router.get("/coordinator/routes")
async def coordinator_list_routes():
    """List all available built-in coordinator routes."""
    routes_info = []
    for action, route in coordinator.routes.items():
        routes_info.append({
            "action": action,
            "description": route.description,
            "requires_agents": route.requires_agents or [],
            "timeout_seconds": route.timeout_seconds
        })
    return {"routes": routes_info}