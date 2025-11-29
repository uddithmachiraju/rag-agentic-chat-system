from fastapi import APIRouter, HTTPException, UploadFile, File, Form
# from fastapi.responses import JSONResponse
import uuid
import shutil
from pathlib import Path
from typing import Optional 

from src.api.endpoints.coordinator.schemas import CoordinatorRequest, CoordinatorResponse
from src.agents.agent_singleton import coordinator_agent
from src.mcp.protocol import MCPMessage, MessageType, AgentType, create_mcp_message

router = APIRouter()
coordinator = coordinator_agent

# Directory to store uploaded files (same as ingestion)
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@router.post("/ingestion/upload", tags=["ingestion"]) 
async def upload_document(file: UploadFile = File(...), user_id: str = Form("anonymous")):
    """Upload a document for ingestion via coordinator."""
    file_id = str(uuid.uuid4().hex[:6])
    file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
    try:
        with open(file_path, "wb") as dest:
            shutil.copyfileobj(file.file, dest)
        # Route to coordinator for processing
        mcp_message = create_mcp_message(
            sender=AgentType.COORDINATOR,
            receiver=AgentType.COORDINATOR,
            message_type=MessageType.REQUEST,
            payload={"action": "process_document", "file_path": str(file_path), "user_id": user_id}
        )
        response = await coordinator._handle_message(mcp_message)
        if not response or response.message_type == MessageType.ERROR:
            error_detail = response.payload.get("error", "Ingestion failed") if response else "Ingestion failed"
            raise HTTPException(status_code=500, detail=error_detail)
        return response.payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ingestion/documents", tags=["ingestion"]) 
async def list_documents(user_id: str = "anonymous"):
    """List all documents, optionally filtered by user_id."""
    payload = {"action": "list_documents"}
    if user_id:
        payload["user_id"] = user_id
    
    mcp_message = create_mcp_message(
        sender=AgentType.COORDINATOR,
        receiver=AgentType.COORDINATOR,
        message_type=MessageType.REQUEST,
        payload=payload
    )
    response = await coordinator._handle_message(mcp_message)
    if not response or response.message_type == MessageType.ERROR:
        error_detail = response.payload.get("error", "List failed") if response else "List failed"
        raise HTTPException(status_code=500, detail=error_detail)
    return response.payload

@router.get("/ingestion/document/{document_id}", tags=["ingestion"]) 
async def get_document(document_id: str):
    """Get document details and chunks via coordinator."""
    mcp_message = create_mcp_message(
        sender=AgentType.COORDINATOR,
        receiver=AgentType.COORDINATOR,
        message_type=MessageType.REQUEST,
        payload={"action": "get_document", "document_id": document_id}
    )
    response = await coordinator._handle_message(mcp_message)
    if not response or response.message_type == MessageType.ERROR:
        error_detail = response.payload.get("error", "Get failed") if response else "Get failed"
        raise HTTPException(status_code=500, detail=error_detail)
    # Return the full MCPMessage as a dict (model_dump), not just the payload
    return response.model_dump()

@router.get("/ingestion/document/{document_id}/status", tags=["ingestion"])
async def get_document_status(document_id: str):
    """Get processing status for a document via coordinator."""
    mcp_message = create_mcp_message(
        sender=AgentType.COORDINATOR,
        receiver=AgentType.COORDINATOR,
        message_type=MessageType.REQUEST,
        payload={"action": "get_processing_status", "document_id": document_id}
    )
    response = await coordinator._handle_message(mcp_message)
    if not response or response.message_type == MessageType.ERROR:
        error_detail = response.payload.get("error", "Status failed") if response else "Status failed"
        raise HTTPException(status_code=500, detail=error_detail)
    return response.payload


@router.post("/ingestion/document/{document_id}/cancel", tags=["ingestion"])
async def cancel_processing(document_id: str):
    """Cancel processing for a document via coordinator."""
    mcp_message = MCPMessage(
        sender=AgentType.COORDINATOR,
        receiver=AgentType.COORDINATOR,
        message_type=MessageType.REQUEST,
        payload={"action": "cancel_processing", "document_id": document_id}
    )
    response = await coordinator._handle_message(mcp_message)
    if not response or response.message_type == MessageType.ERROR:
        error_detail = response.payload.get("error", "Cancel failed") if response else "Cancel failed"
        raise HTTPException(status_code=500, detail=error_detail)
    return response.payload

@router.delete("/ingestion/document/{document_id}", tags=["ingestion"])
async def delete_document(document_id: str):
    """Delete a document via coordinator."""
    mcp_message = create_mcp_message(
        sender=AgentType.COORDINATOR,
        receiver=AgentType.COORDINATOR,
        message_type=MessageType.REQUEST,
        payload={"action": "delete_document", "document_id": document_id}
    )
    response = await coordinator._handle_message(mcp_message)
    if not response or response.message_type == MessageType.ERROR:
        error_detail = response.payload.get("error", "Delete failed") if response else "Delete failed"
        raise HTTPException(status_code=500, detail=error_detail)
    return response.payload

@router.get("/ingestion/documents/{document_id}/chunks", tags=["ingestion"])
async def get_document_chunks(document_id: str, limit: int = 10, offset: int = 0):
    """List chunks for a document (with pagination)."""

    try:
        mcp_message = create_mcp_message(
            sender=AgentType.COORDINATOR,
            receiver=AgentType.COORDINATOR,
            message_type=MessageType.REQUEST,
            payload={"action": "list_document_chunks", "document_id": document_id, "limit": limit, "offset": offset}
        )
        response = await coordinator._handle_message(mcp_message)
        if not response or response.message_type == MessageType.ERROR:
            error_detail = response.payload.get("error", "List chunks failed") if response else "List chunks failed"
            raise HTTPException(status_code=500, detail=error_detail)
        return response.payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# @router.get("/ingestion/chunking/strategies", tags=["ingestion"], response_model=CoordinatorResponse)
# async def get_chunking_strategies():
#     """Get available chuncking strategies."""
#     try:
#         mcp_message = create_mcp_message(
#             sender=AgentType.COORDINATOR, 
#             receiver=AgentType.COORDINATOR, 
#             message_type=MessageType.REQUEST, 
#             payload={"action": "chunking_strategies"}
#         )
#         response = await coordinator._handle_message(mcp_message) 
#         if not response or response.message_type == MessageType.ERROR:
#             error_msg = response.payload.get("error", "Failed to get strategies") if response else "Failed to get strategies"
#             raise HTTPException(status_code=500, detail=error_msg)
#         return CoordinatorResponse(success=True, payload=response.payload)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@router.get("/ingestion/stats", tags=["ingestion"]) 
async def get_ingestion_stats():
    """Get ingestion statistics via coordinator."""
    mcp_message = create_mcp_message(
        sender=AgentType.COORDINATOR,
        receiver=AgentType.COORDINATOR,
        message_type=MessageType.REQUEST,
        payload={"action": "get_ingestion_stats"}
    )
    response = await coordinator._handle_message(mcp_message)
    if not response or response.message_type == MessageType.ERROR:
        raise HTTPException(status_code=500, detail=response.payload.get("error", "Stats failed"))
    return response.payload

@router.get("/ingestion/debug/documents", tags=["ingestion"]) 
async def debug_documents():
    """Debug: List all active and stored documents via coordinator."""
    mcp_message = create_mcp_message(
        sender=AgentType.COORDINATOR,
        receiver=AgentType.COORDINATOR,
        message_type=MessageType.REQUEST,
        payload={"action": "debug_documents"}
    )
    response = await coordinator._handle_message(mcp_message)
    if not response or response.message_type == MessageType.ERROR:
        raise HTTPException(status_code=500, detail=response.payload.get("error", "Debug failed"))
    return response.payload

@router.post("/coordinator/message", tags=["coordinator"], response_model=CoordinatorResponse)
async def coordinator_message(request: CoordinatorRequest):
    """
    Generic endpoint for sending messages to CoordinatorAgent.
    The 'action' field determines the route to execute.
    """
    try:
        # Build MCPMessage
        mcp_message = create_mcp_message(
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
@router.get("/coordinator/health", tags=["coordinator"], response_model=CoordinatorResponse)
async def coordinator_health():
    try:
        health = await coordinator._route_health(create_mcp_message(
            sender=AgentType.COORDINATOR,
            receiver=AgentType.COORDINATOR,
            message_type=MessageType.REQUEST,
            payload={"action": "health"}
        ))
        return CoordinatorResponse(success=True, payload=health.payload)
    except Exception as e:
        return CoordinatorResponse(success=False, error=str(e))

# List all built-in coordinator routes
@router.get("/coordinator/routes", tags=["coordinator"])
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

@router.get("/llm/stats", tags=["llm"], response_model=CoordinatorResponse)
async def get_llm_stats():
    """Get LLM statistics from the LLM agent."""
    try:
        # Create message with get_stats action for the LLM agent
        # The coordinator will route this through _route_get_llm_stats
        mcp_message = create_mcp_message(
            sender=AgentType.COORDINATOR,
            receiver=AgentType.COORDINATOR, 
            message_type=MessageType.REQUEST, 
            payload={"action": "get_llm_stats"} 
        )

        response = await coordinator._handle_message(mcp_message) 
        if not response or response.message_type == MessageType.ERROR:
            error_msg = response.payload.get("error", "Stats failed") if response else "Stats failed"
            return CoordinatorResponse(success=False, error=error_msg)
        
        return CoordinatorResponse(success=True, payload=response.payload)
    except Exception as e:
        return CoordinatorResponse(success=False, error=str(e))
    
@router.post("/retrieval/query/analyze", tags=["retrieval"], response_model=CoordinatorResponse)
async def query_analyze(query: str, context: Optional[str] = None):
    """Analyze a query using the retrieval agent via coordinator."""
    try:
        mcp_message = create_mcp_message(
            sender=AgentType.COORDINATOR,
            receiver=AgentType.COORDINATOR,
            message_type=MessageType.REQUEST,
            payload={"action": "analyze_query", "query": query, "conversation_context": context}
        )

        response = await coordinator._handle_message(mcp_message)
        if not response or response.message_type == MessageType.ERROR:
            error_msg = response.payload.get("error", "Query failed") if response else "Query failed"
            return CoordinatorResponse(success=False, error=error_msg)

        return CoordinatorResponse(success=True, payload=response.payload)
    except Exception as e:
        return CoordinatorResponse(success=False, error=str(e))
    
@router.post("/retrieval/query", tags=["retrieval"], response_model=CoordinatorResponse)
async def perform_retrieval_query(query: str):
    """Perform a retrieval query via the retrieval agent through coordinator."""
    try:
        mcp_message = create_mcp_message(
            sender=AgentType.COORDINATOR,
            receiver=AgentType.COORDINATOR,
            message_type=MessageType.REQUEST,
            payload={"action": "retrieve_documents", "query": query}
        )

        response = await coordinator._handle_message(mcp_message)
        if not response or response.message_type == MessageType.ERROR:
            error_msg = response.payload.get("error", "Query failed") if response else "Query failed"
            return CoordinatorResponse(success=False, error=error_msg)

        return CoordinatorResponse(success=True, payload=response.payload)
    except Exception as e:
        return CoordinatorResponse(success=False, error=str(e))

@router.get("/retrieval/stats", tags=["retrieval"], response_model=CoordinatorResponse)
async def get_retrieval_stats():
    """Get retrieval statistics from the retrieval agent."""
    try:
        # Create message with get_stats action for the retrieval agent
        # The coordinator will route this through _route_get_retrieval_stats
        mcp_message = create_mcp_message(
            sender=AgentType.COORDINATOR,
            receiver=AgentType.COORDINATOR, 
            message_type=MessageType.REQUEST, 
            payload={"action": "get_retrieval_stats"} 
        )

        response = await coordinator._handle_message(mcp_message) 
        if not response or response.message_type == MessageType.ERROR:
            error_msg = response.payload.get("error", "Stats failed") if response else "Stats failed"
            return CoordinatorResponse(success=False, error=error_msg)
        
        return CoordinatorResponse(success=True, payload=response.payload)
    except Exception as e:
        return CoordinatorResponse(success=False, error=str(e))