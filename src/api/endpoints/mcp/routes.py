from fastapi import APIRouter, HTTPException, Query
from datetime import datetime, timedelta, UTC
from typing import List, Optional
from src.api.endpoints.mcp.schemas import (
    MessageResponse, SendMessageRequest
)
from src.mcp.transport import (
    MCPMessage, get_message_bus
)
from src.mcp.protocol import (
    create_mcp_message, AgentType, 
    MessageType, MessageStatus
)

mcp_router = APIRouter(prefix = "/mcp") 

@mcp_router.post("/send", response_model = MessageResponse)
async def send_message(request: SendMessageRequest):
    """Sends a message through the MCP transport system"""
    try:
        bus = await get_message_bus()

        message = create_mcp_message(
            sender = request.sender, 
            receiver = request.receiver, 
            message_type = request.message_type, 
            payload = request.payload, 
            trace_id = request.trace_id, 
            metadata = request.metadata
        )

        success = await bus.send_message(message) 

        return MessageResponse(
            success = success, 
            message_id = message.message_id, 
            trace_id = message.trace_id
        )
    except Exception as e:
        raise HTTPException(
            status_code = 500, detail = str(e)
        )
    
@mcp_router.get("/queues/status")
async def get_queue_status():
    """Get the status of all messages queues in transport"""
    try:
        bus = await get_message_bus()
        transport = bus.transport

        status = {}

        for agent_type in AgentType:
            queue_size = 0
            subsciber_count = 0

            if agent_type in transport.message_queues:
                queue_size = transport.message_queues[agent_type].qsize()

            if agent_type in transport.subscribers:
                subsciber_count = len(transport.subscribers[agent_type])

            status[agent_type.value] = {
                "queue_size": queue_size, 
                "subscriber_count": subsciber_count
            }

        return {
            "total_queues": len(transport.message_queues), 
            "total_messages_in_history": len(transport.message_history), 
            "queue_status": status
        }
    except Exception as e:
        raise HTTPException(
            status_code = 500, detail = str(e) 
        )

@mcp_router.get("/history", response_model = List[MCPMessage])
async def get_message_history(
    trace_id: Optional[str] = Query(None, description = "Filter by trace_id"), 
    agent_type: Optional[AgentType] = Query(None, description = "Filter by Agent Type"), 
    limit: Optional[int] = Query(100, description = "Maximum number of messages to return"),
    message_type: Optional[MessageType] = Query(None, description = "Filter by message type"),
    status: Optional[MessageStatus] = Query(None, description = "Filter by message status") 
    ):
    try:
        bus = await get_message_bus()
        transport = bus.transport

        messages = await transport.get_message_history(trace_id, agent_type, limit)

        if message_type:
            messages = [m for m in messages if m.message_type == message_type]
        
        if status:
            messages = [m for m in messages if m.status == status]

        return messages
    except Exception as e:
        raise HTTPException(
            status_code = 501, detail = str(e)
        )
    
@mcp_router.get("/traces/{trace_id}")
async def get_trace_details(trace_id: str):
    try:
        bus = await get_message_bus()
        transport = bus.transport

        messages = await transport.get_message_history(trace_id = trace_id)

        if not messages:
            raise HTTPException(status_code = 400, detail = "Trace not found")
        
        trace_analysis = {
            "trace_id": trace_id, 
            "message_count": len(messages), 
            "start_time": min(m.timestamp for m in messages), 
            "end_time": max(m.timestamp for m in messages), 
            "duration_seconds": (max(m.timestamp for m in messages) - min(m.timestamp for m in messages)).total_seconds(), 
            "agents_involved": list(set([m.sender for m in messages] + [m.receiver for m in messages])), 
            "message_types": list(set(m.message_type for m in messages)), 
            "status_counts": {}, 
            "messages": messages
        }

        for status in MessageStatus:
            trace_analysis["status_counts"][status.value] = len([m for m in messages if m.status == status])

        return trace_analysis 
    
    except Exception as e:
        raise HTTPException(status_code = 500, detail = str(e)) 

@mcp_router.get("/analytics")
async def get_message_analytics():
    try:
        bus = await get_message_bus()
        transport = bus.transport
        
        all_messages = await transport.get_message_history()
        
        if not all_messages:
            return {"message": "No messages in history"}
        
        # Calculate analytics
        now = datetime.now(UTC)
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)
        
        analytics = {
            "total_messages": len(all_messages),
            "messages_last_hour": len([m for m in all_messages if m.timestamp > hour_ago]),
            "messages_last_day": len([m for m in all_messages if m.timestamp > day_ago]),
            "unique_traces": len(set(m.trace_id for m in all_messages)),
            "agent_activity": {},
            "message_type_counts": {},
            "status_distribution": {},
            "average_processing_time": None,
            "peak_activity_hour": None
        }
        
        # Agent activity
        for agent in AgentType:
            sent = len([m for m in all_messages if m.sender == agent])
            received = len([m for m in all_messages if m.receiver == agent])
            analytics["agent_activity"][agent.value] = {
                "sent": sent,
                "received": received,
                "total": sent + received
            }
        
        # Message type counts
        for msg_type in MessageType:
            count = len([m for m in all_messages if m.message_type == msg_type])
            if count > 0:
                analytics["message_type_counts"][msg_type.value] = count
        
        # Status distribution
        for status in MessageStatus:
            count = len([m for m in all_messages if m.status == status])
            if count > 0:
                analytics["status_distribution"][status.value] = count
        
        return analytics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@mcp_router.post("/cleanup")
async def cleanup_old_messages(max_age_hours: int = Query(24, description = "Maximum age in hours")):
    try:
        bus = await get_message_bus()
        transport = bus.transport

        messages_before = len(transport.message_history)
        await transport.cleanup_old_messages(max_age_hours)
        messages_after = len(transport.message_history)

        return {
            "success": True, 
            "messages_before": messages_before, 
            "messages_after": messages_after, 
            "messages_cleaned": messages_before - messages_after, 
            "max_age_hours": max_age_hours
        }
    except Exception as e:
        raise HTTPException(
            status_code = 501, detail = str(e)
        )

@mcp_router.get("/status")
async def get_status():
    try:
        bus = await get_message_bus()

        return {
            "message_bus_running": bus._running, 
            "transport_type": type(bus.transport).__name__, 
            "timestamp": datetime.now(UTC).isoformat(), 
            "supported_agents": [agent.value for agent in AgentType], 
            "supported_message_types": [msg_type.value for msg_type in MessageType], 
            "supported_statuses": [status.value for status in MessageStatus]
        }
    except Exception as e:
        raise HTTPException(status_code = 500, detail = str(e)) 