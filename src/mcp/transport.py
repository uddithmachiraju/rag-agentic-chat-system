"""MCP transport layer for message passing between agents."""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional
import json
from datetime import datetime, timedelta, UTC

from ..config import get_logger, get_settings
from .protocol import MCPMessage, MessageStatus, AgentType 


class MCPTransport(ABC):
    """Abstract base class for MCP transport implementations."""
    
    @abstractmethod
    async def send_message(self, message: MCPMessage) -> bool:
        """Send a message through the transport."""
        pass
    
    @abstractmethod
    async def receive_messages(self) -> List[MCPMessage]:
        """Receive messages from the transport."""
        pass
    
    @abstractmethod
    async def subscribe(self, agent_type: AgentType, callback: Callable[[MCPMessage], None]) -> None:
        """Subscribe to messages for a specific agent type."""
        pass


class InMemoryTransport(MCPTransport):
    """In-memory message transport for single-process deployment."""
    
    def __init__(self):
        self.logger = get_logger("InMemoryTransport")
        self.settings = get_settings()
        self.message_queues: Dict[AgentType, asyncio.Queue] = {}
        self.subscribers: Dict[AgentType, List[Callable[[MCPMessage], None]]] = {}
        self.message_history: List[MCPMessage] = []
        self._lock = asyncio.Lock()
    
    def _ensure_queue(self, agent_type: AgentType) -> None:
        """Ensure message queue exists for agent type."""
        if agent_type not in self.message_queues:
            self.message_queues[agent_type] = asyncio.Queue()
        if agent_type not in self.subscribers:
            self.subscribers[agent_type] = []
    
    async def send_message(self, message: MCPMessage) -> bool:
        """Send a message to the target agent's queue."""
        try:
            async with self._lock:
                self._ensure_queue(message.receiver)
                
                # Add to message history
                self.message_history.append(message)
                
                # Put message in target queue
                await self.message_queues[message.receiver].put(message)
                
                # Notify subscribers
                for callback in self.subscribers.get(message.receiver, []):
                    try:
                        callback(message)
                    except Exception as e:
                        self.logger.error(f"Subscriber callback error: {e}")
                
                self.logger.debug(
                    f"Message sent: {message.sender} -> {message.receiver} "
                    f"({message.message_type})"
                )
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            return False
    
    async def receive_messages(self, agent_type: AgentType, timeout: float = 1.0) -> List[MCPMessage]:
        """Receive messages for a specific agent type."""
        messages = []
        
        try:
            self._ensure_queue(agent_type)
            queue = self.message_queues[agent_type]
            
            # Get all available messages without blocking
            while True:
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=0.1)
                    messages.append(message)
                    queue.task_done()
                except asyncio.TimeoutError:
                    break
                    
        except Exception as e:
            self.logger.error(f"Failed to receive messages for {agent_type}: {e}")
        
        return messages
    
    async def subscribe(self, agent_type: AgentType, callback: Callable[[MCPMessage], None]) -> None:
        """Subscribe to messages for a specific agent type."""
        async with self._lock:
            self._ensure_queue(agent_type)
            self.subscribers[agent_type].append(callback)
            self.logger.info(f"Subscribed callback for {agent_type}")
    
    async def get_message_history(self, trace_id: Optional[str] = None, agent_type: Optional[AgentType] = None, limit: int = 100) -> List[MCPMessage]:
        """Get message history with optional filtering."""
        
        async with self._lock:
            filtered_messages = self.message_history
            
            if trace_id:
                filtered_messages = [m for m in filtered_messages if m.trace_id == trace_id]
            
            if agent_type:
                filtered_messages = [
                    m for m in filtered_messages 
                    if m.sender == agent_type or m.receiver == agent_type
                ]
            
            return filtered_messages[-limit:]
    
    async def cleanup_old_messages(self, max_age_hours: int = 24) -> None:
        """Clean up old messages from history."""
        
        cutoff_time = datetime.now(UTC) - timedelta(hours=max_age_hours)
        
        async with self._lock:
            self.message_history = [
                m for m in self.message_history 
                if m.timestamp > cutoff_time
            ]
            
            self.logger.info(f"Cleaned up messages older than {max_age_hours} hours")


class MCPMessageBus:
    """Message bus for coordinating MCP communications."""
    
    def __init__(self, transport: MCPTransport):
        self.transport = transport
        self.logger = get_logger("MCPMessageBus")
        self.settings = get_settings()
        self._running = False
    
    async def start(self) -> None:
        """Start the message bus."""
        self._running = True
        self.logger.info("MCP Message Bus started")
    
    async def stop(self) -> None:
        """Stop the message bus."""
        self._running = False
        self.logger.info("MCP Message Bus stopped")
    
    async def send_message(self, message: MCPMessage) -> bool:
        """Send a message through the transport."""
        if not self._running:
            self.logger.warning("Message bus not running, message not sent")
            return False
        
        message.status = MessageStatus.PROCESSING
        return await self.transport.send_message(message)
    
    async def broadcast_message(self, message: MCPMessage, target_agents: List[AgentType]) -> Dict[AgentType, bool]:
        """Broadcast a message to multiple agents."""
        
        results = {}
        for agent_type in target_agents:
            # Create a copy of the message for each recipient
            agent_message = message.model_copy()
            agent_message.receiver = agent_type
            
            results[agent_type] = await self.send_message(agent_message)
        
        return results
    
    async def request_response(self, message: MCPMessage, timeout: float = None) -> Optional[MCPMessage]:
        """Send a request and wait for response."""
        
        timeout = timeout or self.settings.mcp_timeout
        
        # Send the request
        success = await self.send_message(message)
        if not success:
            return None
        
        # Wait for response
        start_time = asyncio.get_event_loop().time()
        
        while (asyncio.get_event_loop().time() - start_time) < timeout:
            messages = await self.transport.receive_messages(message.sender)
            
            for response in messages:
                if (response.trace_id == message.trace_id and 
                    response.metadata.get("original_message_id") == message.message_id):
                    return response
            
            await asyncio.sleep(0.1)
        
        self.logger.warning(f"Request timeout for message {message.message_id}")
        return None


# Global message bus instance
_message_bus: Optional[MCPMessageBus] = None


async def get_message_bus() -> MCPMessageBus:
    """Get the global message bus instance."""
    global _message_bus
    
    if _message_bus is None:
        transport = InMemoryTransport()
        _message_bus = MCPMessageBus(transport)
        await _message_bus.start()
    
    return _message_bus
