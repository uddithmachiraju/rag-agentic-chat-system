"""Base agent class with MCP integration and lifecycle management."""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set
from datetime import datetime

from ..config import LoggerMixin, get_settings
from ..mcp.protocol import (
    MCPMessage, MessageType, AgentType, MessageStatus,
    create_response_message, create_error_message
)
from ..mcp.transport import get_message_bus, MCPMessageBus


class AgentState:
    """Agent state management."""
    
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class BaseAgent(ABC, LoggerMixin):
    """Abstract base class for all agents in the system."""
    
    def __init__(self, agent_type: AgentType):
        self.agent_type = agent_type
        self.settings = get_settings()
        self.state = AgentState.INITIALIZING
        self.message_bus: Optional[MCPMessageBus] = None
        self.active_tasks: Set[str] = set()
        self.metrics = {
            "messages_processed": 0,
            "messages_sent": 0,
            "errors": 0,
            "start_time": None,
            "last_activity": None
        }
        self._running = False
        self._shutdown_event = asyncio.Event()
    
    async def initialize(self) -> bool:
        """Initialize the agent and establish MCP connection."""
        try:
            self.logger.info(f"Initializing {self.agent_type}")
            
            # Get message bus
            self.message_bus = await get_message_bus()
            
            # Subscribe to messages
            await self.message_bus.transport.subscribe(
                self.agent_type, 
                self._handle_incoming_message
            )
            
            # Perform agent-specific initialization
            if await self._initialize_agent():
                self.state = AgentState.READY
                self.metrics["start_time"] = datetime.utcnow()
                self.logger.info(f"{self.agent_type} initialized successfully")
                return True
            else:
                self.state = AgentState.ERROR
                self.logger.error(f"Failed to initialize {self.agent_type}")
                return False
                
        except Exception as e:
            self.state = AgentState.ERROR
            self.logger.error(f"Initialization error for {self.agent_type}: {e}")
            return False
    
    @abstractmethod
    async def _initialize_agent(self) -> bool:
        """Agent-specific initialization logic."""
        pass
    
    async def start(self) -> None:
        """Start the agent message processing loop."""
        if self.state != AgentState.READY:
            raise RuntimeError(f"Agent {self.agent_type} not ready for start")
        
        self._running = True
        self.logger.info(f"Starting {self.agent_type}")
        
        # Start message processing loop
        asyncio.create_task(self._message_processing_loop())
    
    async def stop(self) -> None:
        """Stop the agent gracefully."""
        self.logger.info(f"Stopping {self.agent_type}")
        self._running = False
        
        # Wait for active tasks to complete (with timeout)
        if self.active_tasks:
            self.logger.info(f"Waiting for {len(self.active_tasks)} active tasks to complete")
            
            try:
                await asyncio.wait_for(
                    self._wait_for_tasks_completion(), 
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                self.logger.warning("Tasks did not complete within timeout, forcing shutdown")
        
        self.state = AgentState.SHUTDOWN
        self._shutdown_event.set()
        self.logger.info(f"{self.agent_type} stopped")
    
    async def _wait_for_tasks_completion(self) -> None:
        """Wait for all active tasks to complete."""
        while self.active_tasks:
            await asyncio.sleep(0.1)
    
    async def _message_processing_loop(self) -> None:
        """Main message processing loop."""
        while self._running:
            try:
                # Receive messages from transport
                messages = await self.message_bus.transport.receive_messages(
                    self.agent_type, timeout=1.0
                )
                
                for message in messages:
                    if not self._running:
                        break
                    
                    # Process message asynchronously
                    task_id = f"{message.message_id}_{datetime.utcnow().timestamp()}"
                    self.active_tasks.add(task_id)
                    
                    asyncio.create_task(
                        self._process_message_with_cleanup(message, task_id)
                    )
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.01)
                
            except Exception as e:
                self.logger.error(f"Error in message processing loop: {e}")
                await asyncio.sleep(1.0)  # Back off on error
    
    async def _process_message_with_cleanup(self, message: MCPMessage, task_id: str) -> None:
        """Process a message with proper cleanup."""
        try:
            await self._process_message(message)
        finally:
            self.active_tasks.discard(task_id)
    
    def _handle_incoming_message(self, message: MCPMessage) -> None:
        """Handle incoming message from subscription."""
        # This is called synchronously, so we just log it
        # The actual processing happens in the message loop
        self.logger.debug(f"Received message: {message.message_type} from {message.sender}")
    
    async def _process_message(self, message: MCPMessage) -> None:
        """Process a single message."""
        try:
            self.state = AgentState.PROCESSING
            self.metrics["messages_processed"] += 1
            self.metrics["last_activity"] = datetime.utcnow()
            
            self.logger.debug(
                f"Processing message {message.message_id} "
                f"({message.message_type}) from {message.sender}"
            )
            
            # Route message to appropriate handler
            response = await self._handle_message(message)
            
            if response:
                # Send response back
                await self.send_message(response)
            
            self.state = AgentState.READY
            
        except Exception as e:
            self.metrics["errors"] += 1
            self.state = AgentState.ERROR
            
            self.logger.error(f"Error processing message {message.message_id}: {e}")
            
            # Send error response
            error_response = create_error_message(
                message, 
                str(e), 
                error_code=f"{self.agent_type}_PROCESSING_ERROR"
            )
            await self.send_message(error_response)
            
            self.state = AgentState.READY
    
    @abstractmethod
    async def _handle_message(self, message: MCPMessage) -> Optional[MCPMessage]:
        """Handle a specific message type. Must be implemented by subclasses."""
        pass
    
    async def send_message(self, message: MCPMessage) -> bool:
        """Send a message through the message bus."""
        if not self.message_bus:
            self.logger.error("Message bus not initialized")
            return False
        
        try:
            message.sender = self.agent_type
            success = await self.message_bus.send_message(message)
            
            if success:
                self.metrics["messages_sent"] += 1
                self.logger.debug(
                    f"Sent message {message.message_id} "
                    f"({message.message_type}) to {message.receiver}"
                )
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            return False
    
    async def send_request_and_wait(
        self, 
        message: MCPMessage, 
        timeout: Optional[float] = None
    ) -> Optional[MCPMessage]:
        """Send a request message and wait for response."""
        
        if not self.message_bus:
            self.logger.error("Message bus not initialized")
            return None
        
        message.sender = self.agent_type
        return await self.message_bus.request_response(message, timeout)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics."""
        uptime = None
        if self.metrics["start_time"]:
            uptime = (datetime.utcnow() - self.metrics["start_time"]).total_seconds()
        
        return {
            **self.metrics,
            "agent_type": self.agent_type,
            "state": self.state,
            "active_tasks": len(self.active_tasks),
            "uptime_seconds": uptime
        }
    
    def is_healthy(self) -> bool:
        """Check if agent is healthy."""
        return (
            self.state in [AgentState.READY, AgentState.PROCESSING] and
            self.message_bus is not None and
            self._running
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health_status = {
            "healthy": self.is_healthy(),
            "state": self.state,
            "metrics": self.get_metrics(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Add any agent-specific health checks
        agent_health = await self._agent_health_check()
        if agent_health:
            health_status.update(agent_health)
        
        return health_status
    
    async def _agent_health_check(self) -> Optional[Dict[str, Any]]:
        """Agent-specific health check. Override in subclasses."""
        return None
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.agent_type}, state={self.state})"
