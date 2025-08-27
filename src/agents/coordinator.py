from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Callable, Awaitable, List
from datetime import datetime

from ..config.logging import LoggerMixin
from ..config.settings import get_settings

from ..mcp.protocol import (
    MCPMessage, MessageType, AgentType,
    create_error_message, create_response_message
)

from .base_agent import BaseAgent
from .ingestion import IngestionAgent
from .retrieval import RetrievalAgent
from .llm_response import LLMResponseAgent


@dataclass
class Route:
    action: str
    handler: Callable[[MCPMessage], Awaitable[MCPMessage]]
    description: str
    requires_agents: List[str] = None
    timeout_seconds: float = 30.0


class CoordinatorError(Exception):
    pass


class CoordinatorAgent(BaseAgent, LoggerMixin):
    """
    Orchestrates the multi-agent pipeline using proper MCP communication.
    Routes requests to appropriate agents and handles complex workflows.
    """

    def __init__(self):
        super().__init__(AgentType.COORDINATOR)
        self.settings = get_settings()

        # Initialize all downstream agents
        self.ingestion_agent = IngestionAgent()
        self.retrieval_agent = RetrievalAgent()
        self.llm_agent = LLMResponseAgent()

        # Agent registry for easier management
        self.agents = {
            'ingestion': self.ingestion_agent,
            'retrieval': self.retrieval_agent,
            'llm': self.llm_agent
        }

        # Registry of action routes
        self.routes: Dict[str, Route] = {}

        # Statistics
        self.stats = {
            "requests": 0,
            "errors": 0,
            "avg_latency_ms": 0.0,
            "last_latency_ms": 0.0,
            "agent_requests": {
                "ingestion": 0,
                "retrieval": 0,
                "llm": 0
            },
            "pipeline_requests": 0,  # Full RAG pipeline requests
            "start_time": datetime.utcnow(),
        }

        self._register_builtin_routes()

    def _register_builtin_routes(self):
        """Register all available routes with proper action names."""
        self.routes.update({
            # Document ingestion routes
            "process_document": Route(
                "process_document", 
                self._route_process_document,
                "Queue a document for ingestion and chunk-indexing",
                requires_agents=["ingestion"],
                timeout_seconds=120.0  # Longer timeout for document processing
            ),
            "get_processing_status": Route(
                "get_processing_status", 
                self._route_get_processing_status,
                "Get ingestion status for a document_id",
                requires_agents=["ingestion"]
            ),
            "cancel_processing": Route(
                "cancel_processing",
                self._route_cancel_processing,
                "Cancel document processing",
                requires_agents=["ingestion"]
            ),
            "get_ingestion_stats": Route(
                "get_ingestion_stats", 
                self._route_get_ingestion_stats,
                "Get ingestion agent statistics",
                requires_agents=["ingestion"]
            ),

            # Document retrieval routes
            "retrieve_documents": Route(
                "retrieve_documents", 
                self._route_retrieve_documents,
                "Semantic retrieval over vector store",
                requires_agents=["retrieval"]
            ),
            "add_documents": Route(
                "add_documents",
                self._route_add_documents,
                "Add documents to vector store",
                requires_agents=["retrieval"]
            ),
            "delete_documents": Route(
                "delete_documents",
                self._route_delete_documents,
                "Delete documents from vector store",
                requires_agents=["retrieval"]
            ),
            "get_retrieval_stats": Route(
                "get_retrieval_stats",
                self._route_get_retrieval_stats,
                "Get retrieval agent statistics",
                requires_agents=["retrieval"]
            ),

            # LLM response routes
            "generate_response": Route(
                "generate_response", 
                self._route_generate_response,
                "Generate LLM response from context",
                requires_agents=["llm"]
            ),
            "get_conversation": Route(
                "get_conversation",
                self._route_get_conversation,
                "Get conversation history",
                requires_agents=["llm"]
            ),
            "clear_conversation": Route(
                "clear_conversation",
                self._route_clear_conversation,
                "Clear conversation history",
                requires_agents=["llm"]
            ),
            "get_llm_stats": Route(
                "get_llm_stats",
                self._route_get_llm_stats,
                "Get LLM agent statistics",
                requires_agents=["llm"]
            ),

            # Complex workflows
            "answer_question": Route(
                "answer_question", 
                self._route_answer_question,
                "Full RAG pipeline: retrieve documents and generate answer",
                requires_agents=["retrieval", "llm"],
                timeout_seconds=60.0
            ),
            "process_and_query": Route(
                "process_and_query",
                self._route_process_and_query,
                "Process document then immediately query it",
                requires_agents=["ingestion", "retrieval", "llm"],
                timeout_seconds=180.0
            ),

            # System routes
            "health": Route(
                "health", 
                self._route_health, 
                "Coordinator and all agents health check"
            ),
            "get_stats": Route(
                "get_stats",
                self._route_get_coordinator_stats,
                "Get coordinator statistics"
            ),
            "list_routes": Route(
                "list_routes",
                self._route_list_routes,
                "List all available routes"
            ),
        })

    async def _initialize_agent(self) -> bool:
        """Initialize all downstream agents with proper error handling."""
        self.logger.info("Coordinator initializing downstream agents")
        
        initialization_results = {}
        
        for name, agent in self.agents.items():
            try:
                self.logger.info(f"Initializing {name} agent...")
                success = await agent.initialize()
                initialization_results[name] = success
                
                if success:
                    self.logger.info(f"{name} agent initialized successfully")
                else:
                    self.logger.error(f"Failed to initialize {name} agent")
                    
            except Exception as e:
                self.logger.error(f"Exception initializing {name} agent: {e}")
                initialization_results[name] = False

        # Check if critical agents initialized
        critical_agents = ['ingestion']  # At minimum, we need ingestion
        failed_critical = [name for name in critical_agents if not initialization_results.get(name, False)]
        
        if failed_critical:
            self.logger.error(f"Critical agents failed to initialize: {failed_critical}")
            return False

        # Warn about optional agents that failed
        optional_agents = ['retrieval', 'llm']
        failed_optional = [name for name in optional_agents if not initialization_results.get(name, False)]
        
        if failed_optional:
            self.logger.warning(f"Optional agents failed to initialize: {failed_optional}")

        self.logger.info("Coordinator initialization completed")
        return True

    async def start_all_agents(self) -> None:
        """Start all downstream agents."""
        self.logger.info("Starting all downstream agents")
        
        for name, agent in self.agents.items():
            try:
                if hasattr(agent, 'start'):
                    await agent.start()
                    self.logger.info(f"Started {name} agent")
            except Exception as e:
                self.logger.error(f"Failed to start {name} agent: {e}")

    async def stop_all_agents(self) -> None:
        """Stop all downstream agents gracefully."""
        self.logger.info("Stopping all downstream agents")
        
        for name, agent in self.agents.items():
            try:
                if hasattr(agent, 'stop'):
                    await agent.stop()
                    self.logger.info(f"Stopped {name} agent")
            except Exception as e:
                self.logger.error(f"Failed to stop {name} agent: {e}")

    async def _handle_message(self, message: MCPMessage) -> Optional[MCPMessage]:
        """Central MCP message router with comprehensive error handling."""
        t0 = time.time()
        
        try:
            self.stats["requests"] += 1
            
            if message.message_type != MessageType.REQUEST:
                return create_error_message(message, "Unsupported message type")

            action = (message.payload or {}).get("action")
            if not action:
                return create_error_message(message, "Missing 'action' in payload")

            route = self.routes.get(action)
            if not route:
                available_actions = list(self.routes.keys())
                return create_error_message(
                    message, 
                    f"Unknown action: {action}. Available actions: {available_actions}"
                )

            # Check if required agents are available
            if route.requires_agents:
                missing_agents = []
                for required_agent in route.requires_agents:
                    agent = self.agents.get(required_agent)
                    if not agent or not agent.is_healthy():
                        missing_agents.append(required_agent)
                
                if missing_agents:
                    return create_error_message(
                        message, 
                        f"Required agents not available: {missing_agents}"
                    )

            # Execute route handler with timeout
            self.logger.debug(f"Routing {action} to handler")
            
            try:
                response = await asyncio.wait_for(
                    route.handler(message), 
                    timeout=route.timeout_seconds
                )
            except asyncio.TimeoutError:
                return create_error_message(
                    message, 
                    f"Request timed out after {route.timeout_seconds} seconds"
                )

            # Update latency stats
            latency = (time.time() - t0) * 1000.0
            self._update_latency(latency)
            
            return response

        except Exception as e:
            self.stats["errors"] += 1
            self.logger.exception(f"Error handling message for action: {message.payload.get('action', 'unknown')}")
            return create_error_message(message, f"Internal error: {str(e)}")

    # ==================== DOCUMENT INGESTION ROUTES ====================

    async def _route_process_document(self, message: MCPMessage) -> MCPMessage:
        """Route document processing to ingestion agent."""
        self.stats["agent_requests"]["ingestion"] += 1
        
        payload = message.payload or {}
        if "file_path" not in payload:
            return create_error_message(message, "Missing required field: file_path")

        # Forward to ingestion agent via MCP
        ingestion_message = MCPMessage(
            sender=self.agent_type,
            receiver=AgentType.INGESTION,
            message_type=MessageType.REQUEST,
            payload=payload
        )

        try:
            response = await self._forward_to_agent('ingestion', ingestion_message)
            return response
        except Exception as e:
            return create_error_message(message, f"Document processing failed: {str(e)}")

    async def _route_get_processing_status(self, message: MCPMessage) -> MCPMessage:
        """Route processing status request to ingestion agent."""
        payload = message.payload or {}
        if "document_id" not in payload:
            return create_error_message(message, "Missing required field: document_id")

        return await self._forward_to_agent('ingestion', message)

    async def _route_cancel_processing(self, message: MCPMessage) -> MCPMessage:
        """Route processing cancellation to ingestion agent."""
        payload = message.payload or {}
        if "document_id" not in payload:
            return create_error_message(message, "Missing required field: document_id")

        return await self._forward_to_agent('ingestion', message)

    async def _route_get_ingestion_stats(self, message: MCPMessage) -> MCPMessage:
        """Get ingestion agent statistics."""
        try:
            if not self.ingestion_agent.is_healthy():
                return create_error_message(message, "Ingestion agent not available")
            
            stats = self.ingestion_agent.get_processing_summary()
            return create_response_message(message, stats)
        except Exception as e:
            return create_error_message(message, f"Failed to get ingestion stats: {str(e)}")

    # ==================== DOCUMENT RETRIEVAL ROUTES ====================

    async def _route_retrieve_documents(self, message: MCPMessage) -> MCPMessage:
        """Route document retrieval to retrieval agent."""
        self.stats["agent_requests"]["retrieval"] += 1
        
        payload = message.payload or {}
        
        # Validate required fields - handle both 'query' and 'question'
        query = payload.get("query") or payload.get("question")
        if not query:
            return create_error_message(message, "Missing required field: 'query' or 'question'")

        # Normalize payload format for retrieval agent
        normalized_payload = {
            "action": "retrieve_documents",
            "query": query,
            "filters": self._build_retrieval_filters(payload)
        }
        
        # Add optional fields
        optional_fields = ["user_id", "session_id", "conversation_context"]
        for field in optional_fields:
            if field in payload:
                normalized_payload[field] = payload[field]

        retrieval_message = MCPMessage(
            sender=self.agent_type,
            receiver=AgentType.RETRIEVAL,
            message_type=MessageType.REQUEST,
            payload=normalized_payload
        )

        return await self._forward_to_agent('retrieval', retrieval_message)

    async def _route_add_documents(self, message: MCPMessage) -> MCPMessage:
        """Route add documents to retrieval agent."""
        return await self._forward_to_agent('retrieval', message)

    async def _route_delete_documents(self, message: MCPMessage) -> MCPMessage:
        """Route delete documents to retrieval agent."""
        return await self._forward_to_agent('retrieval', message)

    async def _route_get_retrieval_stats(self, message: MCPMessage) -> MCPMessage:
        """Get retrieval agent statistics."""
        return await self._forward_to_agent('retrieval', message)

    # ==================== LLM RESPONSE ROUTES ====================

    async def _route_generate_response(self, message: MCPMessage) -> MCPMessage:
        """Route LLM response generation to LLM agent."""
        self.stats["agent_requests"]["llm"] += 1
        return await self._forward_to_agent('llm', message)

    async def _route_get_conversation(self, message: MCPMessage) -> MCPMessage:
        """Route conversation retrieval to LLM agent."""
        return await self._forward_to_agent('llm', message)

    async def _route_clear_conversation(self, message: MCPMessage) -> MCPMessage:
        """Route conversation clearing to LLM agent."""
        return await self._forward_to_agent('llm', message)

    async def _route_get_llm_stats(self, message: MCPMessage) -> MCPMessage:
        """Get LLM agent statistics."""
        return await self._forward_to_agent('llm', message)

    # ==================== COMPLEX WORKFLOW ROUTES ====================

    async def _route_answer_question(self, message: MCPMessage) -> MCPMessage:
        """
        Full RAG pipeline: retrieve relevant documents and generate answer.
        Expected payload: question, k (optional), filters (optional), conversation_id (optional)
        """
        self.stats["pipeline_requests"] += 1
        
        payload = message.payload or {}
        question = payload.get("question") or payload.get("query")
        
        if not question:
            return create_error_message(message, "Missing required field: 'question'")

        try:
            # Step 1: Retrieve relevant documents
            self.logger.debug(f"RAG Step 1: Retrieving documents for question: {question[:100]}...")
            
            retrieval_payload = {
                "action": "retrieve_documents",
                "query": question,
                "filters": self._build_retrieval_filters(payload)
            }
            
            # Add conversation context if provided
            if "conversation_id" in payload:
                retrieval_payload["conversation_id"] = payload["conversation_id"]

            retrieve_message = MCPMessage(
                sender=self.agent_type,
                receiver=AgentType.RETRIEVAL,
                message_type=MessageType.REQUEST,
                payload=retrieval_payload
            )

            retrieve_response = await self._forward_to_agent('retrieval', retrieve_message)
            
            if not retrieve_response or retrieve_response.message_type == MessageType.ERROR:
                error_msg = "Retrieval failed"
                if retrieve_response and retrieve_response.payload:
                    error_msg = retrieve_response.payload.get("error", error_msg)
                return create_error_message(message, f"Document retrieval failed: {error_msg}")

            # Extract retrieval results
            retrieval_results = retrieve_response.payload.get("results", [])
            
            if not retrieval_results:
                # No documents found - return empty answer
                return create_response_message(message, {
                    "question": question,
                    "answer": "I couldn't find any relevant documents to answer your question. Please try rephrasing your question or upload more relevant documents.",
                    "sources": [],
                    "confidence_level": "uncertain",
                    "retrieval_results": []
                })

            # Step 2: Generate response using LLM
            self.logger.debug(f"RAG Step 2: Generating response using {len(retrieval_results)} retrieved documents")
            
            llm_payload = {
                "action": "generate_response",
                "query": question,
                "retrieval_results": {
                    "results": retrieval_results,
                    "total_results": len(retrieval_results)
                }
            }
            
            # Pass through optional parameters
            optional_params = ["conversation_id", "user_id"]
            for param in optional_params:
                if param in payload:
                    llm_payload[param] = payload[param]

            llm_message = MCPMessage(
                sender=self.agent_type,
                receiver=AgentType.LLM_RESPONSE,
                message_type=MessageType.REQUEST,
                payload=llm_payload
            )

            llm_response = await self._forward_to_agent('llm', llm_message)
            
            if not llm_response or llm_response.message_type == MessageType.ERROR:
                error_msg = "LLM response generation failed"
                if llm_response and llm_response.payload:
                    error_msg = llm_response.payload.get("error", error_msg)
                return create_error_message(message, f"Response generation failed: {error_msg}")

            # Step 3: Combine results and return comprehensive response
            final_response = {
                "question": question,
                "answer": llm_response.payload.get("answer", ""),
                "response_type": llm_response.payload.get("response_type", "direct_answer"),
                "confidence_level": llm_response.payload.get("confidence_level", "medium"),
                "sources": llm_response.payload.get("sources", []),
                "follow_up_questions": llm_response.payload.get("follow_up_questions", []),
                "retrieval_results": retrieval_results,
                "metrics": {
                    "documents_retrieved": len(retrieval_results),
                    "retrieval_time": retrieve_response.payload.get("retrieval_time", 0),
                    "generation_time": llm_response.payload.get("metrics", {}).get("generation_time", 0),
                    "total_processing_time": retrieve_response.payload.get("processing_time", 0) + 
                                           llm_response.payload.get("metrics", {}).get("total_response_time", 0)
                }
            }

            self.logger.info(
                f"RAG pipeline completed: {len(retrieval_results)} docs retrieved, "
                f"answer generated ({len(final_response['answer'])} chars)"
            )

            return create_response_message(message, final_response)

        except Exception as e:
            self.logger.error(f"RAG pipeline failed: {e}")
            return create_error_message(message, f"RAG pipeline error: {str(e)}")

    async def _route_process_and_query(self, message: MCPMessage) -> MCPMessage:
        """
        Process a document and then immediately query it.
        Expected payload: file_path, question, processing_options (optional)
        """
        payload = message.payload or {}
        
        required_fields = ["file_path", "question"]
        missing_fields = [field for field in required_fields if field not in payload]
        if missing_fields:
            return create_error_message(message, f"Missing required fields: {missing_fields}")

        try:
            # Step 1: Process document
            self.logger.info(f"Process-and-query Step 1: Processing document {payload['file_path']}")
            
            process_response = await self._route_process_document(message)
            
            if not process_response or process_response.message_type == MessageType.ERROR:
                return process_response  # Return the error as-is

            document_id = process_response.payload.get("document_id")
            
            # Step 2: Wait for processing to complete
            self.logger.info(f"Process-and-query Step 2: Waiting for document {document_id} to be processed")
            
            max_wait_time = 120  # 2 minutes
            wait_interval = 5    # 5 seconds
            waited_time = 0
            
            while waited_time < max_wait_time:
                status_message = MCPMessage(
                    sender=self.agent_type,
                    receiver=AgentType.INGESTION,
                    message_type=MessageType.REQUEST,
                    payload={"action": "get_processing_status", "document_id": document_id}
                )
                
                status_response = await self._forward_to_agent('ingestion', status_message)
                
                if status_response and status_response.payload:
                    status = status_response.payload.get("status")
                    
                    if status == "indexed":
                        break
                    elif status == "failed":
                        return create_error_message(message, "Document processing failed")
                
                await asyncio.sleep(wait_interval)
                waited_time += wait_interval
            else:
                return create_error_message(message, "Document processing timed out")

            # Step 3: Query the processed document
            self.logger.info(f"Process-and-query Step 3: Querying processed document")
            
            query_payload = {
                "question": payload["question"],
                "filters": {"document_ids": [document_id]}  # Only query the newly processed document
            }
            
            query_message = MCPMessage(
                sender=self.agent_type,
                receiver=self.agent_type,  # Route back to self for RAG pipeline
                message_type=MessageType.REQUEST,
                payload={**query_payload, "action": "answer_question"}
            )

            return await self._route_answer_question(query_message)

        except Exception as e:
            self.logger.error(f"Process-and-query failed: {e}")
            return create_error_message(message, f"Process-and-query error: {str(e)}")

    # ==================== SYSTEM ROUTES ====================

    async def _route_health(self, message: MCPMessage) -> MCPMessage:
        """Comprehensive health check for coordinator and all agents."""
        health = {
            "coordinator": {
                "healthy": self.is_healthy(),
                "state": self.state,
                "uptime_seconds": (datetime.utcnow() - self.stats["start_time"]).total_seconds(),
                "stats": self.stats
            },
            "agents": {}
        }

        for name, agent in self.agents.items():
            try:
                if hasattr(agent, 'health_check'):
                    agent_health = await agent.health_check()
                    health["agents"][name] = agent_health
                else:
                    health["agents"][name] = {
                        "healthy": agent.is_healthy() if hasattr(agent, 'is_healthy') else False,
                        "state": getattr(agent, 'state', 'unknown')
                    }
            except Exception as e:
                health["agents"][name] = {"healthy": False, "error": str(e)}

        # Overall system health
        agent_healths = [agent.get("healthy", False) for agent in health["agents"].values()]
        health["system"] = {
            "healthy": health["coordinator"]["healthy"] and any(agent_healths),
            "total_agents": len(self.agents),
            "healthy_agents": sum(1 for h in agent_healths if h),
            "available_routes": len(self.routes)
        }

        return create_response_message(message, health)

    async def _route_get_coordinator_stats(self, message: MCPMessage) -> MCPMessage:
        """Get detailed coordinator statistics."""
        stats = self.stats.copy()
        stats.update({
            "uptime_seconds": (datetime.utcnow() - stats["start_time"]).total_seconds(),
            "available_routes": len(self.routes),
            "agent_status": {
                name: agent.is_healthy() if hasattr(agent, 'is_healthy') else False
                for name, agent in self.agents.items()
            },
            "routes": {
                action: {
                    "description": route.description,
                    "requires_agents": route.requires_agents or [],
                    "timeout_seconds": route.timeout_seconds
                }
                for action, route in self.routes.items()
            }
        })
        return create_response_message(message, stats)

    async def _route_list_routes(self, message: MCPMessage) -> MCPMessage:
        """List all available routes and their descriptions."""
        routes_info = {
            action: {
                "description": route.description,
                "requires_agents": route.requires_agents or [],
                "timeout_seconds": route.timeout_seconds
            }
            for action, route in self.routes.items()
        }
        return create_response_message(message, {"routes": routes_info})

    # ==================== HELPER METHODS ====================

    async def _forward_to_agent(self, agent_name: str, message: MCPMessage) -> MCPMessage:
        """Forward message to specified agent with error handling."""
        agent = self.agents.get(agent_name)
        
        if not agent:
            raise CoordinatorError(f"Agent '{agent_name}' not found")
        
        if not agent.is_healthy():
            raise CoordinatorError(f"Agent '{agent_name}' is not healthy")

        try:
            # Use the agent's message handler directly since we're in the same process
            # In a distributed system, this would go through the message bus
            response = await agent._handle_message(message)
            
            if not response:
                raise CoordinatorError(f"No response from {agent_name} agent")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error forwarding message to {agent_name}: {e}")
            raise CoordinatorError(f"Communication with {agent_name} agent failed: {str(e)}")

    def _build_retrieval_filters(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Build retrieval filters from request payload."""
        filters = {
            "max_results": payload.get("k", payload.get("max_results", 6)),
            "similarity_threshold": payload.get("threshold", payload.get("similarity_threshold", 0.7))
        }
        
        # Add specific filters
        if "document_ids" in payload:
            filters["document_ids"] = payload["document_ids"]
        
        if "document_formats" in payload:
            filters["document_formats"] = payload["document_formats"]
        
        if "chunk_types" in payload:
            filters["chunk_types"] = payload["chunk_types"]
        
        if "date_range" in payload:
            filters["date_range"] = payload["date_range"]
        
        if "metadata_filters" in payload:
            filters["metadata_filters"] = payload["metadata_filters"]
        
        # Include any additional filters from the payload
        if "filters" in payload and isinstance(payload["filters"], dict):
            filters.update(payload["filters"])
        
        return filters

    def _update_latency(self, latency_ms: float):
        """Update latency statistics."""
        self.stats["last_latency_ms"] = latency_ms
        
        if self.stats["requests"] == 1:
            self.stats["avg_latency_ms"] = latency_ms
        else:
            # Exponential moving average
            alpha = 0.15
            self.stats["avg_latency_ms"] = (
                alpha * latency_ms + (1 - alpha) * self.stats["avg_latency_ms"]
            )

    async def _agent_health_check(self) -> Optional[Dict[str, Any]]:
        """Coordinator-specific health check."""
        return {
            "routes_available": len(self.routes),
            "agents_managed": len(self.agents),
            "agent_health": {
                name: agent.is_healthy() if hasattr(agent, 'is_healthy') else False
                for name, agent in self.agents.items()
            },
            "request_stats": self.stats
        }

    def get_route_info(self, action: str = None) -> Dict[str, Any]:
        """Get information about available routes."""
        if action:
            route = self.routes.get(action)
            if not route:
                return {"error": f"Route '{action}' not found"}
            
            return {
                "action": action,
                "description": route.description,
                "requires_agents": route.requires_agents or [],
                "timeout_seconds": route.timeout_seconds
            }
        
        return {
            "total_routes": len(self.routes),
            "routes": {
                action: {
                    "description": route.description,
                    "requires_agents": route.requires_agents or [],
                    "timeout_seconds": route.timeout_seconds
                }
                for action, route in self.routes.items()
            }
        }