from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from src.config.logging import setup_logging, get_logger
from .schemas import SearchResponse, SearchRequest, SearchResult

from src.agents.base_agent import AgentType
from src.models.query import Query, QueryFilters
from src.mcp.protocol import create_mcp_message, MessageType
from src.agents.retrieval import RetrievalAgent, RetrievalError

# setup logging
setup_logging()
logger = get_logger("retrieval-api")

router = APIRouter(prefix = "/retrieval")

retrieval_agent: Optional[RetrievalAgent] = None 

async def get_retrieval_agent() -> RetrievalAgent:
    global retrieval_agent

    if retrieval_agent is None:
        retrieval_agent = RetrievalAgent()
        if not await retrieval_agent.initialize():
            raise HTTPException(
                status_code = 503, 
                detail = "Failed to initilize retrieval agent"
            )
    return retrieval_agent 

# endpoints 
@router.post("/search", response_model = SearchResponse)
async def search_documents(request: SearchRequest, agent: RetrievalAgent = Depends(get_retrieval_agent)):
    # Create a Query
    try:
        query = Query(
            text = request.query, 
            query_type = request.query_type, 
            retrieval_strategy = request.retrieval_strategy, 
            user_id = request.user_id, 
            session_id = request.session_id
        )

        # Setup FIlters
        query.filters = QueryFilters(
            max_results = request.max_results, 
            similarity_threshold = request.similarity_threshold, 
            document_ids = request.document_ids, 
            chunk_types = request.chunk_types
        )

        # Create a MCP Message
        message = create_mcp_message(
            sender = AgentType.COORDINATOR, 
            receiver = AgentType.RETRIEVAL, 
            message_type = MessageType.REQUEST, 
            payload = {
                "action": "retrieve_documents",
                "query": request.query,
                "query_type": request.query_type.value,
                "retrieval_strategy": request.retrieval_strategy.value,
                "filters": query.filters.dict(),
                "user_id": request.user_id,
                "session_id": request.session_id,
            }
        )

        response_message = await agent._handle_message(message)
        if response_message is not None and response_message.message_type == MessageType.ERROR:
            raise HTTPException(status_code = 500, detail = response_message.payload.get("error", "search failed"))
        
        payload = response_message.payload
        return SearchResponse(
                query_id = payload["query_id"],
                results = payload["results"],
                total_results = payload["total_results"],
                retrieval_time = payload["retrieval_time"],
                processing_time = payload.get("processing_time"),
                strategy_used = payload["strategy_used"],
                cached = payload.get("cached", False),
            )
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))