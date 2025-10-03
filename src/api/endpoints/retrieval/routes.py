
from fastapi import Body
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from src.config.logging import setup_logging, get_logger
from .schemas import SearchResponse, SearchRequest, SearchResult
from .schemas import AddDocumentsRequest
from fastapi import Query as FastAPIQuery
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
@router.post("/search", response_model=SearchResponse)
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
    

@router.post("/documents/add/")
async def add_documents(request: AddDocumentsRequest, agent: RetrievalAgent = Depends(get_retrieval_agent)):
    """Add document chunks to the vector store."""
    try:
        message = create_mcp_message(
            sender=AgentType.COORDINATOR,
            receiver=AgentType.RETRIEVAL,
            message_type=MessageType.REQUEST,
            payload={
                "action": "add_documents",
                "chunks": [chunk.dict() for chunk in request.chunks],
            }
        )
        response_message = await agent._handle_message(message)
        if response_message is not None and response_message.message_type == MessageType.ERROR:
            raise HTTPException(status_code=500, detail=response_message.payload.get("error", "Failed to add documents"))
        return response_message.payload
    except Exception as e:
        logger.error(f"Add documents failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add document: {str(e)}")


# Delete documents endpoint
@router.delete("/documents/delete/")
async def delete_document(document_id: str = FastAPIQuery(..., description="Document ID to delete"), agent: RetrievalAgent = Depends(get_retrieval_agent)):
    """Delete a document from the vector store by document_id."""
    try:
        message = create_mcp_message(
            sender=AgentType.COORDINATOR,
            receiver=AgentType.RETRIEVAL,
            message_type=MessageType.REQUEST,
            payload={
                "action": "delete_documents",
                "document_id": document_id,
            }
        )
        response_message = await agent._handle_message(message)
        if response_message is not None and response_message.message_type == MessageType.ERROR:
            raise HTTPException(status_code=500, detail=response_message.payload.get("error", "Failed to delete document"))
        return response_message.payload
    except Exception as e:
        logger.error(f"Delete document failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


# Stats endpoint
@router.get("/stats/")
async def get_stats(agent: RetrievalAgent = Depends(get_retrieval_agent)):
    """Get retrieval agent statistics and vector store stats."""
    try:
        message = create_mcp_message(
            sender=AgentType.COORDINATOR,
            receiver=AgentType.RETRIEVAL,
            message_type=MessageType.REQUEST,
            payload={"action": "get_stats"}
        )
        response_message = await agent._handle_message(message)
        if response_message is not None and response_message.message_type == MessageType.ERROR:
            raise HTTPException(status_code=500, detail=response_message.payload.get("error", "Failed to get stats"))
        return response_message.payload
    except Exception as e:
        logger.error(f"Get stats failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


# Health check endpoint
@router.get("/health/")
async def health(agent: RetrievalAgent = Depends(get_retrieval_agent)):
    """Health check for the retrieval agent and its dependencies."""
    try:
        health_info = await agent._agent_health_check()
        return health_info
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")
    
# List all document IDs
@router.get("/documents/list/")
async def list_documents(agent: RetrievalAgent = Depends(get_retrieval_agent)):
    try:
        stats = await agent.vector_store.get_collection_stats()
        # Try to extract document_ids from stats
        document_ids = list(stats.get("document_ids")) if stats.get("document_ids") else None
        if document_ids is None:
            # fallback: try to extract from unique_documents (may be int or set)
            unique_docs = stats.get("unique_documents")
            if isinstance(unique_docs, (list, set)):
                document_ids = list(unique_docs)
            else:
                document_ids = []
        return {"document_ids": document_ids}
    except Exception as e:
        logger.error(f"List documents failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

# Get all chunks for a document
@router.get("/documents/{document_id}/chunks/")
async def get_document_chunks(document_id: str, agent: RetrievalAgent = Depends(get_retrieval_agent)):
    try:
        chunks = await agent.vector_store.get_chunks_by_document(document_id)
        return {"chunks": [chunk.dict() for chunk in chunks]}
    except Exception as e:
        logger.error(f"Get document chunks failed: {e}")
        raise HTTPException(status_code=500, detail=f"Get document chunks failed: {str(e)}")

# Get a chunk by chunk_id
@router.get("/chunks/{chunk_id}/")
async def get_chunk_by_id(chunk_id: str, agent: RetrievalAgent = Depends(get_retrieval_agent)):
    try:
        chunk = await agent.vector_store.get_chunk_by_id(chunk_id)
        if chunk is None:
            raise HTTPException(status_code=404, detail="Chunk not found")
        return chunk.dict()
    except Exception as e:
        logger.error(f"Get chunk by id failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get chunk: {str(e)}")

@router.post("/chunk/preview/")
async def chunk_preview(
    text: str = Body(..., embed=True, description="Text to chunk"),
    chunk_size: int = Body(200, embed=True),
    overlap: int = Body(30, embed=True),
    strategy: str = Body("paragraph", embed=True),
    agent: RetrievalAgent = Depends(get_retrieval_agent)
):
    """Preview chunking for a given text using semantic strategies."""
    try:
        return agent.semantic_chunk_text(text, chunk_size, overlap, strategy)
    except Exception as e:
        logger.error(f"Chunk preview failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chunk preview failed: {str(e)}")

# List chunking strategies
@router.get("/chunk/strategies/")
async def chunk_strategies(agent: RetrievalAgent = Depends(get_retrieval_agent)):
    return {"strategies": agent.available_chunking_strategies()}

# List all indexed documents
@router.get("/documents/list/")
async def list_documents(agent: RetrievalAgent = Depends(get_retrieval_agent)):
    try:
        return {"documents": await agent.list_documents()}
    except Exception as e:
        logger.error(f"List documents failed: {e}")
        raise HTTPException(status_code=500, detail=f"List documents failed: {str(e)}")

# List all chunks for a document
@router.get("/documents/{document_id}/chunks/")
async def get_document_chunks(document_id: str, agent: RetrievalAgent = Depends(get_retrieval_agent)):
    try:
        return {"chunks": await agent.get_document_chunks(document_id)}
    except Exception as e:
        logger.error(f"Get document chunks failed: {e}")
        raise HTTPException(status_code=500, detail=f"Get document chunks failed: {str(e)}")