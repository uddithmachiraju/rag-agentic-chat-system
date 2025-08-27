# src/services/query_service.py

import time
from typing import Any, Dict, List, Optional, Tuple

from src.config.settings import get_settings
from src.config.logging import LoggerMixin

from src.vector_store.embeddings import GeminiEmbeddingService
from src.vector_store.chroma_store import ChromaVectorStore

from src.agents.base_agent import BaseAgent
from src.models.document import DocumentChunk 
from src.mcp.protocol import MCPMessage, AgentType, MessageType

class QueryServiceError(Exception):
    pass

class QueryService(LoggerMixin):
    def __init__(self,
                 vector_store: Optional[ChromaVectorStore] = None,
                 embedder: Optional[GeminiEmbeddingService] = None,
                 llm_agent: Optional[BaseAgent] = None):
        self.settings = get_settings()
        self.vector_store = vector_store or ChromaVectorStore()
        self.embedder = embedder or GeminiEmbeddingService()
        self.llm_agent = llm_agent   # e.g., your LLMResponseAgent instance

        self.initialized = False
        self.stats = {
            "queries": 0,
            "avg_latency_ms": 0.0,
            "last_latency_ms": 0.0,
        }

    async def initialize(self):
        # Initialize dependencies
        await self.vector_store.initialize()
        # embedder is ready after constructor, no async init in your snippet
        self.initialized = True
        self.logger.info("QueryService initialized")

    async def answer(self,
                     question: str,
                     k: int = 6,
                     rerank: bool = True,
                     min_score: float = 0.0,
                     user_id: Optional[str] = None,
                     filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.initialized:
            raise QueryServiceError("QueryService not initialized")

        t0 = time.time()
        try:
            # 1) Embed query
            q_emb = await self.embedder.generate_query_embedding(question)

            # 2) Vector search
            hits = await self.vector_store.search(embedding=q_emb, k=k, filters=filters or {})

            # hits expected format: List[{"id": str, "score": float, "metadata": {...}, "content": str}]
            # 3) Filter by min_score
            hits = [h for h in hits if h.get("score", 0.0) >= min_score]

            # 4) Optional rerank (placeholder: keep as-is or plug a reranker)
            if rerank:
                # Implement rerank if you have a reranker; otherwise pass.
                pass

            # 5) Build context
            context_blocks = []
            sources = []
            for h in hits:
                snippet = h.get("content", "")
                doc_id = h.get("metadata", {}).get("document_id")
                chunk_idx = h.get("metadata", {}).get("chunk_index")
                context_blocks.append(f"[{doc_id}#{chunk_idx}] {snippet}")
                sources.append({
                    "id": h.get("id"),
                    "document_id": doc_id,
                    "chunk_index": chunk_idx,
                    "score": h.get("score"),
                    "file_name": h.get("metadata", {}).get("file_name"),
                })
            context_text = "\n\n".join(context_blocks)

            # 6) Call LLM agent (Coordinator not strictly required here)
            if not self.llm_agent:
                raise QueryServiceError("LLM agent not configured in QueryService")

            prompt_payload = {
                "action": "llm_response",
                "question": question,
                "context": context_text,
                "instructions": (
                    "Use the provided context to answer concisely. "
                    "Cite sources inline using [doc#chunk] where appropriate. "
                    "If unsure, say you don't know."
                ),
            }
            req = MCPMessage(
                sender=AgentType.API,
                receiver=self.llm_agent.agent_type,
                message_type=MessageType.REQUEST,
                payload=prompt_payload
            )
            llm_msg = await self.llm_agent._handle_message(req)
            if not llm_msg or llm_msg.message_type != MessageType.RESPONSE:
                raise QueryServiceError("LLM agent returned invalid response")

            answer_text = llm_msg.payload.get("text") or llm_msg.payload.get("answer") or ""

            # 7) Telemetry
            latency_ms = (time.time() - t0) * 1000.0
            self._update_latency(latency_ms)

            return {
                "question": question,
                "answer": answer_text,
                "sources": sources,
                "hit_count": len(hits),
                "latency_ms": latency_ms,
            }

        except Exception as e:
            latency_ms = (time.time() - t0) * 1000.0
            self._update_latency(latency_ms)
            self.logger.exception("QueryService.answer failed")
            raise QueryServiceError(str(e))

    def _update_latency(self, latency_ms: float):
        self.stats["queries"] += 1
        self.stats["last_latency_ms"] = latency_ms
        if self.stats["queries"] == 1:
            self.stats["avg_latency_ms"] = latency_ms
        else:
            alpha = 0.15
            self.stats["avg_latency_ms"] = (
                alpha * latency_ms + (1 - alpha) * self.stats["avg_latency_ms"]
            )
