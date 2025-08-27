"""Advanced retrieval agent with intelligent search strategies and query processing."""

import time
from typing import Dict, Any, List, Optional
import re

from ..config import get_settings
from ..models.document import DocumentChunk
from ..models.query import (
    Query, QueryFilters, QueryType, RetrievalStrategy, 
    RetrievalResults, SearchResult, ConversationContext
)
from ..mcp.protocol import (
    MCPMessage, MessageType, AgentType, create_response_message, 
    create_error_message
)
from ..vector_store import ChromaVectorStore, GeminiEmbeddingService
from .base_agent import BaseAgent


class RetrievalError(Exception):
    """Custom exception for retrieval operations."""
    pass


class QueryProcessor:
    """Processes and expands user queries for better retrieval."""
    
    def __init__(self):
        self.stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'would', 'could', 'should', 'may',
            'might', 'can', 'do', 'does', 'did', 'have', 'had', 'having'
        }
        
        # Query type patterns
        self.question_patterns = {
            'what': r'\bwhat\b',
            'how': r'\bhow\b',
            'when': r'\bwhen\b',  
            'where': r'\bwhere\b',
            'who': r'\bwho\b',
            'why': r'\bwhy\b',
            'which': r'\bwhich\b'
        }
        
        # Intent patterns
        self.intent_patterns = {
            'definition': r'\b(what is|define|definition of|meaning of)\b',
            'comparison': r'\b(compare|comparison|difference|versus|vs)\b',
            'procedure': r'\b(how to|step|process|procedure|instructions)\b',
            'list': r'\b(list|enumerate|all|every)\b',
            'summary': r'\b(summary|summarize|overview)\b',
            'example': r'\b(example|sample|instance)\b'
        }
    
    def process_query(self, query_text: str, context: Optional[ConversationContext] = None) -> Dict[str, Any]:
        """Process and analyze a user query."""
        
        analysis = {
            'original_query': query_text,
            'processed_query': self._clean_query(query_text),
            'query_type': self._classify_query_type(query_text),
            'intent': self._detect_intent(query_text),
            'key_terms': self._extract_key_terms(query_text),
            'expanded_terms': [],
            'context_enhanced': False
        }
        
        # Apply context if available
        if context:
            analysis.update(self._apply_context(analysis, context))
        
        # Generate query expansion
        analysis['expanded_terms'] = self._expand_query_terms(analysis['key_terms'])
        
        return analysis
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize query text."""
        
        # Convert to lowercase
        cleaned = query.lower().strip()
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove leading question words if they don't add semantic value
        cleaned = re.sub(r'^(please\s+)?(can you\s+)?(tell me\s+)?', '', cleaned)
        
        return cleaned.strip()
    
    def _classify_query_type(self, query: str) -> QueryType:
        """Classify the type of query."""
        
        query_lower = query.lower()
        
        # Check for question patterns
        for question_type, pattern in self.question_patterns.items():
            if re.search(pattern, query_lower):
                return QueryType.SEMANTIC
        
        # Check if it's a simple keyword query
        if len(query.split()) <= 3 and not any(c in query for c in '?!.'):
            return QueryType.SIMPLE
        
        # Default to semantic
        return QueryType.SEMANTIC
    
    def _detect_intent(self, query: str) -> Optional[str]:
        """Detect user intent from query."""
        
        query_lower = query.lower()
        
        for intent, pattern in self.intent_patterns.items():
            if re.search(pattern, query_lower):
                return intent
        
        return None
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from query."""
        
        # Simple tokenization
        words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        
        # Remove stopwords and short words
        key_terms = [
            word for word in words 
            if word not in self.stopwords and len(word) > 2
        ]
        
        return key_terms
    
    def _expand_query_terms(self, key_terms: List[str]) -> List[str]:
        """Expand query terms with synonyms and related terms."""
        
        # Simple term expansion (in production, this could use WordNet or other resources)
        expansion_rules = {
            'cost': ['price', 'expense', 'budget'],
            'revenue': ['income', 'earnings', 'sales'],
            'process': ['procedure', 'method', 'workflow'],
            'analysis': ['evaluation', 'assessment', 'review'],
            'performance': ['results', 'metrics', 'kpi', 'results'],
            'strategy': ['plan', 'approach', 'methodology'],
            'customer': ['client', 'user', 'consumer'],
            'product': ['service', 'solution', 'offering']
        }
        
        expanded = []
        for term in key_terms:
            if term in expansion_rules:
                expanded.extend(expansion_rules[term])
        
        return expanded
    
    def _apply_context(self, analysis: Dict[str, Any], context: ConversationContext) -> Dict[str, Any]:
        """Apply conversational context to enhance query."""
        
        context_updates = {'context_enhanced': True}
        
        # Get recent context
        recent_context = context.get_recent_context(2)
        
        if recent_context['queries']:
            # Look for pronouns or references that need context
            pronouns = ['it', 'this', 'that', 'they', 'them', 'these', 'those']
            
            query_lower = analysis['processed_query']
            has_pronouns = any(pronoun in query_lower.split() for pronoun in pronouns)
            
            if has_pronouns:
                # Extract key terms from previous queries
                prev_terms = []
                for prev_query in recent_context['queries']:
                    prev_terms.extend(self._extract_key_terms(prev_query))
                
                # Add relevant previous terms
                context_updates['context_terms'] = list(set(prev_terms))
                
                # Enhance processed query
                if context_updates['context_terms']:
                    enhanced_query = f"{analysis['processed_query']} {' '.join(context_updates['context_terms'][:3])}"
                    context_updates['processed_query'] = enhanced_query
        
        return context_updates


class RetrievalStrategy:
    """Implements various retrieval strategies."""
    
    def __init__(self, vector_store: ChromaVectorStore, embedding_service: GeminiEmbeddingService):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
    
    async def similarity_search(
        self, 
        query: Query, 
        query_analysis: Dict[str, Any]
    ) -> RetrievalResults:
        """Perform basic similarity search."""
        
        try:
            # Generate query embedding
            query_text = query_analysis.get('processed_query', query.text)
            query_embedding = await self.embedding_service.generate_query_embedding(query_text)
            
            # Prepare filters
            filters = self._prepare_filters(query.filters)
            
            # Perform search
            results = await self.vector_store.search(
                query_embedding=query_embedding,
                max_results=query.filters.max_results,
                similarity_threshold=query.filters.similarity_threshold,
                filters=filters
            )
            
            results.query_id = query.query_id
            results.retrieval_strategy = RetrievalStrategy.SIMILARITY
            
            return results
            
        except Exception as e:
            raise RetrievalError(f"Similarity search failed: {str(e)}")
    
    async def mmr_search(
        self, 
        query: Query, 
        query_analysis: Dict[str, Any]
    ) -> RetrievalResults:
        """Perform Maximal Marginal Relevance search to reduce redundancy."""
        
        try:
            # First, get more results than needed
            initial_results_count = min(query.filters.max_results * 3, 50)
            
            # Generate query embedding
            query_text = query_analysis.get('processed_query', query.text)
            query_embedding = await self.embedding_service.generate_query_embedding(query_text)
            
            # Prepare filters
            filters = self._prepare_filters(query.filters)
            
            # Get initial results
            initial_results = await self.vector_store.search(
                query_embedding=query_embedding,
                max_results=initial_results_count,
                similarity_threshold=query.filters.similarity_threshold * 0.8,  # Lower threshold
                filters=filters
            )
            
            # Apply MMR algorithm
            final_results = self._apply_mmr(
                query_embedding=query_embedding,
                candidates=initial_results.results,
                max_results=query.filters.max_results,
                lambda_param=0.7  # Balance between relevance and diversity
            )
            
            # Create result object
            mmr_results = RetrievalResults(
                query_id=query.query_id,
                results=final_results,
                retrieval_time=initial_results.retrieval_time,
                retrieval_strategy=RetrievalStrategy.MMR,
                filters_applied=filters
            )
            
            return mmr_results
            
        except Exception as e:
            raise RetrievalError(f"MMR search failed: {str(e)}")
    
    async def hybrid_search(
        self, 
        query: Query, 
        query_analysis: Dict[str, Any]
    ) -> RetrievalResults:
        """Perform hybrid search combining semantic and keyword approaches."""
        
        try:
            # Perform semantic search
            semantic_results = await self.similarity_search(query, query_analysis)
            
            # Perform keyword-based reranking
            keyword_scores = self._calculate_keyword_scores(
                query_analysis['key_terms'], 
                semantic_results.results
            )
            
            # Combine scores
            hybrid_results = []
            for result in semantic_results.results:
                keyword_score = keyword_scores.get(result.chunk_id, 0.0)
                
                # Weighted combination (70% semantic, 30% keyword)
                combined_score = (0.7 * result.similarity_score) + (0.3 * keyword_score)
                
                # Update result
                result.similarity_score = combined_score
                hybrid_results.append(result)
            
            # Re-sort by combined score
            hybrid_results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            # Apply final threshold
            filtered_results = [
                r for r in hybrid_results 
                if r.similarity_score >= query.filters.similarity_threshold
            ]
            
            # Create result object
            final_results = RetrievalResults(
                query_id=query.query_id,
                results=filtered_results[:query.filters.max_results],
                retrieval_time=semantic_results.retrieval_time,
                retrieval_strategy=RetrievalStrategy.HYBRID,
                filters_applied=semantic_results.filters_applied
            )
            
            return final_results
            
        except Exception as e:
            raise RetrievalError(f"Hybrid search failed: {str(e)}")
    
    def _prepare_filters(self, query_filters: QueryFilters) -> Dict[str, Any]:
        """Prepare filters for vector store search."""
        
        filters = {}
        
        if query_filters.document_ids:
            filters['document_ids'] = query_filters.document_ids
        
        if query_filters.document_formats:
            # This would need to be implemented based on metadata structure
            pass
        
        if query_filters.chunk_types:
            filters['chunk_types'] = query_filters.chunk_types
        
        if query_filters.date_range:
            # Convert datetime filters if needed
            pass
        
        # Add custom metadata filters
        if query_filters.metadata_filters:
            filters.update(query_filters.metadata_filters)
        
        return filters
    
    def _apply_mmr(
        self, 
        query_embedding: List[float], 
        candidates: List[SearchResult], 
        max_results: int,
        lambda_param: float = 0.7
    ) -> List[SearchResult]:
        """Apply Maximal Marginal Relevance algorithm."""
        
        if not candidates:
            return []
        
        selected = []
        remaining = candidates.copy()
        
        # Select first result (highest similarity)
        if remaining:
            first_result = max(remaining, key=lambda x: x.similarity_score)
            selected.append(first_result)
            remaining.remove(first_result)
        
        # Iteratively select results balancing relevance and diversity
        while len(selected) < max_results and remaining:
            best_score = -1
            best_result = None
            
            for candidate in remaining:
                # Relevance score (similarity to query)
                relevance = candidate.similarity_score
                
                # Diversity score (maximum similarity to already selected)
                max_similarity = 0
                for selected_result in selected:
                    # Simplified diversity calculation (would need actual embeddings)
                    # Using content overlap as proxy
                    similarity = self._calculate_content_similarity(
                        candidate.content, 
                        selected_result.content
                    )
                    max_similarity = max(max_similarity, similarity)
                
                diversity = 1 - max_similarity
                
                # MMR score
                mmr_score = lambda_param * relevance + (1 - lambda_param) * diversity
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_result = candidate
            
            if best_result:
                selected.append(best_result)
                remaining.remove(best_result)
            else:
                break
        
        return selected
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate simple content similarity (Jaccard similarity)."""
        
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def _calculate_keyword_scores(self, key_terms: List[str], results: List[SearchResult]) -> Dict[str, float]:
        """Calculate keyword-based scores for results."""
        
        keyword_scores = {}
        
        for result in results:
            content_lower = result.content.lower()
            
            # Count keyword matches
            matches = 0
            total_terms = len(key_terms)
            
            for term in key_terms:
                if term in content_lower:
                    matches += 1
            
            # Calculate score (0.0 to 1.0)
            score = matches / total_terms if total_terms > 0 else 0.0
            keyword_scores[result.chunk_id] = score
        
        return keyword_scores


class RetrievalAgent(BaseAgent):
    """Advanced retrieval agent with intelligent search capabilities."""
    
    def __init__(self):
        super().__init__(AgentType.RETRIEVAL)
        self.settings = get_settings()
        
        # Initialize components
        self.vector_store: Optional[ChromaVectorStore] = None
        self.embedding_service: Optional[GeminiEmbeddingService] = None
        self.query_processor = QueryProcessor()
        self.retrieval_strategy: Optional[RetrievalStrategy] = None
        
        # Query cache
        self.query_cache: Dict[str, RetrievalResults] = {}
        self.cache_max_size = 1000
        
        # Statistics
        self.stats = {
            'queries_processed': 0,
            'cache_hits': 0,
            'average_response_time': 0.0,
            'total_results_returned': 0,
            'strategy_usage': {
                'similarity': 0,
                'mmr': 0,
                'hybrid': 0
            }
        }
    
    async def _initialize_agent(self) -> bool:
        """Initialize the retrieval agent."""
        
        try:
            self.logger.info("Initializing RetrievalAgent")
            
            # Initialize vector store
            self.vector_store = ChromaVectorStore()
            if not await self.vector_store.initialize():
                self.logger.error("Failed to initialize vector store")
                return False
            
            # Initialize embedding service
            self.embedding_service = GeminiEmbeddingService()
            
            # Initialize retrieval strategies
            self.retrieval_strategy = RetrievalStrategy(
                self.vector_store, 
                self.embedding_service
            )
            
            # Verify vector store health
            health_check = await self.vector_store.health_check()
            if not health_check['healthy']:
                self.logger.error(f"Vector store health check failed: {health_check['errors']}")
                return False
            
            self.logger.info("RetrievalAgent initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RetrievalAgent: {e}")
            return False
    
    async def _handle_message(self, message: MCPMessage) -> Optional[MCPMessage]:
        """Handle incoming MCP messages."""
        
        try:
            if message.message_type == MessageType.REQUEST:
                # Handle retrieval request
                if message.payload.get('action') == 'retrieve_documents':
                    return await self._handle_retrieval_request(message)
                
                elif message.payload.get('action') == 'add_documents':
                    return await self._handle_add_documents_request(message)
                
                elif message.payload.get('action') == 'delete_documents':
                    return await self._handle_delete_documents_request(message)
                
                elif message.payload.get('action') == 'get_stats':
                    return await self._handle_stats_request(message)
                
                else:
                    return create_error_message(
                        message, 
                        f"Unknown action: {message.payload.get('action')}"
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
            return create_error_message(message, str(e))
    
    async def _handle_retrieval_request(self, message: MCPMessage) -> MCPMessage:
        """Handle document retrieval request."""
        
        start_time = time.time()
        
        try:
            # Extract request data
            request_data = message.payload
            
            # Validate required fields
            if 'query' not in request_data:
                return create_error_message(message, "Missing required field: query")
            
            # Create Query object
            query = Query(
                text=request_data['query'],
                query_type=QueryType(request_data.get('query_type', 'semantic')),
                retrieval_strategy=RetrievalStrategy(
                    request_data.get('retrieval_strategy', 'similarity')
                ),
                user_id=request_data.get('user_id'),
                session_id=request_data.get('session_id')
            )
            
            # Apply filters if provided
            if 'filters' in request_data:
                filters_data = request_data['filters']
                query.filters = QueryFilters(**filters_data)
            
            # Apply conversation context if provided
            if 'conversation_context' in request_data:
                context_data = request_data['conversation_context']
                query.conversation_context = ConversationContext(**context_data)
            
            # Check cache first
            cache_key = self._generate_cache_key(query)
            if cache_key in self.query_cache:
                self.stats['cache_hits'] += 1
                cached_result = self.query_cache[cache_key]
                
                response_payload = {
                    'query_id': query.query_id,
                    'results': [result.dict() for result in cached_result.results],
                    'total_results': cached_result.total_results,
                    'retrieval_time': cached_result.retrieval_time,
                    'cached': True
                }
                
                return create_response_message(message, response_payload)
            
            # Process the query
            results = await self._process_query(query)
            
            # Cache results
            self._cache_results(cache_key, results)
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_stats(query.retrieval_strategy, processing_time, len(results.results))
            
            # Prepare response
            response_payload = {
                'query_id': query.query_id,
                'results': [result.dict() for result in results.results],
                'total_results': results.total_results,
                'retrieval_time': results.retrieval_time,
                'processing_time': processing_time,
                'strategy_used': results.retrieval_strategy.value,
                'cached': False
            }
            
            return create_response_message(message, response_payload)
            
        except Exception as e:
            error_msg = f"Retrieval request failed: {str(e)}"
            self.logger.error(error_msg)
            return create_error_message(message, error_msg)
    
    async def _handle_add_documents_request(self, message: MCPMessage) -> MCPMessage:
        """Handle request to add documents to vector store."""
        
        try:
            request_data = message.payload
            
            if 'chunks' not in request_data:
                return create_error_message(message, "Missing required field: chunks")
            
            # Reconstruct DocumentChunk objects
            chunks = []
            for chunk_data in request_data['chunks']:
                chunk = DocumentChunk(**chunk_data)
                chunks.append(chunk)
            
            # Generate embeddings
            self.logger.info(f"Generating embeddings for {len(chunks)} chunks")
            
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = await self.embedding_service.generate_embeddings_batch(chunk_texts)
            
            # Add to vector store
            success = await self.vector_store.add_documents(chunks, embeddings)
            
            if success:
                response_payload = {
                    'success': True,
                    'chunks_added': len(chunks),
                    'message': f'Successfully added {len(chunks)} chunks to vector store'
                }
            else:
                response_payload = {
                    'success': False,
                    'message': 'Failed to add chunks to vector store'
                }
            
            return create_response_message(message, response_payload)
            
        except Exception as e:
            error_msg = f"Add documents request failed: {str(e)}"
            self.logger.error(error_msg)
            return create_error_message(message, error_msg)
    
    async def _handle_delete_documents_request(self, message: MCPMessage) -> MCPMessage:
        """Handle request to delete documents from vector store."""
        
        try:
            request_data = message.payload
            
            if 'document_id' not in request_data:
                return create_error_message(message, "Missing required field: document_id")
            
            document_id = request_data['document_id']
            
            # Delete from vector store
            success = await self.vector_store.delete_document(document_id)
            
            if success:
                response_payload = {
                    'success': True,
                    'document_id': document_id,
                    'message': f'Successfully deleted document {document_id}'
                }
            else:
                response_payload = {
                    'success': False,
                    'document_id': document_id,
                    'message': f'Failed to delete document {document_id}'
                }
            
            return create_response_message(message, response_payload)
            
        except Exception as e:
            error_msg = f"Delete documents request failed: {str(e)}"
            self.logger.error(error_msg)
            return create_error_message(message, error_msg)
    
    async def _handle_stats_request(self, message: MCPMessage) -> MCPMessage:
        """Handle statistics request."""
        
        try:
            # Get comprehensive statistics
            stats = await self._get_comprehensive_stats()
            
            return create_response_message(message, stats)
            
        except Exception as e:
            error_msg = f"Stats request failed: {str(e)}"
            self.logger.error(error_msg)
            return create_error_message(message, error_msg)
    
    async def _process_query(self, query: Query) -> RetrievalResults:
        """Process a query and return results."""
        
        try:
            # Process and analyze query
            query_analysis = self.query_processor.process_query(
                query.text, 
                query.conversation_context
            )
            
            self.logger.debug(f"Query analysis: {query_analysis}")
            
            # Apply retrieval strategy
            if query.retrieval_strategy == RetrievalStrategy.SIMILARITY:
                results = await self.retrieval_strategy.similarity_search(query, query_analysis)
            elif query.retrieval_strategy == RetrievalStrategy.MMR:
                results = await self.retrieval_strategy.mmr_search(query, query_analysis)
            elif query.retrieval_strategy == RetrievalStrategy.HYBRID:
                results = await self.retrieval_strategy.hybrid_search(query, query_analysis)
            else:
                # Default to similarity search
                results = await self.retrieval_strategy.similarity_search(query, query_analysis)
            
            # Post-process results
            results = await self._post_process_results(results, query_analysis)
            
            self.logger.info(
                f"Processed query '{query.text[:50]}...' - "
                f"Found {len(results.results)} results in {results.retrieval_time:.3f}s"
            )
            
            return results
            
        except Exception as e:
            raise RetrievalError(f"Query processing failed: {str(e)}")
    
    async def _post_process_results(self, results: RetrievalResults, query_analysis: Dict[str, Any]) -> RetrievalResults:
        """Post-process retrieval results."""
        
        try:
            # Add highlighting information
            key_terms = query_analysis.get('key_terms', [])
            
            for result in results.results:
                # Simple highlighting
                highlighted_content = result.content
                for term in key_terms:
                    # Case-insensitive highlighting
                    pattern = re.compile(re.escape(term), re.IGNORECASE)
                    highlighted_content = pattern.sub(f"**{term}**", highlighted_content)
                
                result.highlighted_content = highlighted_content
            
            return results
            
        except Exception as e:
            self.logger.error(f"Post-processing failed: {e}")
            return results  # Return original results if post-processing fails
    
    def _generate_cache_key(self, query: Query) -> str:
        """Generate cache key for query."""
        
        import hashlib
        
        # Create key from query components
        key_components = [
            query.text,
            query.query_type.value,
            query.retrieval_strategy.value,
            str(query.filters.max_results),
            str(query.filters.similarity_threshold),
            str(sorted(query.filters.document_ids or [])),
            str(sorted(query.filters.chunk_types or []))
        ]
        
        key_string = '|'.join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _cache_results(self, cache_key: str, results: RetrievalResults) -> None:
        """Cache retrieval results."""
        
        # Manage cache size
        if len(self.query_cache) >= self.cache_max_size:
            # Remove oldest 10% of entries
            keys_to_remove = list(self.query_cache.keys())[:self.cache_max_size // 10]
            for key in keys_to_remove:
                del self.query_cache[key]
        
        self.query_cache[cache_key] = results
    
    def _update_stats(self, strategy: RetrievalStrategy, processing_time: float, results_count: int) -> None:
        """Update agent statistics."""
        
        self.stats['queries_processed'] += 1
        self.stats['total_results_returned'] += results_count
        
        # Update average response time
        if self.stats['queries_processed'] == 1:
            self.stats['average_response_time'] = processing_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.stats['average_response_time'] = (
                alpha * processing_time + 
                (1 - alpha) * self.stats['average_response_time']
            )
        
        # Update strategy usage
        strategy_name = strategy.value.lower()
        if strategy_name in self.stats['strategy_usage']:
            self.stats['strategy_usage'][strategy_name] += 1
    
    async def _get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        
        stats = self.stats.copy()
        
        # Add vector store stats
        if self.vector_store:
            try:
                vector_stats = await self.vector_store.get_collection_stats()
                stats['vector_store'] = vector_stats
            except Exception as e:
                stats['vector_store'] = {'error': str(e)}
        
        # Add embedding service stats
        if self.embedding_service:
            stats['embedding_service'] = self.embedding_service.get_stats()
        
        # Add cache statistics
        stats['cache'] = {
            'size': len(self.query_cache),
            'max_size': self.cache_max_size,
            'hit_rate': (
                self.stats['cache_hits'] / max(1, self.stats['queries_processed']) * 100
            )
        }
        
        return stats
    
    async def _agent_health_check(self) -> Optional[Dict[str, Any]]:
        """Perform agent-specific health check."""
        
        health_info = {
            'vector_store_healthy': False,
            'embedding_service_healthy': False,
            'cache_size': len(self.query_cache),
            'statistics': self.stats.copy()
        }
        
        # Check vector store health
        if self.vector_store:
            try:
                vector_health = await self.vector_store.health_check()
                health_info['vector_store_healthy'] = vector_health['healthy']
                health_info['vector_store'] = vector_health
            except Exception as e:
                health_info['vector_store_error'] = str(e)
        
        # Check embedding service health
        if self.embedding_service:
            try:
                embedding_health = await self.embedding_service.health_check()
                health_info['embedding_service_healthy'] = embedding_health['healthy']
                health_info['embedding_service'] = embedding_health
            except Exception as e:
                health_info['embedding_service_error'] = str(e)
        
        return health_info
    
    async def close(self) -> None:
        """Close agent resources."""
        
        try:
            if self.vector_store:
                await self.vector_store.close()
            
            self.logger.info("RetrievalAgent resources closed")
            
        except Exception as e:
            self.logger.error(f"Error closing RetrievalAgent: {e}")
