"""Embedding service using Google Gemini API."""

import asyncio
import time
from typing import List, Optional, Dict, Any
import hashlib
import json

import google.generativeai as genai
# from google.generativeai.types import EmbedContentResponse

from ..config import get_settings, LoggerMixin


class EmbeddingError(Exception):
    """Custom exception for embedding operations."""
    pass


class GeminiEmbeddingService(LoggerMixin):
    """Google Gemini embedding service with caching and rate limiting."""
    
    def __init__(self):
        self.settings = get_settings()
        self.model_name = self.settings.gemini_embedding_model
        self.api_key = self.settings.google_gemini_api_key
                
        # Rate limiting
        self.requests_per_minute = 100  # Gemini free tier limit
        self.request_times: List[float] = []
        
        # Caching
        self.embedding_cache: Dict[str, List[float]] = {}
        self.cache_max_size = 10000
        
        # Statistics
        self.stats = {
            'embeddings_generated': 0,
            'cache_hits': 0,
            'api_calls': 0,
            'total_tokens_processed': 0,
            'average_embedding_time': 0.0,
            'errors': 0
        }
        
        # Configure Gemini
        if self.api_key:
            genai.configure(api_key = self.api_key)
        else:
            raise EmbeddingError("GEMINI_API_KEY not configured")
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        
        if not text or not text.strip():
            raise EmbeddingError("Text cannot be empty")
        
        # Check cache first
        cache_key = self._get_cache_key(text)
        if cache_key in self.embedding_cache:
            self.stats['cache_hits'] += 1
            self.logger.debug(f"Cache hit for text: {text[:50]}...")
            return self.embedding_cache[cache_key]
        
        # Rate limiting
        await self._enforce_rate_limit()
        
        try:
            start_time = time.time()
            
            # Generate embedding using Gemini
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="RETRIEVAL_DOCUMENT"
            )
            
            embedding = result['embedding']
            generation_time = time.time() - start_time
            
            # Update statistics
            self.stats['embeddings_generated'] += 1
            self.stats['api_calls'] += 1
            self.stats['total_tokens_processed'] += len(text.split())
            self._update_average_time(generation_time)
            
            # Cache the result
            self._cache_embedding(cache_key, embedding)
            
            self.logger.debug(
                f"Generated embedding for text ({len(text)} chars) "
                f"in {generation_time:.2f}s"
            )
            
            return embedding
            
        except Exception as e:
            self.stats['errors'] += 1
            error_msg = f"Failed to generate embedding: {str(e)}"
            self.logger.error(error_msg)
            raise EmbeddingError(error_msg)
    
    async def generate_embeddings_batch(
        self, 
        texts: List[str], 
        batch_size: int = 10
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts with batching."""
        
        if not texts:
            return []
        
        embeddings = []
        
        # Process in batches to avoid rate limits
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Process batch concurrently
            batch_tasks = [
                self.generate_embedding(text) 
                for text in batch if text and text.strip()
            ]
            
            if batch_tasks:
                try:
                    batch_embeddings = await asyncio.gather(*batch_tasks)
                    embeddings.extend(batch_embeddings)
                    
                    # Small delay between batches
                    if i + batch_size < len(texts):
                        await asyncio.sleep(0.1)
                        
                except Exception as e:
                    self.logger.error(f"Batch embedding failed: {e}")
                    # Continue with individual processing for failed batch
                    for text in batch:
                        try:
                            if text and text.strip():
                                embedding = await self.generate_embedding(text)
                                embeddings.append(embedding)
                        except Exception as individual_error:
                            self.logger.error(f"Individual embedding failed: {individual_error}")
                            # Add zero vector as placeholder
                            embeddings.append([0.0] * 768)  # Gemini embedding dimension
        
        self.logger.info(f"Generated {len(embeddings)} embeddings from {len(texts)} texts")
        return embeddings
    
    async def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding specifically for query text."""
        
        try:
            # Rate limiting
            await self._enforce_rate_limit()
            
            start_time = time.time()
            
            # Generate embedding with query task type
            result = genai.embed_content(
                model=self.model_name,
                content=query,
                task_type="RETRIEVAL_QUERY"
            )
            
            embedding = result['embedding']
            generation_time = time.time() - start_time
            
            # Update statistics
            self.stats['api_calls'] += 1
            self.stats['total_tokens_processed'] += len(query.split())
            self._update_average_time(generation_time)
            
            self.logger.debug(
                f"Generated query embedding in {generation_time:.2f}s"
            )
            
            return embedding
            
        except Exception as e:
            self.stats['errors'] += 1
            error_msg = f"Failed to generate query embedding: {str(e)}"
            self.logger.error(error_msg)
            raise EmbeddingError(error_msg)
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _cache_embedding(self, cache_key: str, embedding: List[float]) -> None:
        """Cache an embedding with size management."""
        
        # Remove oldest entries if cache is full
        if len(self.embedding_cache) >= self.cache_max_size:
            # Remove 10% of oldest entries
            keys_to_remove = list(self.embedding_cache.keys())[:self.cache_max_size // 10]
            for key in keys_to_remove:
                del self.embedding_cache[key]
        
        self.embedding_cache[cache_key] = embedding
    
    async def _enforce_rate_limit(self) -> None:
        """Enforce API rate limiting."""
        
        current_time = time.time()
        
        # Remove requests older than 1 minute
        self.request_times = [
            req_time for req_time in self.request_times 
            if current_time - req_time < 60
        ]
        
        # Check if we're at the limit
        if len(self.request_times) >= self.requests_per_minute:
            # Wait until we can make another request
            oldest_request = min(self.request_times)
            wait_time = 60 - (current_time - oldest_request)
            
            if wait_time > 0:
                self.logger.warning(f"Rate limit reached, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
        
        # Record this request
        self.request_times.append(current_time)
    
    def _update_average_time(self, generation_time: float) -> None:
        """Update average embedding generation time."""
        
        total_embeddings = self.stats['embeddings_generated']
        if total_embeddings == 1:
            self.stats['average_embedding_time'] = generation_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.stats['average_embedding_time'] = (
                alpha * generation_time + 
                (1 - alpha) * self.stats['average_embedding_time']
            )
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this service."""
        return 768  # Gemini embedding dimension
    
    def get_stats(self) -> Dict[str, Any]:
        """Get embedding service statistics."""
        
        stats = self.stats.copy()
        stats.update({
            'cache_size': len(self.embedding_cache),
            'cache_hit_rate': (
                self.stats['cache_hits'] / max(1, self.stats['embeddings_generated']) * 100
            ),
            'model_name': self.model_name,
            'embedding_dimension': self.get_embedding_dimension()
        })
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        
        self.embedding_cache.clear()
        self.logger.info("Embedding cache cleared")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the embedding service."""
        
        health_status = {
            'healthy': True,
            'service': 'GeminiEmbeddingService',
            'model': self.model_name,
            'errors': [],
            'stats': self.get_stats()
        }
        
        try:
            # Test embedding generation
            test_text = "Health check test"
            start_time = time.time()
            
            test_embedding = await self.generate_embedding(test_text)
            
            response_time = time.time() - start_time
            
            health_status.update({
                'test_successful': True,
                'response_time': response_time,
                'embedding_dimension': len(test_embedding)
            })
            
        except Exception as e:
            health_status.update({
                'healthy': False,
                'test_successful': False,
                'errors': [str(e)]
            })
        
        return health_status
