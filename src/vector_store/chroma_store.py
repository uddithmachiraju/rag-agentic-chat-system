"""ChromaDB implementation of vector store."""

import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from ..config import get_settings
from ..models.document import DocumentChunk, ChunkType
from ..models.query import SearchResult, RetrievalResults, RetrievalStrategy
from .base_store import BaseVectorStore, VectorStoreError
from .embeddings import GeminiEmbeddingService


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB implementation of vector store with Gemini embeddings."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        self.settings = get_settings()
        
        # ChromaDB configuration
        self.db_path = self.config.get('db_path', self.settings.chroma_db_path)
        self.collection_name = self.config.get('collection_name', 'document_chunks')
        
        # ChromaDB client and collection
        self.client: Optional[chromadb.Client] = None
        self.collection: Optional[chromadb.Collection] = None
        
        # Embedding service
        self.embedding_service = GeminiEmbeddingService()
        
        # Statistics
        self.stats = {
            'chunks_stored': 0,
            'searches_performed': 0,
            'last_update': None,
            'collection_size': 0
        }
    
    async def initialize(self) -> bool:
        """Initialize the ChromaDB vector store."""
        
        try:
            self.logger.info(f"Initializing ChromaDB at {self.db_path}")
            
            # Create ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Document chunks for RAG system"}
            )
            
            # Update stats
            self.stats['collection_size'] = self.collection.count()
            self.is_initialized = True
            
            self.logger.info(
                f"ChromaDB initialized successfully. "
                f"Collection '{self.collection_name}' has {self.stats['collection_size']} chunks"
            )
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to initialize ChromaDB: {str(e)}"
            self.logger.error(error_msg)
            raise VectorStoreError(error_msg)
    
    async def add_documents(
        self, 
        chunks: List[DocumentChunk], 
        embeddings: List[List[float]]
    ) -> bool:
        """Add document chunks with their embeddings to ChromaDB."""
        
        if not self.is_initialized:
            raise VectorStoreError("Vector store not initialized")
        
        if len(chunks) != len(embeddings):
            raise VectorStoreError("Number of chunks and embeddings must match")
        
        try:
            self.logger.info(f"Adding {len(chunks)} chunks to ChromaDB")
            
            # Prepare data for ChromaDB
            ids = []
            documents = []
            metadatas = []
            embeddings_list = []
            
            for chunk, embedding in zip(chunks, embeddings):
                ids.append(chunk.chunk_id)
                documents.append(chunk.content)
                
                # Prepare metadata
                metadata = {
                    'document_id': chunk.document_id,
                    'chunk_index': chunk.chunk_index,
                    'chunk_type': chunk.chunk_type.value,
                    'page_number': chunk.page_number or 0,
                    'created_at': chunk.created_at.isoformat(),
                    'content_length': len(chunk.content),
                    'start_char': chunk.start_char or 0,
                    'end_char': chunk.end_char or 0,
                    'heading_level': chunk.heading_level or 0,
                    'parent_section': chunk.parent_section or "",
                    'embedding_model': chunk.embedding_model or self.embedding_service.model_name
                }
                
                # Add custom metadata
                if chunk.metadata:
                    for key, value in chunk.metadata.items():
                        # Ensure values are JSON serializable
                        if isinstance(value, (str, int, float, bool)):
                            metadata[f"custom_{key}"] = value
                        else:
                            metadata[f"custom_{key}"] = str(value)
                
                metadatas.append(metadata)
                embeddings_list.append(embedding)
            
            # Add to ChromaDB collection
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings_list
            )
            
            # Update chunk embeddings
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding_vector = embedding
                chunk.embedding_model = self.embedding_service.model_name
            
            # Update statistics
            self.stats['chunks_stored'] += len(chunks)
            self.stats['collection_size'] = self.collection.count()
            self.stats['last_update'] = datetime.utcnow()
            
            self.logger.info(f"Successfully added {len(chunks)} chunks to ChromaDB")
            return True
            
        except Exception as e:
            error_msg = f"Failed to add documents to ChromaDB: {str(e)}"
            self.logger.error(error_msg)
            raise VectorStoreError(error_msg)
    
    async def update_document(
        self, 
        chunk_id: str, 
        chunk: DocumentChunk, 
        embedding: List[float]
    ) -> bool:
        """Update a specific document chunk in ChromaDB."""
        
        if not self.is_initialized:
            raise VectorStoreError("Vector store not initialized")
        
        try:
            # Prepare metadata
            metadata = {
                'document_id': chunk.document_id,
                'chunk_index': chunk.chunk_index,
                'chunk_type': chunk.chunk_type.value,
                'page_number': chunk.page_number or 0,
                'created_at': chunk.created_at.isoformat(),
                'content_length': len(chunk.content),
                'start_char': chunk.start_char or 0,
                'end_char': chunk.end_char or 0,
                'heading_level': chunk.heading_level or 0,
                'parent_section': chunk.parent_section or "",
                'embedding_model': self.embedding_service.model_name,
                'updated_at': datetime.utcnow().isoformat()
            }
            
            # Add custom metadata
            if chunk.metadata:
                for key, value in chunk.metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        metadata[f"custom_{key}"] = value
                    else:
                        metadata[f"custom_{key}"] = str(value)
            
            # Update in ChromaDB
            self.collection.update(
                ids=[chunk_id],
                documents=[chunk.content],
                metadatas=[metadata],
                embeddings=[embedding]
            )
            
            # Update chunk
            chunk.embedding_vector = embedding
            chunk.embedding_model = self.embedding_service.model_name
            
            self.logger.debug(f"Updated chunk {chunk_id} in ChromaDB")
            return True
            
        except Exception as e:
            error_msg = f"Failed to update chunk {chunk_id}: {str(e)}"
            self.logger.error(error_msg)
            raise VectorStoreError(error_msg)
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete all chunks for a document from ChromaDB."""
        
        if not self.is_initialized:
            raise VectorStoreError("Vector store not initialized")
        
        try:
            # Query for all chunks of this document
            results = self.collection.get(
                where={"document_id": document_id}
            )
            
            if results['ids']:
                # Delete all chunks
                self.collection.delete(ids=results['ids'])
                
                deleted_count = len(results['ids'])
                self.stats['collection_size'] = self.collection.count()
                
                self.logger.info(f"Deleted {deleted_count} chunks for document {document_id}")
                return True
            else:
                self.logger.warning(f"No chunks found for document {document_id}")
                return True
                
        except Exception as e:
            error_msg = f"Failed to delete document {document_id}: {str(e)}"
            self.logger.error(error_msg)
            raise VectorStoreError(error_msg)
    
    async def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a specific chunk from ChromaDB."""
        
        if not self.is_initialized:
            raise VectorStoreError("Vector store not initialized")
        
        try:
            self.collection.delete(ids=[chunk_id])
            self.stats['collection_size'] = self.collection.count()
            
            self.logger.debug(f"Deleted chunk {chunk_id} from ChromaDB")
            return True
            
        except Exception as e:
            error_msg = f"Failed to delete chunk {chunk_id}: {str(e)}"
            self.logger.error(error_msg)
            raise VectorStoreError(error_msg)
    
    async def search(
        self,
        query_embedding: List[float],
        max_results: int = 5,
        similarity_threshold: float = 0.3,
        filters: Optional[Dict[str, Any]] = None
    ) -> RetrievalResults:
        """Perform similarity search in ChromaDB with detailed debugging."""
        
        if not self.is_initialized:
            raise VectorStoreError("Vector store not initialized")
        
        start_time = time.time()
        
        try:
            # Debug: Check collection state
            collection_count = self.collection.count()
            self.logger.info(f"Collection has {collection_count} documents")
            
            if collection_count == 0:
                self.logger.warning("Collection is empty - no documents to search")
                return self._empty_results(filters, time.time() - start_time)
            
            # Debug: Validate query embedding
            if not query_embedding or len(query_embedding) == 0:
                self.logger.error("Query embedding is empty")
                return self._empty_results(filters, time.time() - start_time)
            
            self.logger.info(f"Query embedding dimension: {len(query_embedding)}")
            
            # Build where clause for filtering
            where_clause = {}
            if filters:
                for key, value in filters.items():
                    if key == 'document_ids' and isinstance(value, list):
                        if len(value) == 1:
                            where_clause['document_id'] = value[0]
                        else:
                            where_clause['document_id'] = {"$in": value}
                    elif key == 'chunk_types' and isinstance(value, list):
                        if len(value) == 1:
                            where_clause['chunk_type'] = value[0]
                        else:
                            where_clause['chunk_type'] = {"$in": value}
                    elif key == 'min_page':
                        where_clause['page_number'] = {"$gte": value}
                    elif key == 'max_page':
                        if 'page_number' in where_clause:
                            where_clause['page_number']['$lte'] = value
                        else:
                            where_clause['page_number'] = {"$lte": value}
                    else:
                        where_clause[key] = value
            
            self.logger.info(f"Where clause: {where_clause}")
            
            # Debug: Try query without similarity threshold first
            debug_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(max_results * 2, 20),  # Get more results for debugging
                where=where_clause if where_clause else None
            )
            
            self.logger.info(f"Raw query returned {len(debug_results.get('ids', [[]]))} result sets")
            if debug_results['ids'] and debug_results['ids'][0]:
                self.logger.info(f"First result set has {len(debug_results['ids'][0])} results")
                if debug_results['distances'] and debug_results['distances'][0]:
                    distances = debug_results['distances'][0]
                    similarities = [1.0 - d for d in distances]
                    self.logger.info(f"Similarity scores: {similarities[:5]}...")  # Show first 5
                    self.logger.info(f"Max similarity: {max(similarities) if similarities else 'N/A'}")
                    self.logger.info(f"Min similarity: {min(similarities) if similarities else 'N/A'}")
            
            # Perform actual search with original parameters
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=max_results,
                where=where_clause if where_clause else None
            )
            
            retrieval_time = time.time() - start_time
            
            # Process results
            search_results = []
            
            if results['ids'] and results['ids'][0]:  # ChromaDB returns nested lists
                ids = results['ids'][0]
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                
                self.logger.info(f"Processing {len(ids)} raw results")
                
                filtered_count = 0
                for i, (chunk_id, content, metadata, distance) in enumerate(
                    zip(ids, documents, metadatas, distances)
                ):
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    similarity_score = 1.0 - distance
                    
                    self.logger.debug(f"Result {i}: similarity={similarity_score:.4f}, threshold={similarity_threshold}")
                    
                    # Apply similarity threshold
                    if similarity_score < similarity_threshold:
                        filtered_count += 1
                        continue
                    
                    # Create search result
                    search_result = SearchResult(
                        chunk_id=chunk_id,
                        document_id=metadata.get('document_id', ''),
                        content=content,
                        similarity_score=similarity_score,
                        chunk_metadata=metadata,
                        document_metadata={},  # Will be populated by retrieval agent
                        page_number=metadata.get('page_number'),
                        chunk_index=metadata.get('chunk_index', i)
                    )
                    
                    search_results.append(search_result)
                
                self.logger.info(f"Filtered out {filtered_count} results below threshold {similarity_threshold}")
                self.logger.info(f"Final results count: {len(search_results)}")
            else:
                self.logger.warning("No results returned from ChromaDB query")
            
            # Create retrieval results
            retrieval_results = RetrievalResults(
                query_id="",  # Will be set by retrieval agent
                results=search_results,
                retrieval_time=retrieval_time,
                retrieval_strategy=RetrievalStrategy.SIMILARITY,
                filters_applied=filters or {}
            )
            
            # Update statistics
            self.stats['searches_performed'] += 1
            
            self.logger.info(
                f"ChromaDB search completed: {len(search_results)} results "
                f"in {retrieval_time:.3f}s"
            )
            
            return retrieval_results
            
        except Exception as e:
            error_msg = f"Search failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise VectorStoreError(error_msg)

    def _empty_results(self, filters, retrieval_time):
        """Helper to create empty results."""
        return RetrievalResults(
            query_id="",
            results=[],
            retrieval_time=retrieval_time,
            retrieval_strategy=RetrievalStrategy.SIMILARITY,
            filters_applied=filters or {}
        )
    
    async def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Retrieve a specific chunk by ID from ChromaDB."""
        
        if not self.is_initialized:
            raise VectorStoreError("Vector store not initialized")
        
        try:
            results = self.collection.get(
                ids=[chunk_id],
                include=['documents', 'metadatas', 'embeddings']
            )
            
            if not results['ids'] or not results['ids'][0]:
                return None
            
            # Extract data
            content = results['documents'][0]
            metadata = results['metadatas'][0]
            embedding = results['embeddings'][0] if results['embeddings'] else None
            
            # Reconstruct DocumentChunk
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                document_id=metadata.get('document_id', ''),
                chunk_index=metadata.get('chunk_index', 0),
                chunk_type=ChunkType(metadata.get('chunk_type', 'text')),
                content=content,
                page_number=metadata.get('page_number'),
                start_char=metadata.get('start_char'),
                end_char=metadata.get('end_char'),
                heading_level=metadata.get('heading_level'),
                parent_section=metadata.get('parent_section'),
                embedding_vector=embedding,
                embedding_model=metadata.get('embedding_model'),
                created_at=datetime.fromisoformat(metadata.get('created_at', datetime.utcnow().isoformat()))
            )
            
            # Add custom metadata
            custom_metadata = {}
            for key, value in metadata.items():
                if key.startswith('custom_'):
                    custom_metadata[key[7:]] = value  # Remove 'custom_' prefix
            
            if custom_metadata:
                chunk.metadata = custom_metadata
            
            return chunk
            
        except Exception as e:
            error_msg = f"Failed to get chunk {chunk_id}: {str(e)}"
            self.logger.error(error_msg)
            raise VectorStoreError(error_msg)
    
    async def get_chunks_by_document(self, document_id: str) -> List[DocumentChunk]:
        """Retrieve all chunks for a document from ChromaDB."""
        
        if not self.is_initialized:
            raise VectorStoreError("Vector store not initialized")
        
        try:
            results = self.collection.get(
                where={"document_id": document_id},
                include=['documents', 'metadatas', 'embeddings']
            )
            
            chunks = []
            
            if results['ids']:
                for i, chunk_id in enumerate(results['ids']):
                    content = results['documents'][i]
                    metadata = results['metadatas'][i]
                    embedding = results['embeddings'][i] if results['embeddings'] else None
                    
                    # Reconstruct DocumentChunk
                    chunk = DocumentChunk(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        chunk_index=metadata.get('chunk_index', i),
                        chunk_type=ChunkType(metadata.get('chunk_type', 'text')),
                        content=content,
                        page_number=metadata.get('page_number'),
                        start_char=metadata.get('start_char'),
                        end_char=metadata.get('end_char'),
                        heading_level=metadata.get('heading_level'),
                        parent_section=metadata.get('parent_section'),
                        embedding_vector=embedding,
                        embedding_model=metadata.get('embedding_model'),
                        created_at=datetime.fromisoformat(metadata.get('created_at', datetime.utcnow().isoformat()))
                    )
                    
                    # Add custom metadata
                    custom_metadata = {}
                    for key, value in metadata.items():
                        if key.startswith('custom_'):
                            custom_metadata[key[7:]] = value
                    
                    if custom_metadata:
                        chunk.metadata = custom_metadata
                    
                    chunks.append(chunk)
                
                # Sort by chunk index
                chunks.sort(key=lambda x: x.chunk_index)
            
            return chunks
            
        except Exception as e:
            error_msg = f"Failed to get chunks for document {document_id}: {str(e)}"
            self.logger.error(error_msg)
            raise VectorStoreError(error_msg)
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the ChromaDB collection."""
        
        if not self.is_initialized:
            raise VectorStoreError("Vector store not initialized")
        
        try:
            # Get basic collection info
            collection_count = self.collection.count()
            
            # Get sample of metadata to analyze
            sample_results = self.collection.get(
                limit=min(1000, collection_count),
                include=['metadatas']
            )
            
            # Analyze content
            document_ids = set()
            chunk_types = {}
            pages_distribution = {}
            
            if sample_results['metadatas']:
                for metadata in sample_results['metadatas']:
                    # Count unique documents
                    document_ids.add(metadata.get('document_id', ''))
                    
                    # Count chunk types
                    chunk_type = metadata.get('chunk_type', 'unknown')
                    chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
                    
                    # Page distribution
                    page_num = metadata.get('page_number', 0)
                    pages_distribution[page_num] = pages_distribution.get(page_num, 0) + 1
            
            stats = {
                'total_chunks': collection_count,
                'unique_documents': len(document_ids),
                'chunk_types': chunk_types,
                'sample_size': len(sample_results['metadatas']) if sample_results['metadatas'] else 0,
                'collection_name': self.collection_name,
                'embedding_dimension': self.embedding_service.get_embedding_dimension(),
                'database_path': self.db_path,
                **self.stats
            }
            
            return stats
            
        except Exception as e:
            error_msg = f"Failed to get collection stats: {str(e)}"
            self.logger.error(error_msg)
            raise VectorStoreError(error_msg)
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on ChromaDB."""
        
        health_status = {
            'healthy': True,
            'service': 'ChromaVectorStore',
            'errors': [],
            'collection_accessible': False,
            'embedding_service_healthy': False
        }
        
        try:
            # Check if initialized
            if not self.is_initialized:
                health_status['healthy'] = False
                health_status['errors'].append("Vector store not initialized")
                return health_status
            
            # Check collection accessibility
            try:
                count = self.collection.count()
                health_status['collection_accessible'] = True
                health_status['collection_count'] = count
            except Exception as e:
                health_status['healthy'] = False
                health_status['errors'].append(f"Collection not accessible: {str(e)}")
            
            # Check embedding service
            try:
                embedding_health = await self.embedding_service.health_check()
                health_status['embedding_service_healthy'] = embedding_health['healthy']
                health_status['embedding_service'] = embedding_health
                
                if not embedding_health['healthy']:
                    health_status['healthy'] = False
                    health_status['errors'].extend(embedding_health.get('errors', []))
            except Exception as e:
                health_status['healthy'] = False
                health_status['errors'].append(f"Embedding service error: {str(e)}")
            
            # Add statistics
            if health_status['healthy']:
                health_status['stats'] = await self.get_collection_stats()
            
        except Exception as e:
            health_status['healthy'] = False
            health_status['errors'].append(f"Health check failed: {str(e)}")
        
        return health_status
    
    async def close(self) -> None:
        """Close ChromaDB connection."""
        
        try:
            if self.client:
                # ChromaDB doesn't have explicit close method
                self.client = None
                self.collection = None
                self.is_initialized = False
                
                self.logger.info("ChromaDB connection closed")
                
        except Exception as e:
            self.logger.error(f"Error closing ChromaDB: {e}")
