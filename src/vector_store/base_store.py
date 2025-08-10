from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime

from ..config.logging import LoggerMixin 
from ..models.document import DocumentChunk 

class VectorStoreError(Exception):
    pass 

class BaseVectorStore(LoggerMixin, ABC):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.is_initialized = False 

    @abstractmethod
    async def initialize(self) -> bool:
        pass 

    @abstractmethod
    async def add_documents(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> bool:
        pass

    @abstractmethod
    async def update_document(self, chunk_id: str, chunk: DocumentChunk, embedding: List[float]) -> bool:
        pass

    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        pass
    
    @abstractmethod
    async def delete_chunk(self, chunk_id: str) -> bool:
        pass

    # @abstractmethod
    # async def search(self, query_embedding: List[float], max_results: int = 5, similarity_threshold: float = 0.7, filters: Optional[Dict[str, Any]] = None) -> RetrievalResults:
    #     """Perform similarity search."""
    #     pass

    @abstractmethod
    async def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        pass
    
    @abstractmethod
    async def get_chunks_by_document(self, document_id: str) -> List[DocumentChunk]:
        pass
    
    @abstractmethod
    async def get_collection_stats(self) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        pass
    
    async def close(self) -> None:
        pass