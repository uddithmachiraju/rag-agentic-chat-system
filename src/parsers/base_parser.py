import re 
from abc import ABC, abstractmethod 
from typing import Dict, Any, List, Optional
from pathlib import Path 
from ..config.logging import LoggerMixin
from ..models.document import Document, DocumentChunk, DocumentMetadata, ProcessingOptions, ChunkType

class ParseResult:
    def __init__(
            self,
            success: bool,
            chunks: List[DocumentChunk] = None,
            metadata: DocumentMetadata = None,
            error_message: str = None,
            processing_time: float = 0.0 
        ):
        self.success = success
        self.chunks = chunks or []
        self.metadata = metadata or DocumentMetadata()
        self.error_message = error_message
        self.processing_time = processing_time

class BaseParser(ABC, LoggerMixin):
    def __init__(self):
        self.supported_extensions: List[str] = []
        self.mime_types: List[str] = [] 

    @abstractmethod
    async def parse(self, file_path: str, document_id: str, options: ProcessingOptions = None) -> ParseResult:
        pass 

    @abstractmethod
    def can_parse(self, file_path: str, mime_type: str = None) -> bool:
        pass 

    def _get_file_extension(self, file_path: str) -> str:
        return Path(file_path).suffix.lower().lstrip(".") 
    
    def _validate_file(self, file_path: str) -> bool:
        try:
            file_path_obj = Path(file_path)
            return file_path_obj.exists() and file_path_obj.is_file()
        except Exception as e:
            self.logger.error(f"File validation error: {e}")
            return False
    
    async def _preprocess_content(self, content: str) -> str:
        if not content:
            return ""
        content = content.strip()

        content = re.sub(r'\s+', ' ', content)
        content = content.replace('\x00', '')
        return content 
    
    def _create_chunk(self, content: str, chunk_index: int, chunk_type: str = "text", page_number: int = None, metadata: Dict[str, Any] = None, document_id: str = None) -> DocumentChunk:
        return DocumentChunk( 
            document_id = document_id,
            content = content, 
            chunk_index = chunk_index,
            chunk_type = ChunkType(chunk_type),
            page_number = page_number,
            metadata = metadata or {} 
        )