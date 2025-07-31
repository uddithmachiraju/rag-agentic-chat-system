"""Document data models and schemas for the agentic RAG system."""

import hashlib
from datetime import datetime, UTC
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


class DocumentFormat(str, Enum):
    """Supported document formats."""
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    CSV = "csv"
    TXT = "txt"
    MARKDOWN = "md"


class ProcessingStatus(str, Enum):
    """Document processing status."""
    UPLOADED = "uploaded"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXED = "indexed"
    FAILED = "failed"
    DELETED = "deleted"


class ChunkType(str, Enum):
    """Types of document chunks."""
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    HEADER = "header"
    FOOTER = "footer"
    METADATA = "metadata"


class DocumentMetadata(BaseModel):
    """Document metadata extracted during processing."""
    
    title: Optional[str] = None
    author: Optional[str] = None
    created_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    subject: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    language: Optional[str] = None
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    character_count: Optional[int] = None
    
    # Format-specific metadata
    pdf_metadata: Dict[str, Any] = Field(default_factory=dict)
    docx_metadata: Dict[str, Any] = Field(default_factory=dict)
    csv_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Custom metadata
    custom_fields: Dict[str, Any] = Field(default_factory=dict)


class ProcessingOptions(BaseModel):
    """Options for document processing."""
    
    chunk_size: int = Field(1000, ge=100, le=5000)
    chunk_overlap: int = Field(200, ge=0, le=1000)
    preserve_formatting: bool = True
    extract_images: bool = False
    extract_tables: bool = True
    ocr_enabled: bool = False
    language_detection: bool = True
    
    # Format-specific options
    pdf_options: Dict[str, Any] = Field(default_factory=dict)
    docx_options: Dict[str, Any] = Field(default_factory=dict)
    csv_options: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('chunk_overlap')
    @classmethod
    def chunk_overlap_must_be_less_than_size(cls, v, info):
        if info.data.get('chunk_size') is not None and v >= info.data['chunk_size']:
            raise ValueError('chunk_overlap must be less than chunk_size')
        return v


class DocumentChunk(BaseModel):
    """Individual document chunk with metadata."""
    
    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    chunk_index: int
    chunk_type: ChunkType = ChunkType.TEXT
    content: str
    
    # Position information
    page_number: Optional[int] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    
    # Structural information
    heading_level: Optional[int] = None
    parent_section: Optional[str] = None
    
    # Embedding information
    embedding_vector: Optional[List[float]] = None
    embedding_model: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    
    @field_validator('content')
    @classmethod
    def content_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Chunk content cannot be empty')
        return v.strip()
    
    def get_content_hash(self) -> str:
        """Generate hash of chunk content for deduplication."""
        return hashlib.sha256(self.content.encode()).hexdigest()


class Document(BaseModel):
    """Main document model."""
    
    document_id: str = Field(default_factory=lambda: str(uuid4()))
    file_name: str
    original_path: str
    stored_path: Optional[str] = None
    
    # File information
    file_size: int = Field(ge=0)
    file_format: DocumentFormat
    mime_type: str
    file_hash: str
    
    # Processing information
    status: ProcessingStatus = ProcessingStatus.UPLOADED
    processing_options: ProcessingOptions = Field(default_factory=ProcessingOptions)
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    
    # Content information
    chunks: List[DocumentChunk] = Field(default_factory=list)
    total_chunks: int = 0
    
    # Timestamps
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    processed_at: Optional[datetime] = None
    indexed_at: Optional[datetime] = None
    last_accessed: Optional[datetime] = None
    
    # User information
    user_id: Optional[str] = None
    
    # Processing metrics
    processing_time: Optional[float] = None
    parsing_time: Optional[float] = None
    chunking_time: Optional[float] = None
    embedding_time: Optional[float] = None
    
    # Error information
    error_message: Optional[str] = None
    error_details: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('file_format', mode='before')
    @classmethod
    def validate_file_format(cls, v, info):
        if isinstance(v, str):
            # Extract format from filename if not provided
            if info.data and 'file_name' in info.data:
                file_path = Path(info.data['file_name'])
                extension = file_path.suffix.lower().lstrip('.')
                
                # Map extensions to formats
                extension_map = {
                    'pdf': DocumentFormat.PDF,
                    'docx': DocumentFormat.DOCX,
                    'pptx': DocumentFormat.PPTX,
                    'csv': DocumentFormat.CSV,
                    'txt': DocumentFormat.TXT,
                    'md': DocumentFormat.MARKDOWN
                }
                
                if extension in extension_map:
                    return extension_map[extension]
        
        return v
    
    @model_validator(mode='after')
    def validate_total_chunks(self):
        self.total_chunks = len(self.chunks)
        return self
    
    def add_chunk(self, chunk: DocumentChunk) -> None:
        """Add a chunk to the document."""
        chunk.document_id = self.document_id
        chunk.chunk_index = len(self.chunks)
        self.chunks.append(chunk)
        self.total_chunks = len(self.chunks)
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Get a chunk by its ID."""
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None
    
    def get_chunks_by_type(self, chunk_type: ChunkType) -> List[DocumentChunk]:
        """Get all chunks of a specific type."""
        return [chunk for chunk in self.chunks if chunk.chunk_type == chunk_type]
    
    def update_status(self, status: ProcessingStatus, error_message: Optional[str] = None) -> None:
        """Update document processing status."""
        self.status = status
        
        if status == ProcessingStatus.FAILED and error_message:
            self.error_message = error_message
        
        if status == ProcessingStatus.INDEXED:
            self.indexed_at = datetime.now(UTC)
        elif status in [ProcessingStatus.PARSING, ProcessingStatus.CHUNKING, ProcessingStatus.EMBEDDING]:
            if not self.processed_at:
                self.processed_at = datetime.now(UTC)
    
    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of the file."""
        hash_sha256 = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get document summary information."""
        return {
            "document_id": self.document_id,
            "file_name": self.file_name,
            "file_format": self.file_format,
            "file_size": self.file_size,
            "status": self.status,
            "total_chunks": self.total_chunks,
            "uploaded_at": self.uploaded_at,
            "processed_at": self.processed_at,
            "processing_time": self.processing_time
        }
    
    model_config = {
        "json_schema_serialization_defaults_required": True,
    }


class DocumentCollection(BaseModel):
    """Collection of documents with metadata."""
    
    collection_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: Optional[str] = None
    documents: List[Document] = Field(default_factory=list)
    
    # Collection metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    user_id: Optional[str] = None
    
    # Collection statistics
    total_documents: int = 0
    total_chunks: int = 0
    total_size: int = 0
    
    def add_document(self, document: Document) -> None:
        """Add a document to the collection."""
        self.documents.append(document)
        self._update_statistics()
    
    def remove_document(self, document_id: str) -> bool:
        """Remove a document from the collection."""
        for i, doc in enumerate(self.documents):
            if doc.document_id == document_id:
                del self.documents[i]
                self._update_statistics()
                return True
        return False
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """Get a document by ID."""
        for doc in self.documents:
            if doc.document_id == document_id:
                return doc
        return None
    
    def _update_statistics(self) -> None:
        """Update collection statistics."""
        self.total_documents = len(self.documents)
        self.total_chunks = sum(doc.total_chunks for doc in self.documents)
        self.total_size = sum(doc.file_size for doc in self.documents)
        self.updated_at = datetime.now(UTC)
    
    def get_documents_by_format(self, file_format: DocumentFormat) -> List[Document]:
        """Get all documents of a specific format."""
        return [doc for doc in self.documents if doc.file_format == file_format]
    
    def get_documents_by_status(self, status: ProcessingStatus) -> List[Document]:
        """Get all documents with a specific status."""
        return [doc for doc in self.documents if doc.status == status]