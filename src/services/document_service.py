import asyncio 
import hashlib 
import logging 
import os 
import uuid
from pathlib import Path 
from datetime import datetime, UTC
from typing import Optional, List, Dict, Any 
from ..config import get_settings, LoggerMixin

from src.models.document import (
    Document, DocumentCollection, DocumentMetadata, 
    DocumentChunk, ProcessingOptions, ProcessingStatus, 
    DocumentFormat
) 

from src.parsers.registry import ParserRegistry 

class DocumentServiceError(Exception):
    pass 

class DocumentService(LoggerMixin):
    def __init__(self, parser_registry: ParserRegistry = None): 
        self.settings = get_settings() 
        self.logger.info("Initilized DocumentService") 
        self.documents: Dict[str, Document] = {} 
        self.collections: Dict[str, DocumentCollection] = {} 
        self.parser_registry = parser_registry or ParserRegistry() 

        self.stats = {
            "documents_processed": 0,
            "total_chuncks_created": 0,
            "processing_failures": 0,
            "last_activity": None 
        }

    async def add_document(self, file_path: str, user_id: str, processing_options: Optional[Dict[str, Any]] = None) -> Document:
        try:
            self.logger.info(f"Adding document: {file_path}")
            abs_file_path = str(Path(os.getcwd()) / file_path)
            if not Path(abs_file_path).exists():
                raise DocumentServiceError(f"File not found {file_path}")

            options = ProcessingOptions(**processing_options) if processing_options else ProcessingOptions()
            document_id = str(uuid.uuid4())
            file_format = self._determine_format(file_path)
            mime_type = self._get_mime_type(file_path)
            file_size = Path(abs_file_path).stat().st_size

            # Initialize document with status UPLOADED
            document = Document(
                document_id=document_id,
                file_name=Path(file_path).name,
                original_path=abs_file_path,
                file_size=file_size,
                file_format=file_format,
                mime_type=mime_type,
                file_hash="",  # Will be updated
                user_id=user_id,
                processing_options=options,
                metadata=DocumentMetadata(),
                chunks=[],
                total_chunks=0,
                processing_time=0.0
            )
            document.update_status(ProcessingStatus.UPLOADED)

            parser = self.parser_registry.get_parser(file_path, mime_type)
            if not parser:
                raise DocumentServiceError(f"No suitable parser found for: {file_path}")

            # Update status before parsing
            document.update_status(ProcessingStatus.PARSING)

            parse_result = await parser.parse(abs_file_path, document.document_id, options)
            if not parse_result.success:
                document.update_status(ProcessingStatus.FAILED)
                raise DocumentServiceError(f"Parsing failed: {parse_result.error_message}")

            file_hash = self._calculate_file_hash(abs_file_path)
            document.file_hash = file_hash

            existing_doc = self._find_document_by_hash(file_hash)
            if existing_doc:
                self.logger.warning(f"Duplicate document detected: {file_path}")
                return existing_doc

            # Update status before chunking
            document.update_status(ProcessingStatus.CHUNKING)

            document.metadata = parse_result.metadata or DocumentMetadata()
            document.chunks = parse_result.chunks
            document.total_chunks = len(parse_result.chunks)
            document.processing_time = parse_result.processing_time

            # Embedding can be set here if needed, otherwise skip to Indexed
            # document.update_status(ProcessingStatus.EMBEDDING)

            document.update_status(ProcessingStatus.INDEXED)
            self.documents[document_id] = document

            self.stats["documents_processed"] += 1
            self.stats["total_chuncks_created"] += len(document.chunks)
            self.stats["last_activity"] = datetime.now(UTC)

            self.logger.info(f"Document added successfully: {document_id} ({len(document.chunks)})")

            return document

        except Exception as e:
            self.logger.error(f"Failed to add document {file_path}: {e}")
            raise DocumentServiceError(str(e))  

    def get_document(self, document_id: str) -> Optional[Document]:
        return self.documents.get(document_id) 
    
    def list_documents(self, user_id: Optional[str] = None, status: Optional[ProcessingStatus] = None) -> List[Document]:
        documents = list(self.documents.values()) 

        if user_id: 
            documents = [doc for doc in documents if doc.user_id == user_id]

        if status:
            documents = [doc for doc in documents if doc.status == status] 

        return documents
    
    def delete_document(self, document_id: str) -> str:
        try:
            document = self.documents.get(document_id)
            if not document:
                return False 
            else:
                self.logger.info("Not yet implemented the deletion part") 
        except Exception as e:
            self.logger.error(f"Failed to delete document {document_id}: {e}") 
            return False 

    def _determine_format(self, file_path: str):
        """Determines the document format from file extension."""
        extension = Path(file_path).suffix.lower().lstrip(".")

        format_mapping = {
            'pdf': DocumentFormat.PDF,
            'docx': DocumentFormat.DOCX,
            'pptx': DocumentFormat.PPTX,
            'csv': DocumentFormat.CSV,
            'txt': DocumentFormat.TXT,
            'md': DocumentFormat.MARKDOWN,
            'markdown': DocumentFormat.MARKDOWN
        }

        return format_mapping.get(extension, DocumentFormat.TXT)  
    
    def _get_mime_type(self, file_path: str) -> str:
        """Get MIME type for file."""
        extension = Path(file_path).suffix.lower().lstrip('.')

        mime_types = {
            'pdf': 'application/pdf',
            'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            'csv': 'text/csv',
            'txt': 'text/plain',
            'md': 'text/markdown',
            'markdown': 'text/markdown'
        }
        
        return mime_types.get(extension, 'application/octet-stream')

    def _calculate_file_hash(self, file_path: str) -> str: 
        """Calcuate SHA-256 hash of file."""
        hash_sha256 = hashlib.sha256() 
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)

        return hash_sha256.hexdigest() 
    
    def _find_document_by_hash(self, file_hash: str) -> Optional[Document]:
        """Find existing document with same hash."""
        
        for document in self.documents.values():
            if document.file_hash == file_hash:
                return document
        
        return None