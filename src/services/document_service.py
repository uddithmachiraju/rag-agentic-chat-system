import asyncio 
import hashlib 
import logging 
import os 
from pathlib import Path 
from datetime import datetime, UTC
from typing import Optional, List, Dict, Any 
from ..config import get_settings, LoggerMixin

from src.models.document import (
    Document, DocumentCollection, DocumentMetadata, 
    DocumentChunk, ProcessingOptions, ProcessingStatus, 
    DocumentFormat
) 

class DocumentServiceError(Exception):
    pass 

class DocumentService(LoggerMixin):
    def __init__(self):
        self.settings = get_settings() 
        self.logger.info("Initilized DocumentService") 
        self.documents: Dict[str, Document] = {} 
        self.collections: Dict[str, DocumentCollection] = {} 

        self.stats = {
            "documents_processed": 0,
            "total_chuncks_created": 0,
            "processing_failures": 0,
            "last_activity": None 
        }

    async def add_document(self, file_path, user_id, processing_options: Optional[Dict[str, Any]] = None) -> Document:
        try:
            self.logger.info(f"Adding document: {file_path}") 
            if not Path(str(os.getcwd()) + "/" + file_path).exists():
                raise DocumentServiceError(f"File not found {file_path}") 
            
            # parse processing options
            options = None 
            if processing_options:
                options = ProcessingOptions(**processing_options)
            else:
                options = ProcessingOptions() 

            # Get appropiate parser
            parser = None 
            # if not parser:
            #     raise DocumentServiceError(f"No suitable parser found for: {file_path}") 

            # parse the document 
            parse_result = None 

            # if not parse_result.success:
            #     raise DocumentServiceError(f"Parsing failed: {parse_result.error_message}") 
            # Calculate the hash for any duplicates 
            file_hash = self._calculate_file_hash(file_path)  

            # Check for duplicates
            existing_doc = self._find_document_by_hash(file_hash)  

            if existing_doc:
                self.logger.warning(f"Duplicate document detected: {file_path}") 
                return existing_doc
            
            # Create a Document 
            document = Document(
                file_name = Path(file_path).name, 
                original_path = file_path,
                file_size = Path(file_path).stat().st_size,
                file_format = self._determine_format(file_path),
                mime_type = self._get_mime_type(file_path),
                file_hash = file_hash,
                user_id = user_id,
                processing_options = options
            )
            # print(str(os.getcwd()) + "/" + file_path)
            document.update_status(ProcessingStatus.INDEXED)
            # print(document)

            self.documents[document.document_id] = document

            self.stats["documents_processed"] += 1
            self.stats["total_chuncks_created"] += len(document.chunks) 
            self.stats["last_activity"] = datetime.now(UTC) 

            self.logger.info(f"Document added successfully: {document.document_id}" f"({len(document.chunks)})") 

            return document 
            
        except Exception as e:
            print(f"Error") 
            self.logger.info("Not yet completed the implementation")  

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