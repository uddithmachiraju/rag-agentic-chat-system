import asyncio 
import time 
import json 
from pathlib import Path 
from typing import Dict, Any, List, Optional, Type, Union
import hashlib 
import shutil 
from datetime import datetime

from ..config.logging import LoggerMixin
from ..config.settings import get_settings 

from ..models.document import (
    Document, DocumentChunk, DocumentFormat, ProcessingStatus, 
    ProcessingOptions, DocumentMetadata
)

from ..mcp.protocol import (MCPMessage, MessageType, AgentType, 
    create_response_message, create_error_message, create_mcp_message,
    DocumentProcessingRequest, DocumentProcessingResponse
)

from ..parsers.registry import (
    BaseParser, PDFParser, ParserRegistry,
)

from ..parsers.base_parser import ParseResult

from ..agents.base_agent import BaseAgent 

from ..vector_store.embeddings import GeminiEmbeddingService
from ..vector_store.chroma_store import ChromaVectorStore

class IngestionError(Exception):
    """Custom exception for ingestion process errors."""
    pass

class DocumentValidator:
    """Validates documents before processing."""
    
    def __init__(self):
        self.max_file_size = 100 * 1024 * 1024  # 100MB default
        self.allowed_extensions = {
            'pdf', 'docx', 'pptx', 'csv', 'txt', 'md', 'markdown'
        }
        self.dangerous_extensions = {
            'exe', 'bat', 'cmd', 'com', 'pif', 'scr', 'vbs', 'js'
        }
    
    def validate_file(self, file_path: str, options: ProcessingOptions = None) -> Dict[str, Any]:
        """Validate file for processing."""
        
        validation_result = {
            'valid': False,
            'errors': [],
            'warnings': [],
            'file_info': {}
        }
        
        try:
            file_path_obj = Path(file_path)
            
            # Check if file exists
            if not file_path_obj.exists():
                validation_result['errors'].append("File does not exist")
                return validation_result
            
            if not file_path_obj.is_file():
                validation_result['errors'].append("Path is not a file")
                return validation_result
            
            # Get file info
            file_stat = file_path_obj.stat()
            file_extension = file_path_obj.suffix.lower().lstrip('.')
            
            validation_result['file_info'] = {
                'size': file_stat.st_size,
                'extension': file_extension,
                'created': datetime.fromtimestamp(file_stat.st_ctime),
                'modified': datetime.fromtimestamp(file_stat.st_mtime),
                'name': file_path_obj.name
            }
            
            # Check file size
            max_size = options.max_file_size_mb * 1024 * 1024 if options else self.max_file_size
            if file_stat.st_size > max_size:
                validation_result['errors'].append(
                    f"File size ({file_stat.st_size:,} bytes) exceeds maximum allowed "
                    f"({max_size:,} bytes)"
                )
            
            # Check extension
            if file_extension not in self.allowed_extensions:
                validation_result['errors'].append(f"File extension '{file_extension}' not supported")
            
            if file_extension in self.dangerous_extensions:
                validation_result['errors'].append(f"File extension '{file_extension}' is potentially dangerous")
            
            # Check if file is empty
            if file_stat.st_size == 0:
                validation_result['errors'].append("File is empty")
            
            # Additional format-specific validation
            format_validation = self._validate_file_format(file_path, file_extension)
            validation_result['warnings'].extend(format_validation.get('warnings', []))
            validation_result['errors'].extend(format_validation.get('errors', []))
            
            # Set valid flag
            validation_result['valid'] = len(validation_result['errors']) == 0
            
        except Exception as e:
            validation_result['errors'].append(f"Validation failed: {str(e)}")
        
        return validation_result
    
    def _validate_file_format(self, file_path: str, extension: str) -> Dict[str, List[str]]:
        """Perform format-specific validation."""
        
        result = {'warnings': [], 'errors': []}
        
        try:
            if extension == 'pdf':
                # Basic PDF validation
                with open(file_path, 'rb') as f:
                    header = f.read(8)
                    if not header.startswith(b'%PDF-'):
                        result['errors'].append("File does not appear to be a valid PDF")
            
            elif extension in ['docx', 'pptx']:
                # Check if it's a valid zip file (Office docs are zip-based)
                import zipfile
                try:
                    with zipfile.ZipFile(file_path, 'r') as zf:
                        # Check for required Office document files
                        files = zf.namelist()
                        if extension == 'docx' and 'word/document.xml' not in files:
                            result['warnings'].append("May not be a valid DOCX file")
                        elif extension == 'pptx' and 'ppt/presentation.xml' not in files:
                            result['warnings'].append("May not be a valid PPTX file")
                except zipfile.BadZipFile:
                    result['errors'].append(f"File is not a valid {extension.upper()} file")
            
            elif extension == 'csv':
                # Basic CSV validation
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        sample = f.read(1024)
                        if not any(delimiter in sample for delimiter in [',', ';', '\t', '|']):
                            result['warnings'].append("File may not contain valid CSV data")
                except Exception:
                    result['warnings'].append("Could not validate CSV format")
            
        except Exception as e:
            result['warnings'].append(f"Format validation failed: {str(e)}")
        
        return result


class DocumentStorage:
    """Manages document storage and organization."""
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.original_files_path = self.storage_path / "originals"
        self.processed_files_path = self.storage_path / "processed"
        self.metadata_path = self.storage_path / "metadata"
        
        for path in [self.original_files_path, self.processed_files_path, self.metadata_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    def store_original_file(self, source_path: str, document_id: str) -> str:
        """Store original file and return storage path."""
        
        source_path_obj = Path(source_path)
        extension = source_path_obj.suffix
        
        # Create storage filename
        stored_filename = f"{document_id}{extension}"
        stored_path = self.original_files_path / stored_filename
        
        # Copy file to storage
        shutil.copy2(source_path, stored_path)
        
        return str(stored_path)
    
    def get_document_path(self, document_id: str) -> Optional[str]:
        """Get path to stored document."""
        
        # Look for file with this document_id
        for file_path in self.original_files_path.iterdir():
            if file_path.stem == document_id:
                return str(file_path)
        
        return None
    
    def delete_document(self, document_id: str) -> bool:
        """Delete stored document and its metadata."""
        
        try:
            # Delete original file
            file_path = self.get_document_path(document_id)
            if file_path and Path(file_path).exists():
                Path(file_path).unlink()
            
            # Delete metadata file
            metadata_file = self.metadata_path / f"{document_id}.json"
            if metadata_file.exists():
                metadata_file.unlink()
            
            return True
            
        except Exception:
            return False
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        
        stats = {
            'total_files': 0,
            'total_size': 0,
            'files_by_extension': {},
            'oldest_file': None,
            'newest_file': None
        }
        
        try:
            oldest_time = float('inf')
            newest_time = 0
            
            for file_path in self.original_files_path.iterdir():
                if file_path.is_file():
                    stats['total_files'] += 1
                    
                    file_stat = file_path.stat()
                    stats['total_size'] += file_stat.st_size
                    
                    extension = file_path.suffix.lower()
                    stats['files_by_extension'][extension] = stats['files_by_extension'].get(extension, 0) + 1
                    
                    if file_stat.st_mtime < oldest_time:
                        oldest_time = file_stat.st_mtime
                        stats['oldest_file'] = file_path.name
                    
                    if file_stat.st_mtime > newest_time:
                        newest_time = file_stat.st_mtime
                        stats['newest_file'] = file_path.name
            
        except Exception as e:
            stats['error'] = str(e)
        
        return stats


class IngestionAgent(BaseAgent, LoggerMixin): 
    """Advanced document ingestion agent with comprehensive processing pipeline."""
    
    def __init__(self):
        super().__init__(AgentType.INGESTION)
        self.settings = get_settings()
        self.validator = DocumentValidator()
        self.parser_registry = ParserRegistry()
        self.storage = DocumentStorage(self.settings.upload_directory)
        
        # Processing state
        self.active_documents: Dict[str, Document] = {}
        self.processing_queue: asyncio.Queue = asyncio.Queue()
        self.processing_tasks: Dict[str, asyncio.Task] = {}

        # Embedding service
        self.embedding_service = GeminiEmbeddingService()
        self.vector_store = ChromaVectorStore() 
        
        # Statistics
        self.stats = {
            'documents_processed': 0,
            'documents_failed': 0,
            'total_chunks_created': 0,
            'total_processing_time': 0.0,
            'last_activity': None
        }
    
    async def _initialize_agent(self) -> bool:
        """Initialize the ingestion agent."""
        try:
            self.logger.info("Initializing IngestionAgent")
            
            # Validate storage directory
            if not self.storage.storage_path.exists():
                self.logger.error(f"Storage directory does not exist: {self.storage.storage_path}")
                return False
            
            # Start processing queue worker
            asyncio.create_task(self._process_queue_worker())
            
            self.logger.info("IngestionAgent initialized successfully")

            await self.vector_store.initialize() 
            self.logger.info("Vector store initialized successfully")

            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize IngestionAgent: {e}")
            return False
    
    async def _handle_message(self, message: MCPMessage) -> Optional[MCPMessage]:
        """Handle incoming MCP messages."""
        
        try:
            if message.message_type == MessageType.REQUEST:
                # Handle document processing request
                if message.payload.get('action') == 'process_document':
                    return await self._handle_process_document_request(message)
                
                elif message.payload.get('action') == 'get_processing_status':
                    return await self._handle_status_request(message)
                
                elif message.payload.get('action') == 'cancel_processing':
                    return await self._handle_cancel_request(message)
                
                elif message.payload.get('action') == 'get_stats':
                    return await self._handle_stats_request(message)
                
                elif message.payload.get('action') == 'get_document':
                    return await self.get_document(message) 

                elif message.payload.get('action') == 'list_documents':
                    return await self.list_all_documents(message)

                elif message.payload.get('action') == 'list_every_document':
                    return await self.list_every_document(message)
                
                else:
                    return create_error_message(
                        message, 
                        f"Unknown action: {message.payload.get('action')}"
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
            return create_error_message(message, str(e))
    
    async def _handle_process_document_request(self, message: MCPMessage) -> MCPMessage:
        """Handle document processing request."""
        
        try:
            # Extract request data
            request_data = message.payload
            
            # Validate required fields
            if 'file_path' not in request_data:
                return create_error_message(message, "Missing required field: file_path")
            
            file_path = request_data['file_path']
            file_name = request_data.get('file_name', Path(file_path).name)
            user_id = request_data.get('user_id')
            
            # Parse processing options
            options_data = request_data.get('processing_options', {})
            options = ProcessingOptions(**options_data)
            
            # Create processing request
            processing_request = DocumentProcessingRequest(
                file_path=file_path,
                file_name=file_name,
                file_type=Path(file_path).suffix.lower().lstrip('.'),
                user_id=user_id,
                processing_options=options_data
            )
            
            # Add to processing queue
            document_id = await self._queue_document_processing(processing_request, options)
            
            response_payload = {
                'document_id': document_id,
                'status': 'queued',
                'message': 'Document queued for processing'
            }
            
            return create_response_message(message, response_payload)
            
        except Exception as e:
            self.logger.error(f"Error handling process document request: {e}")
            return create_error_message(message, str(e))
    
    async def _handle_status_request(self, message: MCPMessage) -> MCPMessage:
        """Handle processing status request."""
        
        try:
            document_id = message.payload.get('document_id')
            
            if not document_id:
                return create_error_message(message, "Missing document_id")
            
            if document_id in self.active_documents:
                document = self.active_documents[document_id]
                
                response_payload = {
                    'document_id': document_id,
                    'status': document.status.value,
                    'progress': {
                        'total_chunks': document.total_chunks,
                        'processing_time': document.processing_time,
                        'error_message': document.error_message
                    }
                }
            else:
                # Try to load from storage and report status if present
                stored_doc = await self.load_document_from_storage(document_id)
                if stored_doc:
                    response_payload = {
                        'document_id': document_id,
                        'status': stored_doc.status.value if hasattr(stored_doc.status, 'value') else str(stored_doc.status),
                        'progress': {
                            'total_chunks': stored_doc.total_chunks,
                            'processing_time': stored_doc.processing_time,
                            'error_message': stored_doc.error_message
                        }
                    }
                else:
                    response_payload = {
                        'document_id': document_id,
                        'status': 'not_found',
                        'message': 'Document not found in active processing or storage'
                    }
            
            return create_response_message(message, response_payload)
            
        except Exception as e:
            self.logger.error(f"Error handling status request: {e}")
            return create_error_message(message, str(e))
    
    async def _handle_cancel_request(self, message: MCPMessage) -> MCPMessage:
        """Handle processing cancellation request."""
        
        try:
            document_id = message.payload.get('document_id')
            
            if not document_id:
                return create_error_message(message, "Missing document_id")
            
            # Cancel processing task if it exists
            cancelled = False
            if document_id in self.processing_tasks:
                task = self.processing_tasks[document_id]
                if not task.done():
                    task.cancel()
                    cancelled = True
                    
                    # Update document status
                    if document_id in self.active_documents:
                        self.active_documents[document_id].update_status(
                            ProcessingStatus.FAILED, 
                            "Processing cancelled by user"
                        )
            
            response_payload = {
                'document_id': document_id,
                'cancelled': cancelled,
                'message': 'Processing cancelled' if cancelled else 'No active processing found'
            }
            
            return create_response_message(message, response_payload)
            
        except Exception as e:
            self.logger.error(f"Error handling cancel request: {e}")
            return create_error_message(message, str(e))
    
    async def _handle_stats_request(self, message: MCPMessage) -> MCPMessage:
        """Handle statistics request."""
        
        try:
            # Get current statistics
            current_stats = self.stats.copy()
            current_stats['active_documents'] = len(self.active_documents)
            current_stats['queue_size'] = self.processing_queue.qsize()
            current_stats['storage_stats'] = self.storage.get_storage_stats()
            current_stats['supported_formats'] = self.parser_registry.get_supported_extensions()
            
            return create_response_message(message, current_stats)
            
        except Exception as e:
            self.logger.error(f"Error handling stats request: {e}")
            return create_error_message(message, str(e))
    
    async def _queue_document_processing(
        self, 
        request: DocumentProcessingRequest, 
        options: ProcessingOptions
    ) -> str:
        """Queue document for processing."""
        
        # Validate file
        validation_result = self.validator.validate_file(request.file_path, options)
        
        if not validation_result['valid']:
            raise IngestionError(f"File validation failed: {'; '.join(validation_result['errors'])}")
        
        # Create document record
        document = Document(
            file_name=request.file_name,
            original_path=request.file_path,
            file_size=validation_result['file_info']['size'],
            file_format=DocumentFormat(request.file_type),
            mime_type=self._get_mime_type(request.file_type),
            file_hash=self._calculate_file_hash(request.file_path),
            user_id=request.user_id,
            processing_options=options
        )
        
        # Store in active documents
        self.active_documents[document.document_id] = document
        
        # Add to processing queue
        await self.processing_queue.put((document, request))
        
        self.logger.info(f"Document queued for processing: {document.document_id}")
        
        return document
    
    async def _process_queue_worker(self):
        """Background worker to process queued documents."""
        
        while True:
            try:
                # Get next document from queue
                document, request = await self.processing_queue.get()
                
                # Create processing task
                task = asyncio.create_task(
                    self._process_document(document, request)
                )
                
                self.processing_tasks[document.document_id] = task
                
                # Wait for completion and cleanup
                try:
                    await task
                finally:
                    self.processing_tasks.pop(document.document_id, None)
                    self.processing_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in queue worker: {e}")
                await asyncio.sleep(1)  # Brief pause before continuing
    
    async def _process_document(self, document: Document, request: DocumentProcessingRequest):
        """Process a single document."""
        
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting document processing: {document.document_id}")
            
            # Update status to parsing
            document.update_status(ProcessingStatus.PARSING)
            
            # Store original file
            stored_path = self.storage.store_original_file(
                request.file_path, 
                document.document_id
            )
            document.stored_path = stored_path
            
            # Get appropriate parser
            parser = self.parser_registry.get_parser(request.file_path)
            if not parser:
                raise IngestionError(f"No parser available for file type: {request.file_type}")
            
            # Parse document
            parse_start = time.time()
            parse_result = await parser.parse(request.file_path, document_id = document.document_id, options = document.processing_options)
            document.parsing_time = time.time() - parse_start
            
            if not parse_result.success:
                raise IngestionError(f"Parsing failed: {parse_result.error_message}")
            
            # Update status to chunking
            document.update_status(ProcessingStatus.CHUNKING)
            
            # Process chunks
            chunk_start = time.time()
            await self._process_chunks(document, parse_result)
            document.chunking_time = time.time() - chunk_start
            
            # Update metadata
            if parse_result.metadata:
                document.metadata = parse_result.metadata
        
            # Update statistics
            document.processing_time = time.time() - start_time
            self.stats['documents_processed'] += 1
            self.stats['total_chunks_created'] += document.total_chunks
            self.stats['total_processing_time'] += document.processing_time
            self.stats['last_activity'] = datetime.utcnow()
            
            self.logger.info(
                f"Document processing completed: {document.document_id} "
                f"({document.total_chunks} chunks in {document.processing_time:.2f}s)"
            )

            document.update_status(ProcessingStatus.EMBEDDING)

            if not self.vector_store.is_initialized:
                await self.vector_store.initialize() 

            embedding_tasks = [self.embedding_service.generate_embedding(chunk.content) for chunk in document.chunks]
            embeddings = await asyncio.gather(*embedding_tasks)
            for chunk, embedding in zip(document.chunks, embeddings):
                chunk.embedding_vector = embedding
                chunk.embedding_model = self.embedding_service.model_name 

            # Store embeddings in vector store  
            items = []
            for chunk in document.chunks:
                items.append(
                    {
                        "id": chunk.chunk_id,
                        "embedding": chunk.embedding_vector,
                        "metadata": {
                            "document_id": document.document_id,
                            "file_name": document.file_name,
                            "chunk_index": chunk.chunk_index,
                            "content": chunk.content,
                            "created_at": chunk.created_at.isoformat()
                        }
                    }
                )
            await self.vector_store.add_documents(chunks = document.chunks, embeddings = embeddings)

            document.update_status(ProcessingStatus.INDEXED)

            await self.save_document_metadata(document) 
                                
            # Send completion notification
            await self._send_processing_completion_notification(document)
            
        except Exception as e:
            error_message = str(e)
            self.logger.error(f"Document processing failed: {document.document_id} - {error_message}")
            
            # Update document status
            document.update_status(ProcessingStatus.FAILED, error_message)
            document.processing_time = time.time() - start_time
            
            # Update statistics
            self.stats['documents_failed'] += 1
            self.stats['last_activity'] = datetime.utcnow()
            
            # Send failure notification
            await self._send_processing_failure_notification(document, error_message)
        
        finally:
            # Clean up active document after a delay (keep for status queries)
            asyncio.create_task(self._cleanup_document(document.document_id, delay=300))  # 5 minutes
    
    async def _process_chunks(self, document: Document, parse_result: ParseResult):
        """Process and validate document chunks."""
        self.logger.info(f"Processing {len(parse_result.chunks)} chunks for document {document.document_id}")
        for chunk in parse_result.chunks:
            # Set document ID
            chunk.document_id = document.document_id
            
            # Validate chunk content
            if not chunk.content or not chunk.content.strip():
                self.logger.warning(f"Empty chunk detected in document {document.document_id}")
                continue
            
            # Add chunk to document
            document.add_chunk(chunk)
        
        self.logger.info(f"Processed {len(document.chunks)} chunks for document {document.document_id}")
    
    async def _send_processing_completion_notification(self, document: Document):
        """Send notification when processing is complete."""
        
        try:
            # Create notification message
            notification_payload = {
                'event': 'document_processed',
                'document_id': document.document_id,
                'status': document.status.value,
                'chunks_created': document.total_chunks,
                'processing_time': document.processing_time,
                'file_name': document.file_name
            }
            
            # Send to coordinator or other interested agents
            # Build a proper NOTIFICATION message addressed to the Coordinator
            notification_message = create_mcp_message(
                sender=self.agent_type,
                receiver=AgentType.COORDINATOR,
                message_type=MessageType.NOTIFICATION,
                payload=notification_payload
            )

            await self.send_message(notification_message)
            
        except Exception as e:
            self.logger.error(f"Failed to send completion notification: {e}")
    
    async def _send_processing_failure_notification(self, document: Document, error_message: str):
        """Send notification when processing fails."""
        
        try:
            # Create failure notification
            notification_payload = {
                'event': 'document_processing_failed',
                'document_id': document.document_id,
                'status': document.status.value,
                'error_message': error_message,
                'processing_time': document.processing_time,
                'file_name': document.file_name
            }
            notification_message = create_mcp_message(
                sender=self.agent_type,
                receiver=AgentType.COORDINATOR,
                message_type=MessageType.NOTIFICATION,
                payload=notification_payload
            )

            await self.send_message(notification_message)
            
        except Exception as e:
            self.logger.error(f"Failed to send failure notification: {e}")
    
    async def _cleanup_document(self, document_id: str, delay: int = 300):
        """Clean up document from active processing after delay."""
        
        await asyncio.sleep(delay)
        
        try:
            if document_id in self.active_documents:
                document = self.active_documents.pop(document_id)
                self.logger.debug(f"Cleaned up document from active processing: {document_id}")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up document {document_id}: {e}")
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file."""
        
        hash_sha256 = hashlib.sha256()
        
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            
            return hash_sha256.hexdigest()
            
        except Exception as e:
            self.logger.error(f"Failed to calculate file hash: {e}")
            return ""
    
    def _get_mime_type(self, file_extension: str) -> str:
        """Get MIME type for file extension."""
        
        mime_types = {
            'pdf': 'application/pdf',
            'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            'csv': 'text/csv',
            'txt': 'text/plain',
            'md': 'text/markdown',
            'markdown': 'text/markdown'
        }
        
        return mime_types.get(file_extension.lower(), 'application/octet-stream')
    
    async def _agent_health_check(self) -> Optional[Dict[str, Any]]:
        """Perform agent-specific health check."""
        
        health_info = {
            'storage_accessible': self.storage.storage_path.exists(),
            'queue_size': self.processing_queue.qsize(),
            'active_documents': len(self.active_documents),
            'active_tasks': len(self.processing_tasks),
            'supported_formats': len(self.parser_registry.get_supported_extensions()),
            'statistics': self.stats.copy()
        }
        
        # Check parser availability
        parser_health = {}
        for ext in self.parser_registry.get_supported_extensions():
            try:
                parser = self.parser_registry.parsers[ext]()
                parser_health[ext] = True
            except Exception as e:
                parser_health[ext] = False
        
        health_info['parser_availability'] = parser_health
        
        return health_info
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get comprehensive processing summary."""
        
        summary = {
            'statistics': self.stats.copy(),
            'active_processing': {
                'documents': len(self.active_documents),
                'queue_size': self.processing_queue.qsize(),
                'active_tasks': len(self.processing_tasks)
            },
            'storage': self.storage.get_storage_stats(),
            'supported_formats': self.parser_registry.get_supported_extensions(),
            'current_documents': []
        }
        
        # Add current document summaries
        for doc_id, document in self.active_documents.items():
            summary['current_documents'].append({
                'document_id': doc_id,
                'file_name': document.file_name,
                'status': document.status.value,
                'chunks': document.total_chunks,
                'processing_time': document.processing_time,
                'uploaded_at': document.uploaded_at.isoformat()
            })
        
        return summary
    
    async def ingest_document(self, file_path: Union[str, Path], options: Optional[ProcessingOptions] = None) -> Dict[str, Any]:
        options = options or ProcessingOptions()
        file_path_str = str(file_path)
        file_name = Path(file_path_str).name
        file_extension = Path(file_path_str).suffix.lower().lstrip('.')

        processing_request = DocumentProcessingRequest(
            file_path=file_path_str,
            file_name=file_name,
            file_type=file_extension,
            processing_options=options.dict() if hasattr(options, "dict") else {},
        )

        # Validate file
        validation_result = self.validator.validate_file(file_path_str, options)
        if not validation_result['valid']:
            raise IngestionError(f"Validation errors: {', '.join(validation_result['errors'])}")

        # Create Document instance
        document = Document(
            file_name=file_name,
            original_path=file_path_str,
            file_size=validation_result['file_info']['size'],
            file_format=DocumentFormat(file_extension),
            mime_type=self._get_mime_type(file_extension),
            file_hash=self._calculate_file_hash(file_path_str),
            user_id=processing_request.user_id,
            processing_options=options,
        )
        self.active_documents[document.document_id] = document

        # Queue or process directly
        document_id = document.document_id
        await self.processing_queue.put((document, processing_request))

        task = self.processing_tasks.get(document_id)
        if task:
            await task  # wait for background processing
        else:
            # fallback synchronous direct processing
            await self._process_document(document, processing_request)

        # Return detailed info after processing
        processed_doc = self.active_documents.get(document_id)
        return {
            "document_id": processed_doc.document_id,
            "file_name": processed_doc.file_name,
            "status": processed_doc.status.value,
            "total_chunks": processed_doc.total_chunks,
            "processing_time": processed_doc.processing_time,
            "parsing_time": processed_doc.parsing_time,
            "chunking_time": processed_doc.chunking_time,
            "embedding_time": processed_doc.embedding_time,
            "error_message": processed_doc.error_message or "",
        }
    
    async def save_document_metadata(self, document: Document):
        """Save document metadata to persistent storage."""
        try:
            metadata_file = self.storage.metadata_path / f"{document.document_id}.json"
            self.logger.info(f"Saving metadata to: {metadata_file}")
            document_data = {
                "document_id": document.document_id,
                "file_name": document.file_name,
                "status": document.status.value,
                "total_chunks": document.total_chunks,
                "processing_time": getattr(document, 'processing_time', 0),
                "parsing_time": getattr(document, 'parsing_time', 0),
                "chunking_time": getattr(document, 'chunking_time', 0),
                "embedding_time": getattr(document, 'embedding_time', 0),
                "file_size": document.file_size,
                "file_format": document.file_format.value,
                "mime_type": document.mime_type,
                "file_hash": document.file_hash,
                "user_id": document.user_id,
                "uploaded_at": document.uploaded_at.isoformat(),
                "processed_at": document.processed_at.isoformat(),
                "error_message": getattr(document, 'error_message', None),
                "metadata": document.metadata.model_dump() if document.metadata else {},
                "chunks": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "chunk_index": chunk.chunk_index,
                        "content": chunk.content,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                        "page_number": chunk.page_number,
                        "embedding_vector": chunk.embedding_vector,
                        "embedding_model": chunk.embedding_model,
                        "created_at": chunk.created_at.isoformat()
                    }
                    for chunk in document.chunks
                ]
            }

            # Ensure the metadata directory exists
            self.storage.metadata_path.mkdir(parents=True, exist_ok=True)

            with open(metadata_file, 'w') as f:
                json.dump(document_data, f, indent=4, default=str)
                
            self.logger.info(f"Saved metadata for document {document.document_id}")
            
        except Exception as e:
            self.logger.error(f"Error saving document metadata: {e}")

    async def load_document_from_storage(self, document_id: str) -> Optional[Document]:
        """Load a document from persistent storage."""
        try:
            metadata_file = self.storage.metadata_path / f"{document_id}.json"
            if not metadata_file.exists():
                return None
            
            with open(metadata_file, 'r') as f:
                doc_data = json.load(f)
            
            # Reconstruct document object
            document = Document(
                document_id=doc_data["document_id"],
                file_name=doc_data["file_name"],
                original_path=self.storage.get_document_path(doc_data["document_id"]),
                file_size=doc_data["file_size"],
                file_format=DocumentFormat(doc_data["file_format"]),
                mime_type=doc_data["mime_type"],
                file_hash=doc_data["file_hash"],
                user_id=doc_data["user_id"]
            )
            
            # Restore timestamps
            document.uploaded_at = datetime.fromisoformat(doc_data["uploaded_at"])
            if doc_data.get("processed_at"):
                document.processed_at = datetime.fromisoformat(doc_data["processed_at"])
            
            # Restore status and timing
            document.status = ProcessingStatus(doc_data["status"])
            document.processing_time = doc_data.get("processing_time", 0)
            document.parsing_time = doc_data.get("parsing_time", 0)
            document.chunking_time = doc_data.get("chunking_time", 0)
            document.embedding_time = doc_data.get("embedding_time", 0)
            document.error_message = doc_data.get("error_message")
            
            # Restore metadata
            if doc_data.get("metadata"):
                document.metadata = DocumentMetadata(**doc_data["metadata"])
            
            # Restore chunks
            for chunk_data in doc_data.get("chunks", []):
                chunk = DocumentChunk(
                    chunk_id=chunk_data["chunk_id"],
                    document_id=chunk_data["document_id"],
                    chunk_index=chunk_data["chunk_index"],
                    content=chunk_data["content"],
                    start_char=chunk_data.get("start_char", 0),
                    end_char=chunk_data.get("end_char", 0),
                    page_number=chunk_data.get("page_number"),
                    section=chunk_data.get("section"),
                    embedding_vector=chunk_data.get("embedding_vector"),
                    embedding_model=chunk_data.get("embedding_model")
                )
                chunk.created_at = datetime.fromisoformat(chunk_data["created_at"])
                document.chunks.append(chunk)
            
            document.total_chunks = len(document.chunks)
            
            self.logger.info(f"Successfully loaded document {document_id} from storage")
            return document
            
        except Exception as e:
            self.logger.error(f"Error loading document from storage: {e}")
            return None
        
    async def get_document(self, message: MCPMessage) -> Optional[Document]:
        """Get document from active memory or load from storage."""
        document_id = message.payload.get("document_id")
        # First check active documents
        if document_id in self.active_documents:
            document = self.active_documents[document_id]
            return create_response_message(message, document.model_dump())

        # Then try to load from storage
        loaded_doc = await self.load_document_from_storage(document_id)
        if loaded_doc:
            return create_response_message(message, loaded_doc.model_dump())
        else:
            return create_response_message(message, None)

    async def list_all_documents(self, message: MCPMessage) -> MCPMessage:
        """List all documents (active + stored). Returns an MCPMessage with payload {'documents': [...]}

        Expects optional payload: {'user_id': <user_id>} to filter documents.
        """
        try:
            user_id = (message.payload or {}).get("user_id", "anonymous") 
            documents = []

            # Get active documents
            for doc_id, doc in self.active_documents.items():
                if user_id is None or doc.user_id == user_id:
                    documents.append(doc.model_dump())

            # Get stored documents
            try:
                for metadata_file in self.storage.metadata_path.glob("*.json"):
                    doc_id = metadata_file.stem

                    stored_doc = await self.load_document_from_storage(doc_id)
                    if stored_doc and (user_id == "anonymous" or stored_doc.user_id == user_id):
                        documents.append(stored_doc.model_dump())

            except Exception as e:
                self.logger.error(f"Error listing stored documents: {e}")

            return create_response_message(message, {"documents": documents})

        except Exception as e:
            self.logger.exception(f"Failed to list documents: {e}")
            return create_error_message(message, str(e))

    async def list_every_document(self, message: MCPMessage) -> MCPMessage:
        """Return every document in active memory and stored metadata without filtering by user."""
        try:
            documents = []

            # Active documents
            for doc_id, doc in self.active_documents.items():
                documents.append(doc.model_dump())

            # Stored documents
            try:
                for metadata_file in self.storage.metadata_path.glob("*.json"):
                    doc_id = metadata_file.stem
                    stored_doc = await self.load_document_from_storage(doc_id)
                    if stored_doc:
                        documents.append(stored_doc.model_dump())
            except Exception as e:
                self.logger.error(f"Error listing stored documents: {e}")

            return create_response_message(message, {"documents": documents})

        except Exception as e:
            self.logger.exception(f"Failed to list every document: {e}")
            return create_error_message(message, str(e))