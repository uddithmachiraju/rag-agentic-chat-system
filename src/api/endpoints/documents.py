import uuid
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from src.config.logging import LoggerMixin
from src.services.document_service import DocumentService, DocumentServiceError
from src.agents.ingestion import IngestionAgent

document_router = APIRouter()
# document_service = DocumentService()
ingestion_agent = IngestionAgent()

# Directory to store uploaded files
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

class UploadHandler(LoggerMixin):
    async def upload(self, file: UploadFile = File(...), user_id: str = Form("anonymous")):
        file_id = str(uuid.uuid4().hex[:6]) 
        file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
        
        self.logger.info(f"Received upload from user: {user_id}, saving to: {file_path}")

        try:
            with open(file_path, "wb") as dest:
                shutil.copyfileobj(file.file, dest)
            self.logger.info(f"File saved successfully: {file_path}")

            result = await ingestion_agent.ingest_document(str(file_path))
            self.logger.info(f"Document Processed: {result["document_id"]}") 

            return JSONResponse({
                "user_id": user_id,
                "document_id": result["document_id"],
                "file_name": result["file_name"],
                "total_chunks": result["total_chunks"],
                "status": result["status"]
            })

        except Exception as e:
            self.logger.exception("Unexpected error during upload")
            raise HTTPException(status_code=500, detail="Internal server error")

upload_handler = UploadHandler()

@document_router.post("/upload")
async def upload_endpoint(file: UploadFile = File(...), user_id: str = Form("anonymous")):
    return await upload_handler.upload(file=file, user_id=user_id)

class DocumentHandler(LoggerMixin):
    async def list_documents(self, user_id: str = None):
        try:
            docs = []
            for doc_id, doc in ingestion_agent.active_documents.items():
                if user_id is None or doc.user_id == user_id:
                    docs.append(doc)
            self.logger.info(f"Found {len(docs)} documents in ingestion agent.")
            return [
                {
                    "document_id": doc.document_id,
                    "file_name": doc.file_name,
                    "total_chunks": doc.total_chunks,
                    "status": doc.status,
                    "file_hash": doc.file_hash,
                    "metadata": doc.metadata.model_dump() if doc.metadata else {},
                    "uploaded_at": doc.uploaded_at.isoformat() if doc.uploaded_at else None,
                    "processing_time": doc.processing_at.total_seconds() if doc.processing_time else None,
                    "user_id": doc.user_id
                }
                for doc in docs 
            ]
        except Exception as e:
            self.logger.error(f"An error occurred while listing documents: {e}")
            return {
                "status" : f"Error: {str(e)}" 
            }
        
    async def get_document(self, document_id: str):
        try:
            doc = None 
            if document_id in ingestion_agent.active_documents:
                doc = ingestion_agent.active_documents[document_id]
                self.logger.info(f"Found document in active documents: {document_id}")
            else:
                doc = await self._load_document_from_storage(document_id)
                if doc:
                    self.logger.info(f"Loaded document from storage: {document_id}") 

            if not doc:
                self.logger.error(f"No document found with {document_id}")
                raise HTTPException(404, "Document not found")
            self.logger.error(f"Getting the content from {document_id}") 
            response = {
                "document_id": doc.document_id,
                "file_name": doc.file_name,
                "total_chunks": doc.total_chunks,
                "status": doc.status.value if hasattr(doc.status, 'value') else str(doc.status),
                "file_size": doc.file_size,
                "file_format": doc.file_format.value if hasattr(doc.file_format, 'value') else str(doc.file_format),
                "mime_type": doc.mime_type,
                "file_hash": doc.file_hash,
                "user_id": doc.user_id,
                "uploaded_at": doc.uploaded_at.isoformat(),
                "processing_time": getattr(doc, 'processing_at', 0),
                "parsing_time": getattr(doc, 'parsing_time', 0),
                "chunking_time": getattr(doc, 'chunking_time', 0),
                "metadata": doc.metadata.model_dump() if doc.metadata else {},
                "chunks": doc.chunks 
            }
            return response 
        except Exception as e:
            self.logger.error(f"An error occurred while listing documents: {e}")
            return {
                "status": f"Error {str(e)}" 
            }
        
    async def _load_document_from_storage(self, document_id: str):
        try:
            stored_path = ingestion_agent.storage.get_document_path(document_id)
            if not stored_path:
                return None
            
            metadata_file = ingestion_agent.storage.metadata_path / f"{document_id}.json"
            if metadata_file.exists():
                import json
                with open(metadata_file, 'r') as f:
                    doc_data = json.load(f)
                
                self.logger.info(f"Found metadata for document {document_id}, but reconstruction not implemented")
                return None
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error loading document from storage: {e}")
            return None
        
    async def ingest_document(self, file: UploadFile = File(...)):
        file_id = str(uuid.uuid4().hex[:6]) 
        file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
        
        try:
            with open(file_path, "wb") as dest:
                shutil.copyfileobj(file.file, dest)
            self.logger.info(f"File saved successfully: {file_path}")

            # Ingest and index the document
            result = await ingestion_agent.ingest_document(file_path)
            return {"result": result}
        
        except DocumentServiceError as e:
            self.logger.error(f"Ingestion Agent Error: {e}")
            raise HTTPException(status_code=400, detail=str(e))

        except Exception as e:
            self.logger.exception("Unexpected error during upload")
            raise HTTPException(status_code=500, detail="Internal server error")
        
    async def get_processing_status(self, document_id: str):
        try:
            if document_id in ingestion_agent.active_documents:
                doc = ingestion_agent.active_documents[document_id]
                return {
                    "document_id": document_id,
                    "status": doc.status, 
                    "processing_time": getattr(doc, 'processing_time', 0),
                    "parsing_time": getattr(doc, 'parsing_time', 0),
                    "chunking_time": getattr(doc, 'chunking_time', 0),
                    "total_chunks": doc.total_chunks,
                    "error_message": getattr(doc, 'error_message', None),
                    "file_name": doc.file_name,
                    "uploaded_at": doc.uploaded_at.isoformat()
                }
            else:
                # Check if document exists in storage
                stored_path = ingestion_agent.storage.get_document_path(document_id)
                if stored_path:
                    return {
                        "document_id": document_id, 
                        "status": "completed_and_stored",
                        "message": "Document processed and moved to storage"
                    }
                else:
                    return {
                        "document_id": document_id, 
                        "status": "not_found",
                        "message": "Document not found in active processing or storage"
                    }
        except Exception as e:
            self.logger.error(f"Error getting processing status: {e}")
            raise HTTPException(500, str(e))
        
    async def get_ingestion_stats(self):
        try:
            return ingestion_agent.get_processing_summary()
        except Exception as e:
            self.logger.error(f"Error getting ingestion stats: {e}")
            raise HTTPException(500, str(e))
        
    async def delete_document(self, document_id: str):
        try:
            if document_id in ingestion_agent.active_documents:
                del ingestion_agent.active_documents[document_id]
                self.logger.info(f"Removed document {document_id} from active processing")
            
            deleted = ingestion_agent.storage.delete_document(document_id)
            
            if deleted:
                return {
                    "document_id": document_id,
                    "status": "deleted",
                    "message": "Document successfully deleted"
                }
            else:
                raise HTTPException(404, f"Document {document_id} not found for deletion")
                
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error deleting document {document_id}: {e}")
            raise HTTPException(500, str(e))
        
doc_handler = DocumentHandler() 
        
@document_router.get("/documents/{user_id}")
async def list_documents(user_id: str = None):
    return await doc_handler.list_documents(user_id) 

@document_router.get("/document/{document_id}")
async def get_document(document_id: str):
    return await doc_handler.get_document(document_id) 

@document_router.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    return await doc_handler.ingest_document(file=file)

@document_router.get("/document/{document_id}/status")
async def get_document_status(document_id: str):
    return await doc_handler.get_processing_status(document_id)

@document_router.delete("/document/{document_id}")
async def delete_document(document_id: str):
    return await doc_handler.delete_document(document_id)

@document_router.get("/ingestion/stats")
async def get_ingestion_stats():
    return await doc_handler.get_ingestion_stats()

# Debug endpoint to see what documents are available
@document_router.get("/debug/documents")
async def debug_documents():
    """Debug endpoint to see active documents and stats."""
    try:
        active_docs = []
        for doc_id, doc in ingestion_agent.active_documents.items():
            active_docs.append({
                "document_id": doc_id,
                "file_name": doc.file_name,
                "status": doc.status.value if hasattr(doc.status, 'value') else str(doc.status),
                "user_id": doc.user_id,
                "total_chunks": doc.total_chunks
            })
        
        storage_stats = ingestion_agent.storage.get_storage_stats()
        
        return {
            "active_documents": active_docs,
            "total_active": len(active_docs),
            "ingestion_stats": ingestion_agent.stats,
            "storage_stats": storage_stats,
            "queue_size": ingestion_agent.processing_queue.qsize(),
            "supported_formats": ingestion_agent.parser_registry.get_supported_extensions()
        }
    except Exception as e:
        return {"error": str(e)}