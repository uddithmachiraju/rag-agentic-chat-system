import uuid
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from src.config.logging import LoggerMixin
from src.services.document_service import DocumentService, DocumentServiceError

document_router = APIRouter()
document_service = DocumentService()

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

            # Ingest and index the document
            doc = await document_service.add_document(str(file_path), user_id=user_id)
            self.logger.info(f"Document processed: {doc.document_id}, Chunks: {doc.total_chunks}")

            return JSONResponse({
                "user_id": user_id,
                "document_id": doc.document_id,
                "file_name": doc.file_name,
                "total_chunks": doc.total_chunks,
                "status": "success"
            })

        except DocumentServiceError as e:
            self.logger.error(f"DocumentServiceError: {e}")
            raise HTTPException(status_code=400, detail=str(e))

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
            docs = document_service.list_documents(user_id = user_id)
            self.logger.info(f"Getting the list of documents for {user_id}")
            return [
                {
                    "document_id": doc.document_id,
                    "file_name": doc.file_name,
                    "total_chunks": doc.total_chunks,
                    "status": doc.status,
                    "file_hash": doc.file_hash 
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
            doc = document_service.get_document(document_id)
            if not doc:
                self.logger.error(f"No document found with {document_id}")
                raise HTTPException(404, "Document not found")
            self.logger.error(f"Getting the content from {document_id}") 
            return {
                "document_id" : doc.document_id,
                "file_name": doc.file_name,
                "total_chunks": doc.total_chunks,
                "status": doc.status,
                "metadata": doc.metadata.model_dump() if doc.metadata else {},
                "chunks": [chunk.content[:200] + "...." for chunk in doc.chunks[:5]]
            }
        except Exception as e:
            self.logger.error(f"An error occurred while listing documents: {e}")
            return {
                "status": f"Error {str(e)}" 
            }
        
doc_handler = DocumentHandler() 
        
@document_router.get("/documents/{user_id}")
async def list_documents(user_id: str = None):
    return await doc_handler.list_documents(user_id) 

@document_router.get("/document/{document_id}")
async def get_document(document_id: str):
    return await doc_handler.get_document(document_id) 