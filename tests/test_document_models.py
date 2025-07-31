import pytest 
from datetime import datetime 
from typing import Any, Dict 

from src.models.document import (
    Document, 
    DocumentChunk, 
    DocumentCollection, 
    DocumentFormat, 
    ProcessingStatus,
    ChunkType, 
    ProcessingOptions, 
    DocumentMetadata
)

def test_create_document():
    doc = Document(
        file_name = "example.pdf",
        original_path = "data/example.pdf",
        file_size = 1024,
        file_format = DocumentFormat.PDF, 
        mime_type = "application/pdf",
        file_hash = "dummyhash123" 
    )

    assert doc.file_name == "example.pdf"
    assert doc.status == ProcessingStatus.UPLOADED 
    assert doc.total_chunks == 0 
    assert isinstance(doc.uploaded_at, datetime) 

def test_add_chunk_to_document():
    doc = Document(
        file_name = "example.pdf",
        original_path = "data/test_data/example.pdf",
        file_size = 1024,
        file_format = DocumentFormat.PDF, 
        mime_type = "application/pdf",
        file_hash = "dummyhash123" 
    )

    chunk = DocumentChunk(
        document_id = doc.document_id,
        chunk_index = 0,
        content = "This is a test chunk" 
    ) 

    doc.add_chunk(chunk = chunk) 

    assert doc.total_chunks == 1 
    assert doc.chunks[0].content == "This is a test chunk" 

def test_chunk_content_hash():
    chunk = DocumentChunk(
        document_id = "doc1",
        chunk_index = 0,
        content = "hash this content" 
    )

    content_hash = chunk.get_content_hash()
    assert isinstance(content_hash, str) 
    assert len(content_hash) == 64

def test_document_status_update():
    doc = Document(
        file_name = "example.pdf",
        original_path = "data/test_data/example.pdf",
        file_size = 1024,
        file_format = DocumentFormat.PDF, 
        mime_type = "application/pdf",
        file_hash = "dummyhash123" 
    )

    doc.update_status(ProcessingStatus.PARSING)
    assert doc.status == ProcessingStatus.PARSING
    assert doc.processed_at is not None 

def test_document_metadata_defaults():
    metadata = DocumentMetadata()

    assert metadata.keywords == []
    assert metadata.pdf_metadata == {}
    assert metadata.custom_fields == {} 

def test_processing_option_validation():
    opts = ProcessingOptions(
        chunk_size = 500, 
        chunk_overlap = 100
    )

    assert opts.chunk_size == 500 
    assert opts.chunk_overlap == 100 

    with pytest.raises(ValueError):
        ProcessingOptions(chunk_size = 500, chunk_overlap = 600) 

def test_document_collection_add_remove():
    doc = Document(
        file_name = "example.md",
        original_path = "data/test_data/example.md",
        file_size = 512, 
        file_format = DocumentFormat.MARKDOWN,
        mime_type = "text/markdown",
        file_hash = "123" 
    )

    collection = DocumentCollection(name = "test-collection")
    collection.add_document(doc) 

    assert collection.total_documents == 1
    assert collection.total_size == 512 

    removed = collection.remove_document(doc.document_id)

    assert removed 
    assert collection.total_documents == 0 

def test_document_get_summary():
    doc = Document(
        file_name = "test.csv",
        original_path = "data/test_data/test.csv",
        file_size = 2048, 
        file_format = DocumentFormat.CSV,
        mime_type = "text/csv",
        file_hash = "123" 
    )

    summary = doc.get_summary()
    assert isinstance(summary, dict)
    assert summary["file_format"] == DocumentFormat.CSV 

def test_collection_filtering_by_format_and_status():
    doc1 = Document(
        file_name="example.pdf",
        original_path="data/test_data/example.pdf",
        file_size=100,
        file_format=DocumentFormat.PDF,
        mime_type="application/pdf",
        file_hash="1"
    )
    
    doc2 = Document(
        file_name="test.csv",
        original_path="data/test_data/test.csv",
        file_size=200,
        file_format=DocumentFormat.CSV,
        mime_type="text/csv",
        file_hash="2"
    )
    
    doc2.update_status(ProcessingStatus.INDEXED)

    collection = DocumentCollection(name="Test")
    collection.add_document(doc1)
    collection.add_document(doc2)
    
    pdf_docs = collection.get_documents_by_format(DocumentFormat.PDF)
    assert len(pdf_docs) == 1
    
    indexed_docs = collection.get_documents_by_status(ProcessingStatus.INDEXED)
    assert len(indexed_docs) == 1
    assert indexed_docs[0].file_format == DocumentFormat.CSV