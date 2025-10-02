# RAG Agentic Chat System

## Introduction
A modular Retrieval-Augmented Generation (RAG) system for document ingestion, semantic search, and agentic chat, built with FastAPI and ChromaDB.

## Features
- **Document Ingestion**: Add and chunk documents for semantic search.
- **Semantic & Hybrid Search**: Query documents using multiple retrieval strategies (similarity, MMR, hybrid).
- **Agentic Coordination**: Modular agents for retrieval, ingestion, and coordination.
- **Vector Store**: Uses ChromaDB for fast vector search and storage.
- **Chunking**: Supports paragraph, sentence, and word-based chunking.

## Coordinator API Routes

All user interaction is through the Coordinator. The Coordinator agent handles all ingestion, retrieval, and management by routing requests to the appropriate sub-agents. Use only the following endpoints:

### Ingestion (via Coordinator)
- `POST /ingestion/upload` — Upload a document for ingestion
- `GET /ingestion/documents/{user_id}` — List all documents for a user
- `GET /ingestion/document/{document_id}` — Get document details and chunks
- `GET /ingestion/document/{document_id}/status` — Get processing status for a document
- `DELETE /ingestion/document/{document_id}` — Delete a document
- `GET /ingestion/ingestion/stats` — Get ingestion statistics
- `GET /ingestion/debug/documents` — Debug: List all active and stored documents

### Coordinator System
- `POST /coordinator/message` — Generic endpoint for sending messages to CoordinatorAgent
- `GET /coordinator/health` — Coordinator and agent health check
- `GET /coordinator/routes` — List all available built-in coordinator routes

> All other APIs are internal and not intended for direct use. Always interact with the system through the Coordinator endpoints above.

## Quickstart
1. **Install dependencies**
   ```bash
   make install-dependencies
   ```
2. **Run the API server**
   ```bash
   make run-api
   ```
3. **Access the API docs**
   - Open [http://localhost:8000/docs](http://localhost:8000/docs)

## Project Structure
- `src/agents/` — Agent logic (retrieval, ingestion, coordination)
- `src/api/` — FastAPI endpoints
- `src/models/` — Data models
- `src/vector_store/` — ChromaDB vector store integration
- `src/parsers/` — Document parsers
- `src/services/` — Service layer (not used anywhere in the codebase, just sits there)
