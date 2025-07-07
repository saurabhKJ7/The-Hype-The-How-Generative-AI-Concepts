from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import List, Dict
import shutil
from .services.document_processor import DocumentProcessor
from .services.embedding_service import EmbeddingService
from .services.search_service import SearchService
from .models.document import Document, SearchQuery, MultiMethodSearchResults

app = FastAPI(title="Indian Legal Document Search System")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
doc_processor = DocumentProcessor()
embedding_service = EmbeddingService()
search_service = SearchService()

# In-memory storage (replace with proper database in production)
documents: Dict[str, Document] = {}

# Create data directory if it doesn't exist
os.makedirs("data/uploads", exist_ok=True)

@app.post("/upload")
async def upload_document(file: UploadFile, doc_type: str):
    """Upload and process a legal document."""
    try:
        # Save file
        file_path = f"data/uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Process document
        document = doc_processor.process_file(file_path, doc_type)
        
        # Get embeddings for chunks
        document.chunks = embedding_service.embed_chunks(document.chunks)
        
        # Store document
        documents[document.id] = document
        
        return {"message": "Document processed successfully", "document_id": document.id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search(query: SearchQuery) -> MultiMethodSearchResults:
    """Search across documents using multiple similarity methods."""
    try:
        # Get all chunks from relevant documents
        all_chunks = []
        for doc in documents.values():
            if not query.doc_type or doc.doc_type == query.doc_type:
                all_chunks.extend(doc.chunks)
                
        if not all_chunks:
            raise HTTPException(status_code=404, detail="No documents found")
            
        # Get query embedding
        query_embedding = embedding_service.embed_query(query.query)
        if not query_embedding:
            raise HTTPException(status_code=500, detail="Failed to get query embedding")
            
        # Perform multi-method search
        results = search_service.multi_search(
            query_embedding=query_embedding,
            chunks=all_chunks,
            top_k=query.top_k
        )
        results.query = query.query
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents():
    """List all uploaded documents."""
    return {
        "documents": [
            {
                "id": doc.id,
                "title": doc.title,
                "doc_type": doc.doc_type,
                "upload_date": doc.upload_date
            }
            for doc in documents.values()
        ]
    }

@app.get("/document/{document_id}")
async def get_document(document_id: str):
    """Get document details by ID."""
    if document_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found")
    doc = documents[document_id]
    return {
        "id": doc.id,
        "title": doc.title,
        "doc_type": doc.doc_type,
        "upload_date": doc.upload_date,
        "chunk_count": len(doc.chunks)
    } 