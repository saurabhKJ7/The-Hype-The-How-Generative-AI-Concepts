from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uvicorn

from app.core.document_processor import DocumentProcessor
from app.models.document import ProcessedDocument, DocumentType
from app.services.vector_store import VectorStore

app = FastAPI(
    title="Intelligent Document Chunker",
    description="Enterprise document processing pipeline with intelligent chunking",
    version="1.0.0"
)

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
vector_store = VectorStore()

@app.post("/documents/process", response_model=ProcessedDocument)
async def process_document(file: UploadFile = File(...)):
    """
    Process a document through the intelligent chunking pipeline.
    """
    content = await file.read()
    
    # Process the document
    processed_doc = await doc_processor.process(
        content=content,
        filename=file.filename,
        content_type=file.content_type
    )
    
    return processed_doc

@app.get("/documents/search")
async def search_documents(query: str, limit: int = 5):
    """
    Search through processed documents using semantic similarity.
    """
    results = await vector_store.search(query, limit=limit)
    return results

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 