from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime

class DocumentChunk(BaseModel):
    text: str
    metadata: Dict
    embedding: Optional[List[float]] = None
    legal_entities: Optional[List[str]] = None

class Document(BaseModel):
    id: str
    title: str
    content: str
    doc_type: str  # e.g., "income_tax", "gst", "court_judgment", "property"
    chunks: List[DocumentChunk] = []
    upload_date: datetime = datetime.now()
    file_path: str
    
class SearchQuery(BaseModel):
    query: str
    doc_type: Optional[str] = None
    top_k: int = 5
    
class SearchResult(BaseModel):
    chunk: DocumentChunk
    score: float
    method: str
    source_doc: str
    
class MultiMethodSearchResults(BaseModel):
    query: str
    cosine_results: List[SearchResult]
    euclidean_results: List[SearchResult]
    mmr_results: List[SearchResult]
    hybrid_results: List[SearchResult]
    metrics: Dict[str, Dict[str, float]]  # Method -> Metric -> Score 