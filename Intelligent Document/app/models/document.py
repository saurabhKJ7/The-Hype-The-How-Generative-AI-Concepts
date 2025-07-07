from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime

class DocumentType(str, Enum):
    TECHNICAL = "technical"
    API_REFERENCE = "api_reference"
    SUPPORT_TICKET = "support_ticket"
    POLICY = "policy"
    TUTORIAL = "tutorial"
    UNKNOWN = "unknown"

class DocumentChunk(BaseModel):
    content: str
    metadata: Dict = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    chunk_type: str = "text"  # text, code, policy_clause, etc.
    chunk_index: int
    parent_section: Optional[str] = None

class ProcessedDocument(BaseModel):
    id: str
    filename: str
    doc_type: DocumentType
    chunks: List[DocumentChunk]
    metadata: Dict = Field(default_factory=dict)
    processed_at: datetime = Field(default_factory=datetime.utcnow)
    total_chunks: int
    embedding_model: str
    chunking_strategy: str

class SearchResult(BaseModel):
    chunk: DocumentChunk
    score: float
    document_id: str
    document_metadata: Dict 