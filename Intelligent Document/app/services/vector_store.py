import os
from typing import List, Dict, Any
import numpy as np
import json
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

from app.models.document import DocumentChunk, SearchResult
from app.services.embeddings import EmbeddingService

class VectorStore:
    def __init__(self):
        self.store_path = Path(os.getenv("VECTOR_STORE_PATH", "./data/vector_store"))
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        self.vectors_path = self.store_path / "vectors.npy"
        self.metadata_path = self.store_path / "metadata.json"
        
        # Initialize or load vectors and metadata
        if self.vectors_path.exists() and self.metadata_path.exists():
            self.vectors = np.load(str(self.vectors_path))
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.vectors = np.array([]).reshape(0, 768)  # TF-IDF vector size
            self.metadata = []
        
        # Initialize the embedding service
        self.embedding_service = EmbeddingService()
    
    async def add_document(self, doc_id: str, chunks: List[DocumentChunk]) -> None:
        # Convert chunks to vectors
        new_vectors = []
        new_metadata = []
        
        for chunk in chunks:
            # Generate embedding if not already present
            if not chunk.embedding:
                chunk.embedding = await self.embedding_service.embed_text(chunk.content)
            
            new_vectors.append(chunk.embedding)
            new_metadata.append({
                "doc_id": doc_id,
                "chunk_index": chunk.chunk_index,
                "chunk_type": chunk.chunk_type,
                "content": chunk.content,
                "metadata": chunk.metadata
            })
        
        if new_vectors:
            # Add to vectors array
            new_vectors_array = np.array(new_vectors)
            self.vectors = np.vstack([self.vectors, new_vectors_array]) if self.vectors.size > 0 else new_vectors_array
            
            # Update metadata
            start_idx = len(self.metadata)
            for i, meta in enumerate(new_metadata):
                meta["index"] = start_idx + i
                self.metadata.append(meta)
            
            # Save to disk
            self._save_store()
    
    async def search(self, query: str, limit: int = 5) -> List[SearchResult]:
        # Get query embedding
        query_embedding = await self.embedding_service.embed_text(query)
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], self.vectors)[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[-limit:][::-1]
        
        # Convert to SearchResult objects
        results = []
        for i, idx in enumerate(top_indices):
            if idx < len(self.metadata):
                meta = self.metadata[idx]
                chunk = DocumentChunk(
                    content=meta["content"],
                    chunk_type=meta["chunk_type"],
                    chunk_index=meta["chunk_index"],
                    metadata=meta["metadata"]
                )
                
                results.append(SearchResult(
                    chunk=chunk,
                    score=float(similarities[idx]),
                    document_id=meta["doc_id"],
                    document_metadata=meta["metadata"]
                ))
        
        return results
    
    def _save_store(self) -> None:
        # Save vectors
        np.save(str(self.vectors_path), self.vectors)
        
        # Save metadata
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f) 