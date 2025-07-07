from openai import OpenAI
import os
from dotenv import load_dotenv
from typing import List, Dict
import numpy as np
from ..models.document import DocumentChunk

load_dotenv()

class EmbeddingService:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "text-embedding-3-small"
        
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts using OpenAI's API."""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            return []
            
    def embed_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Add embeddings to document chunks."""
        texts = [chunk.text for chunk in chunks]
        embeddings = self.get_embeddings(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
            
        return chunks
        
    def embed_query(self, query: str) -> List[float]:
        """Get embedding for a search query."""
        embeddings = self.get_embeddings([query])
        return embeddings[0] if embeddings else [] 