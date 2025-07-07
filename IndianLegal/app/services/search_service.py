import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from ..models.document import DocumentChunk, SearchResult, MultiMethodSearchResults

class SearchService:
    def __init__(self):
        self.methods = {
            "cosine": self._cosine_search,
            "euclidean": self._euclidean_search,
            "mmr": self._mmr_search,
            "hybrid": self._hybrid_search
        }
    
    def multi_search(self, query_embedding: List[float], chunks: List[DocumentChunk], 
                    top_k: int = 5) -> MultiMethodSearchResults:
        """Perform search using all methods and return combined results."""
        query_embedding = np.array(query_embedding).reshape(1, -1)
        chunk_embeddings = np.array([chunk.embedding for chunk in chunks])
        
        results = MultiMethodSearchResults(
            query="",  # Will be set by the API
            cosine_results=self._cosine_search(query_embedding, chunks, chunk_embeddings, top_k),
            euclidean_results=self._euclidean_search(query_embedding, chunks, chunk_embeddings, top_k),
            mmr_results=self._mmr_search(query_embedding, chunks, chunk_embeddings, top_k),
            hybrid_results=self._hybrid_search(query_embedding, chunks, chunk_embeddings, top_k),
            metrics=self._calculate_metrics(chunks, top_k)
        )
        return results
    
    def _cosine_search(self, query_embedding: np.ndarray, chunks: List[DocumentChunk], 
                      chunk_embeddings: np.ndarray, top_k: int) -> List[SearchResult]:
        """Search using cosine similarity."""
        similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append(SearchResult(
                chunk=chunks[idx],
                score=float(similarities[idx]),
                method="cosine",
                source_doc=chunks[idx].metadata.get("source", "unknown")
            ))
        return results
    
    def _euclidean_search(self, query_embedding: np.ndarray, chunks: List[DocumentChunk], 
                         chunk_embeddings: np.ndarray, top_k: int) -> List[SearchResult]:
        """Search using Euclidean distance."""
        distances = euclidean_distances(query_embedding, chunk_embeddings)[0]
        # Convert distances to similarities (smaller distance = higher similarity)
        similarities = 1 / (1 + distances)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append(SearchResult(
                chunk=chunks[idx],
                score=float(similarities[idx]),
                method="euclidean",
                source_doc=chunks[idx].metadata.get("source", "unknown")
            ))
        return results
    
    def _mmr_search(self, query_embedding: np.ndarray, chunks: List[DocumentChunk], 
                    chunk_embeddings: np.ndarray, top_k: int, lambda_param: float = 0.5) -> List[SearchResult]:
        """Search using Maximal Marginal Relevance."""
        similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
        selected_indices = []
        unselected_indices = list(range(len(chunks)))
        
        for _ in range(top_k):
            if not unselected_indices:
                break
                
            # Calculate MMR scores
            mmr_scores = []
            for idx in unselected_indices:
                if not selected_indices:
                    mmr_score = similarities[idx]
                else:
                    selected_embeddings = chunk_embeddings[selected_indices]
                    redundancy = np.max(cosine_similarity(
                        chunk_embeddings[idx].reshape(1, -1), 
                        selected_embeddings
                    ))
                    mmr_score = lambda_param * similarities[idx] - (1 - lambda_param) * redundancy
                mmr_scores.append((idx, mmr_score))
            
            # Select chunk with highest MMR score
            selected_idx, max_mmr_score = max(mmr_scores, key=lambda x: x[1])
            selected_indices.append(selected_idx)
            unselected_indices.remove(selected_idx)
        
        results = []
        for idx in selected_indices:
            results.append(SearchResult(
                chunk=chunks[idx],
                score=float(similarities[idx]),
                method="mmr",
                source_doc=chunks[idx].metadata.get("source", "unknown")
            ))
        return results
    
    def _hybrid_search(self, query_embedding: np.ndarray, chunks: List[DocumentChunk], 
                      chunk_embeddings: np.ndarray, top_k: int) -> List[SearchResult]:
        """Search using hybrid similarity (cosine + legal entity match)."""
        # Calculate cosine similarities
        cosine_scores = cosine_similarity(query_embedding, chunk_embeddings)[0]
        
        # Calculate legal entity overlap scores
        query_text = chunks[0].text  # Assuming first chunk contains query text
        entity_scores = np.zeros(len(chunks))
        for i, chunk in enumerate(chunks):
            if chunk.legal_entities and len(chunk.legal_entities) > 0:
                overlap = len(set(chunk.legal_entities))
                entity_scores[i] = overlap / len(chunk.legal_entities)
        
        # Combine scores (0.6 * cosine + 0.4 * entity)
        hybrid_scores = 0.6 * cosine_scores + 0.4 * entity_scores
        top_indices = np.argsort(hybrid_scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append(SearchResult(
                chunk=chunks[idx],
                score=float(hybrid_scores[idx]),
                method="hybrid",
                source_doc=chunks[idx].metadata.get("source", "unknown")
            ))
        return results
    
    def _calculate_metrics(self, chunks: List[DocumentChunk], top_k: int) -> Dict[str, Dict[str, float]]:
        """Calculate evaluation metrics for each method."""
        metrics = {}
        for method in self.methods.keys():
            metrics[method] = {
                "precision@5": 0.0,  # Will be calculated by the API based on user feedback
                "recall": 0.0,       # Will be calculated by the API based on user feedback
                "diversity": self._calculate_diversity_score(chunks[:top_k])
            }
        return metrics
    
    def _calculate_diversity_score(self, chunks: List[DocumentChunk]) -> float:
        """Calculate diversity score based on unique legal entities and content."""
        if not chunks:
            return 0.0
            
        # Count unique legal entities
        unique_entities = set()
        for chunk in chunks:
            if chunk.legal_entities:
                unique_entities.update(chunk.legal_entities)
                
        # Calculate diversity based on unique entities ratio
        total_entities = sum(len(chunk.legal_entities) if chunk.legal_entities else 0 for chunk in chunks)
        if total_entities == 0:
            return 0.0
            
        return len(unique_entities) / total_entities 